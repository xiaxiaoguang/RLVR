"""Model: LlamaForCausalLM (RoPE, SwiGLU; ~85M non-embedding params by default) over the
char tokenizer; save/load; generation.

    model = build_model(cfg["model"], tok)
    save_model(model, tok, cfg, "checkpoints/pt-mid-s0")
    model, tok = load_model("checkpoints/pt-mid-s0")
    completions = generate(model, tok, prompts, temperature=1.0, num_return_sequences=32)

Checkpoints use save_pretrained (config.json + model.safetensors) plus our
tokenizer.json and the resolved run_config.yaml, so they also load in TRL.
Generation left-pads, prepends <bos> like training, and cuts each completion
right after its first <eos> (a completion without <eos> scores 0).
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
import yaml
from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM
from transformers.utils import logging as hf_logging

from .task import EOS
from .precision import model_autocast

hf_logging.disable_progress_bar()
from .tokenizer import CharTokenizer


def build_model(mcfg: dict, tok: CharTokenizer) -> LlamaForCausalLM:
    """mcfg: the `model` section of a resolved config."""
    assert mcfg["pos"] == "rope"
    config = LlamaConfig(
        vocab_size=tok.vocab_size,
        hidden_size=mcfg["d_model"],
        intermediate_size=mcfg["d_ff"],
        num_hidden_layers=mcfg["n_layer"],
        num_attention_heads=mcfg["n_head"],
        num_key_value_heads=mcfg["n_head"],
        max_position_embeddings=mcfg["max_len"],
        bos_token_id=tok.bos_id, eos_token_id=tok.eos_id, pad_token_id=tok.pad_id,
        tie_word_embeddings=False,
    )
    config._attn_implementation = mcfg.get("attn_impl", "sdpa")
    model = LlamaForCausalLM(config)
    model.generation_config = generation_config(tok)
    return model


def generation_config(tok: CharTokenizer) -> GenerationConfig:
    # top_k=0 / top_p=1: sample from the full distribution (HF defaults to top_k=50).
    # generate() always passes do_sample explicitly; True here only keeps the config valid.
    return GenerationConfig(bos_token_id=tok.bos_id, eos_token_id=tok.eos_id, pad_token_id=tok.pad_id,
                            do_sample=True, top_k=0, top_p=1.0)


def n_params(model, embeddings: bool = True) -> int:
    n = sum(p.numel() for p in model.parameters())
    if not embeddings:
        n -= model.get_input_embeddings().weight.numel() + model.get_output_embeddings().weight.numel()
    return n


def save_model(model, tok: CharTokenizer, cfg: dict | None, path: str | Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if model.config.vocab_size != tok.vocab_size:
        raise ValueError("model and tokenizer vocabulary sizes differ")
    if cfg is not None:
        from .task import make_task
        task = make_task(cfg["task"])
        if tok.vocab != task.tokenizer().vocab:
            raise ValueError("model tokenizer does not match task config")
        if hasattr(task, "fingerprint"):
            model.config.task_fingerprint = task.fingerprint
    model.save_pretrained(path)
    tok.save(path / "tokenizer.json")
    if cfg is not None:
        (path / "run_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def load_model(path: str | Path, device: str | torch.device = "cuda") -> tuple[LlamaForCausalLM, CharTokenizer]:
    path = Path(path)
    tok = CharTokenizer.load(path / "tokenizer.json")
    model = LlamaForCausalLM.from_pretrained(path).to(device).eval()
    model.generation_config = generation_config(tok)
    if (path / "run_config.yaml").exists():
        from .task import make_task
        cfg = yaml.safe_load((path / "run_config.yaml").read_text())
        task = make_task(cfg["task"])
        if tok.vocab != task.tokenizer().vocab or model.config.vocab_size != tok.vocab_size:
            raise ValueError("checkpoint tokenizer/task vocabulary mismatch")
        if hasattr(task, "fingerprint") and getattr(model.config, "task_fingerprint", None) != task.fingerprint:
            raise ValueError("checkpoint task fingerprint mismatch")
    return model, tok


def env_info(device: str | torch.device) -> dict:
    """{torch, gpu, gpu_mem_gb} of the machine a run is on (gpu fields None off CUDA)."""
    device = torch.device(device)
    cuda = device.type == "cuda"
    return {"torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(device) if cuda else None,
            "gpu_mem_gb": torch.cuda.get_device_properties(device).total_memory / 2**30 if cuda else None}


def load_run_config(path: str | Path) -> dict:
    return yaml.safe_load((Path(path) / "run_config.yaml").read_text())


def left_pad(tok: CharTokenizer, prompts: list[str], device) -> tuple[torch.Tensor, torch.Tensor]:
    ids = [tok.encode(p, add_bos=True) for p in prompts]
    n = max(map(len, ids))
    input_ids = torch.full((len(ids), n), tok.pad_id, dtype=torch.long)
    mask = torch.zeros((len(ids), n), dtype=torch.long)
    for i, x in enumerate(ids):
        input_ids[i, n - len(x):] = torch.tensor(x)
        mask[i, n - len(x):] = 1
    return input_ids.to(device), mask.to(device)


def cut_at_eos(text: str) -> str:
    i = text.find(EOS)
    return text if i < 0 else text[:i + len(EOS)]


def checkpoint_ids(ckpt: str | Path) -> tuple[str, str, int | None]:
    """(model_id, run_name, step): checkpoints/pt-mid-s0 -> (pt-mid-s0, pt-mid-s0, None);
    runs/pt-mid-s0/ckpt/step-3000 -> (pt-mid-s0-step-3000, pt-mid-s0, 3000)."""
    ckpt = Path(ckpt)
    if ckpt.parent.name == "ckpt":
        run = ckpt.parent.parent.name
        return f"{run}-{ckpt.name}", run, int(ckpt.name.split("-")[-1])
    return ckpt.name, ckpt.name, None


@torch.no_grad()
def generate_ids(model, tok: CharTokenizer, prompts: list[str], max_new_tokens: int = 128,
                 temperature: float = 0.0, num_return_sequences: int = 1,
                 batch_size: int = 1024, progress: str = "") -> list[list[int]]:
    """Sampled completion token ids (prompt excluded), grouped by prompt: out[i*n + j] is
    sample j of prompt i. Each row ends with its first <eos>; a row without <eos> has all
    max_new_tokens tokens. temperature=0 means greedy. batch_size counts prompts.
    progress: a label; if set, prints prompts done, rate and ETA after the first batch and
    then at most every 10 s."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    out: list[list[int]] = []
    t0 = t_log = time.monotonic()
    n_tok = 0
    for s in range(0, len(prompts), batch_size):
        input_ids, mask = left_pad(tok, prompts[s:s + batch_size], device)
        kw = dict(do_sample=True, temperature=temperature) if temperature > 0 else dict(do_sample=False)
        with model_autocast(device):
            seqs = model.generate(input_ids=input_ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                                  num_return_sequences=num_return_sequences, **kw)
        new = seqs[:, input_ids.shape[1]:]
        is_eos = new == tok.eos_id
        n = torch.where(is_eos.any(1), is_eos.int().argmax(1) + 1, new.shape[1])
        out.extend(row[:k] for row, k in zip(new.tolist(), n.tolist()))
        if progress:
            n_tok += new.numel()
            done, now = min(s + batch_size, len(prompts)), time.monotonic()
            if s == 0 or now - t_log >= 10 or done == len(prompts):
                t_log = now
                print(f"[{progress}] {done}/{len(prompts)} prompts x {num_return_sequences}, "
                      f"{done / (now - t0):.1f} prompts/s, {n_tok / (now - t0):.0f} tok/s, "
                      f"{now - t0:.0f}s, eta {(len(prompts) - done) * (now - t0) / done:.0f}s", flush=True)
    model.train(was_training)
    return out


def generate(model, tok: CharTokenizer, prompts: list[str], max_new_tokens: int = 128,
             temperature: float = 0.0, num_return_sequences: int = 1,
             batch_size: int = 1024, progress: str = "") -> list[str]:
    """Completions as text (see generate_ids); <pad> tokens are dropped."""
    return [tok.decode(ids) for ids in generate_ids(model, tok, prompts, max_new_tokens, temperature,
                                                    num_return_sequences, batch_size, progress)]
