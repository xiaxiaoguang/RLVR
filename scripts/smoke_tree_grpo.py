"""Synthetic tree-v1 plumbing only: tiny random model, saved artifacts, one GRPO step.

No pretrained claims or reward-improvement criterion. A separate known-answer
pair checks that the real binary reward can produce finite, nonzero gradients.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import grpo as grpo_script
from rlvr.config import load_config, task_and_splits
from rlvr.grpo import group_advantages, grpo_update, pack, token_logprobs
from rlvr.io import problems_to_rows, read_jsonl, write_jsonl
from rlvr.model import build_model, generate_ids, load_model, save_model
from rlvr.precision import autocast_dtype


def run(output: Path, device: str = "cpu") -> dict:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite synthetic artifacts in {output}; choose a fresh --output")
    dev = torch.device(device)
    if dev.type == "cuda":
        if not os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ["CUDA_VISIBLE_DEVICES"] == "-1":
            raise SystemExit("No assigned CUDA_VISIBLE_DEVICES; do not select another GPU")
        if not torch.cuda.is_available():
            raise SystemExit("CUDA unavailable; host driver/allocation must be fixed before GPU validation")
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize(dev)
    start = time.perf_counter()
    torch.manual_seed(0)
    rng = random.Random(0)
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config("configs/tasks/tree_v1.yaml", [
        "model.n_layer=1", "model.d_model=32", "model.d_ff=64", "model.n_head=2",
        f"paths.data={output / 'data'}", f"paths.runs={output / 'runs'}",
        f"paths.checkpoints={output / 'checkpoints'}"])
    task, splits = task_and_splits(cfg)
    tok = task.tokenizer()
    data = [splits.sample("rl", d, rng) for d in (1, 2)]
    eval_probs = [splits.sample("eval", 2, rng)]
    ds = output / "data/datasets/synthetic.jsonl"
    write_jsonl(ds, problems_to_rows(task, data, origin="synthetic-random-smoke"))
    write_jsonl(output / "data/eval/v1/E2.jsonl", problems_to_rows(task, eval_probs, eval_set="E2"))
    model = build_model(cfg["model"], tok).to(dev).eval()
    base = output / "checkpoints/pt-synthetic-tree-s0"
    save_model(model, tok, cfg, base)
    loaded, loaded_tok = load_model(base, dev)
    prompt = task.format_prompt(data[1])
    assert generate_ids(model, tok, [prompt], 4) == generate_ids(loaded, loaded_tok, [prompt], 4)
    del loaded

    # Explicit correct/incorrect completions, scored by the unchanged task reward.
    # These are a unit probe, never substituted for sampled rollout rewards.
    correct = task.format_target(data[1])
    completions = [tok.encode(correct), tok.encode("A:aaaaaa<eos>")]
    if task.reward(tok.decode(completions[1]), data[1]) != 0:
        completions[1] = tok.encode("A:bbbbbb<eos>")
    rewards = torch.tensor([[task.reward(tok.decode(c), data[1]) for c in completions]])
    assert rewards.tolist() == [[1.0, 0.0]]
    prompts = [tok.encode(prompt, add_bos=True)] * 2
    ids, cm = pack(prompts, completions, tok.pad_id, dev)
    logp, mask = token_logprobs(model, ids, cm)
    assert int(mask.sum()) == sum(map(len, completions)) and (logp[~mask] == 0).all()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    stats = grpo_update(model, opt, prompts, completions, group_advantages(rewards).flatten(),
                        {**cfg["rl"], "micro_batch_tokens": 128}, tok.pad_id)
    assert all(torch.isfinite(torch.tensor(stats[k])) for k in ("loss", "grad_norm", "entropy"))
    assert stats["grad_norm"] > 0
    # Reload the untouched random base for the actual sampled-rollout script.
    del model, opt, logp
    # Inherit the tree checkpoint's 768-token completion budgets. Tiny B/G and
    # architecture do not change the task's conservative length requirements.
    args = ["--base", str(base), "--dataset", str(ds), "--device", str(dev),
            "rl.steps=1", "rl.prompts_per_step=2", "rl.group_size=2",
            "rl.eval_sets=[E2]", "rl.probe_set=''", "rl.kl_every=0",
            "rl.gen_batch_size=1", "rl.micro_batch_tokens=128", "eval.batch_size=1", "rl.tag=synthetic"]
    assert grpo_script.main(["--preflight", *args])["status"] == "preflight_passed"
    assert not (output / "runs").exists(), "preflight must not create training artifacts"
    summary = grpo_script.main(args)
    run_dir = output / "runs" / summary["run_name"]
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    assert len(metrics) == 1 and metrics[0]["rollouts"] == 4
    assert all(torch.isfinite(torch.tensor(metrics[0][k])) for k in ("loss", "grad_norm", "entropy"))
    final, final_tok = load_model(run_dir / "ckpt/step-1", dev)
    assert final.config.task_fingerprint == task.fingerprint and final_tok.vocab == tok.vocab
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    result = {"label": "synthetic-random-smoke-only", "seed": 0, "device": str(dev),
              "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "precision": "bf16" if autocast_dtype(dev) else "fp32",
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "compiled_arches": torch.cuda.get_arch_list(), "task_fingerprint": task.fingerprint,
              "vocab_size": tok.vocab_size, "known_answer_probe": stats, "sampled_metrics": metrics[0],
              "wall_seconds": time.perf_counter() - start, "summary": str(run_dir / "summary.json"),
              "gpu_probe_seconds": None, "peak_cuda_bytes": None}
    if dev.type == "cuda":
        result.update(gpu_probe_seconds=result["wall_seconds"], peak_cuda_bytes=torch.cuda.max_memory_allocated(dev),
                      gpu=torch.cuda.get_device_name(dev), capability=torch.cuda.get_device_capability(dev))
    (output / "measurements.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_num_threads(1)
    run(args.output, args.device)
