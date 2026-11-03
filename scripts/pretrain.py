"""Pretrain one base model on the streamed corpus (rlvr.stream), single GPU or DDP.

    torchrun --standalone --nproc_per_node=4 scripts/pretrain.py --config configs/pretrain/mid.yaml [key=value ...]
    python scripts/pretrain.py --config configs/pretrain/mid.yaml           # one GPU
    torchrun ... scripts/pretrain.py --config ... --bench 200               # throughput -> suggested steps

The eval sets (scripts/make_eval_sets.py) must exist: their pids are excluded
from the stream and they are scored during training. Next-token cross-entropy on
the target only. Each micro-batch holds >= pretrain.micro_target_tokens
supervised tokens; the loss is the token sum divided by that budget, so every
supervised token weighs the same in every level. AdamW, linear warmup, cosine
decay to min_lr_frac. Greedy eval runs on rank 0 while the others wait.

Output:
  runs/pt-{level}-s{seed}/run_config.yaml   resolved config + git commit + world size
  runs/pt-{level}-s{seed}/metrics.jsonl     step, loss, loss_d{1..}, lr, grad_norm, tok_per_s, mfu, tokens ...
  runs/pt-{level}-s{seed}/eval.jsonl        step, eval_set, depth, group, greedy_acc, format_err, ...
  runs/pt-{level}-s{seed}/stats.json        examples / supervised tokens by kind, depth, group, function; rejections
  runs/pt-{level}-s{seed}/ckpt/step-{N}/    checkpoints at pretrain.save_at (P9)
  runs/pt-{level}-s{seed}/bench.json        --bench only: throughput and suggested pretrain.steps
  checkpoints/pt-{level}-s{seed}/           final model (save_pretrained + tokenizer + run_config)
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402

from rlvr.config import ROOT, load_config, resolve  # noqa: E402
from rlvr.evaluate import greedy_eval  # noqa: E402
from rlvr.io import load_problems, read_jsonl  # noqa: E402
from rlvr.model import build_model, n_params, save_model  # noqa: E402
from rlvr.stream import GROUP_IDS, KINDS, REJECT_KINDS, PretrainStream  # noqa: E402
from rlvr.task import Task  # noqa: E402
from rlvr.tokenizer import IGNORE_INDEX, CharTokenizer  # noqa: E402

A100_BF16_FLOPS = 312e12


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                        cwd=ROOT, text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def lr_lambda(warmup: int, total: int, min_frac: float):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        t = min(1.0, (step - warmup) / max(1, total - warmup))
        return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * t))
    return f


def token_losses(model, batch) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token CE [B, T-1] (0 where masked) and the validity mask."""
    logits = model(input_ids=batch["input_ids"], position_ids=batch["position_ids"]).logits
    labels = batch["labels"][:, 1:]
    valid = labels != IGNORE_INDEX
    ce = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), labels.flatten().clamp(min=0),
                         reduction="none").view_as(labels)
    return ce * valid, valid


class Stats:
    """Corpus statistics accumulated from batch metadata (doc: 在线统计)."""

    def __init__(self, n_funcs: int):
        self.examples, self.target_tokens = Counter(), Counter()
        self.func_counts = torch.zeros(n_funcs, dtype=torch.long)
        self.rejects = torch.zeros(len(REJECT_KINDS), dtype=torch.long)
        self.noisy = 0
        self.tokens = self.padded_tokens = 0

    def add(self, b: dict) -> None:
        groups = {v: k for k, v in GROUP_IDS.items()}
        for k, d, g, n in zip(b["kind"].tolist(), b["depth"].tolist(), b["group"].tolist(), b["n_target"].tolist()):
            key = f"{KINDS[k]}|d{d}|{groups[g]}"
            self.examples[key] += 1
            self.target_tokens[key] += n
        self.func_counts += b["func_counts"]
        self.rejects += b["rejects"]
        self.noisy += int(b["noisy"].sum())
        self.tokens += int((b["input_ids"] != 0).sum())
        self.padded_tokens += b["input_ids"].numel()

    def merge(self, others: list["Stats"]) -> dict:
        ex, tt = Counter(), Counter()
        funcs, rej = torch.zeros_like(self.func_counts), torch.zeros_like(self.rejects)
        noisy = tokens = padded = 0
        for s in others:
            ex.update(s.examples)
            tt.update(s.target_tokens)
            funcs += s.func_counts
            rej += s.rejects
            noisy, tokens, padded = noisy + s.noisy, tokens + s.tokens, padded + s.padded_tokens
        total_tgt = sum(tt.values())
        out = {"examples": sum(ex.values()), "target_tokens": total_tgt, "tokens": tokens,
               "pad_frac": 1 - tokens / max(1, padded), "noisy_examples": noisy,
               "rejects": dict(zip(REJECT_KINDS, rej.tolist()))}
        for i, name in ((0, "kind"), (1, "depth"), (2, "group")):
            agg_ex, agg_tt = Counter(), Counter()
            for key in tt:
                part = key.split("|")[i]
                agg_ex[part] += ex[key]
                agg_tt[part] += tt[key]
            out[f"by_{name}"] = {k: {"examples": agg_ex[k], "target_tokens": agg_tt[k],
                                     "target_token_frac": agg_tt[k] / max(1, total_tgt)} for k in sorted(agg_ex)}
        out["by_cell"] = {k: {"examples": ex[k], "target_tokens": tt[k]} for k in sorted(tt)}
        out["comp_func_exposure"] = funcs.tolist()
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pretrain/mid.yaml")
    ap.add_argument("--bench", type=int, default=0, help="run N steps, report throughput and suggested steps, exit")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing run / final checkpoint")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    ddp = "WORLD_SIZE" in os.environ
    if ddp:
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30))
        rank, world, local_rank = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local_rank = 0, 1, 0
    master = rank == 0
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = load_config(args.config, args.overrides)
    pc, seed = cfg["pretrain"], cfg["seed"]
    run_name = f"pt-{pc['level']}-s{seed}"
    run_dir = resolve(cfg["paths"]["runs"]) / run_name
    final_dir = resolve(cfg["paths"]["checkpoints"]) / run_name
    eval_dir = resolve(cfg["paths"]["data"]) / "eval" / cfg["eval"]["version"]
    if not eval_dir.exists():
        raise SystemExit(f"{eval_dir} missing; run scripts/make_eval_sets.py first")
    existing = [d for d in (final_dir, run_dir / "metrics.jsonl") if d.exists()]
    if existing and not args.bench and not args.overwrite:
        raise SystemExit(f"{run_name} already exists ({', '.join(map(str, existing))}); "
                         "move it away or pass --overwrite")
    run_cfg = {**cfg, "git_commit": git_commit()}
    if master:
        run_dir.mkdir(parents=True, exist_ok=True)
        if not args.bench:
            (run_dir / "run_config.yaml").write_text(
                yaml.safe_dump({**run_cfg, "world_size": world}, sort_keys=False))

    accum = max(1, round(pc["target_tokens_per_step"] / (pc["micro_target_tokens"] * world)))
    steps = args.bench or pc["steps"]
    torch.manual_seed(seed)
    task = Task(cfg["task"])
    tok = CharTokenizer(task.alphabet)
    raw = build_model(cfg["model"], tok).to(device)
    n_nonemb = n_params(raw, embeddings=False)
    if pc["compile"]:
        # Per decoder layer: compiling the whole model also lowers RoPE setup to a CPU
        # kernel, which needs a C++20 host compiler (RHEL 8's g++ 8.5 is too old).
        # The layers share one code object, so raise the per-code recompile limit (default 8).
        torch._dynamo.config.recompile_limit = 8 * len(raw.model.layers)
        for layer in raw.model.layers:
            layer.compile(dynamic=True)
    model = raw
    if ddp:
        model = DDP(model, device_ids=[local_rank])
    if master:
        print(f"{run_name}: {n_params(raw) / 1e6:.1f}M params ({n_nonemb / 1e6:.1f}M non-embedding), "
              f"world {world}, accum {accum}, ~{accum * world * pc['micro_target_tokens'] / 1e3:.0f}k "
              f"supervised tokens/step, {steps} steps", flush=True)

    eval_pids = frozenset(r["pid"] for f in sorted(eval_dir.glob("*.jsonl")) for r in read_jsonl(f))
    loader = torch.utils.data.DataLoader(
        PretrainStream(cfg, rank, eval_pids), batch_size=None, num_workers=pc["num_workers"],
        pin_memory=True, persistent_workers=pc["num_workers"] > 0,
        prefetch_factor=8 if pc["num_workers"] > 0 else None)
    batches = iter(loader)
    eval_sets = {name: load_problems(task, eval_dir / f"{name}.jsonl") for name in pc["eval_sets"]} if master else {}

    decay = [p for p in raw.parameters() if p.dim() >= 2]
    no_decay = [p for p in raw.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": pc["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=pc["lr"], betas=(0.9, 0.95), fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda(max(1, int(pc["warmup_frac"] * steps)), steps, pc["min_lr_frac"]))
    save_steps = {max(1, round(f * steps)) for f in pc["save_at"]} if not args.bench else set()

    stats = Stats(task.n_funcs)
    D = task.max_depth + 1
    metrics_f = (run_dir / "metrics.jsonl").open("w") if master and not args.bench else None
    eval_f = (run_dir / "eval.jsonl").open("w") if master and not args.bench else None

    def evaluate(step):
        if master:
            t = time.time()
            rows = greedy_eval(raw, tok, task, eval_sets, step, cfg["eval"]["max_new_tokens"],
                               cfg["eval"]["batch_size"])
            for r in rows:
                eval_f.write(json.dumps(r) + "\n")
            eval_f.flush()
            summary = {r["eval_set"]: round(r["greedy_acc"], 3) for r in rows if r["depth"] == "all"}
            print(f"[eval step {step}] {summary} ({time.time() - t:.0f}s)", flush=True)
            raw.train()
        if ddp:
            dist.barrier()

    def save(path):
        if master:
            save_model(raw, tok, {**run_cfg, "world_size": world}, path)
        if ddp:
            dist.barrier()

    model.train()
    loss_sum = torch.zeros(D, device=device)
    tok_cnt = torch.zeros(D, device=device)
    t0, tok_win, tgt_win, total_tgt, total_tok = time.time(), 0, 0, 0, 0
    bench_t0, bench_tok, bench_skip = None, 0, min(20, steps // 4)
    for step in range(1, steps + 1):
        for micro in range(accum):
            b = next(batches)
            stats.add(b)
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()
                 if k in ("input_ids", "labels", "position_ids", "depth")}
            sync = nullcontext() if (not ddp or micro == accum - 1) else model.no_sync()
            with sync:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ce, valid = token_losses(model, b)
                loss = ce.sum() / (pc["micro_target_tokens"] * accum)
                loss.backward()
            with torch.no_grad():
                loss_sum.index_add_(0, b["depth"], ce.sum(1).detach())
                tok_cnt.index_add_(0, b["depth"], valid.sum(1).float())
            n_tok = int((b["input_ids"] != tok.pad_id).sum())
            n_tgt = int(valid.sum())
            tok_win, tgt_win = tok_win + n_tok, tgt_win + n_tgt
            if bench_t0 is not None:
                bench_tok += n_tok
        gn = torch.nn.utils.clip_grad_norm_(raw.parameters(), pc["grad_clip"])
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)

        if args.bench and step == bench_skip:
            torch.cuda.synchronize()
            bench_t0, bench_tok = time.time(), 0

        if not args.bench and (step % pc["log_every"] == 0 or step == steps):
            win = torch.tensor([tok_win, tgt_win], device=device, dtype=torch.float64)
            if ddp:
                dist.all_reduce(loss_sum)
                dist.all_reduce(tok_cnt)
                dist.all_reduce(win)
            dt = time.time() - t0
            total_tok += int(win[0])
            total_tgt += int(win[1])
            tps = float(win[0]) / dt
            m = {"step": step, "loss": float(loss_sum.sum() / tok_cnt.sum().clamp(min=1)),
                 **{f"loss_d{d}": float(loss_sum[d] / tok_cnt[d]) for d in range(1, D) if tok_cnt[d] > 0},
                 "lr": sched.get_last_lr()[0], "grad_norm": float(gn), "tok_per_s": tps,
                 "target_tok_per_s": float(win[1]) / dt,
                 "mfu": 6 * n_nonemb * tps / (A100_BF16_FLOPS * world),
                 "tokens": total_tok, "target_tokens": total_tgt}
            if master:
                metrics_f.write(json.dumps(m) + "\n")
                metrics_f.flush()
                if step % (pc["log_every"] * 10) == 0 or step == steps:
                    eta = (steps - step) * dt / pc["log_every"] / 60
                    print(f"step {step}/{steps} loss {m['loss']:.4f} "
                          + " ".join(f"d{d} {m[f'loss_d{d}']:.3f}" for d in range(1, D) if f"loss_d{d}" in m)
                          + f" lr {m['lr']:.2e} gn {m['grad_norm']:.2f} {tps / 1e6:.2f}M tok/s "
                          f"mfu {m['mfu']:.2f} eta {eta:.0f}m", flush=True)
            loss_sum.zero_()
            tok_cnt.zero_()
            t0, tok_win, tgt_win = time.time(), 0, 0

        if not args.bench:
            if step % pc["eval_every"] == 0 or step == steps:
                evaluate(step)
                t0 = time.time()
            if step in save_steps:
                save(run_dir / "ckpt" / f"step-{step}")
                t0 = time.time()

    if args.bench:
        torch.cuda.synchronize()
        dt = time.time() - bench_t0
        tok_s = torch.tensor([bench_tok / dt], device=device, dtype=torch.float64)
        if ddp:
            dist.all_reduce(tok_s)
        sec_per_step = dt / (steps - bench_skip)
        suggested = math.ceil(pc["target_minutes"] * 60 / sec_per_step)
        res = {"world_size": world, "accum": accum, "sec_per_step": sec_per_step, "tok_per_s": float(tok_s),
               "mfu": 6 * n_nonemb * float(tok_s) / (A100_BF16_FLOPS * world),
               "target_minutes": pc["target_minutes"], "suggested_steps": suggested,
               "suggested_tokens": float(tok_s) * pc["target_minutes"] * 60}
        if master:
            (run_dir / "bench.json").write_text(json.dumps(res, indent=1))
            print(json.dumps(res, indent=1))
            print(f"-> set pretrain.steps={suggested} for ~{pc['target_minutes']} min of training")
    else:
        all_stats = [None] * world
        if ddp:
            dist.all_gather_object(all_stats, stats)
        else:
            all_stats = [stats]
        if master:
            st = {"run_name": run_name, "world_size": world, "steps": steps, **stats.merge(all_stats)}
            st["comp_func_exposure"] = {task.fname(i): c for i, c in enumerate(st["comp_func_exposure"])}
            (run_dir / "stats.json").write_text(json.dumps(st, indent=1))
        save(final_dir)
        if master:
            print(f"saved {final_dir}", flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
