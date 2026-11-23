"""GRPO on one base checkpoint and one RL dataset (single GPU).

    python scripts/grpo.py --base checkpoints/pt-weak-s0 --dataset mixed_500 [key=value ...]
    python scripts/grpo.py --base runs/pt-mid-s0/ckpt/step-2375 --dataset depth_3 rl.group_size=4 seed=1
    python scripts/grpo.py --config configs/rl/smoke.yaml --base checkpoints/pt-weak-s0 --dataset depth_5

Task, model, splits, paths and the eval version are taken from the base checkpoint's
run_config.yaml (changing them here is an error), so the same command works for chain-v1 and
tree-v1 checkpoints; everything else is defaults <- --config <- key=value (rlvr/params.py: rl.*, data.*).
The dataset is a name under <data>/datasets/ (p-based ones: <data>/datasets/{model_id}/, built
with scripts/build_dataset.py --model-id) or a .jsonl path. data.sampler=auto uses the sampler
of the dataset spec (online_filter / learnability), else static.

Each step: sample B problems, G completions each at rl.temperature, exact-match reward, group
advantages, one policy-gradient update (rlvr/grpo.py). Greedy eval of rl.eval_sets at step 0,
every rl.eval_every steps and at the end; rl.probe_set every rl.probe_every steps.

Output: runs/{run}/ with run = rl-{model_id}-{dataset}-G{G}-s{seed}[-{tag}]
  run_config.yaml   resolved config + git commit + env (torch version, GPU name and memory; not in ckpt/)
  metrics.jsonl     per step: mean_reward, signal_frac, entropy, mean_len, trunc_frac, grad_norm, loss,
                    lr, kl_to_base (every rl.kl_every), reward_d{depth}, sampler state, seconds, gpu_mem_gb (CUDA)
  rollouts.jsonl    per (step, problem): pid, depth, group, str_len, rewards[G], lens[G], p_online,
                    completions[G] (rl.log_completions)
  eval.jsonl        greedy_eval rows + `rollouts` (cumulative count, the x axis of the curve fits)
  eval_problems.jsonl  per (eval step, eval set, problem): step, eval_set, pid, correct, n_correct, err_type,
                    err_step (rl.log_eval_problems)
  summary.json      written last: first / final accuracy and step score per eval set, timing (minutes: whole run,
                    eval_minutes: part of it), peak GPU memory of training / evals (marks the run complete)
  ckpt/step-{N}/    rl.save_at fractions and, with rl.save_final, the last step
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Rollout batches change shape every step; without this the caching allocator fragments.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from rlvr.config import git_commit, load_config, resolve  # noqa: E402
from rlvr.evaluate import greedy_eval  # noqa: E402
from rlvr.grpo import group_advantages, grpo_update, signal_groups  # noqa: E402
from rlvr.io import load_problems, read_jsonl  # noqa: E402
from rlvr.model import checkpoint_ids, generate_ids, load_model, load_run_config, save_model  # noqa: E402
from rlvr.params import PARAMS, PARAMS_DEFAULTS, merge  # noqa: E402
from rlvr.sampler import ProblemSampler  # noqa: E402

FROZEN_SECTIONS = ("task", "model", "splits")


def rl_config(config: str | None, overrides: list[str]) -> dict:
    """Resolved RL config. The base checkpoint (rl.base) supplies task, model, splits, paths, the
    eval version and length budgets, and its pretrain section (provenance)."""
    user = load_config(config, overrides)
    if not user["rl"]["base"]:
        raise SystemExit("rl.base is not set: pass --base <checkpoint dir>")
    ck = load_run_config(resolve(user["rl"]["base"]))
    inherit = {k: ck[k] for k in (*FROZEN_SECTIONS, "paths") if k in ck}
    inherit["pretrain"] = {k: v for k, v in ck.get("pretrain", {}).items() if f"pretrain.{k}" in PARAMS}
    inherit["eval"] = {k: ck["eval"][k] for k in ("version", "max_new_tokens") if k in ck.get("eval", {})}
    if "max_new_tokens" in ck.get("rl", {}):
        inherit["rl"] = {"max_new_tokens": ck["rl"]["max_new_tokens"]}
    cfg = load_config(config, overrides, base=inherit)
    for k in FROZEN_SECTIONS:
        if k in ck and cfg[k] != merge(PARAMS_DEFAULTS, {k: ck[k]})[k]:
            raise SystemExit(f"config changes '{k}', which is fixed by the base checkpoint {user['rl']['base']}")
    return cfg


def dataset_path(cfg: dict, model_id: str) -> Path:
    """<data>/datasets/{model_id}/{name}.jsonl (p-based) before <data>/datasets/{name}.jsonl; or a path."""
    name = cfg["data"]["dataset"]
    if not name:
        raise SystemExit("data.dataset is not set: pass --dataset <name or .jsonl path>")
    if name.endswith(".jsonl"):
        return resolve(name)
    root = resolve(cfg["paths"]["data"]) / "datasets"
    for p in (root / model_id / f"{name}.jsonl", root / f"{name}.jsonl"):
        if p.exists():
            return p
    raise SystemExit(f"dataset '{name}' not found under {root} (p-based specs: "
                     f"python scripts/build_dataset.py configs/datasets/{name}.yaml --model-id {model_id})")


def run_name_for(cfg: dict) -> str:
    model_id = checkpoint_ids(resolve(cfg["rl"]["base"]))[0]
    name = f"rl-{model_id}-{Path(cfg['data']['dataset']).stem}-G{cfg['rl']['group_size']}-s{cfg['seed']}"
    return name + (f"-{cfg['rl']['tag']}" if cfg["rl"]["tag"] else "")


def sampler_kind(cfg: dict, ds_path: Path) -> str:
    kind = cfg["data"]["sampler"]
    if kind != "auto":
        return kind
    meta = ds_path.with_suffix("").with_suffix(".meta.json")
    spec = json.loads(meta.read_text()).get("spec", {}) if meta.exists() else {}
    return spec.get("sampler") or "static"


def initial_p(cfg: dict, rows: list[dict], model_id: str) -> tuple[np.ndarray | None, str]:
    """Starting p per dataset problem from the base model's labels (NaN where unlabelled)."""
    if not cfg["data"]["p_init_labels"]:
        return None, "none"
    if all(r.get("p_hat") is not None for r in rows):
        return np.array([r["p_hat"] for r in rows], dtype=float), "dataset"
    labels = resolve(cfg["paths"]["data"]) / "labels" / f"{model_id}.parquet"
    if labels.exists():
        p = pd.read_parquet(labels, columns=["pid", "p_hat"]).set_index("pid").p_hat
        return p.reindex([r["pid"] for r in rows]).to_numpy(dtype=float), str(labels)
    return None, "none"


def summarise_eval(rows: list[dict]) -> dict:
    """{eval_set: {all: acc, d2: acc, ...}} from greedy_eval rows of one step."""
    out: dict = defaultdict(dict)
    by_depth: dict = defaultdict(list)
    for r in rows:
        if r.get("aggregation") or "func" in r:
            continue
        if r["depth"] == "all":
            out[r["eval_set"]]["all"] = r["greedy_acc"]
        else:
            by_depth[(r["eval_set"], r["depth"])].append((r["greedy_acc"], r["n"]))
    for (name, d), v in sorted(by_depth.items()):
        out[name][f"d{d}"] = sum(a * n for a, n in v) / sum(n for _, n in v)
    return dict(out)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML overrides for rl.* / data.*")
    ap.add_argument("--base", default=None, help="base checkpoint dir (= rl.base)")
    ap.add_argument("--dataset", default=None, help="dataset name or .jsonl path (= data.dataset)")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing run directory's logs")
    ap.add_argument("--allow-unknown-commit", action="store_true",
                    help="run although the git commit cannot be read (a copy of the repo without .git, or no git)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    overrides = list(args.overrides)
    if args.base:
        overrides.append(f"rl.base={args.base}")
    if args.dataset:
        overrides.append(f"data.dataset={args.dataset}")

    cfg = rl_config(args.config, overrides)
    rc, dc, ec, seed = cfg["rl"], cfg["data"], cfg["eval"], cfg["seed"]
    base = resolve(rc["base"])
    model_id = checkpoint_ids(base)[0]
    run_name = run_name_for(cfg)
    run_dir = resolve(cfg["paths"]["runs"]) / run_name
    if (run_dir / "metrics.jsonl").exists() and not args.overwrite:
        raise SystemExit(f"{run_dir} already exists; move it away, pass --overwrite or set rl.tag")
    commit = git_commit()
    if commit == "unknown":  # git_commit()'s fallback
        if not args.allow_unknown_commit:
            raise SystemExit("git commit unknown (no .git here, or git is not installed), so the run could not "
                             "record its code version; copy .git with the code, or pass --allow-unknown-commit")
        print("warning: git commit unknown; run_config.yaml will not record the code version", flush=True)
    ds_path = dataset_path(cfg, model_id)
    eval_dir = resolve(cfg["paths"]["data"]) / "eval" / ec["version"]
    if not eval_dir.exists():
        raise SystemExit(f"{eval_dir} missing; run scripts/make_eval_sets.py first")

    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(seed)
    model, tok = load_model(base, device)
    from rlvr.task import make_task
    task = make_task(cfg["task"])
    if hasattr(task, "validate_lengths"):
        task.validate_lengths(cfg)

    rows = read_jsonl(ds_path)
    problems = [task.from_dict(r) for r in rows]
    eval_names = list(rc["eval_sets"]) + ([rc["probe_set"]] if rc["probe_set"] else [])
    eval_sets = {name: load_problems(task, eval_dir / f"{name}.jsonl") for name in dict.fromkeys(eval_names)}
    eval_pids = {r["pid"] for f in sorted(eval_dir.glob("*.jsonl")) for r in read_jsonl(f)}
    leaked = eval_pids & {p.pid for p in problems}
    if leaked:
        raise SystemExit(f"{len(leaked)} dataset problems are in the eval sets of {eval_dir}")

    B, G, steps = rc["prompts_per_step"], rc["group_size"], rc["steps"]
    kind = sampler_kind(cfg, ds_path)
    p_init, p_source = initial_p(cfg, rows, model_id)
    sampler = ProblemSampler([p.depth for p in problems], kind, dc["order"], seed, steps * B, p_init,
                             dc["p_ema"], dc["p_edge"], dc["explore_frac"])
    run_cfg = merge(cfg, {"git_commit": commit, "data": {"sampler": kind}})
    cuda = device.type == "cuda"
    # Only in the run's run_config.yaml, not in checkpoints: load_config() rejects unregistered keys.
    env = {"torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(device) if cuda else None,
           "gpu_mem_gb": torch.cuda.get_device_properties(device).total_memory / 2**30 if cuda else None}
    run_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("summary.json", "eval_problems.jsonl"):  # --overwrite: not rewritten until / unless needed
        (run_dir / stale).unlink(missing_ok=True)
    (run_dir / "run_config.yaml").write_text(yaml.safe_dump({**run_cfg, "env": env}, sort_keys=False))
    print(f"{run_name}: base {model_id}, dataset {ds_path.name} ({len(problems)} problems, sampler {kind}, "
          f"p init {p_source}), B {B} x G {G} x {steps} steps = {B * G * steps} rollouts", flush=True)

    ref_model = None
    if rc["kl_coef"] > 0 or rc["kl_every"] > 0:
        ref_model, _ = load_model(base, device)
        ref_model.requires_grad_(False)
    if rc["compile"]:
        torch._dynamo.config.recompile_limit = 8 * len(model.model.layers)
        for layer in model.model.layers:
            layer.compile(dynamic=True)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": rc["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=rc["lr"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, rc["warmup_steps"])))
    save_steps = {max(1, round(f * steps)) for f in rc["save_at"]} | ({steps} if rc["save_final"] else set())

    metrics_f = (run_dir / "metrics.jsonl").open("w")
    rollouts_f = (run_dir / "rollouts.jsonl").open("w")
    eval_f = (run_dir / "eval.jsonl").open("w")
    eval_problems_f = (run_dir / "eval_problems.jsonl").open("w") if rc["log_eval_problems"] else None
    evals: dict[int, dict] = defaultdict(dict)  # step -> {eval_set: {all, d2, ...}}
    step_evals: dict[int, dict] = defaultdict(dict)  # step -> {eval_set: {step_score, step_frac}}
    eval_seconds = 0.0
    eval_mem_gb: list[float] = []  # peak GPU memory per eval (CUDA only)

    def evaluate(step: int, names: list[str]) -> None:
        nonlocal eval_seconds
        if not names:
            return
        t = time.monotonic()
        records = [] if eval_problems_f else None
        if cuda:  # the step's peak is already in its metrics row; the next step resets again
            torch.cuda.reset_peak_memory_stats(device)
        res = greedy_eval(model, tok, task, {n: eval_sets[n] for n in names}, step,
                          ec["max_new_tokens"], ec["batch_size"], ec["pass_k_samples"], ec["pass_k"],
                          records=records, progress=True)
        if cuda:
            eval_mem_gb.append(torch.cuda.max_memory_allocated(device) / 2**30)
        for r in res:
            eval_f.write(json.dumps({**r, "rollouts": step * B * G}) + "\n")
        eval_f.flush()
        if eval_problems_f:
            for p in records:
                row = {"step": step, **{k: p[k] for k in ("eval_set", "pid", "correct", "n_correct", "err_type",
                                                          "err_step")}}
                eval_problems_f.write(json.dumps(row, separators=(",", ":")) + "\n")
            eval_problems_f.flush()
        evals[step].update(summarise_eval(res))
        step_evals[step].update({r["eval_set"]: {"step_score": r["step_score"], "step_frac": r["step_frac"]}
                                 for r in res if r["depth"] == "all"})
        eval_seconds += time.monotonic() - t
        print(f"[eval step {step}] " + " ".join(f"{n} {evals[step][n]['all']:.3f} "
                                                f"(steps {step_evals[step][n]['step_frac']:.3f})" for n in names)
              + f" ({time.monotonic() - t:.0f}s" + (f", {eval_mem_gb[-1]:.1f} GB" if cuda else "") + ")", flush=True)

    def due(step: int, every: int) -> bool:
        return step in (0, steps) or (every > 0 and step % every == 0)

    def names_due(step: int) -> list[str]:
        names = list(rc["eval_sets"]) if due(step, rc["eval_every"]) else []
        if rc["probe_set"] and rc["probe_set"] not in names and due(step, rc["probe_every"]):
            names.append(rc["probe_set"])
        return names

    t_run = time.monotonic()  # summary minutes: the whole run, step-0 eval included
    evaluate(0, names_due(0))  # also confirms the right checkpoint was loaded
    t_start = time.monotonic()  # ETA: training steps only
    window: list[dict] = []
    for step in range(1, steps + 1):
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.monotonic()
        idx = sampler.sample(B)
        batch = [problems[i] for i in idx]
        prompts = [task.format_prompt(p) for p in batch]
        comp_ids = generate_ids(model, tok, prompts, rc["max_new_tokens"], rc["temperature"], G,
                                rc["gen_batch_size"])
        texts = [tok.decode(c) for c in comp_ids]
        r = torch.tensor([task.reward(texts[i * G + j], p, rc["reward"]) for i, p in enumerate(batch)
                          for j in range(G)]).view(B, G)
        adv = group_advantages(r, rc["adv_std_norm"])
        prompt_ids = [ids for p in prompts for ids in [tok.encode(p, add_bos=True)] * G]
        t1 = time.monotonic()
        model.train()
        with_kl = rc["kl_every"] > 0 and (step % rc["kl_every"] == 0 or step == 1)
        up = grpo_update(model, opt, prompt_ids, comp_ids, adv.flatten(), rc, tok.pad_id, ref_model, with_kl)
        lr = sched.get_last_lr()[0]
        sched.step()
        group_mean = r.mean(1)
        sampler.update(idx, group_mean)
        t2 = time.monotonic()

        sig = signal_groups(r)
        lens = [len(c) for c in comp_ids]
        ended = [bool(c) and c[-1] == tok.eos_id for c in comp_ids]
        depths = torch.tensor([p.depth for p in batch])
        m = {"step": step, "rollouts": step * B * G, "mean_reward": float(r.mean()),
             "signal_frac": float(sig.float().mean()), "mean_len": float(np.mean(lens)),
             "trunc_frac": 1 - float(np.mean(ended)), "lr": lr, **up,
             **{f"reward_d{d}": float(r[depths == d].mean()) for d in sorted(set(depths.tolist()))},
             **{f"signal_d{d}": float(sig[depths == d].float().mean()) for d in sorted(set(depths.tolist()))},
             "n_unique": len(set(idx.tolist())), **sampler.stats(), "sec_rollout": t1 - t0, "sec_update": t2 - t1,
             **({"gpu_mem_gb": torch.cuda.max_memory_allocated(device) / 2**30} if cuda else {})}
        metrics_f.write(json.dumps(m) + "\n")
        metrics_f.flush()
        for i, p in enumerate(batch):
            row = {"step": step, "pid": p.pid, "depth": p.depth, "group": p.group, "str_len": len(p.x),
                   "rewards": r[i].tolist(), "lens": lens[i * G:(i + 1) * G], "p_online": float(sampler.p[idx[i]])}
            if rc["log_completions"]:
                row["completions"] = texts[i * G:(i + 1) * G]
            rollouts_f.write(json.dumps(row, separators=(",", ":")) + "\n")
        window.append(m)
        if step % rc["log_every"] == 0 or step == steps:
            avg = {k: float(np.mean([w[k] for w in window])) for k in
                   ("mean_reward", "signal_frac", "entropy", "grad_norm", "mean_len", "sec_rollout", "sec_update")}
            eta = (steps - step) * (time.monotonic() - t_start) / step / 60
            print(f"step {step}/{steps} reward {avg['mean_reward']:.3f} signal {avg['signal_frac']:.2f} "
                  f"entropy {avg['entropy']:.4f} gn {avg['grad_norm']:.2f} len {avg['mean_len']:.0f} "
                  f"{avg['sec_rollout']:.1f}+{avg['sec_update']:.1f}s/step eta {eta:.0f}m", flush=True)
            window = []
            rollouts_f.flush()
        evaluate(step, names_due(step))
        if step in save_steps:
            save_model(model, tok, run_cfg, run_dir / "ckpt" / f"step-{step}")

    for f in (metrics_f, rollouts_f, eval_f, eval_problems_f):
        if f:
            f.close()
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    k = max(1, min(10, steps // 4))
    first, last = min(evals), max(evals)
    summary = {
        "run_name": run_name, "model_id": model_id, "base": rc["base"], "dataset": Path(dc["dataset"]).stem,
        "n_problems": len(problems), "sampler": kind, "p_init": p_source, "seed": seed, "tag": rc["tag"],
        "group_size": G, "prompts_per_step": B, "steps": steps, "rollouts": steps * B * G, "lr": rc["lr"],
        "git_commit": run_cfg["git_commit"], "minutes": (time.monotonic() - t_run) / 60,
        "eval_minutes": eval_seconds / 60,
        "gpu_mem_gb": max((m["gpu_mem_gb"] for m in metrics if "gpu_mem_gb" in m), default=None),
        "eval_gpu_mem_gb": max(eval_mem_gb, default=None),
        "train_reward_first": float(np.mean([m["mean_reward"] for m in metrics[:k]])),
        "train_reward_last": float(np.mean([m["mean_reward"] for m in metrics[-k:]])),
        "signal_frac_mean": float(np.mean([m["signal_frac"] for m in metrics])),
        "eval_first": evals[first], "eval_final": evals[last],
        "steps_first": step_evals[first], "steps_final": step_evals[last],
        "ckpt": str(run_dir / "ckpt" / f"step-{steps}") if rc["save_final"] else None,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{run_name}: done in {summary['minutes']:.1f} min; "
          + " ".join(f"{n} {evals[first][n]['all']:.3f}->{evals[last][n]['all']:.3f}" for n in evals[last]),
          flush=True)
    return summary


if __name__ == "__main__":
    main()
