"""Run a series of GRPO runs: base checkpoints x datasets x seeds (x hyperparameter grid).

    python scripts/rl_sweep.py --name repro --gpus 4 5 6 7 \\
        --models checkpoints/pt-weak-s0 checkpoints/pt-mid-s0 \\
        --datasets easy_500 mixed_500 depth_3 --seeds 0 1
    python scripts/rl_sweep.py --name G --gpus 0 1 --models checkpoints/pt-mid-s0 --datasets mixed_500 \\
        --grid rl.group_size=4,8,32 --grid rl.lr=3e-6,1e-5 rl.steps=300      # key=value goes to every run
    python scripts/rl_sweep.py --name repro ... --dry-run                    # list the runs and exit
    python scripts/rl_sweep.py --name repro ... --summarize-only             # rebuild the summary

One scripts/grpo.py process per GPU slot at a time (--per-gpu N for N runs per GPU). A run whose
summary.json exists is skipped, so an interrupted sweep continues where it stopped (--rerun
forces). Models may be final checkpoints or intermediate ones (runs/pt-mid-s0/ckpt/step-2375),
chain-v1 or tree-v1: each run takes task, model and paths from its checkpoint. Grid keys other
than rl.group_size are appended to the run name through rl.tag (e.g. -lr3e-06).

Output: <runs>/rl_sweep/{name}/
  jobs.json                 the planned runs and their commands
  logs/{run}.log            stdout / stderr of each run
  summary.csv / summary.md  one row per finished run: greedy accuracy first -> final per eval set, deltas,
                            E2 by depth, training reward, signal fraction, minutes
  curves.png                greedy accuracy against rollouts, one panel per eval set
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from grpo import rl_config, run_name_for  # noqa: E402

from rlvr.config import ROOT, resolve  # noqa: E402
from rlvr.io import read_jsonl  # noqa: E402


def grid_points(specs: list[str]) -> list[dict[str, str]]:
    """["rl.group_size=4,8", "rl.lr=3e-6,1e-5"] -> the 4 combinations as {key: value}."""
    keys, values = [], []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--grid must look like key=v1,v2: {spec!r}")
        k, v = spec.split("=", 1)
        keys.append(k)
        values.append(v.split(","))
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def plan(args) -> list[dict]:
    user_tag = next((o.split("=", 1)[1] for o in args.overrides if o.startswith("rl.tag=")), "")
    overrides = [o for o in args.overrides if not o.startswith("rl.tag=")]
    jobs = []
    for model, dataset, seed, point in itertools.product(args.models, args.datasets, args.seeds,
                                                         grid_points(args.grid)):
        tag = "-".join(filter(None, [user_tag] + [f"{k.split('.')[-1]}{v}" for k, v in point.items()
                                                  if k != "rl.group_size"]))
        ov = [*overrides, *(f"{k}={v}" for k, v in point.items()), f"seed={seed}", f"rl.base={model}",
              f"data.dataset={dataset}"] + ([f"rl.tag={tag}"] if tag else [])
        cfg = rl_config(args.config, ov)
        run = run_name_for(cfg)
        cmd = [sys.executable, "scripts/grpo.py", "--overwrite"] + (["--config", args.config] if args.config else []) + ov
        jobs.append({"run": run, "run_dir": str(resolve(cfg["paths"]["runs"]) / run), "model": model,
                     "dataset": dataset, "seed": seed, "grid": point, "cmd": cmd,
                     "sweep_root": str(resolve(cfg["paths"]["runs"]) / "rl_sweep")})
    names = [j["run"] for j in jobs]
    if len(set(names)) != len(names):
        raise SystemExit("two planned runs share a name; vary rl.tag or the grid")
    return jobs


def run_jobs(jobs: list[dict], slots: list[str], log_dir: Path) -> list[str]:
    q: queue.Queue = queue.Queue()
    for j in jobs:
        q.put(j)
    failed, lock, t0 = [], threading.Lock(), time.time()

    def worker(gpu: str):
        while True:
            try:
                job = q.get_nowait()
            except queue.Empty:
                return
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
            with (log_dir / f"{job['run']}.log").open("w") as log:
                rc = subprocess.call(job["cmd"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            with lock:
                if rc != 0:
                    failed.append(job["run"])
                print(f"[{(time.time() - t0) / 60:6.1f}m] gpu {gpu} {'ok  ' if rc == 0 else 'FAIL'} "
                      f"{job['run']} ({q.qsize()} queued)", flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in slots]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return failed


def summarise(jobs: list[dict], out: Path) -> pd.DataFrame:
    rows, curves = [], {}
    for j in jobs:
        path = Path(j["run_dir"]) / "summary.json"
        if not path.exists():
            continue
        s = json.loads(path.read_text())
        row = {"run": s["run_name"], "model": s["model_id"], "dataset": s["dataset"], "sampler": s["sampler"],
               "G": s["group_size"], "seed": s["seed"], "tag": s["tag"], "steps": s["steps"],
               "rollouts": s["rollouts"], "minutes": round(s["minutes"], 1),
               "reward_first": s["train_reward_first"], "reward_last": s["train_reward_last"],
               "signal_frac": s["signal_frac_mean"]}
        for name, final in s["eval_final"].items():
            first = s["eval_first"].get(name, {})
            row[f"{name}_first"], row[f"{name}_final"] = first.get("all"), final["all"]
            row[f"{name}_delta"] = final["all"] - first["all"] if "all" in first else None
            if name == "E2":
                for d, v in final.items():
                    if d != "all":
                        row[f"E2_{d}_first"], row[f"E2_{d}_final"] = first.get(d), v
        rows.append(row)
        ev = pd.DataFrame(read_jsonl(Path(j["run_dir"]) / "eval.jsonl"))
        curves[s["run_name"]] = ev[(ev.depth == "all") & (ev.group == "all")]
    df = pd.DataFrame(rows)
    if df.empty:
        print("no finished runs to summarise")
        return df
    df.to_csv(out / "summary.csv", index=False)

    def fmt(v):
        return "" if v is None or v != v else f"{v:.3f}" if isinstance(v, float) else str(v)

    cols = [c for c in df.columns if not c.startswith("E2_d")]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df[cols].itertuples(index=False)]
    (out / "summary.md").write_text("\n".join(md) + "\n")

    sets = [n for n in ("E2", "E3", "E4", "probe") if f"{n}_final" in df]
    if sets:
        fig, axes = plt.subplots(1, len(sets), figsize=(4.6 * len(sets), 3.8), squeeze=False)
        for ax, name in zip(axes[0], sets):
            for run, ev in curves.items():
                c = ev[ev.eval_set == name].sort_values("rollouts")
                ax.plot(c.rollouts, c.greedy_acc, marker=".", lw=1, label=run.removeprefix("rl-"))
            ax.set(title=f"{name} greedy accuracy", xlabel="rollouts", ylim=(-0.02, 1.02))
            ax.grid(alpha=0.3)
        if len(curves) <= 16:
            axes[0][-1].legend(fontsize=6, loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.tight_layout()
        fig.savefig(out / "curves.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    show = ["run", "minutes", "reward_first", "reward_last", "signal_frac"] + [f"{n}_{k}" for n in sets
                                                                              for k in ("first", "final")]
    print(df[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"summary -> {out}")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="default", help="sweep name (output directory)")
    ap.add_argument("--models", nargs="+", required=True, help="base checkpoint dirs")
    ap.add_argument("--datasets", nargs="+", required=True, help="dataset names (or .jsonl paths)")
    ap.add_argument("--seeds", nargs="+", default=["0"])
    ap.add_argument("--grid", action="append", default=[], help="key=v1,v2,... (repeatable; full product)")
    ap.add_argument("--config", default=None, help="YAML overrides for every run")
    ap.add_argument("--gpus", nargs="+", default=["0"])
    ap.add_argument("--per-gpu", type=int, default=1, help="concurrent runs per GPU (~13 GB each at B 64 x G 8)")
    ap.add_argument("--rerun", action="store_true", help="also rerun finished runs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("overrides", nargs="*", help="key=value for every run")
    args = ap.parse_args()
    # key=value given after a list option (--datasets a b rl.steps=200) is an override, not a list item
    for name in ("models", "datasets", "seeds", "gpus"):
        items = getattr(args, name)
        args.overrides += [v for v in items if "=" in v]
        setattr(args, name, [v for v in items if "=" not in v])
    args.seeds = [int(v) for v in args.seeds]

    jobs = plan(args)
    out = Path(jobs[0]["sweep_root"]) / args.name
    todo = [j for j in jobs if args.rerun or not (Path(j["run_dir"]) / "summary.json").exists()]
    print(f"{len(jobs)} runs planned, {len(jobs) - len(todo)} already finished -> {out}")
    if args.dry_run:
        for j in jobs:
            print(("run  " if j in todo else "done ") + j["run"])
        return
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "jobs.json").write_text(json.dumps(jobs, indent=1))
    failed = []
    if not args.summarize_only and todo:
        slots = [g for g in args.gpus for _ in range(args.per_gpu)]
        failed = run_jobs(todo, slots, out / "logs")
    summarise(jobs, out)
    if failed:
        sys.exit(f"failed: {failed} (logs in {out / 'logs'})")


if __name__ == "__main__":
    main()
