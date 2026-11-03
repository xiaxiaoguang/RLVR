"""Pilot sweep (doc: 试点扫描): comp_frac x pos_offset_max on short single-GPU runs.

    python scripts/sweep.py --gpus 0 1 2 3 4 5 6 7                     # train + calibrate + summarise
    python scripts/sweep.py --gpus 0 1 --comp-fracs 0.001 0.01 --pos-offsets 128
    python scripts/sweep.py --summarize-only                           # rebuild the summary
    python scripts/sweep.py --gpus 0 1 2 3 pretrain.steps=800          # extra overrides go to pretrain.py

Each run is pretrain.level=pilot-cf{comp_frac}-po{offset} from --config (default
configs/pretrain/pilot.yaml), followed by calibrate.py --skip-pool on a calib
subsample. One job per GPU at a time. Logs: runs/sweep/{level}.log.

Output (runs/sweep/):
  summary.csv / summary.md   mean p_hat per depth (all groups), d2-L and d4-H, E1 greedy, final loss
  summary.png                comp_frac -> mean p_hat per depth, one panel per pos_offset
  suggestion.json            comp_frac per level by log-linear interpolation of the targets below

Targets (doc): weak d3 mean p ~ 0.1, mid d3 ~ 0.45, strong d4 ~ 0.5. The full runs
see ~25x more tokens, which shifts the curves to smaller comp_frac, so treat the
suggestion as an upper bound and confirm on the 25% checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rlvr.config import ROOT, load_config, resolve  # noqa: E402

TARGETS = {"weak": ("d3", 0.1), "mid": ("d3", 0.45), "strong": ("d4", 0.5)}


def level_name(cf: float, po: int) -> str:
    return f"pilot-cf{cf:g}-po{po}"


def run_job(gpu: str, level: str, args, log_dir: Path) -> int:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    cf, po = level.split("-cf")[1].split("-po")
    cmds = [
        [sys.executable, "scripts/pretrain.py", "--config", args.config, "--overwrite", f"pretrain.level={level}",
         f"pretrain.comp_frac={cf}", f"pretrain.pos_offset_max={po}", f"seed={args.seed}", *args.overrides],
        [sys.executable, "scripts/calibrate.py", f"checkpoints/pt-{level}-s{args.seed}", f"label.k={args.k}",
         "--skip-pool", "--calib-per-cell", str(args.calib_per_cell)],
    ]
    with (log_dir / f"{level}.log").open("w") as log:
        for cmd in cmds:
            log.write("$ " + " ".join(cmd) + "\n")
            log.flush()
            rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            # calibrate.py exits 1 when the gate fails; that is a result, not an error.
            if rc != 0 and cmd[1] != "scripts/calibrate.py":
                return rc
    return 0


def summarize(levels: list[str], seed: int, runs: Path, out: Path) -> None:
    rows = []
    for level in levels:
        calib = runs / f"pt-{level}-s{seed}" / "calib"
        if not (calib / "calib_labels.parquet").exists():
            print(f"{level}: no calibration yet")
            continue
        df = pd.read_parquet(calib / "calib_labels.parquet")
        gate = json.loads((calib / "gate.json").read_text())
        cf, po = level.split("-cf")[1].split("-po")
        row = {"level": level, "comp_frac": float(cf), "pos_offset_max": int(po),
               "E1": gate["checks"]["depth1_greedy"]["value"],
               "final_loss": gate["checks"].get("final_loss", {}).get("value")}
        row.update({f"d{d}": float(v) for d, v in df.groupby("depth").p_hat.mean().items()})
        for d, g in ((2, "L"), (4, "H"), (5, "H")):
            sel = df[(df.depth == d) & (df.group == g)].p_hat
            row[f"d{d}-{g}"] = float(sel.mean())
            row[f"d{d}-{g}_mid"] = float(((sel >= 0.4) & (sel <= 0.6)).mean())
        rows.append(row)
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values(["pos_offset_max", "comp_frac"])
    df.to_csv(out / "summary.csv", index=False)
    fmt = lambda v: f"{v:.3g}" if isinstance(v, float) else str(v)  # noqa: E731
    md = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    md += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    depth_cols = sorted((c for c in df.columns if c[1:].isdigit()), key=lambda c: int(c[1:]))
    offsets = sorted(df.pos_offset_max.unique())
    fig, axes = plt.subplots(1, len(offsets), figsize=(5.5 * len(offsets), 4), squeeze=False)
    for ax, po in zip(axes[0], offsets):
        sub = df[df.pos_offset_max == po]
        for c in depth_cols:
            ax.plot(sub.comp_frac, sub[c], marker="o", label=c)
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("comp_frac")
        ax.set_ylabel("mean p_hat")
        ax.set_title(f"pos_offset_max={po}")
        ax.grid(alpha=0.3)
    axes[0][-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "summary.png", dpi=130)
    plt.close(fig)

    # Log-linear interpolation of comp_frac reaching each target (p rises with comp_frac).
    suggestion = {}
    for po in offsets:
        sub = df[df.pos_offset_max == po].sort_values("comp_frac")
        for lv, (col, target) in TARGETS.items():
            x, y = np.log(sub.comp_frac.to_numpy()), sub[col].to_numpy() if col in sub else None
            if y is None or len(x) < 2 or not (y.min() <= target <= y.max()):
                suggestion[f"po{po}/{lv}"] = None
                continue
            order = np.argsort(y)
            suggestion[f"po{po}/{lv}"] = float(math.exp(np.interp(target, y[order], x[order])))
    (out / "suggestion.json").write_text(json.dumps({"targets": TARGETS, "comp_frac": suggestion}, indent=1))
    print("suggested comp_frac (None = target outside the swept range):", json.dumps(suggestion, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pretrain/pilot.yaml")
    ap.add_argument("--gpus", nargs="*", default=["0"])
    ap.add_argument("--comp-fracs", nargs="*", type=float, default=[0.0003, 0.001, 0.003, 0.01, 0.03, 0.1])
    ap.add_argument("--pos-offsets", nargs="*", type=int, default=[0, 128])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib-per-cell", type=int, default=50)
    ap.add_argument("--k", type=int, default=16, help="samples per calib problem")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("overrides", nargs="*", help="key=value overrides for pretrain.py")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    runs = resolve(cfg["paths"]["runs"])
    out = runs / "sweep"
    out.mkdir(parents=True, exist_ok=True)
    levels = [level_name(cf, po) for po in args.pos_offsets for cf in args.comp_fracs]

    if not args.summarize_only:
        jobs: queue.Queue = queue.Queue()
        for lv in levels:
            jobs.put(lv)
        failed = []

        def worker(gpu):
            while True:
                try:
                    lv = jobs.get_nowait()
                except queue.Empty:
                    return
                print(f"[gpu {gpu}] {lv}", flush=True)
                if run_job(gpu, lv, args, out) != 0:
                    failed.append(lv)
                    print(f"[gpu {gpu}] {lv} FAILED (see {out / (lv + '.log')})", flush=True)

        threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if failed:
            print("failed:", failed)
    summarize(levels, args.seed, runs, out)


if __name__ == "__main__":
    main()
