"""Calibrate a base model: p heatmap, RL-pool pass-rate labels, gate check.

    python scripts/calibrate.py checkpoints/pt-mid-s0 [--skip-pool] [key=value ...]
    python scripts/calibrate.py runs/pt-mid-s0/ckpt/step-3000 --skip-pool   # model_id pt-mid-s0-step-3000

p_hat = correct / label.k samples at label.temperature (doc: 校准热力图, 难度标注).

Output:
  runs/{model_id}/calib/calib_labels.parquet   p_hat for the calib set (depth 1-8 x H/L/mixed x 200)
  runs/{model_id}/calib/heatmap.png            mean p_hat and signal fraction per (depth, group)
  runs/{model_id}/calib/gate.json              gate check (also printed); exit code 1 if it fails
  data/labels/{model_id}.parquet               p_hat for the whole RL pool (+ origin), for build_dataset.py

Gate (doc: 门槛检查):
  1. E1 (depth 1) greedy accuracy >= gate.depth1_greedy, overall and for every function
  2. at each gate.spread_depths depth, >= gate.min_spread_frac of calib problems have 0 < p_hat < 1
  3. each gate.matched_cells RL-pool cell has >= gate.min_matched problems with p_hat in gate.p_range
  4. pretraining loss at the checkpoint's step >= gate.min_final_loss (else the task is still too easy)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rlvr.config import load_config, resolve  # noqa: E402
from rlvr.evaluate import greedy_eval, label_pass_rates  # noqa: E402
from rlvr.io import load_problems, read_jsonl  # noqa: E402
from rlvr.model import load_model  # noqa: E402
from rlvr.task import GROUPS, Task  # noqa: E402


def ids_for(ckpt: Path) -> tuple[str, str, int | None]:
    """(model_id, run_name, step): checkpoints/pt-mid-s0 -> (pt-mid-s0, pt-mid-s0, None);
    runs/pt-mid-s0/ckpt/step-3000 -> (pt-mid-s0-step-3000, pt-mid-s0, 3000)."""
    if ckpt.parent.name == "ckpt":
        run = ckpt.parent.parent.name
        return f"{run}-{ckpt.name}", run, int(ckpt.name.split("-")[-1])
    return ckpt.name, ckpt.name, None


def heatmap(df: pd.DataFrame, title: str, path: Path) -> None:
    depths = sorted(df.depth.unique())
    mean = df.pivot_table(index="group", columns="depth", values="p_hat", aggfunc="mean").reindex(GROUPS)
    sig = df.assign(s=(df.p_hat > 0) & (df.p_hat < 1)).pivot_table(
        index="group", columns="depth", values="s", aggfunc="mean").reindex(GROUPS)
    fig, axes = plt.subplots(1, 2, figsize=(4 + 1.1 * len(depths), 3.4))
    for ax, m, name in ((axes[0], mean, "mean p_hat"), (axes[1], sig, "frac 0 < p_hat < 1")):
        im = ax.imshow(m.to_numpy(dtype=float), vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(depths)), [str(d) for d in depths])
        ax.set_yticks(range(len(GROUPS)), GROUPS)
        ax.set_xlabel("depth")
        ax.set_title(name)
        for (i, j), v in np.ndenumerate(m.to_numpy(dtype=float)):
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="w" if v < 0.6 else "k", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="checkpoint dir, e.g. checkpoints/pt-mid-s0")
    ap.add_argument("--skip-pool", action="store_true", help="skip RL-pool labelling (gate 3 not checked)")
    ap.add_argument("--calib-per-cell", type=int, default=0,
                    help="label only the first N calib problems per (depth, group) cell (pilot sweep)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    ckpt = resolve(args.ckpt)
    model_id, run_name, ckpt_step = ids_for(ckpt)
    cfg = load_config(ckpt / "run_config.yaml", args.overrides)
    lc, gc, ec = cfg["label"], cfg["gate"], cfg["eval"]
    task = Task(cfg["task"])
    model, tok = load_model(ckpt, args.device)
    data = resolve(cfg["paths"]["data"])
    eval_dir = data / "eval" / ec["version"]
    out = resolve(cfg["paths"]["runs"]) / model_id / "calib"
    out.mkdir(parents=True, exist_ok=True)

    def label(problems, **extra):
        t = time.time()
        df = label_pass_rates(model, tok, task, problems, lc["k"], lc["temperature"], lc["batch_size"],
                              ec["max_new_tokens"], model_id, extra)
        print(f"labelled {len(problems)} problems x {lc['k']} in {time.time() - t:.0f}s", flush=True)
        return df

    gate = {"model_id": model_id, "checks": {}}

    # 1. depth-1 greedy accuracy
    rows = greedy_eval(model, tok, task, {"E1": load_problems(task, eval_dir / "E1.jsonl")},
                       max_new_tokens=ec["max_new_tokens"], batch_size=ec["batch_size"])
    e1 = next(r["greedy_acc"] for r in rows if r["depth"] == "all")
    weak_funcs = {r["func"]: round(r["greedy_acc"], 3) for r in rows
                  if "func" in r and r["greedy_acc"] < gc["depth1_greedy"]}
    gate["checks"]["depth1_greedy"] = {"value": e1, "min": gc["depth1_greedy"],
                                       "pass": e1 >= gc["depth1_greedy"] and not weak_funcs,
                                       "funcs_below_min": weak_funcs}

    # 4. pretraining loss (memorised / trivial task -> loss ~ 0)
    metrics_path = resolve(cfg["paths"]["runs"]) / run_name / "metrics.jsonl"
    if metrics_path.exists():
        m = [r for r in read_jsonl(metrics_path) if ckpt_step is None or r["step"] <= ckpt_step]
        if m:
            gate["checks"]["final_loss"] = {"step": m[-1]["step"], "value": m[-1]["loss"],
                                            "min": gc["min_final_loss"],
                                            "pass": m[-1]["loss"] >= gc["min_final_loss"]}

    # 2. calibration heatmap + spread over depths
    calib_rows = read_jsonl(eval_dir / "calib.jsonl")
    if args.calib_per_cell:
        kept, per_cell = [], Counter()
        for r in calib_rows:
            per_cell[(r["depth"], r["group"])] += 1
            if per_cell[(r["depth"], r["group"])] <= args.calib_per_cell:
                kept.append(r)
        calib_rows = kept
    calib = label([task.from_dict(r) for r in calib_rows])
    calib.to_parquet(out / "calib_labels.parquet", index=False)
    heatmap(calib, f"{model_id}: p_hat at T={lc['temperature']}, k={lc['k']}", out / "heatmap.png")
    spread = {}
    for d in gc["spread_depths"]:
        p = calib.p_hat[calib.depth == d]
        spread[d] = {"mean_p": float(p.mean()), "signal_frac": float(((p > 0) & (p < 1)).mean())}
    gate["checks"]["spread"] = {"by_depth": spread, "min_signal_frac": gc["min_spread_frac"],
                                "pass": all(v["signal_frac"] >= gc["min_spread_frac"] for v in spread.values())}
    gate["calib_mean_p"] = {f"d{d}-{g}": round(float(v), 3)
                            for (d, g), v in calib.groupby(["depth", "group"]).p_hat.mean().items()}

    # 3. RL-pool labels and matched-p cells
    if not args.skip_pool:
        pool_rows = read_jsonl(data / "rl_pool" / "candidates.jsonl")
        pool = label([task.from_dict(r) for r in pool_rows], origin=[r.get("origin", "base") for r in pool_rows])
        (data / "labels").mkdir(parents=True, exist_ok=True)
        pool.to_parquet(data / "labels" / f"{model_id}.parquet", index=False)
        lo, hi = gc["p_range"]
        cells = {}
        for c in gc["matched_cells"]:
            sel = pool[(pool.depth == c["depth"]) & (pool.group == c["group"])]
            n = int(((sel.p_hat >= lo) & (sel.p_hat <= hi)).sum())
            cells[f"d{c['depth']}-{c['group']}"] = {"n_in_range": n, "n_cell": len(sel),
                                                     "mean_p": float(sel.p_hat.mean())}
        gate["checks"]["matched_cells"] = {"p_range": [lo, hi], "min": gc["min_matched"], "cells": cells,
                                           "pass": all(v["n_in_range"] >= gc["min_matched"] for v in cells.values())}

    gate["pass"] = all(c["pass"] for c in gate["checks"].values())
    (out / "gate.json").write_text(json.dumps(gate, indent=1))
    print(json.dumps(gate, indent=1))
    print(f"{model_id}: gate {'PASS' if gate['pass'] else 'FAIL'}; outputs in {out}")
    sys.exit(0 if gate["pass"] else 1)


if __name__ == "__main__":
    main()
