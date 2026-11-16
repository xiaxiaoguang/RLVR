"""Winner's-curse check of the second labelling for one model's RL-pool labels.

    python scripts/relabel_report.py pt-mid-s0      # -> runs/pt-mid-s0/calib/relabel_report.json

For each label.relabel_cells cell (d2-L, d4-H, d5-H by default):
  first_selected    problems with a first-pass p_hat in gate.p_range: their mean first-pass p_hat
                    and mean independent second estimate (p_hat_relabel), and the share whose
                    second estimate is still in range. A gap between the two means is the
                    selection bias of choosing on a 32-sample p_hat.
  relabel_selected  problems with p_hat_relabel in gate.p_range (what the matched-p datasets and
                    gate check 3 use): count, count per str_len, means of both estimates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from rlvr.config import load_config, resolve  # noqa: E402


def _mean(s: pd.Series) -> float | None:
    return round(float(s.mean()), 4) if len(s) else None


def report(pool: pd.DataFrame, cells: list[dict], p_range: list[float]) -> dict:
    lo, hi = p_range
    lens = sorted(int(v) for v in pool.str_len.unique())
    out = {}
    for c in cells:
        sel = pool[(pool.depth == c["depth"]) & (pool.group == c["group"])]
        first = sel[(sel.p_hat >= lo) & (sel.p_hat <= hi)]
        again = sel[(sel.p_hat_relabel >= lo) & (sel.p_hat_relabel <= hi)]
        out[f"d{c['depth']}-{c['group']}"] = {
            "n_cell": len(sel),
            "mean_p_hat_cell": _mean(sel.p_hat),
            "n_relabelled": int(sel.p_hat_relabel.notna().sum()),
            "first_selected": {
                "n": len(first),
                "mean_p_hat": _mean(first.p_hat),
                "mean_p_hat_relabel": _mean(first.p_hat_relabel),
                "frac_relabel_in_range": round(float(first.p_hat_relabel.between(lo, hi).mean()), 4)
                if len(first) else None},
            "relabel_selected": {
                "n": len(again),
                "n_by_len": {int(k): int(v) for k, v in again.str_len.value_counts().reindex(lens, fill_value=0).items()},
                "mean_p_hat_relabel": _mean(again.p_hat_relabel),
                "mean_p_hat": _mean(again.p_hat)},
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id", help="e.g. pt-mid-s0 (reads data/labels/{model_id}.parquet)")
    args = ap.parse_args()
    run_cfg = resolve("checkpoints") / args.model_id.split("-step-")[0] / "run_config.yaml"
    cfg = load_config(run_cfg if run_cfg.exists() else None)
    pool = pd.read_parquet(resolve(cfg["paths"]["data"]) / "labels" / f"{args.model_id}.parquet")
    if "p_hat_relabel" not in pool:
        raise SystemExit(f"{args.model_id}: no p_hat_relabel column (label.relabel_k = 0?)")
    res = {"model_id": args.model_id, "p_range": cfg["gate"]["p_range"],
           "relabel_range": cfg["label"]["relabel_range"], "relabel_k": cfg["label"]["relabel_k"],
           "cells": report(pool, cfg["label"]["relabel_cells"], cfg["gate"]["p_range"])}
    out = resolve(cfg["paths"]["runs"]) / args.model_id / "calib" / "relabel_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
