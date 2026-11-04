"""Gate checks after pretraining (doc: 门槛检查), as pure functions of eval / label tables.

scripts/calibrate.py computes the inputs (E1 greedy rows, calib labels, RL-pool labels,
metrics.jsonl) and calls these, so the checks can be tested without a GPU. Every check
returns a dict with a boolean `pass`. `gate.hard_checks` decides which checks can fail
the gate; the others are reported as warnings (e.g. weak/strong are not meant to have
p ~ 0.5 problems in the matched cells, so only the atom check is hard for them).

Checks:
  depth1_greedy  E1 greedy accuracy >= gate.depth1_greedy overall; per function, fail only
                 when the Wilson upper bound (z = gate.depth1_func_z) is below it, because
                 50 problems per function cannot separate 0.98 from 0.99
  spread         at each gate.spread_depths depth, >= gate.min_spread_frac of the calib
                 problems have 0 < p_hat < 1
  matched_cells  each gate.matched_cells RL-pool cell has >= gate.min_matched problems with p
                 in gate.p_range in total and >= gate.min_matched_per_len at every input length
                 (build_dataset.py stratifies the matched-p datasets on str_len); p is the
                 independent relabel for relabelled cells (label.relabel_*), else p_hat
  final_loss     composition loss at gate.loss_depths >= gate.min_final_loss (in-distribution
                 compositions not fully saturated); the overall loss is reported but not
                 checked, because atoms dominate it and it mostly tracks comp_frac
  target         mean calib p_hat at gate.target.depth within gate.target.p_range (the level's
                 design target from the pilot sweep); skipped when gate.target is null
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .difficulty import wilson

CHECKS = ("depth1_greedy", "spread", "matched_cells", "final_loss", "target")

# Columns that calibrate.py adds to data/labels/{model_id}.parquet for relabelled problems.
RELABEL_COLS = {"n_correct": "n_correct_relabel", "k": "k_relabel", "p_hat": "p_hat_relabel",
                "ci_lo": "ci_lo_relabel", "ci_hi": "ci_hi_relabel"}


def check_config(gc: dict, lc: dict) -> None:
    """Fail fast (before any labelling) on gate / relabel settings that cannot work."""
    unknown = [c for c in gc["hard_checks"] if c not in CHECKS]
    if unknown:
        raise ValueError(f"gate.hard_checks: unknown {unknown}; choose from {list(CHECKS)}")
    if lc["relabel_k"]:
        cells = {(c["depth"], c["group"]) for c in lc["relabel_cells"]}
        missing = [c for c in gc["matched_cells"] if (c["depth"], c["group"]) not in cells]
        if missing:
            raise ValueError(f"gate.matched_cells {missing} are not in label.relabel_cells")
    t = gc.get("target")
    if t is not None and (set(t) != {"depth", "p_range"} or len(t["p_range"]) != 2):
        raise ValueError(f"gate.target must be {{depth, p_range: [lo, hi]}}, got {t}")


def _in_cells(df: pd.DataFrame, cells: list[dict]) -> np.ndarray:
    m = np.zeros(len(df), dtype=bool)
    for c in cells:
        m |= ((df.depth == c["depth"]) & (df.group == c["group"])).to_numpy()
    return m


def relabel_mask(pool: pd.DataFrame, lc: dict) -> np.ndarray:
    """Problems that get the second, independent labelling: in label.relabel_cells with a
    first-pass p_hat in label.relabel_range (a wide screen around the matched range)."""
    if not lc["relabel_k"]:
        return np.zeros(len(pool), dtype=bool)
    lo, hi = lc["relabel_range"]
    return _in_cells(pool, lc["relabel_cells"]) & (pool.p_hat >= lo).to_numpy() & (pool.p_hat <= hi).to_numpy()


def attach_relabel(pool: pd.DataFrame, mask: np.ndarray, again: pd.DataFrame | None) -> pd.DataFrame:
    """Add the *_relabel columns (NaN where not relabelled). `again` is the labelling of
    pool[mask] in the same order (None when nothing was screened in)."""
    n = int(mask.sum())
    if n and (again is None or len(again) != n or not (again.pid.to_numpy() == pool.pid.to_numpy()[mask]).all()):
        raise ValueError("relabel rows are not aligned with the pool")
    out = pool.copy()
    for src, dst in RELABEL_COLS.items():
        col = np.full(len(out), np.nan)
        if n:
            col[mask] = again[src].to_numpy(dtype=float)
        out[dst] = col
    return out


def depth1_greedy(rows: list[dict], gc: dict) -> dict:
    """rows: greedy_eval output for E1 (an 'all' row plus one row per function)."""
    e1 = next(r["greedy_acc"] for r in rows if r["depth"] == "all")
    thr, z = gc["depth1_greedy"], gc["depth1_func_z"]
    below, funcs = {}, []
    for r in rows:
        if "func" not in r:
            continue
        n = r["n"]
        upper = wilson(round(r["greedy_acc"] * n), n, z)[1]
        funcs.append((r["greedy_acc"], r["func"]))
        if upper < thr:
            below[r["func"]] = {"acc": round(r["greedy_acc"], 3), "n": n, "upper": round(upper, 3)}
    worst = {f: round(a, 3) for a, f in sorted(funcs)[:3]}
    return {"value": e1, "min": thr, "func_z": z, "funcs_below_min": below, "worst_funcs": worst,
            "pass": e1 >= thr and not below}


def spread(calib: pd.DataFrame, gc: dict) -> dict:
    by_depth = {}
    for d in gc["spread_depths"]:
        p = calib.p_hat[calib.depth == d]
        by_depth[d] = {"mean_p": float(p.mean()), "signal_frac": float(((p > 0) & (p < 1)).mean())}
    return {"by_depth": by_depth, "min_signal_frac": gc["min_spread_frac"],
            "pass": all(v["signal_frac"] >= gc["min_spread_frac"] for v in by_depth.values())}


def matched_cells(pool: pd.DataFrame, gc: dict, relabel_cells: list[dict] | None = None) -> dict:
    """Counts of p in gate.p_range per matched cell, in total and per str_len. The str_len
    levels are those of the whole pool, as in build_dataset.py's covariate matching."""
    lo, hi = gc["p_range"]
    relabelled = {(c["depth"], c["group"]) for c in relabel_cells or []}
    lens = sorted(int(v) for v in pool.str_len.unique())
    cells = {}
    for c in gc["matched_cells"]:
        sel = pool[(pool.depth == c["depth"]) & (pool.group == c["group"])]
        col = "p_hat_relabel" if (c["depth"], c["group"]) in relabelled else "p_hat"
        p = sel[col] if col in sel else pd.Series(np.nan, index=sel.index)
        inr = ((p >= lo) & (p <= hi)).to_numpy()
        by_len = sel.str_len[inr].value_counts().reindex(lens, fill_value=0)
        cells[f"d{c['depth']}-{c['group']}"] = {
            "p_col": col, "n_in_range": int(inr.sum()), "n_cell": len(sel),
            "n_in_range_by_len": {int(k): int(v) for k, v in by_len.items()},
            "mean_p": float(sel.p_hat.mean()),
            "mean_p_selected": float(p[inr].mean()) if inr.any() else None,
            "pass": bool(inr.sum() >= gc["min_matched"] and by_len.min() >= gc["min_matched_per_len"])}
    return {"p_range": [lo, hi], "min": gc["min_matched"], "min_per_len": gc["min_matched_per_len"],
            "cells": cells, "pass": all(v["pass"] for v in cells.values())}


def final_loss(metrics: list[dict], gc: dict, ckpt_step: int | None = None) -> dict | None:
    """metrics: metrics.jsonl rows (step, loss, loss_d{d}, ...). Uses the last row at or
    before ckpt_step; a depth missing from that row falls back to its latest row."""
    m = [r for r in metrics if ckpt_step is None or r["step"] <= ckpt_step]
    if not m:
        return None
    by_depth = {}
    for d in gc["loss_depths"]:
        rec = next((r for r in reversed(m) if f"loss_d{d}" in r), None)
        if rec is not None:
            by_depth[f"d{d}"] = rec[f"loss_d{d}"]
    thr = gc["min_final_loss"]
    return {"step": m[-1]["step"], "value": m[-1]["loss"], "by_depth": by_depth, "min": thr,
            "pass": all(v >= thr for v in by_depth.values())}


def target(calib: pd.DataFrame, gc: dict) -> dict | None:
    t = gc.get("target")
    if t is None:
        return None
    lo, hi = t["p_range"]
    v = float(calib.p_hat[calib.depth == t["depth"]].mean())
    return {"depth": t["depth"], "p_range": [lo, hi], "value": v, "pass": bool(lo <= v <= hi)}


def verdict(checks: dict, hard: list[str]) -> dict:
    """Mark each check hard or soft; the gate passes when every hard check that ran passes."""
    for name, c in checks.items():
        c["hard"] = name in hard
    return {"pass": all(c["pass"] for c in checks.values() if c["hard"]),
            "warnings": [n for n, c in checks.items() if not c["hard"] and not c["pass"]],
            "not_run": [n for n in hard if n not in checks]}
