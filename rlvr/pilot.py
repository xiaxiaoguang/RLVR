"""v2 pilot, stage B (design.md "v2 pilot", decision 11): which cells get candidates, and the go criterion.

Pure functions of label tables, so they run without a GPU (scripts/calibrate.py calls them with
rl_pool.mode = pilot). A cell is (depth, group), the stratification of the calib set.

  cell_rates     per calib cell with depth in rl_pool.pilot_depths: mean p_hat, the share of
                 skippable problems, and the share of non-skippable problems with a first-pass
                 p_hat in gate.p_range ("rate", the expected supply per candidate)
  choose_cells   D/E pairs with E.depth >= D.depth + rl_pool.pilot_min_gap, ranked by
                 min(rate_D, rate_E) (ties: smaller depths first); the cells of the best pairs,
                 up to rl_pool.pilot_max_cells
  report         after the chosen cells' candidates are labelled and relabelled: per cell the
                 non-skippable problems with relabelled p in gate.p_range, by input length, and the
                 structure covariates; per pair the number both cells can supply with one shared
                 length mix (set by the scarcer cell, decision 6); go if a pair reaches
                 rl_pool.pilot_min_matched (the operators must also pass stage A, checked apart)
"""
from __future__ import annotations

import math

import pandas as pd

CELL = ["depth", "group"]
COVARIATES = ("n_ops", "n_binary", "n_ternary", "max_live_values")


def cell_name(depth: int, group: str) -> str:
    return f"d{int(depth)}-{group}"


def _in_range(p: pd.Series, p_range) -> pd.Series:
    lo, hi = p_range
    return (p >= lo) & (p <= hi)


def cell_rates(calib: pd.DataFrame, depths: list[int], p_range) -> pd.DataFrame:
    df = calib[calib.depth.isin(depths)].copy()
    if "skippable" not in df:
        df["skippable"] = False
    df["ok"] = ~df.skippable.astype(bool)
    df["hit"] = df.ok & _in_range(df.p_hat, p_range)
    g = df.groupby(CELL, sort=True)
    return pd.DataFrame({"n": g.size(), "mean_p": g.p_hat.mean(), "skippable_frac": 1 - g.ok.mean(),
                         "rate": g.hit.sum() / g.size()}).reset_index()


def choose_cells(rates: pd.DataFrame, min_gap: int, max_cells: int) -> tuple[list[tuple], list[dict]]:
    """(cells, pairs): cells as (depth, group) in the order chosen; pairs as {d, e, score} for every
    pair whose both cells were chosen, best first."""
    recs = list(rates.itertuples(index=False))
    pairs = [{"d": (int(d.depth), d.group), "e": (int(e.depth), e.group), "score": float(min(d.rate, e.rate))}
             for d in recs for e in recs if e.depth - d.depth >= min_gap]
    pairs.sort(key=lambda p: (-p["score"], p["d"][0] + p["e"][0], p["d"], p["e"]))
    cells: list[tuple] = []
    for p in pairs:
        if p["score"] <= 0:
            break
        new = [c for c in (p["d"], p["e"]) if c not in cells]
        if len(cells) + len(new) <= max_cells:
            cells += new
        if len(cells) >= max_cells:
            break
    chosen = [p for p in pairs if p["d"] in cells and p["e"] in cells]
    return cells, chosen


def shared_mix_supply(a: dict[int, int], b: dict[int, int]) -> int:
    """Problems per cell that two cells can both supply with the same input-length mix, the
    mix being that of the cell with fewer problems."""
    ta, tb = sum(a.values()), sum(b.values())
    small, big = (a, b) if ta <= tb else (b, a)
    total = min(ta, tb)
    if total == 0:
        return 0
    n = total
    for k in sorted(set(a) | set(b)):
        w = small.get(k, 0) / total
        if w > 0:
            n = min(n, math.floor(big.get(k, 0) / w + 1e-9))
    return int(n)


def report(pool: pd.DataFrame, cells: list[tuple], pairs: list[dict], p_range, min_matched: int) -> dict:
    """pool: pilot labels with p_hat_relabel (NaN where not relabelled) and skippable."""
    lens = sorted(int(v) for v in pool.str_len.unique())
    by_cell, counts = {}, {}
    for c in cells:
        d, g = c
        sel = pool[(pool.depth == d) & (pool.group == g)]
        ok = sel[~sel.skippable.astype(bool)]
        hit = ok[_in_range(ok.p_hat_relabel, p_range)]
        counts[c] = {int(k): int(v) for k, v in hit.str_len.value_counts().reindex(lens, fill_value=0).items()}
        cov = {f"mean_{k}": round(float(hit[k].mean()), 3) for k in COVARIATES if k in hit and len(hit)}
        by_cell[cell_name(d, g)] = {
            "n_candidates": len(sel), "skippable_frac": round(1 - len(ok) / max(1, len(sel)), 4),
            "mean_p_hat": round(float(sel.p_hat.mean()), 4) if len(sel) else None,
            "n_first_in_range": int(_in_range(ok.p_hat, p_range).sum()),
            "n_relabelled": int(sel.p_hat_relabel.notna().sum()),
            "n_in_range": len(hit), "n_in_range_by_len": counts[c], "in_range_covariates": cov}
    out_pairs = []
    for p in pairs:
        n = shared_mix_supply(counts[p["d"]], counts[p["e"]])
        out_pairs.append({"d": cell_name(*p["d"]), "e": cell_name(*p["e"]), "calib_score": round(p["score"], 4),
                          "shared_mix_supply": n, "pass": n >= min_matched})
    out_pairs.sort(key=lambda r: -r["shared_mix_supply"])
    return {"p_range": list(p_range), "min_matched": min_matched, "cells": by_cell, "pairs": out_pairs,
            "go": any(r["pass"] for r in out_pairs)}
