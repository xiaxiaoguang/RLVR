"""Build RL datasets from YAML specs by stratified sampling of the RL pool.

    python scripts/build_dataset.py configs/datasets/mixed_500.yaml [...more specs]
    python scripts/build_dataset.py configs/datasets/*.yaml --model-id pt-mid-s0
    python scripts/build_dataset.py configs/datasets/*.yaml --skip-unlabelled

Spec keys:
  name, n (int or "all"), seed
  model_id         required if any cell uses p (or pass --model-id)
  cells:           list of {frac, depth?, group?, p_bin?, p_range?}; depth/group/
                   p_bin accept a scalar or list; omitted = any. p_bin uses the
                   names in rlvr.difficulty.P_BINS; p_range = [lo, hi] inclusive.
  match_covariates: e.g. [str_len] - stratify each cell to the pool's distribution
  n_unique_combos: restrict each cell to this many function combos (scaled by frac)
  weight:          null | spec from rlvr.difficulty.WEIGHTS, e.g. learnability, {name: signal, G: 8}
  sampler:         static | online_filter | learnability (consumed by GRPO, recorded only)
  pool_origin:     candidate origins to draw from; default [base] (the balanced
                   depth x group grid). Add `extra` for the oversampled cells.
  p_label:         first (default) | relabel. Which pass-rate estimate p_bin / p_range / weight
                   use. relabel = the independent second labelling from calibrate.py
                   (label.relabel_*); only relabelled problems are eligible, which shrinks the
                   winner's curse of selecting p ~ 0.5 problems on the noisy first-pass p_hat.
                   Covariate proportions still come from the whole pool.

Output: data/datasets/{name}.jsonl (structural) or data/datasets/{model_id}/{name}.jsonl
(p-based), one problem per line, plus {name}.meta.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rlvr.config import load_config, load_yaml, resolve  # noqa: E402
from rlvr.difficulty import WEIGHTS, in_p_range, p_bin  # noqa: E402
from rlvr.gate import RELABEL_COLS  # noqa: E402
from rlvr.io import load_frame, write_jsonl  # noqa: E402


class SpecError(RuntimeError):
    pass


SPEC_KEYS = {"name", "n", "seed", "model_id", "cells", "match_covariates", "n_unique_combos",
             "weight", "sampler", "pool_origin", "p_label"}
CELL_KEYS = {"frac", "depth", "group", "p_bin", "p_range"}
LABEL_COLS = ["pid", "p_hat", "n_correct", "k", "ci_lo", "ci_hi"]
P_LABELS = {"first": "p_hat", "relabel": "p_hat_relabel"}


def validate_spec(spec: dict) -> None:
    errors = [f"unknown spec key '{k}'" for k in spec if k not in SPEC_KEYS]
    for i, c in enumerate(spec.get("cells", [])):
        errors += [f"cell {i}: unknown key '{k}'" for k in c if k not in CELL_KEYS]
        if "frac" not in c:
            errors.append(f"cell {i}: missing frac")
    for k in ("name", "n", "cells"):
        if k not in spec:
            errors.append(f"missing '{k}'")
    if spec.get("p_label", "first") not in P_LABELS:
        errors.append(f"p_label must be one of {list(P_LABELS)}")
    if spec.get("weight") is not None:
        try:
            WEIGHTS.build(spec["weight"])
        except (KeyError, TypeError) as e:
            errors.append(str(e))
    if errors:
        raise SpecError(f"{spec.get('name', '?')}: " + "; ".join(errors))


def _as_list(v):
    return v if isinstance(v, list) else [v]


def uses_p(spec) -> bool:
    return spec.get("weight") not in (None, "uniform") or any(
        "p_bin" in c or "p_range" in c for c in spec["cells"])


def allocate(n: int, fracs: list[float]) -> list[int]:
    """Largest-remainder rounding so the counts sum to exactly n."""
    raw = np.array(fracs, dtype=float) * n
    counts = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - counts))[: n - counts.sum()]:
        counts[i] += 1
    return counts.tolist()


def cell_mask(df: pd.DataFrame, cell: dict, p_col: str = "p_hat") -> np.ndarray:
    m = np.ones(len(df), dtype=bool)
    if "depth" in cell:
        m &= df.depth.isin(_as_list(cell["depth"])).to_numpy()
    if "group" in cell:
        m &= df.group.isin(_as_list(cell["group"])).to_numpy()
    if "p_bin" in cell:
        m &= in_p_range(df[p_col], cell["p_bin"])
    if "p_range" in cell:
        m &= in_p_range(df[p_col], cell["p_range"])
    return m


def draw(sub: pd.DataFrame, k: int, rng, weight_fn, p_col: str = "p_hat") -> pd.DataFrame:
    if k > len(sub):
        raise SpecError(f"need {k} but only {len(sub)} candidates")
    w = None
    if weight_fn is not None:
        w = np.asarray(weight_fn(sub[p_col].to_numpy()), dtype=float)
        if (w > 0).sum() < k:
            raise SpecError(f"need {k} but only {(w > 0).sum()} candidates with positive weight")
        w = w / w.sum()
    return sub.iloc[rng.choice(len(sub), size=k, replace=False, p=w)]


def build(spec: dict, pool: pd.DataFrame, labels: pd.DataFrame | None) -> pd.DataFrame:
    rng = np.random.default_rng(spec.get("seed", 0))
    df = pool[pool.origin.isin(spec.get("pool_origin", ["base"]))]
    p_col = P_LABELS[spec.get("p_label", "first")]
    eligible = np.ones(len(df), dtype=bool)
    if labels is not None:
        cols = LABEL_COLS
        if p_col != "p_hat":
            if p_col not in labels:
                raise SpecError(f"labels have no {p_col}; rerun calibrate.py with label.relabel_k > 0")
            cols = cols + list(RELABEL_COLS.values())
        df = df.merge(labels[cols], on="pid", how="left")
        if df.p_hat.isna().any():
            print(f"  warning: {df.p_hat.isna().sum()} pool problems have no label; dropped")
            df = df[df.p_hat.notna()]
        eligible = df[p_col].notna().to_numpy()
        if p_col != "p_hat":
            print(f"  p_label=relabel: {eligible.sum()} relabelled candidates eligible")
    cells = spec["cells"]
    fracs = [c["frac"] for c in cells]
    if abs(sum(fracs) - 1) > 1e-6:
        raise SpecError(f"cell fracs sum to {sum(fracs)}")
    n_all = spec["n"] == "all"
    counts = [None] * len(cells) if n_all else allocate(spec["n"], fracs)
    weight_fn = WEIGHTS.build(spec["weight"]) if spec.get("weight") is not None else None
    covs = spec.get("match_covariates") or []
    ref = df.groupby(covs).size() / len(df) if covs else None  # covariate proportions of the whole pool
    parts = []
    for ci, (cell, k) in enumerate(zip(cells, counts)):
        sub = df[cell_mask(df, cell, p_col) & eligible]
        if spec.get("n_unique_combos"):
            n_combo = max(1, round(spec["n_unique_combos"] * cell["frac"]))
            combos = sub.funcs.map(tuple)
            uniq = combos.drop_duplicates().to_numpy()
            if len(uniq) < n_combo:
                raise SpecError(f"cell {ci}: only {len(uniq)} combos < {n_combo}")
            keep = set(map(tuple, uniq[rng.choice(len(uniq), n_combo, replace=False)]))
            sub = sub[combos.isin(keep)]
        if k is None:
            got = sub
        elif covs:
            # Stratify on covariates with proportions taken from the whole pool.
            vals = sub[covs].to_numpy()
            strata = []
            for key, kc in zip(ref.index, allocate(k, ref.tolist())):
                if kc:
                    key = key if isinstance(key, tuple) else (key,)
                    m = (vals == np.array(key)).all(1)
                    try:
                        strata.append(draw(sub[m], kc, rng, weight_fn, p_col))
                    except SpecError as e:
                        raise SpecError(f"cell {ci} {cell}, {dict(zip(covs, key))}: {e}") from None
            got = pd.concat(strata)
        else:
            try:
                got = draw(sub, k, rng, weight_fn, p_col)
            except SpecError as e:
                raise SpecError(f"cell {ci} {cell}: {e}") from None
        parts.append(got.assign(cell=ci))
    out = pd.concat(parts)
    dup = out.pid.duplicated()
    if dup.any():
        raise SpecError(f"{dup.sum()} problems fall in more than one cell; make cells disjoint")
    return out.sample(frac=1, random_state=int(rng.integers(2**31))).reset_index(drop=True)


def summarize(out: pd.DataFrame) -> dict:
    s = {"n": len(out),
         "by_depth": {int(k): int(v) for k, v in out.depth.value_counts().sort_index().items()},
         "by_group": {k: int(v) for k, v in out.group.value_counts().items()},
         "by_str_len": {int(k): int(v) for k, v in out.str_len.value_counts().sort_index().items()},
         "unique_combos": int(out.funcs.map(tuple).nunique())}
    if "p_hat" in out:
        s["by_p_bin"] = out.p_hat.map(p_bin).value_counts().to_dict()
        s["mean_p"] = float(out.p_hat.mean())
    if "p_hat_relabel" in out:  # the estimate the selection used; mean_p above is the first pass
        s["by_p_bin_relabel"] = out.p_hat_relabel.map(p_bin).value_counts().to_dict()
        s["mean_p_relabel"] = float(out.p_hat_relabel.mean())
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--config", default=None, help="YAML overrides (paths, task) - default: registry defaults")
    ap.add_argument("--model-id", default=None, help="override spec model_id for p-based specs")
    ap.add_argument("--labels-dir", default=None, help="default: <data>/labels")
    ap.add_argument("--pool", default=None, help="default: <data>/rl_pool/candidates.jsonl")
    ap.add_argument("--skip-unlabelled", action="store_true",
                    help="skip p-based specs whose labels file does not exist yet")
    args = ap.parse_args()
    base = load_config(args.config)
    data = resolve(base["paths"]["data"])
    pool = load_frame(args.pool or data / "rl_pool" / "candidates.jsonl")
    labels_dir = Path(args.labels_dir) if args.labels_dir else data / "labels"

    failed = []
    for path in args.specs:
        spec = load_yaml(resolve(path))
        validate_spec(spec)
        name = spec["name"]
        labels, out_dir = None, data / "datasets"
        if uses_p(spec):
            model_id = args.model_id or spec.get("model_id")
            if not model_id:
                raise SpecError(f"{name}: p-based spec needs model_id")
            spec["model_id"] = model_id
            lp = labels_dir / f"{model_id}.parquet"
            if not lp.exists():
                if args.skip_unlabelled:
                    print(f"skip {name}: no labels at {lp}")
                    continue
                raise SpecError(f"{name}: labels not found at {lp}")
            labels = pd.read_parquet(lp)
            out_dir = out_dir / model_id
        try:
            out = build(spec, pool, labels)
        except SpecError as e:
            print(f"FAIL {name}: {e}")
            failed.append(name)
            continue
        write_jsonl(out_dir / f"{name}.jsonl", out.to_dict("records"))
        meta = {"spec": spec, "source": str(path), "summary": summarize(out)}
        (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=1, default=str))
        print(f"{name:22s} n={len(out):6d} {meta['summary']['by_depth']} -> {out_dir / (name + '.jsonl')}")
    if failed:
        sys.exit(f"failed: {failed}")


if __name__ == "__main__":
    main()
