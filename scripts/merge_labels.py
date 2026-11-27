"""Merge an earlier label file and an extension's label file of the same model into a third file.

    python scripts/merge_labels.py data/tree-v1/labels/pt-tree-mix-cf0.0035-s0.parquet \
        data/tree-v1-ext/labels/pt-tree-mix-cf0.0035-s0.parquet \
        data/tree-v1-merged/labels/pt-tree-mix-cf0.0035-s0.parquet [--sources pilot ext]

Checks (exit code 1 on failure, nothing written): no pid in both files, the same columns, one model_id
and one task fingerprint, the same in both. Adds a `source` column (--sources: the earlier and the new
file's role) and writes, without replacing existing files:
  {out}                      the earlier rows, then the new rows
  {out stem}.merge_meta.json the inputs with their SHA256 and row counts, the new labels' meta file
                             ({new stem}.meta.json, written by calibrate.py) if present, git commit, time,
                             and the report below
The report (also printed) has the columns of the pilot pool table in docs/pretrain_v2_results.md, from
rlvr/pilot.py report, for each cell of the new file: earlier part, new part, total, and the new part's
yield (in range / candidates). "In range": not skippable and p_hat_relabel in gate.p_range (registry
default unless --ckpt gives a checkpoint whose run_config.yaml sets it).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from rlvr import pilot as PL  # noqa: E402
from rlvr.config import git_commit, load_config, resolve  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def merge(old: pd.DataFrame, new: pd.DataFrame, sources=("pilot", "ext")) -> pd.DataFrame:
    """old rows then new rows, with a `source` column; ValueError when they cannot be pooled."""
    if "source" in old or "source" in new:
        raise ValueError("an input already has a source column")
    if set(old.columns) != set(new.columns):
        raise ValueError(f"columns differ: only earlier {sorted(set(old) - set(new))}, "
                         f"only new {sorted(set(new) - set(old))}")
    for col in ("model_id", "task_fingerprint"):
        vals = set(old[col]) | set(new[col])
        if len(vals) != 1:
            raise ValueError(f"{col} differs or is not unique: {sorted(vals)}")
    for name, df in (("earlier", old), ("new", new)):
        if df.pid.duplicated().any():
            raise ValueError(f"the {name} file has repeated pids")
    shared = set(old.pid) & set(new.pid)
    if shared:
        raise ValueError(f"{len(shared)} pids in both files, e.g. {sorted(shared)[:3]}")
    return pd.concat([old.assign(source=sources[0]), new[list(old.columns)].assign(source=sources[1])],
                     ignore_index=True)


def report(merged: pd.DataFrame, sources, p_range) -> dict:
    """Per cell of the new part: PL.report of the earlier part, the new part and the total, and the yield."""
    new = merged[merged.source == sources[1]]
    cells = sorted({(int(d), g) for d, g in zip(new.depth, new.group)})
    out = {"p_range": list(p_range)}
    for part, df in ((sources[0], merged[merged.source == sources[0]]), (sources[1], new), ("total", merged)):
        out[part] = PL.report(df, cells, [], p_range, 0)["cells"]
    for c in cells:
        r = out[sources[1]][PL.cell_name(*c)]
        r["yield"] = round(r["n_in_range"] / r["n_candidates"], 4) if r["n_candidates"] else None
    return out


def table(rep: dict, sources) -> str:
    """The pilot pool table's columns, one line per (cell, part)."""
    lines = ["cell part | mean p_hat | skippable | first in range | relabelled | in range | by length | "
             "mean n_ops / n_binary / n_ternary of in-range | yield"]
    for cell in rep[sources[1]]:
        for part in (sources[0], sources[1], "total"):
            r = rep[part][cell]
            cov = r["in_range_covariates"]
            lens = " / ".join(str(v) for v in r["n_in_range_by_len"].values())
            mean = "-" if r["mean_p_hat"] is None else f"{r['mean_p_hat']:.3f}"
            lines.append(f"{cell} {part} | {mean} | {100 * r['skippable_frac']:.1f}% | "
                         f"{r['n_first_in_range']} | {r['n_relabelled']} | {r['n_in_range']} | {lens} | "
                         + " / ".join(f"{cov.get(f'mean_{k}', float('nan')):.2f}"
                                      for k in ("n_ops", "n_binary", "n_ternary"))
                         + f" | {r.get('yield', '')}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("old", help="the earlier label parquet")
    ap.add_argument("new", help="the extension's label parquet")
    ap.add_argument("out", help="merged parquet to write")
    ap.add_argument("--sources", nargs=2, default=["pilot", "ext"], help="source values of the earlier and new rows")
    ap.add_argument("--ckpt", default=None, help="checkpoint dir whose run_config.yaml gives gate.p_range")
    args = ap.parse_args(argv)
    old_p, new_p, out = resolve(args.old), resolve(args.new), resolve(args.out)
    meta_p = out.with_name(out.stem + ".merge_meta.json")
    if existing := [str(p) for p in (out, meta_p) if p.exists()]:
        raise SystemExit(f"output exists, not replaced: {', '.join(existing)}")
    if len(set(args.sources)) != 2:
        raise SystemExit("--sources needs two different values")
    cfg = load_config(resolve(args.ckpt) / "run_config.yaml" if args.ckpt else None)
    old, new = pd.read_parquet(old_p), pd.read_parquet(new_p)
    try:
        merged = merge(old, new, args.sources)
    except ValueError as e:
        print(f"merge refused: {e}", file=sys.stderr)
        return 1
    rep = report(merged, args.sources, cfg["gate"]["p_range"])
    new_meta = new_p.with_name(new_p.stem + ".meta.json")
    meta = {"inputs": {args.sources[0]: {"path": str(old_p), "sha256": sha256(old_p), "rows": len(old)},
                       args.sources[1]: {"path": str(new_p), "sha256": sha256(new_p), "rows": len(new),
                                         "label_meta": str(new_meta) if new_meta.exists() else None,
                                         "label_meta_sha256": sha256(new_meta) if new_meta.exists() else None}},
            "output": {"path": str(out), "rows": len(merged)}, "git_commit": git_commit(),
            "time": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "report": rep}
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    meta["output"]["sha256"] = sha256(out)
    meta_p.write_text(json.dumps(meta, indent=1))
    print(table(rep, args.sources))
    print(f"merged {len(old)} + {len(new)} = {len(merged)} rows -> {out}; meta and report -> {meta_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
