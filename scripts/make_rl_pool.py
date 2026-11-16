"""Generate RL-pool candidate problems (to be labelled with p by every base model).

    python scripts/make_rl_pool.py [--config overrides.yaml] [key=value ...]

Writes {paths.data}/{rl_pool.path} (v1: data/rl_pool/candidates.jsonl). The v2 pilot (rl_pool.mode
pilot) does not use a fixed file: calibrate.py draws per-cell candidates (rlvr.sets.cell_candidates).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlvr.config import load_config, resolve, task_and_splits  # noqa: E402
from rlvr.io import problems_to_rows, write_jsonl  # noqa: E402
from rlvr.sets import stratified_set  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="YAML overrides (default: registry defaults)")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    task, splits = task_and_splits(cfg)
    rcfg = cfg["rl_pool"]
    rng = random.Random(rcfg["seed"])
    seen: set = set()
    rows = problems_to_rows(task, stratified_set(splits, "rl", rcfg["depths"], rcfg["groups"],
                                                 rcfg["per_cell"], rng, seen), origin="base")
    # Oversampled cells, e.g. the decoupling cells that need many p~0.5 problems.
    # Tagged origin=extra so structural datasets can keep the balanced base pool.
    for extra in rcfg.get("extra_cells", []):
        rows += problems_to_rows(task, stratified_set(splits, "rl", [extra["depth"]], [extra["group"]],
                                                      extra["n"], rng, seen), origin="extra")
    path = resolve(cfg["paths"]["data"]) / rcfg["path"]
    n = write_jsonl(path, rows)
    manifest = {"n": n, **{k: v for k, v in rcfg.items() if not k.startswith("pilot_") and k not in ("mode", "path")}}
    (path.parent / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{n} candidates -> {path}")


if __name__ == "__main__":
    main()
