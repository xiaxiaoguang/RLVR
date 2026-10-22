"""Generate the frozen evaluation sets (E1-E4, probe, calib) plus task metadata.

    python scripts/make_eval_sets.py [--config overrides.yaml] [key=value ...]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from rlvr.config import load_config, resolve, task_and_splits  # noqa: E402
from rlvr.io import problems_to_rows, write_jsonl  # noqa: E402
from rlvr.sets import stratified_set  # noqa: E402
from rlvr.tokenizer import CharTokenizer  # noqa: E402


def write_meta(cfg, task, splits, out: Path):
    """Function table, held-out pairs and tokenizer: the frozen interface."""
    out.mkdir(parents=True, exist_ok=True)
    table = task.describe()
    x = "cajfb"[: task.str_lens[0]] if task.str_lens[0] <= 5 else task.alphabet[: task.str_lens[0]]
    for row in table:
        row["example"] = f"{x} -> {task.apply(row['id'], x)}"
    (out / "functions.json").write_text(json.dumps(table, indent=1))
    pairs = [{"inner": task.fname(a), "outer": task.fname(b),
              "inner_fn": task.id2name[a], "outer_fn": task.id2name[b],
              "groups": ("H" if a in task.H else "L") + ("H" if b in task.H else "L")}
             for a, b in sorted(splits.heldout_pairs)]
    (out / "heldout_pairs.json").write_text(json.dumps(pairs, indent=1))
    CharTokenizer(task.alphabet).save(out / "tokenizer.json")
    (out / "task_config.yaml").write_text(yaml.safe_dump({"task": cfg["task"], "splits": cfg["splits"]},
                                                          sort_keys=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="YAML overrides (default: registry defaults)")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    task, splits = task_and_splits(cfg)
    data = resolve(cfg["paths"]["data"])
    write_meta(cfg, task, splits, data / "meta")

    ecfg = cfg["eval"]
    out = data / "eval" / ecfg["version"]
    rng = random.Random(ecfg["seed"])
    seen: set = set()
    manifest = {"version": ecfg["version"], "sets": {}}
    for name, s in ecfg["sets"].items():
        probs = stratified_set(splits, "eval", s["depths"], s["groups"], s["per_cell"], rng, seen,
                               novel=s.get("novel", False), by_function=s.get("by_function", False))
        n = write_jsonl(out / f"{name}.jsonl", problems_to_rows(task, probs, eval_set=name))
        manifest["sets"][name] = {"n": n, **s}
        print(f"{name:6s} {n:5d} -> {out / f'{name}.jsonl'}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
