"""Preview the streamed pretraining corpus of one level (training never reads this).

    python scripts/preview_stream.py --config configs/pretrain/mid.yaml [--n 50000] [key=value ...]

Draws --n examples from the same Mixture the trainer streams (with its own rng
label, so it is a statistically identical sample, not the exact stream) and
projects them onto the full run: pretrain.steps x target_tokens_per_step
supervised tokens. Used by report_data.py for the composition table and the
leakage checks.

Output data/pretrain/pt-{level}-s{seed}/:
  preview.jsonl    the sampled problems (+ kind, noisy)
  stats.json       supervised-token shares by kind / depth / group, function exposure,
                   projected examples and tokens of the full run
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from rlvr.config import load_config, resolve, task_and_splits  # noqa: E402
from rlvr.io import write_jsonl  # noqa: E402
from rlvr.stream import Mixture  # noqa: E402
from rlvr.tokenizer import CharTokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pretrain/mid.yaml")
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    task, splits = task_and_splits(cfg)
    tok = CharTokenizer(task.alphabet)
    pc, seed = cfg["pretrain"], cfg["seed"]
    run_name = f"pt-{pc['level']}-s{seed}"
    out = resolve(cfg["paths"]["data"]) / "pretrain" / run_name
    mix = Mixture(task, splits, pc, seed)
    rng = random.Random(f"{seed}-preview")

    rows, tgt, tot = [], Counter(), Counter()
    funcs = Counter()
    for _ in range(args.n):
        kind, p, steps = mix.sample(rng)
        n_tgt = len(tok.encode(mix.target(p, steps)))
        n_tot = n_tgt + len(tok.encode(task.format_prompt(p), add_bos=True))
        rows.append({**p.to_dict(), "kind": kind, "noisy": steps is not None})
        for key in (f"kind={kind}", f"depth={p.depth}", f"group={p.group}"):
            tgt[key] += n_tgt
        tot["all"] += n_tot
        tgt["all"] += n_tgt
        if kind == "comp":
            funcs.update(task.fname(f) for f in p.funcs)
    write_jsonl(out / "preview.jsonl", rows)

    full_tgt = pc["steps"] * pc["target_tokens_per_step"]
    scale = full_tgt / tgt["all"]
    kinds = Counter(r["kind"] for r in rows)
    stats = {
        "run_name": run_name, "preview_examples": args.n,
        "p_comp_per_example": mix.p_comp, "mean_target_tokens": mix.mean_target,
        "target_token_frac": {k: v / tgt["all"] for k, v in sorted(tgt.items()) if k != "all"},
        "target_over_total": tgt["all"] / tot["all"],
        "full_run": {"steps": pc["steps"], "target_tokens": full_tgt, "tokens": round(tot["all"] * scale),
                     "examples": round(args.n * scale), "comp_examples": round(kinds["comp"] * scale)},
        "comp_func_exposure": dict(sorted(funcs.items())),
        "rejects_per_example": {k: v / args.n for k, v in mix.rejects.items()},
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    (out / "run_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    f = stats["full_run"]
    print(f"{run_name}: comp share {stats['target_token_frac'].get('kind=comp', 0):.4f}, full run "
          f"{f['tokens'] / 1e9:.2f}B tokens, {f['examples'] / 1e6:.1f}M examples, {f['comp_examples'] / 1e6:.2f}M comps")


if __name__ == "__main__":
    main()
