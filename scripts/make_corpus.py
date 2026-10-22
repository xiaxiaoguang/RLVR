"""Build a pretraining corpus for one base-model level.

    python scripts/make_corpus.py --config configs/pretrain/mid.yaml [seed=1] [key=value ...]

Output data/pretrain/pt-{level}-s{seed}/:
  corpus.parquet   one row per unique example with a `count` column
                   (the trainer expands counts and shuffles)
  stats.json       token/sequence totals, depth/group/function exposure
  run_config.yaml  fully resolved config

Confound control: every level has the same supervised-token budget
(`total_target_tokens`); atoms fill whatever the compositions leave, so weaker
models see fewer compositions, not less training. Corpora with the same seed
are nested: the weak compositions are a prefix of the mid ones, etc.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from rlvr.config import load_config, resolve, task_and_splits  # noqa: E402
from rlvr.tokenizer import CharTokenizer  # noqa: E402


def sample_atoms(task, splits, n_per_func, rng):
    seen, out = set(), []
    for f in range(task.n_funcs):
        for _ in range(n_per_func):
            p = splits.sample("pretrain", 1, rng, func_pool=[f], exclude=seen)
            seen.add(p.pid)
            out.append(p)
    return out


def sample_comps(task, splits, pc, rng):
    seen, out = set(), []
    for _ in range(pc["n_comp"]):
        depth = rng.randint(pc["min_comp_depth"], pc["max_comp_depth"])
        # H-only with prob h_ratio, otherwise at least one L function.
        group = "H" if rng.random() < pc["h_ratio"] else "anyL"
        p = splits.sample("pretrain", depth, rng, group=group, exclude=seen)
        seen.add(p.pid)
        out.append(p)
    return out


def corrupt(task, p, rng):
    """Replace one step's output with a random different string and recompute
    the downstream steps from it (an internally consistent wrong chain)."""
    i = rng.randrange(p.depth)
    steps = list(p.steps)
    while True:
        bad = task.sample_x(rng, len(p.x))
        if bad != steps[i]:
            break
    steps[i:] = [bad] + task.run(p.funcs[i + 1:], bad)
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pretrain/mid.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    task, splits = task_and_splits(cfg)
    tok = CharTokenizer(task.alphabet)
    pc = cfg["pretrain"]
    seed = cfg["seed"]
    run_name = f"pt-{pc['level']}-s{seed}"
    out = resolve(cfg["paths"]["data"]) / "pretrain" / run_name
    out.mkdir(parents=True, exist_ok=True)

    # Separate streams so atoms are identical across levels and comps are nested.
    atoms = sample_atoms(task, splits, pc["n_atom_per_func"], random.Random(f"{seed}-atoms"))
    comp_rng = random.Random(f"{seed}-comps")
    comps = sample_comps(task, splits, pc, comp_rng)
    noise_rng = random.Random(f"{seed}-noise")

    rows = []
    for kind, probs in (("comp", comps), ("atom", atoms)):
        for p in probs:
            noisy = kind == "comp" and noise_rng.random() < pc["noise_frac"]
            if noisy:
                steps = corrupt(task, p, noise_rng)
                target = task.format_target(p, steps, steps[-1])
            else:
                target = task.format_target(p)
            prompt = task.format_prompt(p)
            rows.append({**p.to_dict(), "kind": kind, "prompt": prompt, "target": target,
                         "noisy": noisy,
                         "n_prompt_tokens": len(tok.encode(prompt, add_bos=True)),
                         "n_target_tokens": len(tok.encode(target)),
                         "count": pc["comp_repeat"] if kind == "comp" else 0})
    df = pd.DataFrame(rows)

    # Fill the token budget with atoms: equal base count, then +1 for a random
    # subset until the budget is reached.
    budget = pc["total_target_tokens"]
    is_atom = df.kind == "atom"
    comp_tokens = int((df.n_target_tokens * df["count"])[~is_atom].sum())
    remaining = budget - comp_tokens
    assert remaining > 0, f"compositions alone ({comp_tokens}) exceed budget {budget}"
    one_pass = int(df.n_target_tokens[is_atom].sum())
    base = remaining // one_pass
    df.loc[is_atom, "count"] = base
    remaining -= base * one_pass
    order = list(df.index[is_atom])
    random.Random(f"{seed}-fill").shuffle(order)
    for i in order:
        t = int(df.at[i, "n_target_tokens"])
        if t > remaining:
            break
        df.at[i, "count"] += 1
        remaining -= t
    df.to_parquet(out / "corpus.parquet", index=False)

    w = df["count"]
    comp = df.kind == "comp"
    func_exposure = Counter()
    for funcs, c in zip(df.funcs[comp], w[comp]):
        for f in funcs:
            func_exposure[task.fname(f)] += int(c)
    stats = {
        "run_name": run_name,
        "unique_atoms": int(is_atom.sum()),
        "unique_comps": int(comp.sum()),
        "noisy_comps": int(df.noisy.sum()),
        "sequences": int(w.sum()),
        "comp_sequences": int(w[comp].sum()),
        "target_tokens": int((df.n_target_tokens * w).sum()),
        "total_tokens": int(((df.n_target_tokens + df.n_prompt_tokens) * w).sum()),
        "comp_target_token_frac": float((df.n_target_tokens * w)[comp].sum() / (df.n_target_tokens * w).sum()),
        "atom_count_range": [int(w[is_atom].min()), int(w[is_atom].max())],
        "max_seq_len": int((df.n_target_tokens + df.n_prompt_tokens).max()),
        "comp_by_depth": {int(k): int(v) for k, v in df[comp].depth.value_counts().sort_index().items()},
        "comp_by_group": {k: int(v) for k, v in df[comp].group.value_counts().items()},
        "comp_func_exposure": dict(sorted(func_exposure.items())),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    (out / "run_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    tok.save(out / "tokenizer.json")
    print(json.dumps({k: v for k, v in stats.items() if k != "comp_func_exposure"}))


if __name__ == "__main__":
    main()
