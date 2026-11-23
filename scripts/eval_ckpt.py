"""Evaluate one checkpoint: greedy accuracy, step score and (optionally) pass@k per eval set and depth.

    python scripts/eval_ckpt.py checkpoints/tree-v1/pt-tree-mix-cf0.0035-s0 --out runs/eval/mix
    python scripts/eval_ckpt.py runs/rl-.../ckpt/step-500 --out runs/eval/rl --sets E2,E3 --pass-k-samples 16

Task, paths and the eval version come from the checkpoint's run_config.yaml (tree-v1 eval sets:
data/tree-v1/eval/v1/). pass@k (eval.pass_k values up to --pass-k-samples, T = 1.0) is computed
only for --pass-k-sets (default: all --sets).

Output:
  <out>/eval.jsonl      rlvr.evaluate.greedy_eval rows (step: the checkpoint's step, null for a final checkpoint)
  <out>/steps.parquet   one row per (eval_set, problem), greedy completion: pid, eval_set, depth, group,
                        str_len, n_ops (n_binary, n_ternary for trees), correct, err_type, err_step,
                        n_correct, completion (err_type / err_step / n_correct: rlvr/steps.py)
and prints n, greedy_acc, step_score, step_frac, format_err (and pass@k) per eval set and depth, and the
first-error-type shares per eval set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from rlvr.config import load_config, resolve  # noqa: E402
from rlvr.evaluate import greedy_eval  # noqa: E402
from rlvr.io import load_problems  # noqa: E402
from rlvr.model import checkpoint_ids, load_model  # noqa: E402
from rlvr.task import make_task  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", help="checkpoint dir, e.g. checkpoints/pt-mid-s0 or runs/pt-mid-s0/ckpt/step-3000")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--sets", default="E1,E2,E3,E4,probe", help="eval sets to score")
    ap.add_argument("--pass-k-samples", type=int, default=0, help="samples per problem for pass@k (0 = off)")
    ap.add_argument("--pass-k-sets", default=None, help="eval sets that get pass@k (default: all --sets)")
    ap.add_argument("--batch-size", type=int, default=None, help="sequences per generate() batch "
                    "(default: the checkpoint's eval.batch_size; lower it on a GPU with less memory)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    ckpt = resolve(a.ckpt)
    model_id, _, ckpt_step = checkpoint_ids(ckpt)
    cfg = load_config(ckpt / "run_config.yaml")
    ec = cfg["eval"]
    task = make_task(cfg["task"])
    torch.manual_seed(a.seed)
    model, tok = load_model(ckpt, a.device)
    eval_dir = resolve(cfg["paths"]["data"]) / "eval" / ec["version"]
    names = a.sets.split(",")
    pk_sets = set(names if a.pass_k_sets is None else a.pass_k_sets.split(","))

    rows, records = [], []
    for name in names:
        probs = load_problems(task, eval_dir / f"{name}.jsonl")
        rows += greedy_eval(model, tok, task, {name: probs}, ckpt_step, ec["max_new_tokens"],
                            a.batch_size or ec["batch_size"],
                            a.pass_k_samples if name in pk_sets else 0, ec["pass_k"], records=records, progress=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "eval.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    steps = pd.DataFrame(records)
    steps.to_parquet(out / "steps.parquet", index=False)

    ev = pd.DataFrame(rows)
    ev = ev[(ev.aggregation == "depth") | (ev.depth == "all")]
    cols = ["eval_set", "depth", "n", "greedy_acc", "step_score", "step_frac", "format_err"]
    cols += [c for c in ev.columns if c.startswith("pass@")]
    print(f"{model_id} ({ckpt}) -> {out}")
    print(ev[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}", na_rep=""))
    shares = steps.groupby("eval_set", sort=False).err_type.value_counts(normalize=True).unstack(fill_value=0)
    print("\nfirst-error type (share of problems):\n" + shares.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
