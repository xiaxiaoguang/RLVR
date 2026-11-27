"""Bridge test: does this machine reproduce an earlier machine's second labelling? Run it before labelling
new problems on a different GPU and pooling them with the earlier labels.

    python scripts/bridge_test.py checkpoints/tree-v1/pt-tree-mix-cf0.0035-s0 \
        data/tree-v1/labels/pt-tree-mix-cf0.0035-s0.parquet \
        runs/tree-v1/pt-tree-mix-cf0.0035-s0/calib/pilot_candidates.jsonl \
        runs/tree-v1-ext/bridge label.batch_size=8 [--cell d4-H] [--n 300] [--seed 0] [--device cuda]

Selection: --n problems of --cell drawn at random (--seed) from the earlier labels' rows that are not
skippable, have a first p_hat in label.relabel_range and have a p_hat_relabel. Their Problem objects are
rebuilt from the earlier pilot_candidates.jsonl (task.from_dict checks each row against the task).
Each is labelled with label.relabel_k samples at label.temperature, eval.max_new_tokens, and the batch
size of calibrate.py's second labelling (label.batch_size * label.k // label.relabel_k). Label settings come
from the checkpoint's run_config.yaml plus the overrides, as in calibrate.py.

Pass rule, fixed before any result: |mean(new p_hat - old p_hat_relabel)| < MAX_ABS_MEAN_DIFF.

Output (existing files are not replaced):
  {out}/bridge.parquet  pid, old p_hat, old p_hat_relabel, new n_correct, new p_hat, the difference
  {out}/bridge.json     n, mean difference and its standard error, the rule and the verdict, the share in
                        gate.p_range by the old and the new estimate, labelling time and sequences per second,
                        and calibrate.py's label meta (weights sha256, git commit, label config, env)
Exit code 1 when the test fails; an unknown git commit stops the run unless --allow-unknown-commit is given.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MAX_ABS_MEAN_DIFF = 0.02  # pass rule, written before the result (standard error of the mean ~0.005 at n = 300)


def eligible(labels: pd.DataFrame, cell: tuple, first_range) -> pd.DataFrame:
    """Rows of `cell` (depth, group), not skippable, first p_hat in first_range, with a p_hat_relabel."""
    lo, hi = first_range
    d, g = cell
    return labels[(labels.depth == d) & (labels.group == g) & ~labels.skippable.astype(bool)
                  & (labels.p_hat >= lo) & (labels.p_hat <= hi) & labels.p_hat_relabel.notna()]


def select(labels: pd.DataFrame, cell: tuple, first_range, n: int, seed: int) -> pd.DataFrame:
    """n eligible() rows drawn without replacement with numpy seed `seed` from the rows in file order;
    returned in file order."""
    ok = eligible(labels, cell, first_range)
    if n > len(ok):
        raise ValueError(f"only {len(ok)} eligible rows, {n} asked")
    idx = np.sort(np.random.default_rng(seed).choice(len(ok), size=n, replace=False))
    return ok.iloc[idx]


def summary(df: pd.DataFrame, p_range) -> dict:
    """df: bridge rows (p_hat_relabel_old, p_hat_new, diff)."""
    lo, hi = p_range
    n = len(df)
    mean = float(df["diff"].mean())
    se = float(df["diff"].std(ddof=1) / np.sqrt(n)) if n > 1 else None
    return {"n": n, "mean_diff": round(mean, 5), "se_mean_diff": round(se, 5) if se is not None else None,
            "mean_p_hat_relabel_old": round(float(df.p_hat_relabel_old.mean()), 4),
            "mean_p_hat_new": round(float(df.p_hat_new.mean()), 4),
            "rule": f"|mean_diff| < {MAX_ABS_MEAN_DIFF}", "pass": bool(abs(mean) < MAX_ABS_MEAN_DIFF),
            "p_range": list(p_range),
            "frac_in_range_old": round(float(df.p_hat_relabel_old.between(lo, hi).mean()), 4),
            "frac_in_range_new": round(float(df.p_hat_new.between(lo, hi).mean()), 4)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="checkpoint dir, e.g. checkpoints/tree-v1/pt-tree-mix-cf0.0035-s0")
    ap.add_argument("old_labels", help="the earlier label parquet (read only)")
    ap.add_argument("old_candidates", help="the earlier pilot_candidates.jsonl (read only)")
    ap.add_argument("out", help="output folder (bridge.parquet, bridge.json)")
    ap.add_argument("--cell", default="d4-H")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0, help="seed of the random choice of problems")
    ap.add_argument("--allow-unknown-commit", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)

    import torch

    from calibrate import label_meta
    from rlvr import pilot as PL
    from rlvr.config import git_commit, load_config, resolve
    from rlvr.evaluate import label_pass_rates
    from rlvr.io import read_jsonl
    from rlvr.model import checkpoint_ids, env_info, load_model
    from rlvr.task import make_task

    commit = git_commit()
    if commit == "unknown" and not args.allow_unknown_commit:
        raise SystemExit("git commit unknown (no .git here, or git is not installed); copy .git with the code, "
                         "or pass --allow-unknown-commit")
    out = resolve(args.out)
    paths = {"parquet": out / "bridge.parquet", "json": out / "bridge.json"}
    if existing := [str(p) for p in paths.values() if p.exists()]:
        raise SystemExit(f"output exists, not replaced: {', '.join(existing)}")
    (cell,) = PL.parse_cells(args.cell)
    ckpt = resolve(args.ckpt)
    model_id = checkpoint_ids(ckpt)[0]
    cfg = load_config(ckpt / "run_config.yaml", args.overrides)
    lc, ec = cfg["label"], cfg["eval"]
    k = lc["relabel_k"]
    if not k:
        raise SystemExit("label.relabel_k is 0")
    task = make_task(cfg["task"])

    labels = pd.read_parquet(resolve(args.old_labels))
    if set(labels.model_id) != {model_id} or set(labels.task_fingerprint) != {task.fingerprint}:
        raise SystemExit(f"{args.old_labels}: model_id {set(labels.model_id)} / task fingerprint "
                         f"{set(labels.task_fingerprint)} do not match {model_id} / {task.fingerprint}")
    sel = select(labels, cell, lc["relabel_range"], args.n, args.seed)
    rows = {r["pid"]: r for r in read_jsonl(resolve(args.old_candidates))}
    missing = [pid for pid in sel.pid if pid not in rows]
    if missing:
        raise SystemExit(f"{len(missing)} selected pids not in {args.old_candidates}, e.g. {missing[:3]}")
    problems = [task.from_dict(rows[pid]) for pid in sel.pid]
    assert [p.pid for p in problems] == list(sel.pid)

    meta = label_meta(args.ckpt, ckpt, cfg, commit, 1, env_info(args.device), {"bridge": True}, args.overrides)
    torch.manual_seed(1000 * cfg["seed"])  # as calibrate.py's rank 0
    model, tok = load_model(ckpt, args.device)
    bs = max(1, lc["batch_size"] * lc["k"] // k)  # calibrate.py label(): same sequences per batch at any k
    print(f"bridge: {len(problems)} {args.cell} problems x {k} samples, batch {bs}, max_new_tokens "
          f"{ec['max_new_tokens']}, on {args.device}", flush=True)
    t = time.time()
    new = label_pass_rates(model, tok, task, problems, k, lc["temperature"], bs, ec["max_new_tokens"], model_id)
    secs = time.time() - t
    df = pd.DataFrame({"pid": sel.pid.to_numpy(), "p_hat_old": sel.p_hat.to_numpy(),
                       "p_hat_relabel_old": sel.p_hat_relabel.to_numpy(),
                       "n_correct_new": new.n_correct.to_numpy(), "k_new": new.k.to_numpy(),
                       "p_hat_new": new.p_hat.to_numpy()})
    df["diff"] = df.p_hat_new - df.p_hat_relabel_old
    res = {"model_id": model_id, "cell": args.cell, "selection_seed": args.seed,
           "selection": {"first_p_hat_range": lc["relabel_range"], "not_skippable": True, "has_relabel": True,
                         "n_eligible": len(eligible(labels, cell, lc["relabel_range"]))},
           **summary(df, cfg["gate"]["p_range"]),
           "label_seconds": round(secs, 1), "sequences_per_second": round(len(problems) * k / secs, 2),
           "k": k, "batch_size": bs, "max_new_tokens": ec["max_new_tokens"], "temperature": lc["temperature"],
           "old_labels": str(resolve(args.old_labels)), "old_candidates": str(resolve(args.old_candidates)),
           "label_meta": json.loads(json.dumps(meta))}
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(paths["parquet"], index=False)
    paths["json"].write_text(json.dumps(res, indent=1))
    print(json.dumps({k_: v for k_, v in res.items() if k_ != "label_meta"}, indent=1))
    print(f"bridge test {'PASS' if res['pass'] else 'FAIL'}: mean diff {res['mean_diff']} "
          f"(se {res['se_mean_diff']}), rule {res['rule']}; outputs in {out}", flush=True)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
