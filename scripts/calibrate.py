"""Calibrate a base model: p heatmap, RL-pool pass-rate labels, gate check.

    python scripts/calibrate.py checkpoints/pt-mid-s0 [--skip-pool] [key=value ...]
    python scripts/calibrate.py runs/pt-mid-s0/ckpt/step-3000 --skip-pool   # model_id pt-mid-s0-step-3000
    torchrun --standalone --nproc_per_node=4 scripts/calibrate.py checkpoints/pt-mid-s0   # 4 GPUs

Under torchrun each rank labels every world-th problem (sampling seed 1000 * seed + rank) and
the shards are gathered back into the original order; E1 greedy, the gate and all outputs run
on rank 0. Labelling is the whole cost, so 4 GPUs take ~1/4 of the time.

p_hat = correct / label.k samples at label.temperature (design.md "Computing p", calibration heatmap).

Output:
  runs/{model_id}/calib/calib_labels.parquet   p_hat for the calib set (depth 1-8 x [n_bin x] group x 200)
  runs/{model_id}/calib/heatmap.png            mean p_hat and signal fraction per (depth, group), one row per n_bin
  runs/{model_id}/calib/gate.json              gate check (also printed); exit code 1 if it fails
  data/labels/{model_id}.parquet               p_hat for the whole RL pool (+ origin), for build_dataset.py;
                                               *_relabel columns for the second labelling (below)

RL pool (rl_pool.mode): 'file' labels {paths.data}/{rl_pool.path}. 'pilot' (v2 stage B, rlvr/pilot.py)
picks the cells with the most p ~ 0.5 problems in the calib labels, draws rl_pool.pilot_per_cell
candidates per chosen cell (rlvr.sets.cell_candidates), labels them, relabels the non-skippable
ones in label.relabel_range, and writes runs/{model_id}/calib/pilot.json (supply per cell and per
D/E pair with a shared length mix; `go`) and pilot_candidates.jsonl.

--e1-only (v2 stage A): only the depth-1 check (E1 greedy per function, incl. binary operators).

Second labelling (label.relabel_*): RL-pool problems in label.relabel_cells whose first-pass
p_hat is in label.relabel_range are labelled again with label.relabel_k independent samples.
The matched-p datasets (p_label: relabel) and gate check 3 select on that second estimate,
which shrinks the winner's-curse bias of selecting p ~ 0.5 problems on a noisy 32-sample p_hat.

Gate (design.md "Deliverables"; details in rlvr/gate.py). gate.hard_checks fail the gate (exit 1), the
other checks are warnings:
  1. depth1_greedy  E1 greedy accuracy >= gate.depth1_greedy overall; per function via Wilson bound
  2. spread         at each gate.spread_depths depth, >= gate.min_spread_frac of calib 0 < p_hat < 1
  3. matched_cells  >= gate.min_matched problems with p in gate.p_range per matched cell, and
                    >= gate.min_matched_per_len at every input length
  4. final_loss     composition loss at gate.loss_depths >= gate.min_final_loss (soft by default)
  5. target         mean calib p at the level's design depth in gate.target.p_range (soft)
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from rlvr import gate as G  # noqa: E402
from rlvr import pilot as PL  # noqa: E402
from rlvr.config import load_config, resolve, task_and_splits  # noqa: E402
from rlvr.evaluate import greedy_eval, label_pass_rates  # noqa: E402
from rlvr.io import load_problems, problems_to_rows, read_jsonl, write_jsonl  # noqa: E402
from rlvr.model import load_model  # noqa: E402
from rlvr.sets import cell_candidates  # noqa: E402
from rlvr.task import GROUPS  # noqa: E402


def ids_for(ckpt: Path) -> tuple[str, str, int | None]:
    """(model_id, run_name, step): checkpoints/pt-mid-s0 -> (pt-mid-s0, pt-mid-s0, None);
    runs/pt-mid-s0/ckpt/step-3000 -> (pt-mid-s0-step-3000, pt-mid-s0, 3000)."""
    if ckpt.parent.name == "ckpt":
        run = ckpt.parent.parent.name
        return f"{run}-{ckpt.name}", run, int(ckpt.name.split("-")[-1])
    return ckpt.name, ckpt.name, None


def heatmap(df: pd.DataFrame, title: str, path: Path) -> None:
    """Mean p_hat and signal fraction per (group x depth); one row of panels per n_bin."""
    depths = sorted(df.depth.unique())
    nb_col = df.n_bin if "n_bin" in df else pd.Series(0, index=df.index)
    n_bins = sorted(int(v) for v in nb_col.unique())
    fig, axes = plt.subplots(len(n_bins), 2, figsize=(4 + 1.1 * len(depths), 0.4 + 3.0 * len(n_bins)),
                             squeeze=False)
    for row, nb in enumerate(n_bins):
        sub = df[(nb_col == nb).to_numpy()]
        groups = [g for g in GROUPS if g in set(sub.group)]
        mean = sub.pivot_table(index="group", columns="depth", values="p_hat", aggfunc="mean")
        sig = sub.assign(s=(sub.p_hat > 0) & (sub.p_hat < 1)).pivot_table(
            index="group", columns="depth", values="s", aggfunc="mean")
        for ax, m, name in ((axes[row, 0], mean, "mean p_hat"), (axes[row, 1], sig, "frac 0 < p_hat < 1")):
            m = m.reindex(index=groups, columns=depths)
            im = ax.imshow(m.to_numpy(dtype=float), vmin=0, vmax=1, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(depths)), [str(d) for d in depths])
            ax.set_yticks(range(len(groups)), groups)
            ax.set_xlabel("depth (n_ops)" if len(n_bins) > 1 else "depth")
            ax.set_title(name + (f", n_bin {nb}" if len(n_bins) > 1 else ""))
            for (i, j), v in np.ndenumerate(m.to_numpy(dtype=float)):
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="w" if v < 0.6 else "k", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def shard(n: int, rank: int, world: int) -> list[int]:
    """Indices of the problems one rank labels: strided, so every rank gets every cell."""
    return list(range(rank, n, world))


def merge_shards(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Gathered per-rank label frames (each with column _i = problem index) -> original order."""
    parts = [d for d in parts if len(d)]
    return pd.concat(parts).sort_values("_i").drop(columns="_i").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="checkpoint dir, e.g. checkpoints/pt-mid-s0")
    ap.add_argument("--skip-pool", action="store_true", help="skip RL-pool labelling (gate 3 not checked)")
    ap.add_argument("--e1-only", action="store_true", help="v2 stage A: depth-1 check only (no calib, no pool)")
    ap.add_argument("--calib-per-cell", type=int, default=0,
                    help="label only the first N calib problems per (depth, group) cell (pilot sweep)")
    ap.add_argument("--pool-first", type=int, default=0,
                    help="label only the first N RL-pool candidates (smoke tests; writes the same outputs)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    ddp = "WORLD_SIZE" in os.environ
    if ddp:
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=3))  # ranks wait for the slowest shard
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        device = f"cuda:{os.environ['LOCAL_RANK']}"
    else:
        rank, world, device = 0, 1, args.device
    master = rank == 0
    ckpt = resolve(args.ckpt)
    model_id, run_name, ckpt_step = ids_for(ckpt)
    cfg = load_config(ckpt / "run_config.yaml", args.overrides)
    lc, gc, ec, rc = cfg["label"], cfg["gate"], cfg["eval"], cfg["rl_pool"]
    G.check_config(gc, lc)
    task, splits = task_and_splits(cfg)
    torch.manual_seed(1000 * cfg["seed"] + rank)  # independent sampling per rank
    model, tok = load_model(ckpt, device)
    data = resolve(cfg["paths"]["data"])
    eval_dir = data / "eval" / ec["version"]
    out = resolve(cfg["paths"]["runs"]) / model_id / "calib"
    if master:
        out.mkdir(parents=True, exist_ok=True)

    def label(problems, k=None, **extra):
        """Label on all ranks (each its shard) and return the full frame on every rank."""
        k = k or lc["k"]
        bs = max(1, lc["batch_size"] * lc["k"] // k)  # same number of sequences per batch at any k
        t = time.time()
        idx = shard(len(problems), rank, world)
        df = label_pass_rates(model, tok, task, [problems[i] for i in idx], k, lc["temperature"], bs,
                              ec["max_new_tokens"], model_id,
                              {c: [v[i] for i in idx] for c, v in extra.items()}) if idx else pd.DataFrame()
        if ddp:
            df["_i"] = idx
            parts = [None] * world
            dist.all_gather_object(parts, df)
            df = merge_shards(parts)
        if master:
            print(f"labelled {len(problems)} problems x {k} on {world} GPU(s) in {time.time() - t:.0f}s", flush=True)
        return df

    def write_gate(gate, final):
        gate.update(G.verdict(gate["checks"], gc["hard_checks"]))
        (out / "gate.json").write_text(json.dumps(gate, indent=1))
        if final:
            print(json.dumps(gate, indent=1))
        warn = f" (warnings: {', '.join(gate['warnings'])})" if gate["warnings"] else ""
        print(f"{model_id}: {'' if final else 'interim '}gate {'PASS' if gate['pass'] else 'FAIL'}{warn}; "
              f"outputs in {out}", flush=True)

    gate = {"model_id": model_id, "hard_checks": gc["hard_checks"], "checks": {}}

    if master:
        # 1. depth-1 greedy accuracy
        rows = greedy_eval(model, tok, task, {"E1": load_problems(task, eval_dir / "E1.jsonl")},
                           max_new_tokens=ec["max_new_tokens"], batch_size=ec["batch_size"])
        gate["checks"]["depth1_greedy"] = G.depth1_greedy(rows, gc)

        # 4. composition loss at the checkpoint's step (saturated in-distribution compositions)
        metrics_path = resolve(cfg["paths"]["runs"]) / run_name / "metrics.jsonl"
        if metrics_path.exists() and (fl := G.final_loss(read_jsonl(metrics_path), gc, ckpt_step)) is not None:
            gate["checks"]["final_loss"] = fl

    if args.e1_only:
        ok = True
        if master:
            write_gate(gate, final=True)
            ok = gate["pass"]
        if ddp:
            dist.destroy_process_group()
        sys.exit(0 if ok else 1)

    # 2. calibration heatmap + spread over depths
    calib_rows = read_jsonl(eval_dir / "calib.jsonl")
    if args.calib_per_cell:
        kept, per_cell = [], Counter()
        for r in calib_rows:
            key = (r["depth"], r.get("n_bin", 0), r["group"])
            per_cell[key] += 1
            if per_cell[key] <= args.calib_per_cell:
                kept.append(r)
        calib_rows = kept
    calib = label([task.from_dict(r) for r in calib_rows])
    if master:
        calib.to_parquet(out / "calib_labels.parquet", index=False)
        heatmap(calib, f"{model_id}: p_hat at T={lc['temperature']}, k={lc['k']}", out / "heatmap.png")
        gate["checks"]["spread"] = G.spread(calib, gc)
        # 5. the level's design target (pilot sweep), e.g. weak d3 ~ 0.1
        if (tg := G.target(calib, gc)) is not None:
            gate["checks"]["target"] = tg
        gate["calib_mean_p"] = {G.cell_name(d, g, nb): round(float(v), 3)
                                for (d, nb, g), v in calib.groupby(["depth", "n_bin", "group"]).p_hat.mean().items()}
        if not args.skip_pool:  # interim gate.json (as with --skip-pool) while the pool is labelled
            write_gate(dict(gate, checks=dict(gate["checks"])), final=False)

    # 3. RL-pool labels (+ independent second labelling of the screened matched cells) and matched-p cells
    if not args.skip_pool and rc["mode"] == "pilot":
        rates = PL.cell_rates(calib, rc["pilot_n_ops"], gc["p_range"])
        cells, pairs = PL.choose_cells(rates, rc["pilot_min_gap"], rc["pilot_max_cells"])
        problems = [p for c in cells for p in cell_candidates(splits, "rl", c, rc["pilot_per_cell"], rc["pilot_seed"])]
        if problems:
            pool = label(problems, origin=["pilot"] * len(problems))
            lo, hi = lc["relabel_range"]
            mask = ((pool.p_hat >= lo) & (pool.p_hat <= hi) & ~pool.skippable.astype(bool)).to_numpy()
            if lc["relabel_k"]:
                again = label([p for p, m in zip(problems, mask) if m], k=lc["relabel_k"]) if mask.any() else None
                pool = G.attach_relabel(pool, mask, again)
        if master:
            rep = {"model_id": model_id, "chosen_cells": [G.cell_name(d, g, nb) for d, nb, g in cells],
                   "calib_rates": rates.round(4).to_dict(orient="records")}
            if problems and lc["relabel_k"]:
                (data / "labels").mkdir(parents=True, exist_ok=True)
                pool.to_parquet(data / "labels" / f"{model_id}.parquet", index=False)
                write_jsonl(out / "pilot_candidates.jsonl", problems_to_rows(task, problems, origin="pilot"))
                rep.update(PL.report(pool, cells, pairs, gc["p_range"], rc["pilot_min_matched"]))
            else:
                rep.update(go=False, reason="no D/E cell pair with p ~ 0.5 calib problems" if not problems
                           else "label.relabel_k is 0")
            (out / "pilot.json").write_text(json.dumps(rep, indent=1))
            best = rep.get("pairs", [None])[0] if rep.get("pairs") else None
            gate["pilot"] = {"go": rep["go"], "best_pair": best}
            print(f"{model_id}: pilot go={rep['go']} best pair {best}", flush=True)
    elif not args.skip_pool:
        pool_rows = read_jsonl(data / rc["path"])
        if args.pool_first:
            pool_rows = pool_rows[:args.pool_first]
        pool_problems = [task.from_dict(r) for r in pool_rows]
        pool = label(pool_problems, origin=[r.get("origin", "base") for r in pool_rows])
        mask = G.relabel_mask(pool, lc)  # same on every rank: all hold the gathered pool
        if lc["relabel_k"]:
            again = label([p for p, m in zip(pool_problems, mask) if m], k=lc["relabel_k"]) if mask.any() else None
            pool = G.attach_relabel(pool, mask, again)
        if master:
            (data / "labels").mkdir(parents=True, exist_ok=True)
            pool.to_parquet(data / "labels" / f"{model_id}.parquet", index=False)
            relabelled = lc["relabel_cells"] if lc["relabel_k"] else []
            gate["checks"]["matched_cells"] = G.matched_cells(pool, gc, relabelled)

    ok = True
    if master:
        write_gate(gate, final=True)
        ok = gate["pass"]
    if ddp:
        dist.destroy_process_group()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
