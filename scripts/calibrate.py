"""Calibrate a base model: p heatmap, RL-pool pass-rate labels, gate check.

    python scripts/calibrate.py checkpoints/pt-mid-s0 [--skip-pool] [key=value ...]
    python scripts/calibrate.py runs/pt-mid-s0/ckpt/step-3000 --skip-pool   # model_id pt-mid-s0-step-3000
    python scripts/calibrate.py checkpoints/tree-v1/pt-tree-a-atoms-s0 --e1-only   # pilot stage A: E1 check only
    python scripts/calibrate.py checkpoints/tree-v1/pt-tree-mix-cf0.0035-s0 --reuse-calib   # pool after --skip-pool
    torchrun --standalone --nproc_per_node=4 scripts/calibrate.py checkpoints/pt-mid-s0   # 4 GPUs

Under torchrun each rank labels every world-th problem (sampling seed 1000 * seed + rank) and
the shards are gathered back into the original order; E1 greedy, the gate and all outputs run
on rank 0. Labelling is the whole cost, so 4 GPUs take ~1/4 of the time.

p_hat = correct / label.k samples at label.temperature (design.md "Computing p", calibration heatmap).

Output:
  runs/{model_id}/calib/calib_labels.parquet   p_hat for the calib set (depth 1-8 x H/L/mixed x 200)
  runs/{model_id}/calib/heatmap.png            mean p_hat and signal fraction per (depth, group)
  runs/{model_id}/calib/gate.json              gate check (also printed); exit code 1 if it fails
  runs/{model_id}/calib/label_meta.json        how the labels were made: weights sha256, git commit, label config,
                                               seed and ranks, env, flags (not with --e1-only); written before labelling
  data/labels/{model_id}.parquet               p_hat for the whole RL pool (+ origin), for build_dataset.py;
                                               *_relabel columns for the second labelling (below)
  data/labels/{model_id}.meta.json             copy of label_meta.json, written with the parquet
                                               (compare two: python -m rlvr.label_meta a.json b.json)

Existing label files (calib_labels.parquet, data/labels/{model_id}.parquet) are not replaced: the run
stops with exit code 2 before the model loads, unless --overwrite is given. --reuse-calib also stops
when calib/label_meta.json shows the calib labels were made differently (weights, label.k,
label.temperature, eval.max_new_tokens, eval version); the reused labels' meta is kept as `reused_calib`.

An unknown git commit (no .git, or no git) stops the run unless --allow-unknown-commit is given.

Pilot extension (rl_pool.mode pilot): rl_pool.pilot_cells names the cells that get candidates (e.g.
rl_pool.pilot_cells=d4-H) in place of the choice from the calib rates; such a run has no D/E pairs, so
pilot.json and gate.json give go = null. rl_pool.pilot_skip=N drops the first N candidates of each cell's
stream, so a run with N = the earlier pilot_per_cell labels new candidates of the same streams. Give such a
run its own paths.data and paths.runs (its outputs have the same names as the earlier pilot's).

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
import hashlib
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
import transformers  # noqa: E402

from rlvr import gate as G  # noqa: E402
from rlvr import label_meta as LM  # noqa: E402
from rlvr import pilot as PL  # noqa: E402
from rlvr.config import git_commit, load_config, resolve, task_and_splits  # noqa: E402
from rlvr.evaluate import greedy_eval, label_pass_rates  # noqa: E402
from rlvr.io import load_problems, problems_to_rows, read_jsonl, write_jsonl  # noqa: E402
from rlvr.model import checkpoint_ids as ids_for, env_info, load_model  # noqa: E402
from rlvr.task import GROUPS, make_task  # noqa: E402


def heatmap(df: pd.DataFrame, title: str, path: Path) -> None:
    depths = sorted(df.depth.unique())
    mean = df.pivot_table(index="group", columns="depth", values="p_hat", aggfunc="mean").reindex(GROUPS)
    sig = df.assign(s=(df.p_hat > 0) & (df.p_hat < 1)).pivot_table(
        index="group", columns="depth", values="s", aggfunc="mean").reindex(GROUPS)
    fig, axes = plt.subplots(1, 2, figsize=(4 + 1.1 * len(depths), 3.4))
    for ax, m, name in ((axes[0], mean, "mean p_hat"), (axes[1], sig, "frac 0 < p_hat < 1")):
        im = ax.imshow(m.to_numpy(dtype=float), vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(depths)), [str(d) for d in depths])
        ax.set_yticks(range(len(GROUPS)), GROUPS)
        ax.set_xlabel("depth")
        ax.set_title(name)
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


def label_meta(ckpt_arg: str, ckpt: Path, cfg: dict, commit: str, world: int, env: dict, flags: dict,
               overrides: list[str]) -> dict:
    """label_meta.json: what is needed to tell how a labels file was made (and compare two of them)."""
    model_id, _, ckpt_step = ids_for(ckpt)
    weights, sha = ckpt / "model.safetensors", None
    if weights.exists():
        h = hashlib.sha256()
        with weights.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        sha = h.hexdigest()
    return {"model_id": model_id, "ckpt": ckpt_arg, "ckpt_path": str(ckpt), "ckpt_step": ckpt_step,
            "weights_sha256": sha, "git_commit": commit, "label": cfg["label"], "seed": cfg["seed"], "world": world,
            "eval_version": cfg["eval"]["version"], "max_new_tokens": cfg["eval"]["max_new_tokens"],
            "rl_pool": cfg["rl_pool"], **flags, "overrides": list(overrides), "env": env,
            "transformers": transformers.__version__,
            "time": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}


def check_outputs(meta: dict, out: Path, labels_path: Path, overwrite: bool, verbose: bool = True) -> dict:
    """Refuse (RuntimeError) to replace label files unless `overwrite`, and to reuse calib labels that were
    made differently (LM.CALIB_KEYS of calib/label_meta.json). Returns `meta`, with the reused labels' own
    meta as `reused_calib` (None when they have none)."""
    calib_path = out / "calib_labels.parquet"
    targets = ([] if meta["reuse_calib"] else [calib_path]) + ([] if meta["skip_pool"] else [labels_path])
    existing = [str(p) for p in targets if p.exists()]
    if existing and not overwrite:
        hint = "" if meta["reuse_calib"] or not calib_path.exists() else ", or --reuse-calib to keep the calib labels"
        raise RuntimeError(f"label file(s) already exist: {', '.join(existing)}; move them away or pass "
                           f"--overwrite to replace them{hint}")
    if not meta["reuse_calib"]:
        return meta
    old_path = out / "label_meta.json"
    if not old_path.exists():
        if verbose:
            print(f"warning: {calib_path} has no label_meta.json; how these labels were made is unknown", flush=True)
        return {**meta, "reused_calib": None}
    old = json.loads(old_path.read_text())
    old = old.get("reused_calib") or old  # the run that made the labels, not a later run that reused them
    d = LM.diff(old, meta)
    bad = {k: v for k, v in d.items() if k in LM.CALIB_KEYS}
    if bad:
        raise RuntimeError(f"--reuse-calib: {calib_path} was made differently (then | now): "
                           + "; ".join(f"{k}: {a!r} | {b!r}" for k, (a, b) in bad.items())
                           + "; label the calib set again (drop --reuse-calib, pass --overwrite)")
    notes = {k: v for k, v in d.items() if k in ("git_commit", "transformers") or k.startswith("env.")}
    if notes and verbose:
        print("note: reused calib labels were made with (then | now) "
              + "; ".join(f"{k}: {a!r} | {b!r}" for k, (a, b) in notes.items()), flush=True)
    return {**meta, "reused_calib": old}


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="checkpoint dir, e.g. checkpoints/pt-mid-s0")
    ap.add_argument("--skip-pool", action="store_true", help="skip RL-pool labelling (gate 3 not checked)")
    ap.add_argument("--calib-per-cell", type=int, default=0,
                    help="label only the first N calib problems per (depth, group) cell (pilot sweep)")
    ap.add_argument("--pool-first", type=int, default=0,
                    help="label only the first N RL-pool candidates (smoke tests; writes the same outputs)")
    ap.add_argument("--e1-only", action="store_true",
                    help="depth-1 greedy check (and final loss) only, no calibration or pool labelling: pilot stage A")
    ap.add_argument("--reuse-calib", action="store_true",
                    help="read calib/calib_labels.parquet from an earlier run instead of labelling the calib set again "
                         "(e.g. the pilot pool after a --skip-pool run)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace existing label files (calib/calib_labels.parquet, data/labels/{model_id}.parquet)")
    ap.add_argument("--allow-unknown-commit", action="store_true",
                    help="run although the git commit cannot be read (a copy of the repo without .git, or no git)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    commit = git_commit()
    if commit == "unknown":  # git_commit()'s fallback; checked on every rank, before the process group exists
        if not args.allow_unknown_commit:
            raise SystemExit("git commit unknown (no .git here, or git is not installed), so the labels could not "
                             "record their code version; copy .git with the code, or pass --allow-unknown-commit")
        if os.environ.get("RANK", "0") == "0":
            print("warning: git commit unknown; label_meta.json will not record the code version", flush=True)
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
    lc, gc, ec = cfg["label"], cfg["gate"], cfg["eval"]
    G.check_config(gc, lc)
    task = make_task(cfg["task"])
    data = resolve(cfg["paths"]["data"])
    eval_dir = data / "eval" / ec["version"]
    out = resolve(cfg["paths"]["runs"]) / model_id / "calib"
    rc = cfg["rl_pool"]
    try:  # a bad cell name or skip stops the run here, before the model loads
        named_cells = PL.parse_cells(rc["pilot_cells"])
        if rc["pilot_skip"] < 0:
            raise ValueError(f"rl_pool.pilot_skip must be >= 0, got {rc['pilot_skip']}")
    except ValueError as e:
        raise SystemExit(f"rl_pool.pilot_cells / pilot_skip: {e}")
    if not args.e1_only:  # --e1-only labels nothing. Checked on every rank, before the model loads.
        reuse = args.reuse_calib and (out / "calib_labels.parquet").exists()  # as actually reused
        flags = {"skip_pool": args.skip_pool, "calib_per_cell": args.calib_per_cell, "pool_first": args.pool_first,
                 "reuse_calib": reuse}
        meta = label_meta(args.ckpt, ckpt, cfg, commit, world, env_info(device), flags, args.overrides)
        try:
            meta = check_outputs(meta, out, data / "labels" / f"{model_id}.parquet", args.overwrite, verbose=master)
        except RuntimeError as e:
            print(f"{model_id}: {e}", file=sys.stderr, flush=True)
            sys.exit(2)  # not 1: that is a failed gate
        meta = json.dumps(meta, indent=1)
    torch.manual_seed(1000 * cfg["seed"] + rank)  # independent sampling per rank
    model, tok = load_model(ckpt, device)
    if master:
        out.mkdir(parents=True, exist_ok=True)
        if not args.e1_only:
            (out / "label_meta.json").write_text(meta)

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
            torch.cuda.empty_cache()  # generation caches fill the GPU; NCCL needs memory for the gather
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

    if args.e1_only:  # pilot stage A (design.md "v2 pilot"): can every operator be learned?
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
            per_cell[(r["depth"], r["group"])] += 1
            if per_cell[(r["depth"], r["group"])] <= args.calib_per_cell:
                kept.append(r)
        calib_rows = kept
    reused = out / "calib_labels.parquet"
    if args.reuse_calib and reused.exists():
        calib = pd.read_parquet(reused)  # every rank reads it, as label() returns the full frame on every rank
        if master:
            print(f"{model_id}: calib labels reused from {reused} ({len(calib)} problems)", flush=True)
    else:
        calib = label([task.from_dict(r) for r in calib_rows])
    if master:
        calib.to_parquet(out / "calib_labels.parquet", index=False)
        heatmap(calib, f"{model_id}: p_hat at T={lc['temperature']}, k={lc['k']}", out / "heatmap.png")
        gate["checks"]["spread"] = G.spread(calib, gc)
        # 5. the level's design target (pilot sweep), e.g. weak d3 ~ 0.1
        if (tg := G.target(calib, gc)) is not None:
            gate["checks"]["target"] = tg
        gate["calib_mean_p"] = {f"d{d}-{g}": round(float(v), 3)
                                for (d, g), v in calib.groupby(["depth", "group"]).p_hat.mean().items()}
        if not args.skip_pool:  # interim gate.json (as with --skip-pool) while the pool is labelled
            write_gate(dict(gate, checks=dict(gate["checks"])), final=False)

    # 3. RL-pool labels (+ independent second labelling of the screened matched cells) and matched-p cells
    if not args.skip_pool and rc["mode"] == "pilot":
        # v2 pilot stage B (design.md decision 11): D/E cells from the calib rates, 10,000 candidates
        # per chosen cell, second labelling of the non-skippable ones in label.relabel_range
        rates = PL.cell_rates(calib, rc["pilot_depths"], gc["p_range"])
        cells, pairs = PL.select_cells(rc, rates)  # rl_pool.pilot_cells, else the best D/E pairs
        _, splits = task_and_splits(cfg)
        t = time.time()
        problems = PL.candidates(splits, cells, rc)  # after the first rl_pool.pilot_skip of each cell
        if master:
            skip = f" (after the first {rc['pilot_skip']} per cell)" if rc["pilot_skip"] else ""
            print(f"{model_id}: pilot cells {[PL.cell_name(*c) for c in cells]}, {len(problems)} candidates{skip} "
                  f"in {time.time() - t:.0f}s", flush=True)
        if problems:
            pool = label(problems, origin=["pilot"] * len(problems))
            lo, hi = lc["relabel_range"]
            mask = ((pool.p_hat >= lo) & (pool.p_hat <= hi) & ~pool.skippable.astype(bool)).to_numpy()
            if lc["relabel_k"]:
                again = label([p for p, m in zip(problems, mask) if m], k=lc["relabel_k"]) if mask.any() else None
                pool = G.attach_relabel(pool, mask, again)
        if master:
            rep = {"model_id": model_id, "chosen_cells": [PL.cell_name(*c) for c in cells],
                   "calib_rates": rates.round(4).to_dict(orient="records")}
            if problems and lc["relabel_k"]:
                (data / "labels").mkdir(parents=True, exist_ok=True)
                pool.to_parquet(data / "labels" / f"{model_id}.parquet", index=False)
                (data / "labels" / f"{model_id}.meta.json").write_text(meta)
                write_jsonl(out / "pilot_candidates.jsonl", problems_to_rows(task, problems, origin="pilot"))
                rep.update(PL.report(pool, cells, pairs, gc["p_range"], rc["pilot_min_matched"]))
            else:
                rep.update(go=False, reason="no D/E cell pair with p ~ 0.5 calib problems" if not problems
                           else "label.relabel_k is 0")
            if named_cells:  # named cells: no D/E pairs, so the go criterion does not apply
                rep.update(go=None, go_note="cells named by rl_pool.pilot_cells: no D/E pairs, go not evaluated")
            (out / "pilot.json").write_text(json.dumps(rep, indent=1))
            best = rep["pairs"][0] if rep.get("pairs") else None
            gate["pilot"] = {"go": rep["go"], "best_pair": best}
            if named_cells:
                gate["pilot"]["go_note"] = rep["go_note"]
                n = {c: v["n_in_range"] for c, v in rep.get("cells", {}).items()}
                print(f"{model_id}: pilot cells named by rl_pool.pilot_cells, go not evaluated; in range {n}",
                      flush=True)
            else:
                print(f"{model_id}: pilot go={rep['go']} best pair {best}", flush=True)
    elif not args.skip_pool:
        pool_rows = read_jsonl(data / "rl_pool" / "candidates.jsonl")
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
            (data / "labels" / f"{model_id}.meta.json").write_text(meta)
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
