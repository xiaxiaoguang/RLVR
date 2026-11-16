"""Where do compositions break? Step-level diagnosis of a base model, by input length.

    gen      sample completions on RL-pool cells (one GPU), save them with the problems
             python scripts/diag_steps.py gen checkpoints/pt-mid-s0 --out runs/diag/mid [--cells 4-H,3-H]
    replay   replay a run's pretraining stream on the exact rng path of pretrain.py (ranks x workers,
             round-robin micro-batches) and count composition examples per (depth, group, str_len);
             the totals are checked against the run's stats.json
             python scripts/diag_steps.py replay runs/pt-mid-s0 [runs/...] --out runs/diag/replay
    analyze  first wrong step and its type for every completion, tabulated by str_len
             python scripts/diag_steps.py analyze runs/diag/mid [runs/diag/...]

Error types (the first one in step order): missing_step, wrong_func (the step names another Fxx),
copy (the step's input is not the previous step's output), apply (wrong output for the right
function and input), extra_step, format (no parsable answer / no <eos>), answer_copy (answer differs
from the last step). Steps before the first error are correct, so "previous output" is the truth.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlvr.config import load_config, resolve  # noqa: E402
from rlvr.io import read_jsonl  # noqa: E402
from rlvr.task import EOS, Task  # noqa: E402

# ---- gen -----------------------------------------------------------------------------------------

def cmd_gen(a):
    import torch

    from rlvr.model import generate, load_model
    ckpt = resolve(a.ckpt)
    cfg = load_config(ckpt / "run_config.yaml")
    task = Task(cfg["task"])
    torch.manual_seed(a.seed)
    model, tok = load_model(ckpt, a.device)
    cells = {(int(c.split("-")[0]), c.split("-")[1]) for c in a.cells.split(",")}
    rows = read_jsonl(resolve(cfg["paths"]["data"]) / "rl_pool" / "candidates.jsonl")
    per, probs = Counter(), []
    for r in rows:
        key = (r["depth"], r["group"], r["str_len"])
        if (r["depth"], r["group"]) in cells and per[key] < a.per_len:
            per[key] += 1
            probs.append(task.from_dict(r))
    prompts = [task.format_prompt(p) for p in probs]
    mnt = cfg["eval"]["max_new_tokens"]
    t = time.time()
    sampled = generate(model, tok, prompts, mnt, temperature=a.temperature, num_return_sequences=a.k,
                       batch_size=max(1, 512 // a.k))
    greedy = generate(model, tok, prompts, mnt, temperature=0.0, batch_size=512)
    secs = time.time() - t
    recs = []
    for i, p in enumerate(probs):
        base = {"pid": p.pid, "depth": p.depth, "group": p.group, "str_len": len(p.x),
                "funcs": list(p.funcs), "x": p.x}
        recs += [{**base, "sample": j, "completion": sampled[i * a.k + j]} for j in range(a.k)]
        recs.append({**base, "sample": -1, "completion": greedy[i]})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recs).to_parquet(out / "completions.parquet", index=False)
    (out / "gen.json").write_text(json.dumps({"ckpt": str(ckpt), "model_id": ckpt.name, "cells": sorted(cells),
                                              "n_problems": len(probs), "k": a.k, "temperature": a.temperature,
                                              "seconds": round(secs)}, indent=1))
    print(f"{ckpt.name}: {len(probs)} problems x ({a.k} sampled + greedy) in {secs:.0f}s -> {out}", flush=True)


# ---- replay --------------------------------------------------------------------------------------

def _replay_one(job):
    cfg, rank, worker, n_batches, eval_pids = job
    import torch
    torch.set_num_threads(1)
    from rlvr.stream import PretrainStream

    class Replay(PretrainStream):
        def encode(self, kind, p, steps):
            r = super().encode(kind, p, steps)
            r["L"], r["G"] = len(p.x), p.group
            return r

        def collate(self, rows):
            super().collate(rows)  # same rng calls (position offsets) as in training
            return len(rows), [(r["depth"], r["G"], r["L"]) for r in rows if r["kind"] == 1]

    s = Replay(cfg, rank, eval_pids)
    s._setup(worker)
    it, n_ex, cnt = s.batches(), 0, Counter()
    for _ in range(n_batches):
        n, comps = next(it)
        n_ex += n
        cnt.update(comps)
    return n_ex, cnt


def cmd_replay(a):
    jobs, meta, run_dirs = [], {}, {}
    for run in a.runs:
        run = resolve(run)
        cfg = load_config(run / "run_config.yaml")
        pc, world = cfg["pretrain"], cfg.get("world_size", 1)
        accum = max(1, round(pc["target_tokens_per_step"] / (pc["micro_target_tokens"] * world)))
        W = max(1, pc["num_workers"])
        per_rank = pc["steps"] * accum
        eval_dir = resolve(cfg["paths"]["data"]) / "eval" / cfg["eval"]["version"]
        eval_pids = frozenset(r["pid"] for f in sorted(eval_dir.glob("*.jsonl")) for r in read_jsonl(f))
        run_dirs[run.name] = run
        meta[run.name] = {"world": world, "workers": W, "accum": accum, "micro_batches_per_rank": per_rank}
        for rank in range(world):
            for w in range(W):  # DataLoader takes micro-batch t of a rank from worker t % W
                n = len(range(w, per_rank, W)) if pc["num_workers"] else per_rank
                jobs.append((run.name, (cfg, rank, w, a.limit or n, eval_pids)))
    t = time.time()
    with Pool(a.procs) as pool:
        res = pool.map(_replay_one, [j for _, j in jobs], chunksize=1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, m in meta.items():
        n_ex, cnt = 0, Counter()
        for (rn, _), (n, c) in zip(jobs, res):
            if rn == name:
                n_ex += n
                cnt.update(c)
        by = defaultdict(dict)
        for (d, g, L), v in sorted(cnt.items()):
            by[f"d{d}-{g}"][L] = v
        stats_p = run_dirs[name] / "stats.json"
        check = {}
        if stats_p.exists() and not a.limit:
            st = json.loads(stats_p.read_text())
            want = {k.split("|", 1)[1].replace("|", "-"): v["examples"] for k, v in st["by_cell"].items()
                    if k.startswith("comp|")}
            got = {k: sum(v.values()) for k, v in by.items()}
            check = {"examples": [n_ex, st["examples"]], "comp_cells": {k: [got.get(k, 0), want.get(k, 0)]
                                                                          for k in sorted(set(want) | set(got))}}
            check["exact"] = n_ex == st["examples"] and all(x == y for x, y in check["comp_cells"].values())
        report[name] = {**m, "examples": n_ex, "comp_by_cell_and_len": by, "check_vs_stats": check}
        ok = check.get("exact")
        print(f"{name}: {n_ex} examples replayed; matches stats.json: {ok}", flush=True)
        for k in sorted(by):
            print(f"  {k:9s} " + " ".join(f"L{L}={v}" for L, v in sorted(by[k].items())), flush=True)
    (out / "replay.json").write_text(json.dumps(report, indent=1))
    print(f"replay of {len(jobs)} rank x worker streams in {time.time() - t:.0f}s -> {out / 'replay.json'}")


# ---- analyze -------------------------------------------------------------------------------------

def first_error(task: Task, funcs, x: str, text: str):
    """(type, step index or None, detail dict) for one completion."""
    if EOS not in text:
        return "format", None, {"why": "no_eos"}
    body = text.split(EOS, 1)[0]
    pre = body.split("A:", 1)[0]
    ab = re.escape(task.alphabet)
    steps = re.findall(rf"F(\d\d)\(([{ab}]*)\)=([{ab}]*);", pre)
    prev = x
    for j, f in enumerate(funcs):
        if j >= len(steps):
            return "missing_step", j, {}
        fid, arg, out = int(steps[j][0]), steps[j][1], steps[j][2]
        if fid != f:
            return "wrong_func", j, {"want": task.id2name[f], "got": task.id2name[fid] if fid < task.n_funcs else fid}
        if arg != prev:
            return "copy", j, {"len_diff": len(arg) - len(prev),
                               "n_wrong": sum(u != v for u, v in zip(arg, prev)) + abs(len(arg) - len(prev))}
        want = task.apply(f, arg)
        if out != want:
            pos = [i for i, (u, v) in enumerate(zip(out, want)) if u != v]
            return "apply", j, {"func": task.id2name[f], "len_diff": len(out) - len(want), "n_wrong": len(pos),
                                "first_pos": pos[0] if pos else min(len(out), len(want))}
        prev = out
    if len(steps) > len(funcs):
        return "extra_step", len(funcs), {}
    ans = task.parse_answer(text)
    if ans is None:
        return "format", len(funcs), {"why": "answer"}
    if ans != prev:
        return "answer_copy", len(funcs), {}
    return "ok", None, {}


def cmd_analyze(a):
    task = Task(load_config()["task"])
    summary = {}
    for d in a.dirs:
        d = Path(d)
        df = pd.read_parquet(d / "completions.parquet")
        meta = json.loads((d / "gen.json").read_text())
        res = [first_error(task, f, x, c) for f, x, c in zip(df.funcs, df.x, df.completion)]
        df["err"], df["err_step"] = [r[0] for r in res], [r[1] for r in res]
        df["detail"] = [json.dumps(r[2]) for r in res]
        rew = [task.reward(c, task.make_problem(f, x)) for f, x, c in zip(df.funcs, df.x, df.completion)]
        mismatch = int(((df.err == "ok") != (pd.Series(rew) == 1.0)).sum())
        df.drop(columns=["completion"]).to_parquet(d / "errors.parquet", index=False)
        name = meta["model_id"]
        print(f"\n===== {name} ({meta['n_problems']} problems, k={meta['k']} + greedy); "
              f"classifier vs reward mismatches: {mismatch}")
        rep = {}
        for (dep, g), s in df.groupby(["depth", "group"]):
            cell = f"d{dep}-{g}"
            smp, grd = s[s["sample"] >= 0], s[s["sample"] < 0]
            acc = smp.groupby("str_len").err.apply(lambda e: round(float((e == "ok").mean()), 3))
            gacc = grd.groupby("str_len").err.apply(lambda e: round(float((e == "ok").mean()), 3))
            types = (smp.groupby("str_len").err.value_counts(normalize=True).unstack(fill_value=0).round(3))
            where = (smp[smp.err != "ok"].groupby("str_len").err_step.value_counts(normalize=True)
                     .unstack(fill_value=0).round(3))
            # P(step j correct | all earlier steps correct), sampled completions
            cond = {}
            for L, sl in smp.groupby("str_len"):
                row = {}
                for j in range(dep):
                    reached = sl[(sl.err == "ok") | (sl.err_step.fillna(99) >= j)]
                    failed = reached[(reached.err != "ok") & (reached.err_step == j)]
                    row[f"s{j + 1}"] = round(1 - len(failed) / max(1, len(reached)), 3)
                cond[int(L)] = row
            print(f"\n--- {cell}: accuracy by str_len (sampled T={meta['temperature']}) {acc.to_dict()}  "
                  f"greedy {gacc.to_dict()}")
            print("first-error type (share of all samples):\n" + types.to_string())
            print("step of the first error (share of failures, 0 = first step):\n" + where.to_string())
            print("P(step correct | earlier steps correct):\n" + pd.DataFrame(cond).T.to_string())
            # apply errors per function and step, by length: rate among completions reaching that step
            ap = {}
            for L, sl in smp.groupby("str_len"):
                for j in range(dep):
                    reached = sl[(sl.err == "ok") | (sl.err_step.fillna(99) >= j)]
                    f_at = reached.funcs.apply(lambda fs, j=j: task.id2name[fs[j]])
                    bad = (reached.err == "apply") & (reached.err_step == j)
                    for fn, n in f_at.value_counts().items():
                        ap.setdefault(fn, {}).setdefault(int(L), [0, 0])
                        ap[fn][int(L)][0] += int((bad & (f_at == fn)).sum())
                        ap[fn][int(L)][1] += int(n)
            ap_rate = {fn: {L: round(b / max(1, n), 3) for L, (b, n) in sorted(v.items())} for fn, v in ap.items()}
            print("apply-error rate per function (all steps pooled), by str_len:\n"
                  + pd.DataFrame(ap_rate).T.sort_index().to_string())
            rep[cell] = {"acc": acc.to_dict(), "greedy_acc": gacc.to_dict(), "first_error_type": types.to_dict("index"),
                         "first_error_step": where.to_dict("index"), "step_cond_acc": cond, "apply_rate": ap_rate}
        summary[name] = rep
        (d / "analysis.json").write_text(json.dumps(rep, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("ckpt")
    g.add_argument("--out", required=True)
    g.add_argument("--cells", default="4-H,3-H,4-mixed", help="depth-group cells of the RL pool")
    g.add_argument("--per-len", type=int, default=400, help="problems per (cell, str_len)")
    g.add_argument("--k", type=int, default=8)
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", default="cuda")
    r = sub.add_parser("replay")
    r.add_argument("runs", nargs="+")
    r.add_argument("--out", required=True)
    r.add_argument("--procs", type=int, default=16)
    r.add_argument("--limit", type=int, default=0, help="micro-batches per stream (smoke test; skips the check)")
    z = sub.add_parser("analyze")
    z.add_argument("dirs", nargs="+")
    a = ap.parse_args()
    {"gen": cmd_gen, "replay": cmd_replay, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
