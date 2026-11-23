"""Evaluation shared by pretraining and RL: greedy accuracy, step score, pass@k, pass-rate labelling.

    rows = greedy_eval(model, tok, task, {"E1": problems, ...}, step=1000)
    rows = greedy_eval(model, tok, task, sets, step, pass_k_samples=16)   # adds pass@1 .. pass@16
    df = label_pass_rates(model, tok, task, problems, k=32, model_id="pt-mid-s0")

greedy_eval returns eval.jsonl rows: one per (eval_set, depth, group), one per eval_set with
depth/group "all", one per (eval_set, depth) over all groups (aggregation "depth"), for tree
problems one per (depth, n_ops, n_binary, n_ternary) (aggregation "structure"), and for E1 one per
function (key `func`). Columns: n, greedy_acc, format_err, mean_len, step_score (mean number of
leading correct step lines of the greedy completion, rlvr/steps.py) and step_frac (step_score / n_ops
per problem, averaged). pass_k_samples = n > 0 draws n samples per problem at `temperature` and adds
pass@k for each k in pass_k with k <= n (unbiased estimator, pass_at_k); 0 = no sampling, no columns.
The overall row of each eval set (depth and group "all") also has `sec`, the set's wall time
(generation and scoring, greedy and pass@k together).
`records`, if given, gets one dict per (eval_set, problem) for the greedy completion; `progress`
prints generation rate and ETA per eval set (rlvr.model.generate_ids).
"""
from __future__ import annotations

import time
from collections import defaultdict
from math import comb
from typing import Sequence

import numpy as np
import pandas as pd

from .difficulty import wilson
from .model import generate
from .steps import first_error
from .task import Problem, Task

COLUMNS = ("greedy_acc", "format_err", "mean_len", "step_score", "step_frac")


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from c correct of n samples: 1 - C(n - c, k) / C(n, k)."""
    return 1.0 - comb(n - c, k) / comb(n, k)


def _summarise(step, eval_set, key: dict, recs: list[tuple], cols: Sequence[str]) -> dict:
    return {"step": step, "eval_set": eval_set, **key, "n": len(recs),
            **{c: float(np.mean(v)) for c, v in zip(cols, zip(*recs))}}


def greedy_eval(model, tok, task: Task, sets: dict[str, Sequence[Problem]], step: int = 0,
                max_new_tokens: int = 128, batch_size: int = 1024, pass_k_samples: int = 0,
                pass_k: Sequence[int] = (1, 2, 4, 8, 16), temperature: float = 1.0,
                records: list[dict] | None = None, progress: bool = False) -> list[dict]:
    n_s = pass_k_samples
    ks = [k for k in pass_k if k <= n_s]
    cols = COLUMNS + tuple(f"pass@{k}" for k in ks)
    rows = []
    for name, probs in sets.items():
        t0 = time.time()
        prompts = [task.format_prompt(p) for p in probs]
        outs = generate(model, tok, prompts, max_new_tokens, temperature=0.0, batch_size=batch_size,
                        progress=f"{name} greedy" if progress else "")
        if n_s:
            samples = generate(model, tok, prompts, max_new_tokens, temperature=temperature,
                               num_return_sequences=n_s, batch_size=max(1, batch_size // n_s),
                               progress=f"{name} pass@k" if progress else "")
        cells, funcs, structures, depths = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
        for i, (p, o) in enumerate(zip(probs, outs)):
            err, err_step, n_correct = first_error(task, p, o)
            rec = (task.reward(o, p), task.parse_answer(o) is None, len(tok.encode(o)),
                   n_correct, n_correct / len(p.funcs))
            if n_s:
                c = int(sum(task.reward(s, p) for s in samples[i * n_s:(i + 1) * n_s]))
                rec += tuple(pass_at_k(n_s, c, k) for k in ks)
            cells[(p.depth, p.group)].append(rec)
            depths[p.depth].append(rec)
            if hasattr(p, "n_ops"):
                structures[(p.depth, p.n_ops, p.n_binary, p.n_ternary)].append(rec)
            cells[("all", "all")].append(rec)
            if p.depth == 1:
                funcs[p.funcs[0]].append(rec)
            if records is not None:
                records.append({"pid": p.pid, "eval_set": name, "depth": p.depth, "group": p.group,
                                "str_len": len(p.x), "n_ops": len(p.funcs),
                                **{k: getattr(p, k) for k in ("n_binary", "n_ternary") if hasattr(p, k)},
                                "correct": rec[0] == 1.0, "err_type": err, "err_step": err_step,
                                "n_correct": n_correct, "completion": o})
        sec = time.time() - t0
        for (d, g), recs in sorted(cells.items(), key=lambda kv: str(kv[0])):
            rows.append(_summarise(step, name, {"depth": d, "group": g}, recs, cols))
            if d == g == "all":
                rows[-1]["sec"] = sec
        for d, recs in sorted(depths.items()):
            rows.append(_summarise(step, name, {"depth": d, "group": "all", "aggregation": "depth"}, recs, cols))
        for (depth, n_ops, n_binary, n_ternary), recs in sorted(structures.items()):
            rows.append(_summarise(step, name, {"depth": depth, "group": "all", "n_ops": n_ops,
                                                "n_binary": n_binary, "n_ternary": n_ternary,
                                                "aggregation": "structure"}, recs, cols))
        if name == "E1":
            for f, recs in sorted(funcs.items()):
                key = {"depth": 1, "group": "H" if f in task.H else "L",
                       "func": task.fname(f), "label": task.id2name[f]}
                rows.append(_summarise(step, name, key, recs, cols))
    return rows


def label_pass_rates(model, tok, task: Task, problems: Sequence[Problem], k: int = 32,
                     temperature: float = 1.0, batch_size: int = 512, max_new_tokens: int = 128,
                     model_id: str = "", extra: dict[str, Sequence] | None = None) -> pd.DataFrame:
    """p_hat = correct / k from k samples per problem. Columns as in the design doc;
    `extra` adds per-problem columns (e.g. origin)."""
    prompts = [task.format_prompt(p) for p in problems]
    outs = generate(model, tok, prompts, max_new_tokens, temperature=temperature,
                    num_return_sequences=k, batch_size=batch_size)
    rows = []
    for i, p in enumerate(problems):
        n = int(sum(task.reward(o, p) for o in outs[i * k:(i + 1) * k]))
        lo, hi = wilson(n, k)
        structural = {key: getattr(p, key) for key in ("n_ops", "n_binary", "n_ternary", "shape_id",
                       "max_live_values", "task_fingerprint") if hasattr(p, key)}
        structural["skippable"] = task.is_skippable(p)
        rows.append({**structural, "pid": p.pid, "depth": p.depth, "funcs": list(p.funcs), "group": p.group,
                     "str_len": len(p.x), "n_correct": n, "k": k, "p_hat": n / k,
                     "ci_lo": lo, "ci_hi": hi, "model_id": model_id})
    df = pd.DataFrame(rows)
    for col, vals in (extra or {}).items():
        df[col] = list(vals)
    return df
