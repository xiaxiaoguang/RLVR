"""Evaluation shared by pretraining and RL: greedy accuracy and pass-rate labelling.

    rows = greedy_eval(model, tok, task, {"E1": problems, ...}, step=1000)
    df = label_pass_rates(model, tok, task, problems, k=32, model_id="pt-mid-s0")

greedy_eval returns eval.jsonl rows: one per (eval_set, depth, group), one per
eval_set with depth/group "all", and for E1 one per function (key `func`).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
import pandas as pd

from .difficulty import wilson
from .model import generate
from .task import Problem, Task


def _summarise(step, eval_set, key: dict, recs: list[tuple[float, bool, int]]) -> dict:
    acc, fmt, ln = zip(*recs)
    return {"step": step, "eval_set": eval_set, **key, "n": len(recs),
            "greedy_acc": float(np.mean(acc)), "format_err": float(np.mean(fmt)),
            "mean_len": float(np.mean(ln))}


def greedy_eval(model, tok, task: Task, sets: dict[str, Sequence[Problem]], step: int = 0,
                max_new_tokens: int = 128, batch_size: int = 1024) -> list[dict]:
    rows = []
    for name, probs in sets.items():
        outs = generate(model, tok, [task.format_prompt(p) for p in probs], max_new_tokens,
                        temperature=0.0, batch_size=batch_size)
        cells, funcs = defaultdict(list), defaultdict(list)
        for p, o in zip(probs, outs):
            rec = (task.reward(o, p), task.parse_answer(o) is None, len(tok.encode(o)))
            cells[(p.depth, p.group)].append(rec)
            cells[("all", "all")].append(rec)
            if p.depth == 1:
                funcs[p.funcs[0]].append(rec)
        for (d, g), recs in sorted(cells.items(), key=lambda kv: str(kv[0])):
            rows.append(_summarise(step, name, {"depth": d, "group": g}, recs))
        if name == "E1":
            for f, recs in sorted(funcs.items()):
                key = {"depth": 1, "group": "H" if f in task.H else "L",
                       "func": task.fname(f), "label": task.id2name[f]}
                rows.append(_summarise(step, name, key, recs))
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
        rows.append({"pid": p.pid, "depth": p.depth, "funcs": list(p.funcs), "group": p.group,
                     "str_len": len(p.x), "n_correct": n, "k": k, "p_hat": n / k,
                     "ci_lo": lo, "ci_hi": hi, "model_id": model_id})
    df = pd.DataFrame(rows)
    for col, vals in (extra or {}).items():
        df[col] = list(vals)
    return df
