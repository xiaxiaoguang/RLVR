"""Stratified problem-set generation over (depth x group [x function]) cells."""
from __future__ import annotations

import random

from .splits import Splits
from .task import Problem


def cells_for(depths, groups):
    """Valid (depth, group) cells; depth-1 problems cannot be 'mixed'."""
    return [(d, g) for d in depths for g in groups if not (d == 1 and g == "mixed")]


def stratified_set(splits: Splits, pool: str, depths, groups, per_cell: int,
                   rng: random.Random, exclude: set, novel: bool = False,
                   by_function: bool = False) -> list[Problem]:
    """per_cell problems per (depth, group) cell, unique and not in `exclude`
    (which is updated in place so successive sets stay disjoint).
    by_function (depth 1 only): per_cell problems per function in the group."""
    task = splits.task
    out = []
    for depth, group in cells_for(depths, groups):
        if by_function:
            assert depth == 1
            funcs = task.H if group == "H" else task.L if group == "L" else range(task.n_funcs)
            for f in funcs:
                for _ in range(per_cell):
                    p = splits.sample(pool, 1, rng, func_pool=[f], exclude=exclude)
                    exclude.add(p.pid)
                    out.append(p)
            continue
        for _ in range(per_cell):
            p = splits.sample(pool, depth, rng, group=group, novel=novel, exclude=exclude)
            exclude.add(p.pid)
            out.append(p)
    return out
