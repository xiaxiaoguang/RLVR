"""Stratified problem-set generation over (depth [x n_bin] x group [x function]) cells.

depth is the number of operations (n_ops); n_bin > 0 cells hold v2 trees. With the default
n_bins = [0] (chains only) the cells, their order and the random draws are exactly v1's.
"""
from __future__ import annotations

import random

from .splits import Splits
from .task import Problem


def cells_for(depths, groups, n_bins=(0,), task=None):
    """Valid (depth, n_bin, group) cells, depth-major. Chains: depth-1 problems cannot be
    'mixed'; trees: see Task.valid_cell (needs `task`)."""
    if task is None:
        assert tuple(n_bins) == (0,), "tree cells need the task"
        return [(d, 0, g) for d in depths for g in groups if not (d == 1 and g == "mixed")]
    return [(d, nb, g) for d in depths for nb in n_bins for g in groups if task.valid_cell(d, nb, g)]


def stratified_set(splits: Splits, pool: str, depths, groups, per_cell: int,
                   rng: random.Random, exclude: set, novel: bool = False,
                   by_function: bool = False, n_bins=(0,)) -> list[Problem]:
    """per_cell problems per (depth, n_bin, group) cell, unique and not in `exclude`
    (which is updated in place so successive sets stay disjoint).
    by_function (depth 1 only): per_cell problems per function in the group
    (group 'bin' with n_bin 1: per binary operator).
    Input lengths cycle through task.str_lens, so every cell is balanced over length."""
    task = splits.task
    lens = task.str_lens
    out = []
    for depth, n_bin, group in cells_for(depths, groups, n_bins, task):
        if by_function:
            assert depth == 1
            if n_bin:
                for f in task.B:
                    for i in range(per_cell):
                        p = splits.sample(pool, 1, rng, group="bin", exclude=exclude, n_bin=1, bin_pool=[f],
                                          str_len=lens[i % len(lens)])
                        exclude.add(p.pid)
                        out.append(p)
                continue
            funcs = task.H if group == "H" else task.L if group == "L" else range(task.n_funcs)
            for f in funcs:
                for i in range(per_cell):
                    p = splits.sample(pool, 1, rng, func_pool=[f], exclude=exclude,
                                      str_len=lens[i % len(lens)])
                    exclude.add(p.pid)
                    out.append(p)
            continue
        for i in range(per_cell):
            if n_bin:
                p = splits.sample(pool, depth, rng, group=group, exclude=exclude, n_bin=n_bin,
                                  str_len=lens[i % len(lens)])
            else:
                p = splits.sample(pool, depth, rng, group=group, novel=novel, exclude=exclude,
                                  str_len=lens[i % len(lens)])
            exclude.add(p.pid)
            out.append(p)
    return out


def cell_candidates(splits: Splits, pool: str, cell: tuple[int, int, str | None], n: int,
                    seed: int | str) -> list[Problem]:
    """n unique problems of one (n_ops, n_bin, group) cell from the cell's own random stream,
    lengths cycling through task.str_lens. The first k candidates of a cell are the same for
    any n >= k and do not depend on other cells, so per-model pilot pools stay consistent."""
    n_ops, n_bin, group = cell
    rng = random.Random(f"{seed}|{n_ops}|{n_bin}|{group}")
    lens = splits.task.str_lens
    seen: set = set()
    out = []
    for i in range(n):
        p = splits.sample(pool, n_ops, rng, group=group, exclude=seen, n_bin=n_bin, str_len=lens[i % len(lens)])
        seen.add(p.pid)
        out.append(p)
    return out
