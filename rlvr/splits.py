"""Leak-free pools.

Problem level: each pid is hashed (salted) into a bucket in [0, 1000); bucket
ranges define the pretrain / rl / eval pools, which never overlap.

Combination level: a fixed set of ordered function pairs (inner -> outer, i.e.
adjacent in `funcs`) is held out. Any problem containing a held-out pair is
excluded from pretrain and rl pools and from the in-distribution eval sets;
E4 (novel compositions) consists only of problems that contain one. In a tree
(v2) a held-out pair is any child -> parent edge between two unary functions.
"""
from __future__ import annotations

import hashlib
import random
from typing import Sequence

from .task import Problem, Task


class Splits:
    def __init__(self, task: Task, cfg: dict | None = None):
        """cfg: the `splits` section of a resolved config (defaults: rlvr.params)."""
        from .params import PARAMS_DEFAULTS
        cfg = {**PARAMS_DEFAULTS["splits"], **(cfg or {})}
        self.cfg = cfg
        self.task = task
        self.salt = cfg["salt"]
        self.pools = {k: tuple(v) for k, v in cfg["pools"].items()}
        rng = random.Random(cfg["heldout_seed"])
        k = cfg["heldout_pairs_per_group_pair"]
        pairs = []
        for gi in (task.H, task.L):
            for go in (task.H, task.L):
                cand = sorted((a, b) for a in gi for b in go if a != b)
                pairs += rng.sample(cand, k)
        self.heldout_pairs: frozenset[tuple[int, int]] = frozenset(pairs)

    def bucket(self, pid: str) -> int:
        h = hashlib.sha256(f"{self.salt}|{pid}".encode()).digest()
        return int.from_bytes(h[:8], "big") % 1000

    def pool_of(self, pid: str) -> str:
        b = self.bucket(pid)
        for name, (lo, hi) in self.pools.items():
            if lo <= b < hi:
                return name
        raise ValueError(f"bucket {b} not covered by pools")

    def has_heldout_pair(self, funcs: Sequence[int]) -> bool:
        return any((funcs[i], funcs[i + 1]) in self.heldout_pairs for i in range(len(funcs) - 1))

    def has_heldout(self, p: Problem) -> bool:
        """Held-out pair anywhere in a chain or tree (unary child -> unary parent edges)."""
        if not p.args:
            return self.has_heldout_pair(p.funcs)
        nf, f = self.task.n_funcs, p.funcs
        return any(f[i] < nf and a[0] >= 0 and f[a[0]] < nf and (f[a[0]], f[i]) in self.heldout_pairs
                   for i, a in enumerate(p.args))

    def admissible(self, p: Problem, pool: str, novel: bool = False) -> bool:
        """novel=False: in the pool and free of held-out pairs.
        novel=True: in the pool and containing at least one held-out pair."""
        return self.pool_of(p.pid) == pool and self.has_heldout(p) == novel

    def sample(self, pool: str, depth: int, rng: random.Random, group: str | None = None,
               novel: bool = False, exclude: set | None = None, max_tries: int = 100_000,
               n_bin: int = 0, bin_pool: Sequence[int] | None = None, **kw) -> Problem:
        """Rejection-sample a problem in `pool` (optionally novel), not in `exclude`.
        n_bin > 0: a tree with depth = n_ops operations, n_bin of them binary (not novel)."""
        for _ in range(max_tries):
            if n_bin:
                assert not novel, "novel (E4) problems are chains"
                p = self.task.sample_tree(depth, n_bin, rng, group, kw.get("str_len"), bin_pool)
            elif novel:
                funcs = self._sample_novel_funcs(depth, rng, group)
                p = self.task.make_problem(funcs, self.task.sample_x(rng, kw.get("str_len")))
            else:
                p = self.task.sample_problem(depth, rng, group, **kw)
            if exclude is not None and p.pid in exclude:
                continue
            if self.admissible(p, pool, novel):
                return p
        raise RuntimeError(f"could not sample pool={pool} depth={depth} n_bin={n_bin} group={group} novel={novel}")

    def _sample_novel_funcs(self, depth: int, rng: random.Random, group: str | None):
        # Plant one held-out pair at a random position, fill the rest at random,
        # then check the group constraint. Rejection keeps it unbiased enough.
        assert depth >= 2
        pairs = sorted(self.heldout_pairs)
        while True:
            a, b = rng.choice(pairs)
            i = rng.randrange(depth - 1)
            funcs = list(self.task.sample_funcs(depth, rng))
            funcs[i], funcs[i + 1] = a, b
            funcs = tuple(funcs)
            if group is None or self.task.group_of(funcs) == group or \
                    (group == "anyL" and self.task.group_of(funcs) != "H"):
                return funcs
