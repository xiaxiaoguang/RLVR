"""Tree pool isolation and held-out directed, argument-position-aware edges."""

import random

from .splits import Splits


class TreeSplits(Splits):
    def __init__(self, task, cfg):
        from .params import PARAMS_DEFAULTS

        cfg = {**PARAMS_DEFAULTS["splits"], **cfg}
        self.cfg, self.task = cfg, task
        self.salt = cfg["salt"]
        self.pools = {name: tuple(bounds) for name, bounds in cfg["pools"].items()}
        rng = random.Random(cfg["heldout_seed"])
        k = cfg["heldout_pairs_per_group_pair"]
        edges = []
        for parents in (task.H, task.L):
            for children in (task.H, task.L):
                for pa in sorted({task.arities[p] for p in parents}):
                    for ca in sorted({task.arities[c] for c in children}):
                        choices = [
                            (p, slot, c)
                            for p in parents
                            if task.arities[p] == pa
                            for slot in range(pa)
                            for c in children
                            if task.arities[c] == ca
                        ]
                        if k > len(choices):
                            raise ValueError("too many held-out edges for an arity/group cell")
                        edges.extend(rng.sample(choices, k))
        self.heldout_edges = frozenset(edges)

    def has_heldout(self, p):
        return bool(set(self.task.edges(p)) & self.heldout_edges)

    def has_heldout_pair(self, funcs):
        raise ValueError("tree splits require the expression; flattened function pairs are not tree edges")

    def sample(self, pool, depth, rng, group=None, novel=False, exclude=None, max_tries=100_000, **kw):
        if novel and depth < 2:
            raise ValueError("novel operator edges require depth >= 2")
        for _ in range(max_tries):
            expr = self.task.sample_expression(depth, rng, group=group, **kw)
            pid = self.task.problem_id(expr)
            if self.pool_of(pid) != pool or (exclude is not None and pid in exclude):
                continue
            heldout = bool(set(self.task.expression_edges(expr)) & self.heldout_edges)
            if heldout == novel:
                return self.task.make_problem(expr)
        raise RuntimeError(f"could not sample tree pool={pool}, depth={depth}, group={group}, novel={novel}")
