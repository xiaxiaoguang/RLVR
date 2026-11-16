"""Per-step problem samplers for GRPO and the online per-problem pass rate p(t).

    sampler = ProblemSampler(depths, kind="online_filter", seed=0, total=steps * B, p_init=p_hat)
    idx = sampler.sample(B)                  # dataset indices for this step
    sampler.update(idx, r.mean(1))           # group mean reward -> EMA p(t)

Kinds (doc D9, groups A-C):
  static         every problem equally often: shuffled epochs without replacement (order=shuffle),
                 or exactly the same draws sorted by depth over the whole run (order=shallow_to_deep).
  online_filter  uniform over problems whose p(t) is unknown or within [p_edge, 1 - p_edge].
  learnability   probability proportional to p(t) (1 - p(t)); unknown p counts as 0.5.

p(t) is tracked for every kind (it is logged), but only the dynamic kinds sample on it. It starts
from p_init (the base model's labels; NaN = unknown), the first observation of an unknown problem
sets it, later ones update it as an EMA. The dynamic kinds draw without replacement within a
step, fill up uniformly from the rest of the dataset when too few problems qualify, and take
`explore_frac` of each batch uniformly from the whole dataset.
"""
from __future__ import annotations

import numpy as np

KINDS = ("static", "online_filter", "learnability")
ORDERS = ("shuffle", "shallow_to_deep")


class ProblemSampler:
    def __init__(self, depths, kind: str = "static", order: str = "shuffle", seed: int = 0,
                 total: int = 0, p_init=None, p_ema: float = 0.9, p_edge: float = 1 / 32,
                 explore_frac: float = 0.0):
        """depths: depth per dataset problem. total: draws of the whole run (steps x B), needed
        for order=shallow_to_deep. p_init: starting p per problem (NaN = unknown) or None."""
        if kind not in KINDS or order not in ORDERS:
            raise ValueError(f"sampler kind {kind!r} / order {order!r} not in {KINDS} / {ORDERS}")
        if order != "shuffle" and kind != "static":
            raise ValueError("data.order applies to the static sampler only")
        self.depths = np.asarray(depths)
        self.n = len(self.depths)
        if self.n == 0:
            raise ValueError("empty dataset")
        self.kind, self.order = kind, order
        self.rng = np.random.default_rng(seed)
        self.p_ema, self.p_edge, self.explore_frac = p_ema, p_edge, explore_frac
        self.p = np.full(self.n, np.nan) if p_init is None else np.asarray(p_init, dtype=float).copy()
        assert self.p.shape == (self.n,)
        self.seen = np.zeros(self.n, dtype=np.int64)
        self.last = {"n_eligible": self.n, "n_fill": 0}
        self._queue = np.empty(0, dtype=np.int64)
        if order == "shallow_to_deep":
            if total <= 0:
                raise ValueError("shallow_to_deep needs the total number of draws")
            draws = np.concatenate([self.rng.permutation(self.n) for _ in range(-(-total // self.n))])[:total]
            self._queue = draws[np.argsort(self.depths[draws], kind="stable")]

    # ---- sampling ----------------------------------------------------------

    def sample(self, b: int) -> np.ndarray:
        if self.kind == "static":
            while len(self._queue) < b:
                if self.order == "shallow_to_deep":
                    raise RuntimeError("shallow_to_deep schedule exhausted (more draws than `total`)")
                self._queue = np.concatenate([self._queue, self.rng.permutation(self.n)])
            out, self._queue = self._queue[:b], self._queue[b:]
            self.last = {"n_eligible": self.n, "n_fill": 0}
            return out
        w = self.weights()
        n_explore = min(b, int(round(b * self.explore_frac)))
        picked = self._draw(np.ones(self.n), n_explore, np.zeros(self.n, dtype=bool))
        taken = np.zeros(self.n, dtype=bool)
        taken[picked] = True
        main = self._draw(w, b - len(picked), taken)
        taken[main] = True
        fill = self._draw(np.ones(self.n), b - len(picked) - len(main), taken)
        out = np.concatenate([picked, main, fill]).astype(np.int64)
        if len(out) < b:  # dataset smaller than the batch: repeat problems
            out = np.concatenate([out, self.rng.choice(self.n, b - len(out))])
        self.last = {"n_eligible": int((w > 0).sum()), "n_fill": int(b - len(picked) - len(main))}
        return self.rng.permutation(out)

    def _draw(self, w: np.ndarray, k: int, taken: np.ndarray) -> np.ndarray:
        """Up to k distinct indices with probability proportional to w, skipping `taken`."""
        w = np.where(taken, 0.0, w)
        k = min(k, int((w > 0).sum()))
        if k <= 0:
            return np.empty(0, dtype=np.int64)
        return self.rng.choice(self.n, size=k, replace=False, p=w / w.sum())

    def weights(self) -> np.ndarray:
        """Sampling weight per problem for the dynamic kinds (ones for static)."""
        if self.kind == "online_filter":
            return (np.isnan(self.p) | ((self.p >= self.p_edge) & (self.p <= 1 - self.p_edge))).astype(float)
        if self.kind == "learnability":
            p = np.where(np.isnan(self.p), 0.5, self.p)
            return p * (1 - p)
        return np.ones(self.n)

    # ---- online p ----------------------------------------------------------

    def update(self, idx, group_mean) -> None:
        for i, m in zip(np.asarray(idx).tolist(), np.asarray(group_mean, dtype=float).tolist()):
            self.p[i] = m if np.isnan(self.p[i]) else self.p_ema * self.p[i] + (1 - self.p_ema) * m
            self.seen[i] += 1

    def stats(self) -> dict:
        known = ~np.isnan(self.p)
        return {**self.last, "n_seen": int((self.seen > 0).sum()),
                "mean_p_online": float(self.p[known].mean()) if known.any() else None}
