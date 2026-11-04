"""Model-relative difficulty p: bins, Wilson intervals, signal probability."""
from __future__ import annotations

import math

import numpy as np

from .registry import Registry

# Default bins for G=8 (doc §分箱). Edge bins produce almost no gradient.
P_BINS: list[tuple[str, float, float]] = [
    ("0-0.03", 0.0, 1 / 32),
    ("0.03-0.2", 1 / 32, 0.2),
    ("0.2-0.4", 0.2, 0.4),
    ("0.4-0.6", 0.4, 0.6),
    ("0.6-0.8", 0.6, 0.8),
    ("0.8-0.97", 0.8, 31 / 32),
    ("0.97-1", 31 / 32, 1.0 + 1e-9),
]
P_BIN_NAMES = [b[0] for b in P_BINS]


def p_bin(p: float) -> str:
    for name, lo, hi in P_BINS:
        if lo <= p < hi:
            return name
    raise ValueError(p)


def in_p_range(p, spec) -> np.ndarray:
    """spec: a bin name, a list of bin names, or an inclusive [lo, hi] numeric range. NaN is in none."""
    p = np.asarray(p, dtype=float)
    if isinstance(spec, str):
        spec = [spec]
    if len(spec) == 2 and all(isinstance(v, (int, float)) for v in spec):
        lo, hi = spec
        return (p >= lo) & (p <= hi)
    names = np.array([p_bin(v) if not np.isnan(v) else "" for v in p]) if p.size else np.array([], dtype=str)
    return np.isin(names, list(spec))


def wilson(n_correct: int, k: int, z: float = 1.96) -> tuple[float, float]:
    if k == 0:
        return 0.0, 1.0
    ph = n_correct / k
    den = 1 + z * z / k
    c = (ph + z * z / (2 * k)) / den
    h = z * math.sqrt(ph * (1 - ph) / k + z * z / (4 * k * k)) / den
    return max(0.0, c - h), min(1.0, c + h)


def p_signal(p, G: int):
    """Probability a group of G rollouts has mixed rewards: 1 - p^G - (1-p)^G."""
    p = np.asarray(p, dtype=float)
    return 1 - p ** G - (1 - p) ** G


# ---- sampling weights for dataset construction ------------------------------
# Each factory returns fn(p_hat: np.ndarray) -> non-negative weights.

WEIGHTS = Registry("weight")


@WEIGHTS.register()
def uniform():
    return lambda p: np.ones_like(np.asarray(p, dtype=float))


@WEIGHTS.register()
def learnability():
    """SFL: p(1-p)."""
    return lambda p: np.asarray(p, dtype=float) * (1 - np.asarray(p, dtype=float))


@WEIGHTS.register()
def signal(G: int = 8):
    """Probability that a group of G rollouts carries gradient signal."""
    return lambda p: p_signal(p, G)
