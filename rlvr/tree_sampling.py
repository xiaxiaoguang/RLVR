"""Sample ordered unary/binary/ternary tree shapes with exact depth and size.

Dynamic programming counts feasible shapes. Sampling is uniform over shapes
conditional on (n_ops, depth), with no unbounded rejection or truncation.
A leaf is (); an internal node is a tuple of one to three child shapes.
"""

from functools import lru_cache
from itertools import product


@lru_cache(None)
def partitions(total: int, width: int):
    if width == 1:
        return ((total,),)
    return tuple((n, *tail) for n in range(total + 1) for tail in partitions(total - n, width - 1))


@lru_cache(None)
def options(n_ops: int, depth: int, arities: tuple[int, ...]):
    if n_ops < 1 or depth < 1 or depth > n_ops:
        return ()
    out = []
    for arity in arities:
        for sizes in partitions(n_ops - 1, arity):
            ranges = [range(1, min(n, depth - 1) + 1) if n else (0,) for n in sizes]
            for heights in product(*ranges):
                if max(heights) != depth - 1:
                    continue
                weight = 1
                for n, h in zip(sizes, heights):
                    weight *= count(n, h, arities)
                if weight:
                    out.append((tuple(zip(sizes, heights)), weight))
    return tuple(out)


@lru_cache(None)
def count(n_ops: int, depth: int, arities: tuple[int, ...]):
    if n_ops == 0:
        return int(depth == 0)
    return sum(w for _, w in options(n_ops, depth, arities))


def sample_shape(n_ops, depth, arities, rng):
    if n_ops == 0 and depth == 0:
        return ()
    choices = options(n_ops, depth, tuple(arities))
    if not choices:
        raise ValueError(f"no tree with n_ops={n_ops}, depth={depth}, arities={arities}")
    ticket = rng.randrange(sum(w for _, w in choices))
    for children, weight in choices:
        if ticket < weight:
            return tuple(sample_shape(n, h, arities, rng) for n, h in children)
        ticket -= weight
    raise AssertionError("unreachable")
