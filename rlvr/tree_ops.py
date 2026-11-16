"""Length-preserving operators on one, two, or three equal-length strings.

Symbols are integer indices modulo the alphabet size. Arity is explicit and
validated; unary implementations reuse the original ATOMS registry.
"""

from dataclasses import dataclass
from collections.abc import Callable

from .atoms import ATOMS


@dataclass(frozen=True)
class Operator:
    arity: int
    fn: Callable
    version: str = "1"

    def __call__(self, args: list[list[int]], alphabet_size: int) -> list[int]:
        if len(args) != self.arity:
            raise ValueError(f"expected {self.arity} arguments, got {len(args)}")
        if not args[0] or len({len(x) for x in args}) != 1:
            raise ValueError("operator arguments must be nonempty and equal-length")
        return self.fn(args, alphabet_size)


# All arguments are semantically active; none is a concatenation or a length change.
MULTI = {
    "zip_add": Operator(2, lambda a, A: [(x + y) % A for x, y in zip(*a)]),
    "zip_sub": Operator(2, lambda a, A: [(x - y) % A for x, y in zip(*a)]),
    "zip_mul": Operator(2, lambda a, A: [(x * y) % A for x, y in zip(*a)]),
    "gather": Operator(2, lambda a, A: [a[0][y % len(a[0])] for y in a[1]]),
    "gather_add": Operator(2, lambda a, A: [(a[0][y % len(a[0])] + y) % A for y in a[1]]),
    "scatter_add": Operator(2, lambda a, A: _scatter_add(a, A)),
    "rotate_by_sum": Operator(2, lambda a, A: _rotate_by_sum(a)),
    "prefix_add": Operator(2, lambda a, A: _prefix_add(a, A)),
    "zip_add3": Operator(3, lambda a, A: [(x + y + z) % A for x, y, z in zip(*a)]),
    "zip_mul_add": Operator(3, lambda a, A: [(x * y + z) % A for x, y, z in zip(*a)]),
    "gather_mix": Operator(3, lambda a, A: [(a[0][y % len(a[0])] + z) % A for y, z in zip(a[1], a[2])]),
    "select_parity": Operator(3, lambda a, A: [x if z % 2 == 0 else y for x, y, z in zip(*a)]),
}


def _scatter_add(a, A):
    out = [0] * len(a[0])
    for x, y in zip(*a):
        out[y % len(out)] = (out[y % len(out)] + x) % A
    return out


def _rotate_by_sum(a):
    offset = sum(a[1]) % len(a[0])
    return a[0][offset:] + a[0][:offset]


def _prefix_add(a, A):
    total, out = 0, []
    for x, y in zip(*a):
        total = (total + y) % A
        out.append((x + total) % A)
    return out


def build_operator(spec: str | dict) -> Operator:
    name = spec if isinstance(spec, str) else spec["name"]
    if name in MULTI:
        if isinstance(spec, dict) and set(spec) != {"name"}:
            raise ValueError(f"{name} does not accept parameters")
        return MULTI[name]
    fn = ATOMS.build(spec)
    return Operator(1, lambda args, A: fn(args[0], A))
