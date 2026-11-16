"""Atomic string functions, registered as parameterized factories.

Each factory returns f(s: list[int], A: int) -> list[int] over symbol indices in
[0, A). Functions must preserve length (checked when a Task is built). The task
config maps a label to a spec, e.g. `rotl2: {name: rotate, k: 2}`.

Binary operators (v2, `task.binary_functions`) live in BINARY: each factory returns
f(a: list[int], b: list[int], A: int) -> list[int] for two strings of the same length n,
returning length n and depending on both inputs (both checked when a Task is built).
"""
from __future__ import annotations

from .registry import Registry

ATOMS = Registry("atom")
BINARY = Registry("binary")


# ---- position transforms ---------------------------------------------------

@ATOMS.register()
def reverse():
    return lambda s, A: s[::-1]


@ATOMS.register()
def rotate(k: int = 1):
    """Cyclic left rotation by k (negative = right)."""
    def f(s, A):
        j = k % len(s)
        return s[j:] + s[:j]
    return f


@ATOMS.register()
def swap_adjacent():
    """Swap positions (0,1), (2,3), ...; an odd last char stays."""
    def f(s, A):
        out = list(s)
        for i in range(0, len(s) - 1, 2):
            out[i], out[i + 1] = out[i + 1], out[i]
        return out
    return f


@ATOMS.register()
def swap_ends():
    return lambda s, A: list(s) if len(s) < 2 else [s[-1]] + s[1:-1] + [s[0]]


@ATOMS.register()
def swap_halves():
    """Swap first and last floor(n/2) chars; odd length keeps the middle in place."""
    def f(s, A):
        n, h = len(s), len(s) // 2
        return s[n - h:] + s[h:n - h] + s[:h]
    return f


@ATOMS.register()
def reverse_half(first: bool = True):
    """Reverse only the first (or last) floor(n/2) chars; an odd middle stays."""
    def f(s, A):
        n, h = len(s), len(s) // 2
        return s[:h][::-1] + s[h:] if first else s[:n - h] + s[n - h:][::-1]
    return f


@ATOMS.register()
def riffle():
    """Interleave the first ceil(n/2) chars with the rest: a0 b0 a1 b1 ..."""
    def f(s, A):
        h = (len(s) + 1) // 2
        out = []
        for i in range(h):
            out.append(s[i])
            if h + i < len(s):
                out.append(s[h + i])
        return out
    return f


# ---- character transforms --------------------------------------------------

@ATOMS.register()
def shift(k: int = 1):
    """Add k to every character, cyclically."""
    return lambda s, A: [(c + k) % A for c in s]


@ATOMS.register()
def complement():
    """Mirror the alphabet: first<->last, second<->second-to-last, ..."""
    return lambda s, A: [A - 1 - c for c in s]


@ATOMS.register()
def shift_at(k: int = 1, parity: int = 0):
    """Add k only at positions i with i % 2 == parity."""
    return lambda s, A: [(c + k) % A if i % 2 == parity else c for i, c in enumerate(s)]


@ATOMS.register()
def shift_by_pos(k: int = 1):
    """Add k * i to the char at position i, cyclically."""
    return lambda s, A: [(c + k * i) % A for i, c in enumerate(s)]


# ---- sorting -----------------------------------------------------------------

@ATOMS.register()
def sort(descending: bool = False):
    return lambda s, A: sorted(s, reverse=descending)


@ATOMS.register()
def sort_half(first: bool = True, descending: bool = False):
    """Sort only the first (or last) floor(n/2) chars; the rest stays in place."""
    def f(s, A):
        n, h = len(s), len(s) // 2
        if first:
            return sorted(s[:h], reverse=descending) + s[h:]
        return s[:n - h] + sorted(s[n - h:], reverse=descending)
    return f


# ---- conditional transforms ------------------------------------------------

@ATOMS.register()
def max_to_front():
    """Move the first occurrence of the largest char to the front."""
    def f(s, A):
        i = s.index(max(s))
        return [s[i]] + s[:i] + s[i + 1:]
    return f


@ATOMS.register()
def min_to_back():
    """Move the first occurrence of the smallest char to the back."""
    def f(s, A):
        i = s.index(min(s))
        return s[:i] + s[i + 1:] + [s[i]]
    return f


@ATOMS.register()
def mirror_first_half():
    """Overwrite the last floor(n/2) chars with the reversed first floor(n/2)."""
    def f(s, A):
        n, h = len(s), len(s) // 2
        return s[:n - h] + s[:h][::-1]
    return f


# ---- binary operators (v2) ---------------------------------------------------

@BINARY.register()
def add():
    """c_i = (a_i + b_i) mod A."""
    return lambda a, b, A: [(x + y) % A for x, y in zip(a, b)]


@BINARY.register()
def sub():
    """c_i = (a_i - b_i) mod A (backup operator)."""
    return lambda a, b, A: [(x - y) % A for x, y in zip(a, b)]


@BINARY.register()
def alternate():
    """c_i = a_i at even i, b_i at odd i."""
    return lambda a, b, A: [x if i % 2 == 0 else y for i, (x, y) in enumerate(zip(a, b))]


@BINARY.register()
def splice():
    """First floor(n/2) chars of a, the rest of b."""
    return lambda a, b, A: list(a[:len(a) // 2]) + list(b[len(a) // 2:])


@BINARY.register()
def maximum():
    """c_i = max(a_i, b_i)."""
    return lambda a, b, A: [max(x, y) for x, y in zip(a, b)]


@BINARY.register()
def minimum():
    """c_i = min(a_i, b_i) (backup operator)."""
    return lambda a, b, A: [min(x, y) for x, y in zip(a, b)]
