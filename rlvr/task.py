"""String function-composition task: problem generation, formatting, parsing, reward.

A chain applies `depth` length-preserving atomic functions to an input string,
innermost first. Function display names are F00-F23 (one per entry of
task.functions); the name -> function
mapping is shuffled with a fixed seed so the model cannot infer behaviour from
the name.

    Q:F03(F11(F07(abcdefgh)))=
    F07(abcdefgh)=bcdefghi;F11(bcdefghi)=ihgfedcb;F03(ihgfedcb)=jihgfedc;
    A:jihgfedc<eos>

v2 adds binary operators (task.binary_functions, display ids from F24 on, not
shuffled, so the unary ids and every chain's pid stay as in v1). A tree problem has
one random leaf string per input (all the same length); its operations are stored
in post-order with operand references (`args`: >= 0 = result of that step, < 0 =
leaf -r-1), and the CoT writes the steps in that order, operands separated by `,`:

    Q:F25(F03(abcdefgh),F07(ijklmnop))=
    F03(abcdefgh)=bcdefgha;F07(ijklmnop)=jklmnopq;F25(bcdefgha,jklmnopq)=kmoqsuwq;
    A:kmoqsuwq<eos>

`depth` is the number of operations (n_ops) for every problem; a chain is a tree
with n_bin = 0 and keeps the v1 representation (args = leaves = ()). The group
(H / L / mixed) is decided by the unary functions only; a tree without unary
functions has group "bin".
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Sequence

EOS = "<eos>"

GROUPS = ("H", "L", "mixed", "bin")


@dataclass(frozen=True)
class Problem:
    pid: str                 # hash of the expression and inputs, globally unique
    funcs: tuple[int, ...]   # display ids (Fxx), one per operation in post-order (chain: innermost first)
    x: str                   # input string (tree: the first leaf)
    depth: int               # number of operations n_ops (= len(funcs))
    steps: tuple[str, ...] = field(compare=False)  # output after each step
    answer: str = field(compare=False)
    group: str = field(compare=False)              # H / L / mixed (/ bin: no unary function)
    leaves: tuple[str, ...] = field(default=(), compare=False)            # tree inputs, left to right
    args: tuple[tuple[int, ...], ...] = field(default=(), compare=False)  # tree operand refs per step

    # ---- structure variables (design.md: n_ops, tree_depth, n_bin, dep_dist) ----

    @property
    def is_tree(self) -> bool:
        return bool(self.args)

    @property
    def n_ops(self) -> int:
        return self.depth

    @property
    def inputs(self) -> tuple[str, ...]:
        return self.leaves if self.args else (self.x,)

    def operands(self) -> tuple[tuple[int, ...], ...]:
        """Operand refs of every step (chain: each step reads the previous one)."""
        return self.args if self.args else tuple((i - 1,) for i in range(self.depth))

    @property
    def n_bin(self) -> int:
        return sum(len(a) == 2 for a in self.args)

    @property
    def tree_depth(self) -> int:
        """Operations on the longest leaf-to-root path."""
        h: list[int] = []
        for a in self.operands():
            h.append(1 + max(h[r] if r >= 0 else 0 for r in a))
        return h[-1]

    @property
    def dep_dist(self) -> int:
        """Largest number of steps between a step and the step that produced its left
        operand (leaf operands not counted): 1 for chains of depth >= 2."""
        return max([i - a[0] for i, a in enumerate(self.operands()) if a[0] >= 0], default=0)

    def to_dict(self) -> dict:
        d = {
            "pid": self.pid, "funcs": list(self.funcs), "x": self.x,
            "depth": self.depth, "steps": list(self.steps),
            "answer": self.answer, "group": self.group, "str_len": len(self.x),
        }
        if self.args:
            d.update(leaves=list(self.leaves), args=[list(a) for a in self.args], n_bin=self.n_bin,
                     tree_depth=self.tree_depth, dep_dist=self.dep_dist)
        return d


def make_pid(funcs: Sequence[int], x: str) -> str:
    key = ".".join(f"{f:02d}" for f in funcs) + "|" + x
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def make_tree_pid(expr: str) -> str:
    """Trees hash their full prompt expression (it contains every leaf); the prefix keeps
    the key space apart from chain pids."""
    return hashlib.sha1(("T|" + expr).encode()).hexdigest()[:16]


class Task:
    """Holds the frozen task definition (alphabet, function table, groups)."""

    def __init__(self, cfg: dict | None = None):
        """cfg: the `task` section of a resolved config (defaults: rlvr.params)."""
        from .atoms import ATOMS, BINARY
        from .params import PARAMS_DEFAULTS
        cfg = {**PARAMS_DEFAULTS["task"], **(cfg or {})}
        self.cfg = cfg
        self.alphabet: str = cfg["alphabet"]
        self.A = len(self.alphabet)
        self._c2i = {c: i for i, c in enumerate(self.alphabet)}
        sl = cfg["str_len"]
        self.str_lens: list[int] = [sl] if isinstance(sl, int) else list(sl)
        self.cot: bool = cfg["cot"]
        self.max_depth: int = cfg["max_depth"]

        # label -> registry spec; labels are what h_group and reports refer to
        specs: dict = cfg["functions"]
        bspecs: dict = cfg["binary_functions"] or {}
        labels = list(specs)
        assert len(labels) + len(bspecs) <= 100, "display names are two digits"
        assert not set(labels) & set(bspecs), "unary and binary labels must differ"
        perm = list(range(len(labels)))
        random.Random(cfg["name_seed"]).shuffle(perm)
        # display id k -> label
        self.id2name: list[str] = [labels[perm[k]] for k in range(len(labels))]
        self.name2id = {n: k for k, n in enumerate(self.id2name)}
        self.specs = [specs[n] for n in self.id2name]
        self._fns = [ATOMS.build(sp) for sp in self.specs]
        self.n_funcs = len(labels)           # unary functions: display ids 0 .. n_funcs-1
        # binary operators: display ids n_funcs .. in config order, outside the shuffle
        self.n_binary = len(bspecs)
        self.B: tuple[int, ...] = tuple(range(self.n_funcs, self.n_funcs + self.n_binary))
        self.id2name += list(bspecs)
        self.name2id = {n: k for k, n in enumerate(self.id2name)}
        self.specs += list(bspecs.values())
        self._bfns = [BINARY.build(sp) for sp in bspecs.values()]
        unknown = set(cfg["h_group"]) - set(labels)
        assert not unknown, f"h_group labels not in task.functions: {unknown}"
        self.H: tuple[int, ...] = tuple(sorted(self.name2id[n] for n in cfg["h_group"]))
        self.L: tuple[int, ...] = tuple(k for k in range(self.n_funcs) if k not in self.H)
        self._check_length_preserving()
        self._is_h = [k in self.H for k in range(self.n_funcs)]

        ab = re.escape(self.alphabet)
        self._ans_re = re.compile(rf"(?:^|\n)A:([{ab}]+)$")
        self._step_re = re.compile(rf"F\d\d\([{ab}]*(?:,[{ab}]*)*\)=([{ab}]*);")
        self._reward_cache: dict = {}

    def _check_length_preserving(self):
        rng = random.Random(0)
        for k, fn in enumerate(self._fns):
            for n in self.str_lens:
                for _ in range(200):
                    v = [rng.randrange(self.A) for _ in range(n)]
                    out = fn(list(v), self.A)
                    assert len(out) == n and all(0 <= c < self.A for c in out), \
                        f"{self.id2name[k]} {self.specs[k]} is not length preserving / in range"
        for j, fn in enumerate(self._bfns):
            name = self.id2name[self.n_funcs + j]
            for n in self.str_lens:
                uses = [False, False]
                for _ in range(200):
                    a, b, c = ([rng.randrange(self.A) for _ in range(n)] for _ in range(3))
                    out = fn(list(a), list(b), self.A)
                    assert len(out) == n and all(0 <= v < self.A for v in out), \
                        f"{name} is not length preserving / in range"
                    uses[0] |= fn(list(c), list(b), self.A) != out
                    uses[1] |= fn(list(a), list(c), self.A) != out
                assert all(uses), f"binary operator {name} must depend on both inputs (length {n})"

    # ---- semantics -------------------------------------------------------

    def is_binary(self, fid: int) -> bool:
        return fid >= self.n_funcs

    def apply(self, fid: int, s: str, t: str | None = None) -> str:
        """Unary f(s), or binary f(s, t)."""
        v = [self._c2i[c] for c in s]
        if fid >= self.n_funcs:
            assert t is not None and len(t) == len(s), "binary operator needs two strings of equal length"
            out = self._bfns[fid - self.n_funcs](v, [self._c2i[c] for c in t], self.A)
        else:
            out = self._fns[fid](v, self.A)
        return "".join(self.alphabet[c] for c in out)

    def run(self, funcs: Sequence[int], x: str) -> list[str]:
        steps, s = [], x
        for f in funcs:
            s = self.apply(f, s)
            steps.append(s)
        return steps

    def group_of(self, funcs: Sequence[int]) -> str:
        """H / L / mixed from the unary functions; "bin" if there are none."""
        h = [self._is_h[f] for f in funcs if f < self.n_funcs]
        if not h:
            return "bin"
        return "H" if all(h) else ("L" if not any(h) else "mixed")

    def make_problem(self, funcs: Sequence[int], x: str) -> Problem:
        funcs = tuple(int(f) for f in funcs)
        steps = self.run(funcs, x)
        return Problem(make_pid(funcs, x), funcs, x, len(funcs), tuple(steps),
                       steps[-1], self.group_of(funcs))

    def evaluate(self, funcs: Sequence[int], args: Sequence[Sequence[int]], leaves: Sequence[str],
                 bypass: dict[int, int] | None = None) -> list[str]:
        """Results of every step of a tree. bypass {step: operand ref}: that step outputs
        the given operand instead of applying its function (skippable check)."""
        v: list[str] = []
        for i, (f, a) in enumerate(zip(funcs, args)):
            xs = [leaves[-r - 1] if r < 0 else v[r] for r in a]
            if bypass and i in bypass:
                r = bypass[i]
                v.append(leaves[-r - 1] if r < 0 else v[r])
            else:
                v.append(self.apply(f, *xs))
        return v

    def make_tree(self, funcs: Sequence[int], args: Sequence[Sequence[int]], leaves: Sequence[str]) -> Problem:
        """A problem from post-order operations; one with no binary operator is built as a chain."""
        funcs = tuple(int(f) for f in funcs)
        args = tuple(tuple(int(r) for r in a) for a in args)
        leaves = tuple(leaves)
        n_leaf, used = len(leaves), []
        for i, (f, a) in enumerate(zip(funcs, args)):
            assert len(a) == (2 if self.is_binary(f) else 1), f"step {i}: wrong number of operands"
            assert all(-n_leaf <= r < i for r in a), f"step {i}: operand refs {a} out of range"
            used += a
        assert len(funcs) == len(args) >= 1
        assert sorted(used) == sorted([*range(-n_leaf, 0), *range(len(funcs) - 1)]), \
            "every leaf and every step but the last must be used exactly once"
        assert len({len(x) for x in leaves}) == 1, "all leaves must have the same length"
        if n_leaf == 1 and not any(self.is_binary(f) for f in funcs):
            return self.make_problem(funcs, leaves[0])
        steps = self.evaluate(funcs, args, leaves)
        return Problem(make_tree_pid(self._expr(funcs, args, leaves)), funcs, leaves[0], len(funcs),
                       tuple(steps), steps[-1], self.group_of(funcs), leaves, args)

    def from_dict(self, d: dict) -> Problem:
        if d.get("args"):
            return self.make_tree(d["funcs"], d["args"], d["leaves"])
        return self.make_problem(d["funcs"], d["x"])

    def is_skippable(self, p: Problem) -> bool:
        """True if removing any one unary step, or replacing any binary step by one of its
        operands, still gives the right answer (design.md decision 6)."""
        ops, inputs = p.operands(), p.inputs
        for i, a in enumerate(ops):
            for r in a:
                if self.evaluate(p.funcs, ops, inputs, bypass={i: r})[-1] == p.answer:
                    return True
        return False

    # ---- sampling --------------------------------------------------------

    def sample_funcs(self, depth: int, rng: random.Random, group: str | None = None,
                     func_pool: Sequence[int] | None = None) -> tuple[int, ...]:
        """group=None: uniform over func_pool (default all). 'H'/'L': only that
        group. 'mixed': at least one H and one L (rejection sampled).
        'anyL': at least one L function (i.e. L or mixed)."""
        if group == "H":
            pool = self.H
        elif group == "L":
            pool = self.L
        else:
            pool = tuple(func_pool) if func_pool is not None else tuple(range(self.n_funcs))
        if group == "mixed":
            assert depth >= 2, "mixed group needs depth >= 2"
        while True:
            funcs = tuple(rng.choice(pool) for _ in range(depth))
            if group == "mixed" and self.group_of(funcs) != "mixed":
                continue
            if group == "anyL" and self.group_of(funcs) == "H":
                continue
            return funcs

    def sample_x(self, rng: random.Random, str_len: int | None = None) -> str:
        n = str_len if str_len is not None else rng.choice(self.str_lens)
        return "".join(rng.choice(self.alphabet) for _ in range(n))

    def sample_problem(self, depth: int, rng: random.Random, group: str | None = None,
                       func_pool: Sequence[int] | None = None,
                       str_len: int | None = None) -> Problem:
        return self.make_problem(self.sample_funcs(depth, rng, group, func_pool),
                                 self.sample_x(rng, str_len))

    def valid_cell(self, n_ops: int, n_bin: int, group: str | None) -> bool:
        """Whether a (n_ops, n_bin, group) cell can hold problems."""
        n_un = n_ops - n_bin
        if n_ops < 1 or not 0 <= n_bin <= n_ops or (n_bin and not self.n_binary):
            return False
        if n_un == 0:
            return group in (None, "bin")
        if group == "bin":
            return False
        return group != "mixed" or n_un >= 2

    def _skeleton(self, k: int, rng: random.Random):
        """Random full binary tree with k binary nodes: None = leaf, (left, right) = node."""
        if k == 0:
            return None
        i = rng.randrange(k)
        return (self._skeleton(i, rng), self._skeleton(k - 1 - i, rng))

    def sample_tree(self, n_ops: int, n_bin: int, rng: random.Random, group: str | None = None,
                    str_len: int | None = None, bin_pool: Sequence[int] | None = None) -> Problem:
        """n_ops operations, n_bin of them binary. Shape: a random binary skeleton with n_bin
        binary nodes; each of the n_ops - n_bin unary functions then sits on a uniformly
        random edge (above the root, a binary node or a leaf). Unary functions follow `group`
        as in sample_funcs; binary operators are uniform over bin_pool (default all).
        n_bin = 0 gives a v1 chain (sample_problem)."""
        assert self.valid_cell(n_ops, n_bin, group if group != "anyL" else "H"), (n_ops, n_bin, group)
        if n_bin == 0:
            return self.sample_problem(n_ops, rng, group, str_len=str_len)
        n_un = n_ops - n_bin
        skel = self._skeleton(n_bin, rng)
        n_slots = 2 * n_bin + 1
        counts = [0] * n_slots
        for _ in range(n_un):
            counts[rng.randrange(n_slots)] += 1
        unary = iter(self.sample_funcs(n_un, rng, group) if n_un else ())
        pool = tuple(bin_pool) if bin_pool is not None else self.B
        binary = iter([rng.choice(pool) for _ in range(n_bin)])
        n = str_len if str_len is not None else rng.choice(self.str_lens)
        leaves = [self.sample_x(rng, n) for _ in range(n_bin + 1)]
        funcs: list[int] = []
        args: list[tuple[int, ...]] = []
        slots, leaf_ids = iter(counts), iter(range(n_bin + 1))

        def emit(node) -> int:
            if node is None:
                ref = -next(leaf_ids) - 1
            else:
                left, right = emit(node[0]), emit(node[1])
                funcs.append(next(binary))
                args.append((left, right))
                ref = len(funcs) - 1
            for _ in range(next(slots)):
                funcs.append(next(unary))
                args.append((ref,))
                ref = len(funcs) - 1
            return ref

        emit(skel)
        return self.make_tree(funcs, args, leaves)

    # ---- formatting ------------------------------------------------------

    def fname(self, fid: int) -> str:
        return f"F{fid:02d}"

    def _expr(self, funcs: Sequence[int], args: Sequence[Sequence[int]], leaves: Sequence[str]) -> str:
        e: list[str] = []
        for f, a in zip(funcs, args):
            e.append(f"{self.fname(f)}(" + ",".join(leaves[-r - 1] if r < 0 else e[r] for r in a) + ")")
        return e[-1]

    def format_prompt(self, p: Problem) -> str:
        if p.args:
            return f"Q:{self._expr(p.funcs, p.args, p.leaves)}=\n"
        expr = p.x
        for f in p.funcs:
            expr = f"{self.fname(f)}({expr})"
        return f"Q:{expr}=\n"

    def format_target(self, p: Problem, steps: Sequence[str] | None = None,
                      answer: str | None = None) -> str:
        """Target completion. `steps`/`answer` overrides exist for noise injection."""
        steps = p.steps if steps is None else steps
        answer = p.answer if answer is None else answer
        body = ""
        if self.cot and p.args:
            for f, a, s in zip(p.funcs, p.args, steps):
                ops = ",".join(p.leaves[-r - 1] if r < 0 else steps[r] for r in a)
                body += f"{self.fname(f)}({ops})={s};"
            body += "\n"
        elif self.cot:
            prev = p.x
            for f, s in zip(p.funcs, steps):
                body += f"{self.fname(f)}({prev})={s};"
                prev = s
            body += "\n"
        return f"{body}A:{answer}{EOS}"

    # ---- parsing / reward ------------------------------------------------

    def parse_answer(self, text: str) -> str | None:
        """Final answer from a completion, or None if malformed.

        The completion must contain <eos>; everything after it is ignored. The
        text before <eos> must end with a line `A:<answer>` where answer uses
        only alphabet chars, and `A:` must appear exactly once."""
        if EOS not in text:
            return None
        body = text.split(EOS, 1)[0]
        if body.count("A:") != 1:
            return None
        m = self._ans_re.search(body)
        return m.group(1) if m else None

    def parse_steps(self, text: str) -> list[str]:
        """Intermediate results `Fxx(..)=<s>;` written in a completion, in order."""
        body = text.split(EOS, 1)[0].split("A:", 1)[0]
        return self._step_re.findall(body)

    def reward(self, text: str, p: Problem, kind: str | dict = "binary") -> float:
        """kind: a spec for rlvr.rewards.REWARDS, e.g. 'binary' or 'partial'."""
        key = kind if isinstance(kind, str) else repr(sorted(kind.items()))
        fn = self._reward_cache.get(key)
        if fn is None:
            from .rewards import REWARDS
            fn = self._reward_cache[key] = REWARDS.build(kind)
        return fn(self, text, p)

    def describe(self) -> list[dict]:
        return [{"id": k, "name": self.fname(k), "function": self.id2name[k], "spec": self.specs[k],
                 "group": "bin" if k >= self.n_funcs else ("H" if self._is_h[k] else "L")}
                for k in range(self.n_funcs + self.n_binary)]
