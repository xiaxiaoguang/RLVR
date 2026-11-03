"""String function-composition task: problem generation, formatting, parsing, reward.

A problem applies `depth` length-preserving atomic functions to an input string,
innermost first. Function display names are F00-F23 (one per entry of
task.functions); the name -> function
mapping is shuffled with a fixed seed so the model cannot infer behaviour from
the name.

    Q:F03(F11(F07(abcdefgh)))=
    F07(abcdefgh)=bcdefghi;F11(bcdefghi)=ihgfedcb;F03(ihgfedcb)=jihgfedc;
    A:jihgfedc<eos>
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Sequence

EOS = "<eos>"

GROUPS = ("H", "L", "mixed")


@dataclass(frozen=True)
class Problem:
    pid: str                 # hash of funcs + x, globally unique
    funcs: tuple[int, ...]   # display ids (Fxx), innermost first
    x: str                   # input string
    depth: int               # = len(funcs)
    steps: tuple[str, ...] = field(compare=False)  # output after each step
    answer: str = field(compare=False)
    group: str = field(compare=False)              # H / L / mixed

    def to_dict(self) -> dict:
        return {
            "pid": self.pid, "funcs": list(self.funcs), "x": self.x,
            "depth": self.depth, "steps": list(self.steps),
            "answer": self.answer, "group": self.group, "str_len": len(self.x),
        }


def make_pid(funcs: Sequence[int], x: str) -> str:
    key = ".".join(f"{f:02d}" for f in funcs) + "|" + x
    return hashlib.sha1(key.encode()).hexdigest()[:16]


class Task:
    """Holds the frozen task definition (alphabet, function table, groups)."""

    def __init__(self, cfg: dict | None = None):
        """cfg: the `task` section of a resolved config (defaults: rlvr.params)."""
        from .atoms import ATOMS
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
        labels = list(specs)
        assert len(labels) <= 100, "display names are two digits"
        perm = list(range(len(labels)))
        random.Random(cfg["name_seed"]).shuffle(perm)
        # display id k -> label
        self.id2name: list[str] = [labels[perm[k]] for k in range(len(labels))]
        self.name2id = {n: k for k, n in enumerate(self.id2name)}
        self.specs = [specs[n] for n in self.id2name]
        self._fns = [ATOMS.build(sp) for sp in self.specs]
        self.n_funcs = len(labels)
        unknown = set(cfg["h_group"]) - set(labels)
        assert not unknown, f"h_group labels not in task.functions: {unknown}"
        self.H: tuple[int, ...] = tuple(sorted(self.name2id[n] for n in cfg["h_group"]))
        self.L: tuple[int, ...] = tuple(k for k in range(self.n_funcs) if k not in self.H)
        self._check_length_preserving()
        self._is_h = [k in self.H for k in range(self.n_funcs)]

        ab = re.escape(self.alphabet)
        self._ans_re = re.compile(rf"(?:^|\n)A:([{ab}]+)$")
        self._step_re = re.compile(rf"F\d\d\([{ab}]*\)=([{ab}]*);")
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

    # ---- semantics -------------------------------------------------------

    def apply(self, fid: int, s: str) -> str:
        v = [self._c2i[c] for c in s]
        out = self._fns[fid](v, self.A)
        return "".join(self.alphabet[c] for c in out)

    def run(self, funcs: Sequence[int], x: str) -> list[str]:
        steps, s = [], x
        for f in funcs:
            s = self.apply(f, s)
            steps.append(s)
        return steps

    def group_of(self, funcs: Sequence[int]) -> str:
        h = [self._is_h[f] for f in funcs]
        return "H" if all(h) else ("L" if not any(h) else "mixed")

    def make_problem(self, funcs: Sequence[int], x: str) -> Problem:
        funcs = tuple(int(f) for f in funcs)
        steps = self.run(funcs, x)
        return Problem(make_pid(funcs, x), funcs, x, len(funcs), tuple(steps),
                       steps[-1], self.group_of(funcs))

    def from_dict(self, d: dict) -> Problem:
        return self.make_problem(d["funcs"], d["x"])

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

    # ---- formatting ------------------------------------------------------

    def fname(self, fid: int) -> str:
        return f"F{fid:02d}"

    def format_prompt(self, p: Problem) -> str:
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
        if self.cot:
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
        return [{"id": k, "name": self.fname(k), "function": self.id2name[k],
                 "spec": self.specs[k], "group": "H" if self._is_h[k] else "L"}
                for k in range(self.n_funcs)]
