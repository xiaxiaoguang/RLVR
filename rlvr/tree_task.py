"""Versioned string-expression trees, evaluated in left-to-right postorder.

Depth is dependency depth; n_ops is the number of executed operators. JSONL
stores the entire ordered tree, not merely its flattened postorder functions.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass

from .task import EOS, Task
from .tree_ops import build_operator
from .tree_sampling import count, sample_shape


@dataclass(frozen=True)
class Leaf:
    value: str

    def to_dict(self):
        return {"value": self.value}


@dataclass(frozen=True)
class Call:
    op: int
    args: tuple[Leaf | Call, ...]

    def to_dict(self):
        return {"op": f"F{self.op:02d}", "args": [a.to_dict() for a in self.args]}


@dataclass(frozen=True)
class TreeProblem:
    pid: str
    task_fingerprint: str
    expr: Call
    funcs: tuple[int, ...]
    x: str  # first leaf, retained for shared string-length APIs
    depth: int
    steps: tuple[str, ...]
    answer: str
    group: str
    trace: tuple[dict, ...]
    n_binary: int
    n_ternary: int
    n_leaves: int
    max_live_values: int
    shape_id: str

    @property
    def n_ops(self):
        return len(self.funcs)

    def to_dict(self):
        return {
            "schema_version": "tree-v1",
            "task_fingerprint": self.task_fingerprint,
            "pid": self.pid,
            "expr": self.expr.to_dict(),
            "funcs": list(self.funcs),
            "x": self.x,
            "depth": self.depth,
            "n_ops": self.n_ops,
            "n_binary": self.n_binary,
            "n_ternary": self.n_ternary,
            "n_leaves": self.n_leaves,
            "max_live_values": self.max_live_values,
            "shape_id": self.shape_id,
            "steps": list(self.steps),
            "trace": list(self.trace),
            "answer": self.answer,
            "group": self.group,
            "str_len": len(self.x),
        }


class TreeTask(Task):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.max_ops = cfg.get("tree_max_ops", 10)
        if not 1 <= self.max_depth <= self.max_ops <= 12:
            raise ValueError("tree task requires 1 <= max_depth <= tree_max_ops <= 12")
        self.sample_max_tries = cfg.get("sample_max_tries", 1000)
        if self.sample_max_tries < 1:
            raise ValueError("sample_max_tries must be positive")
        if len(self.alphabet) < 2 or not self.str_lens or min(self.str_lens) < 1:
            raise ValueError("tree task requires at least two symbols and positive string lengths")
        frozen = {k: self.cfg[k] for k in ("alphabet", "functions", "h_group", "name_seed", "cot")}
        frozen.update(
            schema="tree-v1",
            semantics="1",
            tokenizer=self.tokenizer().vocab,
            operator_versions=[op.version for op in self._fns],
            display_mapping=self.id2name,
            arities=self.arities,
        )
        self.fingerprint = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
        ab = re.escape(self.alphabet)
        self._step_re = re.compile(rf"F\d\d\([{ab},]*\)=([{ab}]+);")

    def _build_functions(self):
        ops = [build_operator(sp) for sp in self.specs]
        self.arities = tuple(op.arity for op in ops)
        return ops

    def _check_length_preserving(self):
        if not self.str_lens or min(self.str_lens) < 1 or len(self.alphabet) < 2:
            raise ValueError("tree task requires positive lengths and at least two symbols")
        rng = random.Random(0)
        for fid in range(self.n_funcs):
            for n in self.str_lens:
                for _ in range(20):
                    args = [self.sample_x(rng, n) for _ in range(self.arities[fid])]
                    assert len(self.apply(fid, *args)) == n, "operator is not length preserving"

    def tokenizer(self):
        from .tokenizer import CharTokenizer

        return CharTokenizer(self.alphabet, extra_chars=",")

    def apply(self, fid, *args):
        if not 0 <= fid < self.n_funcs:
            raise ValueError(f"invalid operator id: {fid}")
        values = [[self._c2i[c] for c in x] for x in args]
        out = self._fns[fid](values, self.A)
        if len(out) != len(args[0]) or any(not 0 <= c < self.A for c in out):
            raise ValueError("operator must preserve length and alphabet")
        return "".join(self.alphabet[c] for c in out)

    def _read_expr(self, obj, level=1, budget=None):
        budget = [self.max_ops] if budget is None else budget
        if not isinstance(obj, dict):
            raise ValueError("expression nodes must be objects")
        if set(obj) == {"value"}:
            value = obj["value"]
            if not isinstance(value, str) or not value or any(c not in self._c2i for c in value):
                raise ValueError("leaf must be a nonempty string in the task alphabet")
            if len(value) not in self.str_lens:
                raise ValueError("leaf length is outside task.str_len")
            return Leaf(value)
        if set(obj) != {"op", "args"} or not isinstance(obj["op"], str) or not re.fullmatch(r"F\d{2}", obj["op"]):
            raise ValueError("call must contain op=Fxx and args")
        fid = int(obj["op"][1:])
        if fid >= self.n_funcs or not isinstance(obj["args"], list) or len(obj["args"]) != self.arities[fid]:
            raise ValueError("invalid operator id or arity")
        budget[0] -= 1
        if budget[0] < 0 or level > self.max_depth:
            raise ValueError("expression exceeds tree_max_ops or max_depth")
        return Call(fid, tuple(self._read_expr(a, level + 1, budget) for a in obj["args"]))

    def _evaluate(self, expr, overrides=None):
        trace, leaves = [], []

        def visit(node, path):
            if isinstance(node, Leaf):
                leaves.append(node.value)
                return node.value, 0, 1, "L"
            child = [visit(a, (*path, i)) for i, a in enumerate(node.args)]
            args = [c[0] for c in child]
            value = self.apply(node.op, *args)
            if overrides and path in overrides:
                value = overrides[path]
            trace.append({"path": list(path), "op": self.fname(node.op), "args": args, "value": value})
            return (
                value,
                1 + max(c[1] for c in child),
                max(i + c[2] for i, c in enumerate(child)),
                f"{len(child)}(" + ",".join(c[3] for c in child) + ")",
            )

        answer, depth, live, shape = visit(expr, ())
        if len({len(x) for x in leaves}) != 1:
            raise ValueError("all leaves in a problem must have equal length")
        return answer, depth, live, shape, trace, leaves

    def make_problem(self, expr, x=None):
        if x is not None:
            raise ValueError("tree problems require an expression, not a flat funcs/x chain")
        # Validate even programmatically constructed nodes through the same boundary.
        expr = self._read_expr(expr.to_dict() if isinstance(expr, (Call, Leaf)) else expr)
        if not isinstance(expr, Call):
            raise ValueError("a problem must contain at least one operator")
        return self._make_from_expr(expr)

    def problem_id(self, expr):
        canonical = json.dumps(expr.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(f"{self.fingerprint}|{canonical}".encode()).hexdigest()[:16]

    def _make_from_expr(self, expr):
        answer, depth, live, shape, trace, leaves = self._evaluate(expr)
        funcs = tuple(int(t["op"][1:]) for t in trace)
        pid = self.problem_id(expr)
        return TreeProblem(
            pid,
            self.fingerprint,
            expr,
            funcs,
            leaves[0],
            depth,
            tuple(t["value"] for t in trace),
            answer,
            self.group_of(funcs),
            tuple(trace),
            sum(self.arities[f] == 2 for f in funcs),
            sum(self.arities[f] == 3 for f in funcs),
            len(leaves),
            live,
            shape,
        )

    def from_dict(self, row):
        if row.get("schema_version") != "tree-v1" or row.get("task_fingerprint") != self.fingerprint:
            raise ValueError("tree schema/task fingerprint mismatch; use the matching config and data")
        p = self.make_problem(row["expr"])
        for key, value in p.to_dict().items():
            if key in row and row[key] != value:
                raise ValueError(f"tree metadata mismatch: {key}")
        if "prompt" in row and row["prompt"] != self.format_prompt(p):
            raise ValueError("tree metadata mismatch: prompt")
        return p

    def sample_problem(self, depth, rng, group=None, func_pool=None, str_len=None, n_ops=None):
        expr = self.sample_expression(depth, rng, group, func_pool, str_len, n_ops)
        return self._make_from_expr(expr)

    def sample_expression(self, depth, rng, group=None, func_pool=None, str_len=None, n_ops=None):
        if group not in (None, "H", "L", "mixed", "anyL"):
            raise ValueError(f"unknown group: {group}")
        if not 1 <= depth <= self.max_depth:
            raise ValueError("depth outside task.max_depth")
        pool = tuple(range(self.n_funcs)) if func_pool is None else tuple(func_pool)
        if not pool or any(not 0 <= f < self.n_funcs for f in pool):
            raise ValueError("invalid or empty function pool")
        if group in ("H", "L"):
            pool = tuple(f for f in pool if f in (self.H if group == "H" else self.L))
        by_arity = {a: tuple(f for f in pool if self.arities[f] == a) for a in (1, 2, 3)}
        arities = tuple(a for a, fs in by_arity.items() if fs)
        sizes = [n_ops] if n_ops is not None else range(depth, min(self.max_ops, 2 * depth) + 1)
        sizes = [n for n in sizes if 1 <= n <= self.max_ops and count(n, depth, arities)]
        if group == "mixed":
            sizes = [n for n in sizes if n > 1]
        if not sizes or (group in ("mixed", "anyL") and not any(f in self.L for f in pool)):
            raise ValueError(f"infeasible tree request: depth={depth}, n_ops={n_ops}, group={group}")
        if group == "mixed" and not any(f in self.H for f in pool):
            raise ValueError("mixed group requires both H and L operators")
        n = str_len if str_len is not None else rng.choice(self.str_lens)
        if n not in self.str_lens:
            raise ValueError("str_len outside task.str_len")
        for _ in range(self.sample_max_tries):
            shape = sample_shape(rng.choice(sizes), depth, arities, rng)
            ops = []

            def fill(s):
                if not s:
                    return Leaf(self.sample_x(rng, n))
                fid = rng.choice(by_arity[len(s)])
                ops.append(fid)
                return Call(fid, tuple(fill(c) for c in s))

            expr = fill(shape)
            actual_group = self.group_of(ops)
            if group == "mixed" and actual_group != "mixed":
                continue
            if group == "anyL" and actual_group == "H":
                continue
            return expr
        raise RuntimeError("could not satisfy tree group within sample_max_tries")

    def format_prompt(self, p):
        def fmt(node):
            if isinstance(node, Leaf):
                return node.value
            return f"{self.fname(node.op)}(" + ",".join(fmt(a) for a in node.args) + ")"

        return f"Q:{fmt(p.expr)}=\n"

    def format_target(self, p, steps=None, answer=None):
        steps = p.steps if steps is None else steps
        if len(steps) != p.n_ops:
            raise ValueError("one target result required per operator node")
        # Reconstruct the inputs from supplied node results (including noise).
        results, body = {}, ""
        for t, value in zip(p.trace, steps):
            path = tuple(t["path"])
            args = [results.get((*path, i), x) for i, x in enumerate(t["args"])]
            body += f"{t['op']}({','.join(args)})={value};"
            results[path] = value
        body = body + "\n" if self.cot else ""
        return f"{body}A:{steps[-1] if answer is None else answer}{EOS}"

    def corrupt_steps(self, p, rng):
        index = rng.randrange(p.n_ops)
        bad = p.steps[index]
        while bad == p.steps[index]:
            bad = self.sample_x(rng, len(p.x))
        *_, trace, _ = self._evaluate(p.expr, {tuple(p.trace[index]["path"]): bad})
        return [t["value"] for t in trace]

    def is_skippable(self, p):
        """True if replacing any one operator node by one of its operands (for a unary node:
        removing it) still gives the right answer (design.md decision 6)."""
        for t in p.trace:
            path = tuple(t["path"])
            for arg in t["args"]:
                if self._evaluate(p.expr, {path: arg})[0] == p.answer:
                    return True
        return False

    def edges(self, p):
        """(parent operator, argument slot, child operator), never sibling pairs."""
        return self.expression_edges(p.expr)

    def expression_edges(self, expr):
        edges = []

        def walk(node):
            if isinstance(node, Call):
                for slot, child in enumerate(node.args):
                    if isinstance(child, Call):
                        edges.append((node.op, slot, child.op))
                        walk(child)

        walk(expr)
        return edges

    def describe(self):
        return [{**row, "arity": self.arities[row["id"]], "semantic_version": "1"} for row in super().describe()]

    def length_bounds(self):
        """Conservative bounds covering every allowed tree, including padding/offset."""
        n, length, arity = self.max_ops, max(self.str_lens), max(self.arities)
        leaves = 1 + (arity - 1) * n
        prompt = 4 + leaves * length + n * (5 + arity - 1)
        target = (n * (7 + (arity + 1) * length + arity - 1) + 1 if self.cot else 0) + length + 3
        return prompt + target + 1, target

    def validate_lengths(self, cfg):
        total, target = self.length_bounds()
        if (
            (total + 7) // 8 * 8 + cfg["pretrain"]["pos_offset_max"] > cfg["model"]["max_len"]
            or target > cfg["eval"]["max_new_tokens"]
            or target > cfg["rl"]["max_new_tokens"]
        ):
            raise ValueError(
                f"tree length bounds require >= {total} sequence and {target} target tokens "
                "plus padding/position offset; increase model/eval/rl length budgets"
            )
