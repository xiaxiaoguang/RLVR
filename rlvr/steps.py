"""Step checker for chain and tree completions: the first error and the step score.

    err_type, err_step, n_correct = first_error(task, p, text)

Expected lines, in step order: a TreeProblem's post-order trace; for a chain Problem, step j is
p.funcs[j] applied to the previous step's value (p.x for the first). Written lines
`Fxx(arg[,arg...])=value;` are read from the text before <eos> and the first `A:`. A line is
correct only if its operator id, argument list and result all equal the expected line, so
n_correct (the step score) is the number of leading correct lines, 0..n_ops.

err_type is the first problem in step order (names as in scripts/diag_steps.py): wrong_func
(operator id differs), copy (arguments differ), apply (result differs), missing_step (fewer lines,
all correct), extra_step (more lines, the first n_ops correct), format (no <eos> or no parsable
answer), answer_copy (answer differs from the last step's true value), else ok. err_step is the
0-based step of the error (n_ops for extra_step / format / answer_copy, None for ok). A completion
without <eos> still gets its leading correct lines counted. ok implies reward 1.
"""
from __future__ import annotations

import re

from .task import EOS


def expected_steps(p) -> list[tuple[int, tuple[str, ...], str]]:
    """(operator id, arguments, result) per step, in the order the target writes them."""
    if hasattr(p, "trace"):
        return [(int(t["op"][1:]), tuple(t["args"]), t["value"]) for t in p.trace]
    return [(f, (a,), s) for f, a, s in zip(p.funcs, (p.x, *p.steps[:-1]), p.steps)]


def first_error(task, p, text: str) -> tuple[str, int | None, int]:
    want = expected_steps(p)
    n = len(want)
    ab = re.escape(task.alphabet)
    body = text.split(EOS, 1)[0].split("A:", 1)[0]
    got = re.findall(rf"F(\d\d)\(([{ab},]*)\)=([{ab}]*);", body)
    for j, (fid, args, value) in enumerate(want):
        if j >= len(got):
            return "missing_step", j, j
        g_fid, g_args, g_value = got[j]
        if int(g_fid) != fid:
            return "wrong_func", j, j
        if tuple(g_args.split(",")) != args:
            return "copy", j, j
        if g_value != value:
            return "apply", j, j
    if len(got) > n:
        return "extra_step", n, n
    ans = task.parse_answer(text)
    if ans is None:
        return "format", n, n
    if ans != want[-1][2]:
        return "answer_copy", n, n
    return "ok", None, n
