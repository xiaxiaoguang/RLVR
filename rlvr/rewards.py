"""Reward functions. Each factory returns fn(task, text, problem) -> float."""
from __future__ import annotations

from .registry import Registry
from .task import EOS

REWARDS = Registry("reward")


@REWARDS.register()
def binary():
    """1.0 iff the final answer is exactly correct; no format credit."""
    def fn(task, text, p):
        ans = task.parse_answer(text)
        return float(ans is not None and ans == p.answer)
    return fn


@REWARDS.register()
def partial():
    """Fraction of correct steps (last step = final answer); 0 without <eos>. R12 ablation."""
    def fn(task, text, p):
        if EOS not in text:
            return 0.0
        got = task.parse_steps(text)
        ans = task.parse_answer(text)
        if ans is not None:
            got = got[:p.depth - 1] + [ans] if p.depth > 1 else [ans]
        return sum(g == s for g, s in zip(got, p.steps)) / p.depth
    return fn
