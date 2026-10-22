"""Small JSONL / problem-set helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .task import Problem, Task


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def problems_to_rows(task: Task, problems: Iterable[Problem], **extra) -> list[dict]:
    return [{**p.to_dict(), "prompt": task.format_prompt(p), **extra} for p in problems]


def load_problems(task: Task, path: str | Path) -> list[Problem]:
    return [task.from_dict(r) for r in read_jsonl(path)]


def load_frame(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.DataFrame(read_jsonl(path))
