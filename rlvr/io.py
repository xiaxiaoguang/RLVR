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


def combination_keys(frame: pd.DataFrame) -> pd.Series:
    """Operator composition identity, including tree topology but excluding leaf values."""
    if "shape_id" in frame:
        return pd.Series(list(zip(frame.shape_id, frame.funcs.map(tuple))), index=frame.index)
    return frame.funcs.map(tuple)


def check_task_directory(task: Task, data: str | Path) -> None:
    """Refuse to mix a different task with metadata already in a data root."""
    import yaml
    from .task import make_task
    path = Path(data) / "meta" / "task_config.yaml"
    if not path.exists():
        return
    fingerprint_file = path.parent / "task_fingerprint.json"
    if fingerprint_file.exists():
        expected = json.loads(fingerprint_file.read_text())["task_fingerprint"]
        if expected != getattr(task, "fingerprint", None):
            raise ValueError("data directory task fingerprint changed; select a new paths.data namespace")
    old = make_task(yaml.safe_load(path.read_text())["task"])
    if (getattr(old, "fingerprint", "chain-v1") != getattr(task, "fingerprint", "chain-v1")
            or old.describe() != task.describe() or old.tokenizer().vocab != task.tokenizer().vocab):
        raise ValueError("data directory contains a different task; select a new paths.data namespace")
