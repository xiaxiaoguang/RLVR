"""Config loading: registry defaults <- YAML (with `base:` inheritance) <- CLI overrides.

    cfg = load_config("configs/pretrain/weak.yaml", ["pretrain.h_ratio=0.5"])
    cfg = load_config()   # pure defaults

The result is validated against rlvr.params (unknown keys, types, choices).
Modules listed under `imports` are imported first so they can register new
atoms / rewards / weights before anything is built.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import yaml

from .params import PARAMS, PARAMS_DEFAULTS, merge, validate

ROOT = Path(__file__).resolve().parent.parent


def load_yaml(path: str | Path) -> dict:
    """Raw YAML; a top-level `base: <relative path>` is loaded first and merged under."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    if "base" in cfg:
        cfg = merge(load_yaml((path.parent / cfg.pop("base")).resolve()), cfg)
    return cfg


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """["pretrain.n_comp=2000", "task.str_len=[4,5,6]"]; values parsed as YAML."""
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must look like key=value: {ov!r}")
        key, val = ov.split("=", 1)
        d = cfg
        *parents, last = key.split(".")
        for p in parents:
            d = d.setdefault(p, {})
        d[last] = _number(key, yaml.safe_load(val))
    return cfg


def _number(key: str, value):
    """YAML 1.1 reads 3e-6 as a string; accept it for float parameters (rl.lr=3e-6)."""
    if isinstance(value, str) and key in PARAMS and isinstance(PARAMS.get(key).default, float):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def load_config(path: str | Path | None = None, overrides: list[str] | None = None,
                base: dict | None = None) -> dict:
    """base: values merged between the defaults and the YAML (e.g. sections inherited from a checkpoint)."""
    cfg = merge(PARAMS_DEFAULTS, base or {})
    if path:
        cfg = merge(cfg, load_yaml(resolve(path)))
    cfg = apply_overrides(cfg, overrides)
    for mod in cfg.get("imports") or []:
        importlib.import_module(mod)
    validate(cfg)
    return cfg


def task_and_splits(cfg: dict):
    from .splits import Splits
    from .task import make_task
    task = make_task(cfg["task"])
    if cfg["task"]["kind"] == "tree":
        from .tree_splits import TreeSplits
        task.validate_lengths(cfg)
        return task, TreeSplits(task, cfg["splits"])
    return task, Splits(task, cfg["splits"])


def git_commit() -> str:
    """HEAD sha (+ '-dirty' with uncommitted tracked changes), for run_config.yaml."""
    import subprocess
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                        cwd=ROOT, text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
