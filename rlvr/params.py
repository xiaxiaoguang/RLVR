"""Parameter registry: every config key with its default, levels and priority.

This is the single source of defaults. YAML configs only override; `load_config`
merges them onto these defaults and validates the result (unknown keys, types,
choices). Doc ids (P1..P11, D1..D10, R1..R13) follow the design document.

    python -m rlvr.params            # table of all parameters
    python -m rlvr.params --core     # only the core (★) variables
    python -m rlvr.params --yaml     # full default config as YAML

A dict-valued parameter (e.g. task.functions) is one value: an override
replaces it as a whole instead of merging into it.
"""
from __future__ import annotations

import argparse
import copy
import difflib
from dataclasses import dataclass
from typing import Any

from .registry import Registry

CORE, OPTIONAL, FROZEN, REPEAT, INFRA = "core", "optional", "frozen", "repeat", "infra"


@dataclass(frozen=True)
class Param:
    key: str
    default: Any
    doc: str = ""
    code: str = ""                 # design-doc variable id, e.g. "P4"
    priority: str = INFRA          # core / optional / frozen / repeat / infra
    levels: tuple | None = None    # values planned in the experiments
    choices: Any = None            # tuple of values, or a Registry
    types: tuple | None = None     # accepted types; inferred from default if None

    def check(self, value) -> str | None:
        if value is None:
            return None if self.default is None else f"{self.key}: None not allowed"
        types = self.types or _infer_types(self.default)
        if types and (not isinstance(value, types) or (isinstance(value, bool) and bool not in types)):
            return f"{self.key}: expected {'/'.join(t.__name__ for t in types)}, got {type(value).__name__} {value!r}"
        choices = self.choices() if callable(self.choices) else self.choices  # lazy registry lookup
        if choices is not None:
            names = choices.names() if isinstance(choices, Registry) else choices
            v = value.get("name") if isinstance(value, dict) else value
            if v not in names:
                return f"{self.key}: {value!r} not in {list(names)}"
        return None


def _infer_types(default) -> tuple | None:
    if default is None:
        return None
    if isinstance(default, bool):
        return (bool,)
    if isinstance(default, int):
        return (int,)
    if isinstance(default, float):
        return (float, int)
    return (type(default),)


PARAMS = Registry("param")


def param(key: str, default: Any, doc: str = "", **kw) -> Param:
    return PARAMS.add(key, Param(key, default, doc, **kw))


# =============================================================================
# Global
# =============================================================================
param("seed", 0, "global seed (pretraining corpus, RL run)", code="P11/R13", priority=REPEAT, levels=(0, 1, 2))
param("imports", [], "modules imported before validation; use them to register new atoms/rewards/...")
param("paths.data", "data", "root of generated data")
param("paths.runs", "runs", "root of run logs and checkpoints")

# =============================================================================
# Task (frozen after week 1: changing these changes pids, pools and names)
# =============================================================================
param("task.alphabet", "abcdefghij", "input alphabet", code="P8", priority=FROZEN)
param("task.str_len", 5, "input length, or list of lengths sampled uniformly", code="P8",
      priority=FROZEN, types=(int, list))
param("task.name_seed", 12345, "seed of the label -> Fxx display-name shuffle", priority=FROZEN)
param("task.cot", True, "write intermediate steps in the target", code="P7", priority=FROZEN)
param("task.max_depth", 6, "largest depth used anywhere", priority=FROZEN)
param("task.functions", {
    "reverse": "reverse",
    "rotl1": {"name": "rotate", "k": 1},
    "rotr1": {"name": "rotate", "k": -1},
    "rotl2": {"name": "rotate", "k": 2},
    "swap_adjacent": "swap_adjacent",
    "swap_ends": "swap_ends",
    "swap_halves": "swap_halves",
    "shift_p1": {"name": "shift", "k": 1},
    "shift_m1": {"name": "shift", "k": -1},
    "shift_p3": {"name": "shift", "k": 3},
    "complement": "complement",
    "sort_asc": "sort",
    "sort_desc": {"name": "sort", "descending": True},
    "even_pos_p1": {"name": "shift_at", "k": 1, "parity": 0},
    "max_to_front": "max_to_front",
    "mirror_first_half": "mirror_first_half",
}, "label -> atom spec (rlvr.atoms.ATOMS); label order fixes the Fxx shuffle", priority=FROZEN)
param("task.h_group", ["reverse", "rotl1", "swap_adjacent", "swap_halves",
                       "shift_p1", "complement", "sort_asc", "even_pos_p1"],
      "labels of the high-exposure group H; the rest form L", code="P6", priority=FROZEN)

# =============================================================================
# Splits
# =============================================================================
param("splits.salt", "rlvr-split-v1", "salt of the pid -> bucket hash", priority=FROZEN)
param("splits.pools", {"pretrain": [0, 900], "rl": [900, 950], "eval": [950, 1000]},
      "bucket ranges (of 1000) per pool", priority=FROZEN)
param("splits.heldout_seed", 777, "seed choosing held-out function pairs", priority=FROZEN)
param("splits.heldout_pairs_per_group_pair", 4, "held-out ordered pairs per HH/HL/LH/LL", priority=FROZEN)

# =============================================================================
# Evaluation sets and RL candidate pool
# =============================================================================
param("eval.version", "v1", "bump when eval sets change")
param("eval.seed", 1001, "eval-set sampling seed")
param("eval.sets", {
    "E1": {"depths": [1], "groups": ["H", "L"], "per_cell": 50, "by_function": True},
    "E2": {"depths": [2, 3, 4], "groups": ["H", "L", "mixed"], "per_cell": 134},
    "E3": {"depths": [5, 6], "groups": ["H", "L", "mixed"], "per_cell": 100},
    "E4": {"depths": [2, 3, 4], "groups": [None], "per_cell": 200, "novel": True},
    "probe": {"depths": [1, 2, 3, 4, 5, 6], "groups": ["H", "L", "mixed"], "per_cell": 24},
    "calib": {"depths": [1, 2, 3, 4, 5, 6], "groups": ["H", "L", "mixed"], "per_cell": 200},
}, "eval set name -> stratification (E1 atomic, E2 in-dist, E3 deeper, E4 novel pairs)")
param("rl_pool.seed", 2002, "candidate sampling seed")
param("rl_pool.depths", [1, 2, 3, 4, 5, 6], "candidate depths")
param("rl_pool.groups", ["H", "L", "mixed"], "candidate groups")
param("rl_pool.per_cell", 1200, "candidates per (depth, group) cell")
param("rl_pool.extra_cells", [{"depth": 2, "group": "L", "n": 4000}, {"depth": 4, "group": "H", "n": 4000}],
      "oversampled cells (origin=extra) for the matched-p datasets")

# =============================================================================
# Model (P)
# =============================================================================
param("model.n_layer", 4, "transformer layers", code="P1", priority=FROZEN, levels=(2, 4, 6))
param("model.d_model", 256, "hidden size", code="P1", priority=FROZEN)
param("model.n_head", 4, "attention heads", code="P1", priority=FROZEN)
param("model.d_ff", 1024, "FFN size", code="P1", priority=FROZEN)
param("model.max_len", 256, "max sequence length", priority=FROZEN)
param("model.pos", "rope", "position encoding", code="P2", priority=FROZEN, choices=("rope",))

# =============================================================================
# Pretraining (P)
# =============================================================================
param("pretrain.level", "mid", "base-model name tag: pt-{level}-s{seed}")
param("pretrain.n_atom_per_func", 5000, "unique depth-1 problems per function", code="P3", priority=FROZEN)
param("pretrain.n_comp", 20000, "unique composition examples (model strength)", code="P4", priority=CORE,
      levels=(2000, 20000, 200000))
param("pretrain.comp_repeat", 1, "times each composition appears in the corpus")
param("pretrain.min_comp_depth", 2, "smallest composition depth")
param("pretrain.max_comp_depth", 2, "largest composition depth", code="P5", priority=OPTIONAL, levels=(2, 3))
param("pretrain.h_ratio", 0.9, "fraction of compositions using only H functions", code="P6", priority=CORE,
      levels=(0.5, 0.9))
param("pretrain.noise_frac", 0.0, "fraction of compositions with one corrupted step", code="P10",
      priority=OPTIONAL, levels=(0.0, 0.1))
param("pretrain.total_target_tokens", 64_000_000, "supervised-token budget, equal across levels")
param("pretrain.steps", 10_000, "optimizer steps", code="P9", priority=OPTIONAL)
param("pretrain.batch_size", 256, "sequences per step")
param("pretrain.lr", 1e-3, "peak learning rate (AdamW, cosine)")
param("pretrain.warmup_frac", 0.05, "warmup fraction")
param("pretrain.eval_every", 1000, "greedy eval on E1-E4 + checkpoint every N steps")

# =============================================================================
# Difficulty labelling (D10)
# =============================================================================
param("label.k", 32, "samples per problem for p_hat", code="D10", priority=FROZEN)
param("label.temperature", 1.0, "sampling temperature for labelling")
param("label.batch_size", 512, "problems per generate() batch")

# =============================================================================
# RL data (D) - dataset composition itself lives in configs/datasets/*.yaml
# =============================================================================
param("data.dataset", None, "dataset name under data/datasets (D1-D4 are set in its spec)", types=(str,))
param("data.sampler", "static", "per-step problem sampler (B: online_filter, C: learnability)",
      code="D9", priority=OPTIONAL, choices=("static", "online_filter", "learnability"))
param("data.order", "shuffle", "problem order", code="D7", priority=OPTIONAL,
      choices=("shuffle", "shallow_to_deep"))
param("data.p_ema", 0.9, "EMA factor of the online per-problem p(t)")

# =============================================================================
# GRPO (R)
# =============================================================================
param("rl.group_size", 8, "rollouts per prompt G", code="R1", priority=CORE, levels=(4, 8, 32))
param("rl.prompts_per_step", 64, "prompts per step B", code="R2", priority=FROZEN)
param("rl.lr", 3e-5, "constant learning rate", code="R3", priority=OPTIONAL, levels=(1e-5, 3e-5, 1e-4))
param("rl.warmup_steps", 10, "linear warmup steps")
param("rl.steps", 500, "training steps (total rollouts = steps x B x G)", code="R4", priority=FROZEN)
param("rl.temperature", 1.0, "rollout temperature", code="R5", priority=OPTIONAL, levels=(0.7, 1.0))
param("rl.max_new_tokens", 128, "max completion length (depth 6 needs ~111)", code="R6", priority=FROZEN)
param("rl.adv_std_norm", True, "divide advantages by group std (False = Dr.GRPO)", code="R7",
      priority=OPTIONAL, levels=(True, False))
param("rl.loss_agg", "token", "loss aggregation", code="R8", priority=FROZEN, choices=("token", "sequence"))
param("rl.kl_coef", 0.0, "KL penalty to the base model", code="R9", priority=FROZEN)
param("rl.clip_eps", 0.2, "PPO clip (unused when ppo_epochs=1)", code="R10", priority=FROZEN)
param("rl.ppo_epochs", 1, "updates per rollout batch", code="R10", priority=FROZEN)
param("rl.entropy_coef", 0.0, "entropy bonus", code="R11", priority=FROZEN)
param("rl.grad_clip", 1.0, "gradient-norm clip")
param("rl.eval_every", 50, "E1-E4 eval interval (steps)")
param("rl.probe_every", 25, "probe-set eval interval (steps)")


def _reward_choices():
    from .rewards import REWARDS
    return REWARDS


param("rl.reward", "binary", "reward spec (rlvr.rewards.REWARDS)", code="R12", priority=OPTIONAL,
      levels=("binary", "partial"), types=(str, dict), choices=_reward_choices)


# =============================================================================
# Defaults, merging, validation
# =============================================================================

def defaults() -> dict:
    out: dict = {}
    for key in PARAMS:
        d = out
        *parents, last = key.split(".")
        for k in parents:
            d = d.setdefault(k, {})
        d[last] = copy.deepcopy(PARAMS.get(key).default)
    return out


PARAMS_DEFAULTS = defaults()


def merge(base: dict, over: dict, prefix: str = "") -> dict:
    """Deep merge; registered parameters (even dict-valued) are replaced whole."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        key = f"{prefix}{k}"
        if key not in PARAMS and isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v, key + ".")
        else:
            out[k] = copy.deepcopy(v)
    return out


def validate(cfg: dict) -> None:
    errors = []

    def walk(d: dict, prefix: str):
        for k, v in d.items():
            key = f"{prefix}{k}"
            if key in PARAMS:
                if (err := PARAMS.get(key).check(v)) is not None:
                    errors.append(err)
            elif isinstance(v, dict) and any(n.startswith(key + ".") for n in PARAMS):
                walk(v, key + ".")
            else:
                close = difflib.get_close_matches(key, PARAMS.names(), n=2)
                errors.append(f"unknown config key '{key}'" + (f" (did you mean {close}?)" if close else ""))

    walk(cfg, "")
    if errors:
        raise ValueError("invalid config:\n  " + "\n  ".join(errors))


def table(only: str | None = None) -> str:
    rows = ["| code | key | default | levels | priority | doc |", "|---|---|---|---|---|---|"]
    for key in PARAMS:
        p = PARAMS.get(key)
        if only and p.priority != only:
            continue
        dflt = "{...}" if isinstance(p.default, dict) else repr(p.default)
        if len(dflt) > 40:
            dflt = dflt[:37] + "..."
        lv = ", ".join(map(str, p.levels)) if p.levels else ""
        rows.append(f"| {p.code} | `{key}` | {dflt} | {lv} | {p.priority} | {p.doc} |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--core", action="store_true", help="only core variables")
    ap.add_argument("--yaml", action="store_true", help="dump the full default config")
    args = ap.parse_args()
    if args.yaml:
        import yaml
        print(yaml.safe_dump(defaults(), sort_keys=False, width=120))
    else:
        print(table(CORE if args.core else None))


if __name__ == "__main__":
    main()
