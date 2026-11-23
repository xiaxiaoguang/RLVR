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
param("git_commit", "", "written by training scripts into run_config.yaml")
param("world_size", 1, "GPUs used for pretraining; written by scripts/pretrain.py into run_config.yaml")
param("imports", [], "modules imported before validation; use them to register new atoms/rewards/...")
param("paths.data", "data", "root of generated data")
param("paths.runs", "runs", "root of run logs and intermediate checkpoints")
param("paths.checkpoints", "checkpoints", "root of final base-model checkpoints")

# =============================================================================
# Task (frozen after week 1: changing these changes pids, pools and names)
# =============================================================================
param("task.kind", "chain", "versioned task implementation", choices=("chain", "tree"))
param("task.tree_max_ops", 10, "maximum operator nodes in a tree problem (up to 12)")
param("task.sample_max_tries", 1000, "bounded retries for tree sampling constraints")
param("task.alphabet", "abcdefghijklmnopqrstuvwxyz", "input alphabet", code="P8", priority=FROZEN)
param("task.str_len", [6, 7, 8, 9, 10], "input length, or list of lengths sampled uniformly", code="P8",
      priority=FROZEN, types=(int, list))
param("task.name_seed", 12345, "seed of the label -> Fxx display-name shuffle", priority=FROZEN)
param("task.cot", True, "write intermediate steps in the target", code="P7", priority=FROZEN)
param("task.max_depth", 8, "largest depth used anywhere", priority=FROZEN)
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
    # v1 additions
    "rotl3": {"name": "rotate", "k": 3},
    "reverse_first_half": {"name": "reverse_half", "first": True},
    "riffle": "riffle",
    "shift_p5": {"name": "shift", "k": 5},
    "shift_by_pos": {"name": "shift_by_pos", "k": 1},
    "sort_first_half": {"name": "sort_half", "first": True},
    "odd_pos_m1": {"name": "shift_at", "k": -1, "parity": 1},
    "min_to_back": "min_to_back",
}, "label -> atom spec (rlvr.atoms.ATOMS); label order fixes the Fxx shuffle", priority=FROZEN)
# 12 / 12, every category on both sides: position 5/5, character 3/3, sorting 2/1, conditional 2/3.
param("task.h_group", ["reverse", "rotl1", "swap_adjacent", "swap_halves", "riffle",
                       "shift_p1", "complement", "shift_by_pos",
                       "sort_asc", "sort_first_half",
                       "even_pos_p1", "max_to_front"],
      "labels of the high-exposure group H; the rest form L", code="P6", priority=FROZEN)

# =============================================================================
# Splits
# =============================================================================
param("splits.salt", "rlvr-split-v1", "salt of the pid -> bucket hash", priority=FROZEN)
param("splits.pools", {"pretrain": [0, 900], "rl": [900, 950], "eval": [950, 1000]},
      "bucket ranges (of 1000) per pool", priority=FROZEN)
param("splits.heldout_seed", 777, "seed choosing held-out function pairs", priority=FROZEN)
param("splits.heldout_pairs_per_group_pair", 6, "held-out ordered pairs per HH/HL/LH/LL", priority=FROZEN)

# =============================================================================
# Evaluation sets and RL candidate pool
# =============================================================================
param("eval.version", "v2", "bump when eval sets change")
param("eval.seed", 1001, "eval-set sampling seed")
param("eval.sets", {
    "E1": {"depths": [1], "groups": ["H", "L"], "per_cell": 50, "by_function": True},
    "E2": {"depths": [2, 3, 4, 5], "groups": ["H", "L", "mixed"], "per_cell": 150},
    "E3": {"depths": [6, 7, 8], "groups": ["H", "L", "mixed"], "per_cell": 100},
    "E4": {"depths": [2, 3, 4, 5], "groups": [None], "per_cell": 200, "novel": True},
    "probe": {"depths": [1, 2, 3, 4, 5, 6, 7, 8], "groups": ["H", "L", "mixed"], "per_cell": 25},
    "calib": {"depths": [1, 2, 3, 4, 5, 6, 7, 8], "groups": ["H", "L", "mixed"], "per_cell": 200},
}, "eval set name -> stratification (E1 atomic, E2 in-dist, E3 deeper, E4 novel pairs)")
param("rl_pool.seed", 2002, "candidate sampling seed")
param("rl_pool.depths", [1, 2, 3, 4, 5, 6, 7, 8], "candidate depths")
param("rl_pool.groups", ["H", "L", "mixed"], "candidate groups")
param("rl_pool.per_cell", 1000, "candidates per (depth, group) cell")
param("rl_pool.extra_cells", [{"depth": 2, "group": "L", "n": 4000}, {"depth": 4, "group": "H", "n": 4000},
                              {"depth": 5, "group": "H", "n": 4000}],
      "oversampled cells (origin=extra) for the matched-p datasets")
param("rl_pool.mode", "file", "calibrate.py RL-pool step: 'file' labels data/rl_pool/candidates.jsonl; 'pilot' "
      "(v2 stage B, design.md decision 11) picks D/E cells from the calib labels and labels per-cell candidates",
      choices=("file", "pilot"))
param("rl_pool.pilot_depths", [1, 2, 3, 4, 5], "pilot: depths of the candidate (depth, group) cells (the RL range)")
param("rl_pool.pilot_min_gap", 2, "pilot: E needs a depth at least this much above D's")
param("rl_pool.pilot_max_cells", 4, "pilot: cells that get candidates (best D/E pairs by calib p ~ 0.5 share)")
param("rl_pool.pilot_per_cell", 10_000, "pilot: candidates per chosen cell (go criterion: from <= 10,000)")
param("rl_pool.pilot_min_matched", 500, "pilot: go criterion, problems per cell with relabelled p in gate.p_range "
      "after removing skippable ones, with a shared length mix")
param("rl_pool.pilot_seed", 3003, "pilot: candidate seed; each cell has its own stream, so the first N candidates "
      "of a cell do not depend on the other cells")

# =============================================================================
# Model (P)
# =============================================================================
param("model.n_layer", 12, "transformer layers", code="P1", priority=FROZEN, levels=(6, 12, 24))
param("model.d_model", 768, "hidden size", code="P1", priority=FROZEN, levels=(384, 768, 1024))
param("model.n_head", 12, "attention heads (head_dim = d_model / n_head)", code="P1", priority=FROZEN)
param("model.d_ff", 2048, "SwiGLU FFN size", code="P1", priority=FROZEN)
param("model.max_len", 512, "max position (depth 8, length 10 is ~290 tokens; + pos offset)", priority=FROZEN)
param("model.attn_impl", "sdpa", "HF attention implementation", choices=("sdpa", "eager", "flash_attention_2"))
param("model.pos", "rope", "position encoding", code="P2", priority=FROZEN, choices=("rope",))

# =============================================================================
# Pretraining (P)
# =============================================================================
param("pretrain.level", "mid", "base-model name tag: pt-{level}-s{seed}")
# Corpus: streamed and never repeated (rlvr.stream). Shares are of supervised (target) tokens.
param("pretrain.comp_frac", 0.01, "share of supervised tokens from composition examples (model strength)",
      code="P4", priority=CORE, levels=(0.001, 0.01, 0.1))
param("pretrain.comp_depth_mix", {2: 0.6, 3: 0.35, 4: 0.05},
      "composition depth -> share of composition examples", code="P5", priority=OPTIONAL)
param("pretrain.h_only_depths", [4], "composition depths that use only H functions", code="P5b", priority=FROZEN)
param("pretrain.h_ratio", 0.9, "fraction of the other compositions using only H functions", code="P6",
      priority=CORE, levels=(0.5, 0.9))
param("pretrain.noise_frac", 0.0, "fraction of compositions with one corrupted step", code="P10",
      priority=OPTIONAL, levels=(0.0, 0.1))
param("pretrain.pos_offset_max", 128, "random start position in [0, max] per example (0 = off)", code="P2b",
      priority=OPTIONAL, levels=(0, 128))
# Optimisation. Batches are length-bucketed and filled to a supervised-token budget, so
# every level sees the same supervised tokens per step and in total.
param("pretrain.steps", 9_500, "optimizer steps; ~63 min on 4 idle A100s (0.40 s/step), re-check with --bench", code="P9",
      priority=OPTIONAL)
param("pretrain.target_tokens_per_step", 192_000, "global supervised tokens per optimizer step (~270k total)")
param("pretrain.micro_target_tokens", 6_000,
      "supervised tokens per micro-batch per GPU (memory knob: ~1 GB per 1k eager, less compiled)")
param("pretrain.bucket_examples", 20_000, "examples sorted by length together when forming micro-batches")
param("pretrain.num_workers", 4, "data-generation worker processes per GPU")
param("pretrain.lr", 6e-4, "peak learning rate (AdamW, cosine)")
param("pretrain.min_lr_frac", 0.1, "final learning rate as a fraction of the peak")
param("pretrain.warmup_frac", 0.02, "warmup fraction")
param("pretrain.weight_decay", 0.1, "AdamW weight decay (not on norms/biases)")
param("pretrain.grad_clip", 1.0, "gradient-norm clip")
param("pretrain.compile", True, "torch.compile the model")
param("pretrain.target_minutes", 60, "--bench suggests steps so training alone takes this long")
param("pretrain.save_at", [0.25, 0.5], "fractions of training with an extra checkpoint (P9; final always saved)")
param("pretrain.eval_every", 2000, "greedy eval interval (steps; rank 0 only)")
param("pretrain.log_every", 50, "metrics.jsonl interval (steps)")
param("pretrain.eval_sets", ["E1", "E2", "E3", "E4", "probe"], "eval sets scored during pretraining")

# =============================================================================
# Shared evaluation / generation settings
# =============================================================================
param("eval.max_new_tokens", 256, "max completion length at eval (depth 8, length 10 needs ~240)")
param("eval.batch_size", 256, "prompts per greedy generate() batch (256 -> ~6 GB)")
param("eval.pass_k_samples", 0, "samples per problem for pass@k at eval (T = 1.0); 0 = off")
param("eval.pass_k", [1, 2, 4, 8, 16], "k values of the pass@k columns (those <= eval.pass_k_samples)")

# =============================================================================
# Calibration and gate check after pretraining (design.md "Deliverables")
# =============================================================================
param("gate.depth1_greedy", 0.98,
      "min greedy accuracy on E1 (depth 1): overall, and per function via its Wilson upper bound")
param("gate.depth1_func_z", 1.96, "z of the per-function Wilson bound (50 problems: <= 2 errors pass)")
param("gate.p_range", [0.4, 0.6], "p range of the matched-p datasets (inclusive)")
param("gate.min_matched", 300, "min problems with p in p_range in each matched cell (total)")
param("gate.min_matched_per_len", 120,
      "min problems with p in p_range per input length in each matched cell; the matched-p datasets "
      "stratify on str_len, so n=500 over 5 lengths needs 100 each (+ margin)")
param("gate.matched_cells", [{"depth": 2, "group": "L"}, {"depth": 4, "group": "H"}],
      "RL-pool cells that need enough p~0.5 problems (shallow L, deep H)")
param("gate.spread_depths", [2, 3, 4], "depths whose calib p must not be all 0 or all 1 (depth 5 is not in "
      "pretraining, so it is excluded: design.md decision 2)")
param("gate.min_spread_frac", 0.2, "min fraction of calib problems with 0 < p_hat < 1 at each spread depth")
param("gate.loss_depths", [3], "composition depths whose final pretraining loss (loss_d*) is checked")
param("gate.min_final_loss", 1e-3,
      "min per-token loss at gate.loss_depths; lower means those compositions are saturated")
param("gate.target", None, "level design target {depth, p_range: [lo, hi]} for the mean calib p (null = skip)",
      types=(dict,))
param("gate.hard_checks", ["depth1_greedy", "spread", "matched_cells"],
      "checks that fail the gate (exit 1); the others are warnings. Choices: depth1_greedy, spread, "
      "matched_cells, final_loss, target")

# =============================================================================
# Difficulty labelling (D10)
# =============================================================================
param("label.k", 32, "samples per problem for p_hat", code="D10", priority=FROZEN)
param("label.temperature", 1.0, "sampling temperature for labelling")
param("label.batch_size", 16,
      "problems per generate() batch at label.k (x k sequences; 16 x 16 -> ~6 GB); scaled down for relabel_k")
param("label.relabel_k", 64,
      "independent second labelling of the screened matched-cell candidates (0 = off); the matched-p "
      "datasets select on it, which shrinks the winner's-curse bias of selecting on a noisy p_hat")
param("label.relabel_range", [0.3, 0.7], "first-pass p_hat range screened for the second labelling")
param("label.relabel_cells", [{"depth": 2, "group": "L"}, {"depth": 4, "group": "H"}, {"depth": 5, "group": "H"}],
      "RL-pool cells that get the second labelling (the matched-p dataset cells)")

# =============================================================================
# RL data (D) - dataset composition itself lives in configs/datasets/*.yaml
# =============================================================================
param("data.dataset", None, "dataset name under data/datasets (D1-D4 are set in its spec)", types=(str,))
param("data.sampler", "auto", "per-step problem sampler (B: online_filter, C: learnability); "
      "auto = the `sampler` of the dataset spec, else static",
      code="D9", priority=OPTIONAL, choices=("auto", "static", "online_filter", "learnability"))
param("data.order", "shuffle", "static sampler: shuffled epochs, or the same draws sorted by depth over the run",
      code="D7", priority=OPTIONAL, choices=("shuffle", "shallow_to_deep"))
param("data.p_ema", 0.9, "EMA factor of the online per-problem p(t)")
param("data.p_init_labels", True,
      "start the online p(t) from the base model's pass-rate labels (dataset p_hat column, else "
      "<data>/labels/{model_id}.parquet) when they exist; otherwise p is unknown until first sampled")
param("data.p_edge", 1 / 32,
      "online_filter keeps problems with p_edge <= p(t) <= 1 - p_edge or unknown p (edge p bins carry no signal)")
param("data.explore_frac", 0.0,
      "dynamic samplers: share of each batch drawn uniformly from the whole dataset (lets filtered problems return)")

# =============================================================================
# GRPO (R)
# =============================================================================
param("rl.group_size", 8, "rollouts per prompt G", code="R1", priority=CORE, levels=(4, 8, 32))
param("rl.prompts_per_step", 64, "prompts per step B", code="R2", priority=FROZEN)
param("rl.lr", 1e-5, "constant learning rate", code="R3", priority=OPTIONAL, levels=(3e-6, 1e-5, 3e-5))
param("rl.warmup_steps", 10, "linear warmup steps")
param("rl.steps", 500, "training steps (total rollouts = steps x B x G)", code="R4", priority=FROZEN)
param("rl.temperature", 1.0, "rollout temperature", code="R5", priority=OPTIONAL, levels=(0.7, 1.0))
param("rl.max_new_tokens", 256, "max completion length (depth 8, length 10 needs ~240)", code="R6",
      priority=FROZEN)
param("rl.adv_std_norm", True, "divide advantages by group std (False = Dr.GRPO)", code="R7",
      priority=OPTIONAL, levels=(True, False))
param("rl.loss_agg", "token", "loss aggregation", code="R8", priority=FROZEN, choices=("token", "sequence"))
param("rl.kl_coef", 0.0, "KL penalty to the base model", code="R9", priority=FROZEN)
param("rl.clip_eps", 0.2, "PPO clip (unused when ppo_epochs=1)", code="R10", priority=FROZEN)
param("rl.ppo_epochs", 1, "updates per rollout batch", code="R10", priority=FROZEN)
param("rl.entropy_coef", 0.0, "entropy bonus", code="R11", priority=FROZEN)
param("rl.grad_clip", 1.0, "gradient-norm clip")
param("rl.weight_decay", 0.0, "AdamW weight decay during RL")
param("rl.eval_every", 50, "eval interval for rl.eval_sets (steps; step 0 and the last step always)")
param("rl.probe_every", 25, "probe-set eval interval (steps)")
param("rl.eval_sets", ["E1", "E2", "E3", "E4"], "eval sets scored every rl.eval_every steps")
param("rl.probe_set", "probe", "eval set scored every rl.probe_every steps (null = none)", types=(str,))
# Run plumbing (scripts/grpo.py). Task, model, splits and paths always come from the base checkpoint.
param("rl.base", None, "base checkpoint dir, e.g. checkpoints/pt-mid-s0 or runs/pt-mid-s0/ckpt/step-2375",
      types=(str,))
param("rl.tag", "", "suffix of the run name rl-{model}-{dataset}-G{G}-s{seed}[-{tag}] for hyperparameter variants")
param("rl.gen_batch_size", 64, "prompts per rollout generate() batch (x group_size sequences; memory knob)")
param("rl.micro_batch_tokens", 16_000,
      "padded tokens per backward micro-batch; gradients are accumulated over the whole rollout batch, "
      "so this is a memory knob that does not change the update")
param("rl.kl_every", 10, "log kl_to_base on the rollout tokens every N steps (0 = never; every step if kl_coef > 0)")
param("rl.log_every", 10, "print interval (steps); metrics.jsonl gets every step")
param("rl.log_completions", True, "store the sampled completions in rollouts.jsonl (rewards are always stored)")
param("rl.log_eval_problems", True, "write eval_problems.jsonl: one row per (eval step, eval set, problem) with "
      "correct, n_correct, err_type, err_step")
param("rl.save_at", [], "fractions of training with a checkpoint in runs/{run}/ckpt/step-{N}")
param("rl.save_final", True, "save the final model to runs/{run}/ckpt/step-{steps} (~340 MB)")
param("rl.compile", False, "torch.compile the decoder layers for the policy-gradient forward/backward")


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
