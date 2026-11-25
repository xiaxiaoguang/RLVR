# RL post-training (GRPO) on tree-v1: runs and results

This file records the RL experiments: what has been run, how, and what came out. The GRPO algorithm, default
hyperparameters and correctness checks are in [`design.md` → GRPO](design.md#grpo-hanzhong); the command-line usage
and log format are in the [README](../README.md#reinforcement-learning-grpo).

All RL so far has run on the lab machine's **GTX 1080 Ti** (Pascal, 11 GB, sm61), not on Perlmutter. The raw evidence
for every run is under `runs/idea-20261121-01-*/` on that machine. `runs/` is not in git (see
[Where the evidence is](#where-the-evidence-is)).

## Status (2026-11-25)

| stage | dates | result |
| --- | --- | --- |
| Pascal port and CPU readiness | 11-21 | fp32 path for Pascal, batching-only 1080 Ti config, `--preflight`; required CPU suite passes |
| Assets and first CUDA checks | 11-21 | v2 base model from HF (pinned revision) verified; fp32 primitive, tiny synthetic GRPO step and one real forward pass on the 1080 Ti |
| Generation / backward capacity | 11-21 – 11-22 | one real G8 group generates and back-propagates with a ~1.6 GB allocator peak |
| Short real RL (20 steps) | 11-23 | `pt-tree-mix-cf0.0035-s0`, B64/G8, 20 AdamW updates: E2 34.4 → 35.8 %, probe 27.5 → 29.4 % |
| **Paired RL (500 steps × 2 models)** | 11-24 – running | first model past step 280/500; second model not started ([below](#paired-rl-500-steps-two-base-models)) |

The design's phases 0–3 (noise, screening, A–G groups, fits) have **not** started. The paired run is one seed and a
fixed-data comparison of the two v2 base models, not one of the A–G groups. The v2 go criterion is still not met
(405 < 500 E problems), so the D/E comparison is not unlocked.

## Running GRPO on a GTX 1080 Ti

### What differs from the A100 setup

Only execution settings change. B = 64, G = 8, T = 1, the 768-token generation limit, the reward, the model and the
splits are the same as in the design.

- **Precision** (`rlvr/precision.py`): native bf16 autocast on CUDA capability ≥ 8 (A100); no autocast on older GPUs
  or on CPU, so on Pascal the model runs in fp32. Generation and the GRPO log-prob forward use the same selector, and
  log-softmax is always fp32.
- **Batching** (`configs/rl/1080ti.yaml`): `rl.gen_batch_size=1` (prompts per generate call; each expands to G
  completions), `rl.micro_batch_tokens=1024`, `rl.compile=false`, `eval.batch_size=1`. A sequence longer than the
  micro-batch limit forms its own chunk. Batching can change random-number consumption, so rollouts are not bitwise
  identical to an A100 run. `tests/test_grpo.py::test_1080ti_config_only_changes_execution_batching` checks that the
  config changes nothing else.
- **Preflight** (`scripts/grpo.py --preflight`): checks the checkpoint files, loads the model and tokenizer, validates
  task identity, lengths and dataset/eval leakage, and prints precision and batching. It runs no forward pass and
  creates no run directory. `scripts/grpo_1080ti.sh` wraps `scripts/grpo.py` with the environment below and the
  1080 Ti config. It runs preflight by default and trains only with `--run`.

### Environment

uv project `/work/users/lab_admin/envs/RLVR-rl1080` (Python 3.11.18, torch 2.8.0+cu126, transformers 5.24.0,
numpy 2.4.6, trl 1.18.2). Lock: its `uv.lock`, SHA256
`da9cbed8ab7ca6484c55505bdb4b4d7700efcc328330f823246870bc87f292b9`. Keep the cu126 torch wheel: it is the build
verified on sm61.

```bash
cd /home/lab_admin/workspace/RLVR
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/work/users/lab_admin/uv-cache UV_LINK_MODE=hardlink \
       UV_PYTHON_INSTALL_DIR=/work/users/lab_admin/uv-python
uv sync --project /work/users/lab_admin/envs/RLVR-rl1080 --frozen --check

# CPU tests (no GPU)
CUDA_VISIBLE_DEVICES='' uv run --project /work/users/lab_admin/envs/RLVR-rl1080 --frozen \
    python -m pytest -q tests/test_grpo.py tests/test_model.py tests/test_tree.py

# on one GPU: preflight, then the real run (--run); other arguments go to scripts/grpo.py
CUDA_VISIBLE_DEVICES=1 bash scripts/grpo_1080ti.sh --base checkpoints/tree-v1/pt-tree-mix-cf0.0035-s0 \
    --dataset data/tree-v1/rl_pool/pt-tree-mix-cf0.0035-s0/pilot_candidates.jsonl
CUDA_VISIBLE_DEVICES=1 bash scripts/grpo_1080ti.sh --run --base ... --dataset ...
```

If the `.venv` is removed, rebuild it with `uv sync --project /work/users/lab_admin/envs/RLVR-rl1080 --frozen`.

### Assets

The v2 base models come from [`rlvr-cs291k/rlvr-pretrain-v2`](https://huggingface.co/rlvr-cs291k/rlvr-pretrain-v2),
pinned to revision `2cc23a4ebdfe5a64dd4d2220e02194a1f5745736`:

| model | `model.safetensors` SHA256 | size |
| --- | --- | --- |
| `pt-tree-mix-cf0.0035-s0` | `6476c96d6a50ce809791379f64cd1d93332ba031d7c1f5a29077b84201c13ee8` | 340,128,712 B |
| `pt-tree-mix-cf0.006-s0` | `36c50e13f27feaf43f75f97ff1150add99da4459e7b118027bba0ec1e6b8a04d` | 340,128,712 B |

Both have 85.0M parameters (12 layers, d 768, 12 heads, FFN 2048, 49-token vocabulary, context 1024, rotate_by_first
at F10). Other inputs: `data/tree-v1/labels/pt-tree-mix-cf0.0035-s0.parquet`,
`data/tree-v1/rl_pool/pt-tree-mix-cf0.0035-s0/pilot_candidates.jsonl` (40,000 candidates) and
`data/tree-v1/calib/pt-tree-mix-cf0.0035-s0/pilot.json`.

**Evaluation sets.** The canonical `data/tree-v1/eval/v1/` files were not available locally, so they were regenerated
twice from the checkpoint's saved recipe (`runs/idea-20261121-01-eval-prep/{primary,repeat}/`, with only
`paths.data` changed). The two copies match byte for byte, and the counts match the historical ones: E1 1200,
E2 1800, E3 900, E4 800, probe 575, calib 4600. This reproduces the recipe deterministically but is not a
hash-verified restore, since no historical hashes exist. The eval sets are disjoint from all 40,000 RL candidates.

### Measured cost on the 1080 Ti

| what | time | memory (PyTorch allocator peak) |
| --- | --- | --- |
| one G8 group, generate (188 tokens) | 2.3 s | 0.53 GB |
| one G8 group, loss + backward, policy + reference resident | 1.4 s | 1.58 GB |
| one RL step, B64/G8 (~140-token completions) | ~100 s rollout + ~12 s update | — |
| greedy eval, eval batch 2 | E1 5 min, E2 40 min, E3 27 min, E4 19 min, probe 13 min (≈ 1.7 h for all) | — |
| 500 steps, full eval every 50 steps and probe every 25 | ~37 h per model (~16 h training, ~21 h evaluation) | — |

The A100 figure in the design (~3.3 s/step) is for ~100-token chain-v1 completions in bf16 with large generation
batches. On the 1080 Ti, generation dominates training and evaluation because it runs one or two prompts at a time.
Faster generation (static KV cache, or larger generation batches that fit in 11 GB) is the main thing to fix before
the design's 135 runs could run here. Evaluating less often is the cheap option.

## Short real RL (20 steps)

`runs/idea-20261121-01-real-rl/` (2026-11-23 01:47–05:23), analysed in `runs/idea-20261121-01-real-rl-analysis/`.

Setup: `pt-tree-mix-cf0.0035-s0` on its 40,000-candidate pilot pool, static sampler with shuffle, seed 0, B64/G8,
T = 1, 768 tokens, lr 1e-5 with 10 warmup steps, token-mean loss, group-std advantages, one update per batch,
grad clip 1, and KL and entropy coefficients of 0 (KL to the reference is logged only).

| | result |
| --- | --- |
| updates / rollouts / distinct problems | 20 / 10,240 / 1,280 |
| mean training reward | 0.559 (first 5 steps 0.524, last 5 steps 0.568) |
| groups with mixed rewards (signal) | 47.6 %; all-wrong 298, all-right 373 of 1,280 |
| completions ending in `<eos>` | 10,239 / 10,240 |
| d4-H training reward | 377 / 2,432; signal in 98 / 304 groups |
| parameter change base → final | L2 0.389, max abs 1.6e-4 |
| KL to base (logged) | 0 → 8.0e-4 at step 20 |
| E2 greedy (same frozen set) | 620 → 644 / 1800 (34.4 → 35.8 %) |
| probe greedy | 158 → 169 / 575 (27.5 → 29.4 %) |

This run shows that the full pipeline trains a real base model on the 1080 Ti. It is one seed and 20 steps, so the
small gains are not evidence of a stable learning effect.

## Paired RL: 500 steps, two base models

`runs/idea-20261121-01-paired-rl/`, started 2026-11-24 23:30 on GPU 1 with a 100 GPU-hour budget. The runner
scripts and configs are copied into [`experiments/paired-rl/`](../experiments/paired-rl/README.md).

**Question.** The two v2 base models differ only in the share of pretraining spent on compositions (comp_frac 0.0035
vs 0.006). Both use seed 0, 9,500 steps, depth mix .30/.35/.35 and h_ratio .5. Does the base model's composition
exposure change what 500 GRPO steps achieve?

**Setup.** The two models run one after the other (cf0.0035 first), each starting from its original checkpoint: seed 0,
500 steps, B64/G8 (256,000 rollouts each), fp32, T = 1, 768 tokens. Both train on the **same** 40,000-candidate pool
(cf0.0035's pilot pool: cells d2-H/L/mixed and d4-H), in the same fixed problem order, with the static sampler and no
p-label initialisation (`data.p_init_labels=false`). The two resolved configs differ only in `rl.base`. Greedy
evaluation runs on the full E1–E4 every 50 steps and on probe every 25 steps (step 0 and the last step included),
with per-problem logs. A checkpoint is saved at every 10 % of training.

**Interim results, `pt-tree-mix-cf0.0035-s0`** (greedy accuracy, with step score in brackets):

| step | E1 | E2 | E3 | E4 | probe |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.000 | .344 (2.52) | .001 (1.09) | .271 (2.51) | .275 (1.86) |
| 50 | 1.000 | .366 (2.63) | .001 (1.18) | .305 (2.61) | .294 (1.99) |
| 100 | 1.000 | .367 (2.62) | .001 (1.15) | .318 (2.62) | .297 (1.99) |
| 150 | 1.000 | .373 (2.65) | .000 (1.16) | .324 (2.67) | .304 (2.03) |
| 200 | 1.000 | .376 (2.65) | .001 (1.16) | .325 (2.65) | .311 (2.04) |
| 250 | 1.000 | .379 (2.66) | .001 (1.15) | .333 (2.69) | .315 (2.06) |

E2 and E4 rise steadily (+3.5 and +6.1 points by step 250). E3 (depth 6–8, outside the RL depth range) stays near 0
in accuracy, with a small step-score gain. Around step 280, training reward is ~0.65–0.75, signal ~0.15–0.25 and
entropy ~0.01: more groups are coming out all-right, so the learning signal is shrinking.

For reference, the historical greedy scores of cf0.006 (A100, measured at pretraining time) start higher: E2 .546,
E3 .007, E4 .461, probe .400 (cf0.0035: .344 / .002 / .273 / .277). The paired run re-measures step 0 for both
models on the same frozen sets.

**Planned comparison.** For each model: final − step 0 on every set, by depth and structure group, for accuracy, step
score and step fraction; then the difference between the two models' RL gains. With one seed and a fixed order
(cf0.0035 always first), this describes two runs. It is not a significance test of H1–H3.

**Operational notes.**

- The supervisor caps total GPU time (346,420 s of the budget remained at launch) and stops if a foreign process
  appears on the GPU. It never uses GPU 0. Status: `runs/idea-20261121-01-paired-rl/main-status.json`. Log:
  `main-pt-tree-mix-cf0.0035-s0.log`. At ~37 h per model, the pair is expected to finish around 2026-11-28.
- Before each model starts, the worker checks (`validate_sources` in its `common.py`) that the code tree it runs from
  (`runs/idea-20261121-01-branch-integration/worktree`, commit `2f22119`) and the main checkout
  (`/home/lab_admin/workspace/RLVR`, HEAD `2d5973b`, including tracked files such as `.gitignore`) are unchanged.
  While the run is in progress, do not commit, check out or edit tracked files in either directory, or the second
  model will refuse to start. Use a separate worktree for other work.
- Checkpoints hold the model, tokenizer and config only (no optimizer, scheduler, RNG or sampler state), so a stopped
  run cannot be resumed exactly. If the pair is interrupted, report which model and steps are missing, and do not
  compare one model's final step with the other's intermediate step.
- Read the results at any time (CPU only):

  ```bash
  source runs/idea-20261121-01-paired-rl/env.sh
  CUDA_VISIBLE_DEVICES='' uv run --project /work/users/lab_admin/envs/RLVR-rl1080 --frozen \
      python "$PAIRED_RUN/inspect_results.py"
  ```

## Not done yet

- The TRL `GRPOTrainer` cross-check and the 8-problem overfitting test from the design's correctness checks.
- Seed noise (phase 0): every RL result so far uses seed 0.
- The fixed-64 E2 diagnostic baseline (`runs/idea-20261121-01-eval-prep/fixed64.*`, frozen before any model output)
  has not been evaluated.
- Faster generation on Pascal. At ~37 h per 500-step run, the design's 135 runs do not fit on this machine.

## Where the evidence is

Each `runs/idea-20261121-01-<task>/` directory has a `REPORT_ZH.md` (results, in Chinese). Most also have `REPRO.md`
(exact commands, input hashes) and `USER_GUIDE_ZH.md`. These directories exist on the lab machine only.

| directory | content |
| --- | --- |
| `readiness`, `readiness-resume` | CPU readiness of the Pascal path; synthetic tiny-model GRPO run |
| `allocation-investigation`, `assets-gpu` | asset download and hashes, first CUDA checks, GPU-time ledger |
| `eval-prep` | regenerated eval sets (two identical copies), fixed-64 E2 diagnostic set |
| `generation-capacity`, `sampled-capacity` | G1 and G8 generation timing and memory |
| `backward-capacity`, `backward-*` | one real G8 group through loss and backward |
| `optimizer-proposal`, `throughput-diagnostic`, `integrated-throughput` | optimizer-step and throughput checks |
| `branch-integration`, `integration-audit-followup` | merge of the evaluation branch into master (`2f22119`) and its audit |
| `real-rl`, `real-rl-analysis` | 20-step real RL run and its CPU re-analysis |
| `comparison-prep` | inputs for the paired run (data root, pretraining inventory of both models) |
| `paired-rl` | the running 500-step × 2 paired run |

Large weights from these runs were moved to `/data/users/lab_admin/migration/RLVR/` and replaced by symlinks
(`runs/idea-20261121-01-real-rl-analysis/STORAGE_MAP.md`).
