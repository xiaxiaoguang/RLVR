# RLVR data-difficulty experiments

**Status (see `docs/design.md` "Status"): the four chain-v1 base models are trained, calibrated and on
Hugging Face (`rlvr-cs291k/rlvr-pretrain-v1`; results in `docs/pretrain_v1_results.md`). The next task
is tree-v1 (below; adopted on 2026-11-16, design.md decision 10); its pilot (stage A: atomic skills,
stage B: p ~ 0.5 supply) runs before new base models are trained. An earlier binary-operator variant
(v2, 24 unary + 4 binary) was piloted and is superseded (`docs/pretrain_v2_results.md`).** GRPO
(`scripts/grpo.py`) and a checkpoint x dataset sweep runner (`scripts/rl_sweep.py`) are implemented
and smoke-tested on the local round-1 checkpoints; see [Reinforcement learning](#reinforcement-learning-grpo).
The RL experiments of the design doc (phases 0-3) have not been run, and the curve-fit analysis
is not implemented yet.

## Multi-input task (tree-v1)

The new versioned recipe replaces 12 simple/redundant unary operators with 8 binary
and 4 ternary operators, generating genuine expression trees. Use
`bash scripts/make_tree_data.sh` in the `rlvr` environment; outputs are isolated under
`data/tree-v1/`. See [operator definitions, sampling and training setup](docs/tree_v1.md).
The original chain-v1 defaults and saved results remain available. No tree model has
been trained yet. Pilot stage A (design.md "v2 pilot"; atoms only, E1 check per operator):

```bash
python scripts/make_eval_sets.py --config configs/tasks/tree_v1.yaml
CONFIG=configs/pretrain/tree/stage_a.yaml CALIB_ARGS=--e1-only bash scripts/slurm/model_pipeline.sh tree-a-atoms
# -> checkpoints/tree-v1/pt-tree-a-atoms-s0, runs/tree-v1/pt-tree-a-atoms-s0/calib/gate.json
```

## Environment

```bash
conda env create -f environment.yml && conda activate rlvr     # python 3.11 + pip install -e .[dev]
# exact versions instead: pip install -r requirements.lock.txt && pip install -e . --no-deps
```
Dependencies live in `pyproject.toml`; `requirements.lock.txt` pins the full resolution
(torch 2.14.1 / CUDA 13 runtime, transformers 5.20.2, trl 1.16.0, pandas 3.1.0).

```
rlvr/                 library (frozen interface after review)
  task.py             24 atomic functions, Problem, sampling, format_prompt/target, parse_answer, reward
  tokenizer.py        char-level tokenizer, 48 tokens; encode_example() masks the prompt with -100
  registry.py         generic name -> factory Registry (build from 'name' or {name: ..., **params})
  params.py           parameter registry: every config key, default, levels, priority, doc id (P1..R13)
  atoms.py            ATOMS registry: parameterized atomic functions (rotate(k), shift(k), sort(descending), ...)
  rewards.py          REWARDS registry: binary, partial
  splits.py           pid-hash pools (pretrain 90% / rl 5% / eval 5%) + 24 held-out function pairs
  sets.py             stratified (depth x group, balanced over input length) set generation
  stream.py           streamed pretraining corpus: Mixture (comp_frac by tokens, depth mix, H/L,
                      leak-free rejection) and PretrainStream (length-bucketed micro-batches)
  difficulty.py       p bins, Wilson CI, P_sig(p, G); WEIGHTS registry (uniform, learnability, signal)
  config.py, io.py    defaults <- YAML (`base:` inheritance) <- key=value overrides, validated; jsonl helpers
  model.py            LlamaForCausalLM (RoPE, 12 x 768, ~85M) build/save/load; generate() (left pad, cut after <eos>)
  evaluate.py         greedy_eval (eval.jsonl rows, E1 per function); label_pass_rates (p_hat, Wilson CI)
  grpo.py             GRPO core: group advantages, teacher-forced completion log-probs, micro-batched update
  sampler.py          per-step problem samplers (static / online_filter / learnability) + online p(t)
  tree_task.py        tree-v1: TreeTask (Leaf / Call trees, post-order trace, structure features, fingerprint)
  tree_ops.py         tree-v1 operators with declared arity (unary, binary, ternary)
  tree_sampling.py    tree-v1 shape counting and uniform shape sampling for (n_ops, depth)
  tree_splits.py      tree-v1 pools and held-out (parent, slot, child) edges
scripts/
  make_eval_sets.py   data/eval/v2/{E1,E2,E3,E4,probe,calib}.jsonl + data/meta/
  make_rl_pool.py     data/rl_pool/candidates.jsonl (to be labelled with p per model)
  preview_stream.py   data/pretrain/pt-{level}-s{seed}/{preview.jsonl,stats.json}: sample + projected corpus stats
  build_dataset.py    data/datasets/{name}.jsonl from configs/datasets/*.yaml
  report_data.py      data/REPORT.md: distributions, degeneracy, leakage checks, examples
  make_all_data.sh    regenerate everything
  pretrain.py         train one base model (torchrun DDP or 1 GPU; --bench) -> runs/pt-*/, checkpoints/pt-*/
  sweep.py            pilot sweep comp_frac x pos_offset_max (1 GPU per run) -> runs/sweep/summary.*
  calibrate.py        p heatmap + gate check -> runs/pt-*/calib/; RL-pool labels -> data/labels/pt-*.parquet
  pretrain_all.sh     weak+strong then mid+midbal, 4 GPUs each, then calibrate
  slurm/              Perlmutter: pt-{level}.sbatch (1 node, 4-GPU training + quick calibration),
                      calib.sbatch (1 shared GPU, full calibration, chained with --dependency=afterok)
  relabel_report.py   winner's-curse check: first-pass p in [0.4, 0.6] vs the second labelling
  grpo.py             GRPO on one base checkpoint + one RL dataset (1 GPU) -> runs/rl-*/
  rl_sweep.py         series of GRPO runs: checkpoints x datasets x seeds x grid, one per GPU, resumable
  make_tree_data.sh   tree-v1 eval sets, RL pool, stream preview, datasets and report under data/tree-v1/
  push_hf.py          upload final checkpoints (+ calib files, optional RL-pool labels) to a Hugging Face repo
configs/              overrides only: pretrain/{weak,mid,strong,midbal,pilot}.yaml, pretrain/tree/*.yaml,
                      tasks/tree_v1.yaml, datasets/*.yaml, rl/*.yaml
tests/                pytest tests/ (CPU, tiny models): task, registry, model, gate, tree, grpo
```

Regenerate: `bash scripts/make_all_data.sh` (deterministic; `SEED=1` for other preview seeds).

## Pretraining

```bash
# 0. throughput on the GPUs you will use -> suggested pretrain.steps for pretrain.target_minutes
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 scripts/pretrain.py \
    --config configs/pretrain/mid.yaml --bench 100
# 1. pilot sweep (~10 min training + calibration per run, one GPU each) -> runs/sweep/summary.{md,png}
python scripts/sweep.py --gpus 0 1 2 3 4 5 6 7
#    then set comp_frac in configs/pretrain/{weak,mid,strong}.yaml (suggestion.json is an upper bound)
# 2. the four base models: weak+strong, then mid+midbal (4 GPUs each), then calibration
SEED=0 GPUS="0 1 2 3 4 5 6 7" bash scripts/pretrain_all.sh
# single run / calibration / p-based datasets
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 scripts/pretrain.py --config configs/pretrain/mid.yaml
python scripts/calibrate.py checkpoints/pt-mid-s0 [--skip-pool]
python scripts/calibrate.py runs/pt-mid-s0/ckpt/step-2375 --skip-pool    # 25% checkpoint
python scripts/build_dataset.py configs/datasets/*.yaml
```
- **Corpus.** Streamed, never repeated (`rlvr/stream.py`). `pretrain.comp_frac` is the share of
  *supervised tokens* from compositions (depth 2:3:4 = 0.6:0.35:0.05, depth 4 H-only, otherwise
  `h_ratio` H-only); atoms (depth 1, all 24 functions uniform) fill the rest. Rejection sampling
  keeps every example in the pretrain pool, free of held-out pairs and of eval pids. `stats.json`
  in the run dir records what was actually consumed.
- **Confound control.** Same model, steps and supervised tokens per step for every level: batches
  are length-bucketed and filled to `micro_target_tokens` supervised tokens; the loss is the token
  sum over that fixed budget. Only the corpus composition differs.
- **Scale (measured, idle A100-40GB, bf16, per-layer `torch.compile`).** 85M non-embedding params;
  ~184k tok/s per GPU (~30% MFU) at micro_target_tokens 12k. 4 GPUs: ~0.40 s/step, so the default
  9,500 steps x ~270k tokens = ~2.6B tokens in ~63 min, plus ~5 greedy evals on rank 0. Re-run
  `--bench` on a shared machine and set `pretrain.steps` from it.
- **Memory (batch size).** There is no `batch_size` for pretraining: a micro-batch is filled to
  `pretrain.micro_target_tokens` supervised tokens (default 6k, ~6 GB peak), and gradient
  accumulation is derived as `target_tokens_per_step / (micro_target_tokens x GPUs)`, so lowering
  the micro budget on a crowded GPU does not change the optimisation. Generation memory:
  `eval.batch_size` (256 prompts, ~6 GB) and `label.batch_size` (16 problems x k=16, ~6 GB).
  Override on the command line, e.g. `pretrain.micro_target_tokens=3000 eval.batch_size=128`.
- **Positions.** Rows are right-padded (causal mask only, padding after real tokens) and get a
  random start position in [0, `pos_offset_max`] (P2b); generation uses positions from 0.
- **Overwrite guard.** `pretrain.py` refuses to replace an existing `checkpoints/pt-*` or run dir
  without `--overwrite`. The v0 models (`checkpoints/pt-{weak,mid,strong}-s0`, `runs/pt-*-s0`,
  `data/labels/v0/`) use the same names: move them away before `pretrain_all.sh`.
- **Compiler.** Whole-model `torch.compile` lowers RoPE setup to a CPU kernel that needs a C++20
  host compiler (RHEL 8's g++ 8.5 is too old), so only the decoder layers are compiled.
- `gate.json` checks (`rlvr/gate.py`): E1 greedy >= 0.98 overall and per function (Wilson upper
  bound, 50 problems each), >= 20% of calib problems at depths 2-4 with 0 < p_hat < 1, >= 300
  RL-pool problems with p in [0.4, 0.6] in both depth-2 L and depth-4 H and >= 120 at every
  str_len (p from the second labelling), depth-3 composition loss >= 1e-3, and the level's
  design target. Only `gate.hard_checks` fail the gate (exit 1; the rest are warnings): the first
  three for mid, the first two for midbal, only the E1 check for weak/strong (`configs/pretrain/*.yaml`).
- Second labelling: RL-pool problems in d2-L / d4-H / d5-H with a first-pass p_hat in [0.3, 0.7]
  get 64 more independent samples (`label.relabel_*`, `*_relabel` columns); the matched-p specs
  select on it (`p_label: relabel`) to shrink the winner's curse of selecting on a 32-sample p_hat.

## Reinforcement learning (GRPO)

```bash
# one run: base checkpoint + dataset -> runs/rl-{model}-{dataset}-G{G}-s{seed}[-{tag}]/
python scripts/grpo.py --base checkpoints/pt-weak-s0 --dataset mixed_500 [key=value ...]
python scripts/grpo.py --config configs/rl/smoke.yaml --base checkpoints/pt-weak-s0 --dataset mixed_500   # ~3 min
# a series: checkpoints x datasets x seeds (x grid), one run per GPU at a time, finished runs skipped
python scripts/rl_sweep.py --name repro --gpus 4 5 6 7 \
    --models checkpoints/pt-weak-s0 checkpoints/pt-mid-s0 runs/pt-mid-s0/ckpt/step-2375 \
    --datasets easy_500 mixed_500 depth_3 --seeds 0 1 2
python scripts/rl_sweep.py --name G --gpus 0 1 --models checkpoints/pt-mid-s0 --datasets mixed_500 \
    --grid rl.group_size=4,8,32 --grid rl.lr=3e-6,1e-5        # -> runs/rl_sweep/G/summary.{md,csv}, curves.png
# p-based datasets are per model: build them once the labels exist
python scripts/build_dataset.py configs/datasets/learnability.yaml --model-id pt-weak-s0
```
- **What a run inherits.** Task, model, splits, paths, eval version and length budgets come from
  the base checkpoint's `run_config.yaml`, so the same commands work for chain-v1 and tree-v1
  checkpoints and for intermediate (`ckpt/step-N`) or RL-trained checkpoints; overriding `task.*`,
  `model.*` or `splits.*` is an error. Everything else is `rl.*` / `data.*` (`python -m rlvr.params`).
- **Step.** B problems x G samples at `rl.temperature`, exact-match reward (`rl.reward`), advantage
  = reward - group mean, divided by the group std (`rl.adv_std_norm`, unbiased std + 1e-6); groups
  with identical rewards get exactly 0. One AdamW update (constant lr after `rl.warmup_steps`,
  betas 0.9/0.999, no weight decay) on `-(adv x logp)` over completion tokens only, token- or
  sequence-averaged. `rl.ppo_epochs > 1` switches to the clipped ratio; `rl.kl_coef`,
  `rl.entropy_coef` add the usual terms. The update is split into micro-batches of
  `rl.micro_batch_tokens` with gradient accumulation (a memory knob; the gradient is unchanged).
- **Samplers** (`data.sampler`; `auto` takes the dataset spec's `sampler`): `static` = shuffled
  epochs (`data.order=shallow_to_deep`: the same draws sorted by depth over the run);
  `online_filter` = uniform over problems with unknown p or `p_edge <= p(t) <= 1 - p_edge`;
  `learnability` = proportional to p(t)(1 - p(t)). p(t) starts from the base model's labels
  (`data/labels/{model_id}.parquet`, when present) and follows the rollouts as an EMA
  (`data.p_ema`). `data.explore_frac` (default 0) draws that share of each batch from the whole
  dataset; without it a problem filtered out at p = 0 only returns when too few problems qualify.
- **Logs** (`runs/rl-*/`): `metrics.jsonl` per step (mean_reward, signal_frac, entropy, mean_len,
  trunc_frac, grad_norm, kl_to_base every `rl.kl_every`, reward/signal per depth, sampler state),
  `rollouts.jsonl` one row per (step, problem) with the G rewards, lengths and completions,
  `eval.jsonl` (greedy E1-E4 every `rl.eval_every`, probe every `rl.probe_every`, step 0 and last
  step always; `rollouts` = cumulative count), `summary.json` (written last; marks the run
  finished), `ckpt/step-N` (`rl.save_at`, `rl.save_final`; loadable by `calibrate.py` / as `--base`).
- **Cost (measured, A100, weak round-1 checkpoint, B 64 x G 8).** ~3.3 s/step (2.6 s rollout,
  0.7 s update) at ~100-token completions, ~13 GB; a full E1-E4 greedy eval ~55 s. 500 steps with
  the default eval schedule is ~35-40 min. Deeper datasets are slower in proportion to length.
- **Tests.** `tests/test_grpo.py`: advantages, teacher-forced log-probs equal to generation
  scores, update equal to the hand-written policy gradient, micro-batch invariance, samplers,
  and an end-to-end tiny run. Not done yet: the TRL `GRPOTrainer` cross-check, pass@k, the
  A/B/C_mid curve fits.

## Parameters and registries

All defaults live in `rlvr/params.py`; YAML files and CLI arguments only override them, and
the merged config is validated (unknown keys with suggestions, types, choices).

```bash
python -m rlvr.params            # all parameters: doc id, key, default, levels, priority
python -m rlvr.params --core     # the core (★) variables
python -m rlvr.params --yaml     # full default config
python scripts/make_corpus.py --config configs/pretrain/weak.yaml pretrain.h_ratio=0.5 seed=1
```

- **Add a parameter:** one `param("rl.foo", default, "doc", code=..., levels=...)` line in `params.py`.
- **Change a function definition:** edit `task.functions` (label -> atom spec), e.g.
  `rotl2: {name: rotate, k: 2}`, or `task.functions.max_to_front=min_to_back` on the CLI.
  A dict-valued parameter is replaced whole when given in YAML. Labels and their order fix the
  `Fxx` shuffle, so any change here requires regenerating all data.
- **Add a new atom / reward / weight:** decorate a factory with `@ATOMS.register()` (or
  `REWARDS`, `WEIGHTS`) in any module and list that module under `imports:` in the config.
  Atoms are checked for length preservation when the Task is built.

## Conventions and decisions not spelled out in the design doc

- **Groups.** Each problem is `H` (only H functions), `L` (only L) or `mixed`. Pools and
  eval sets are stratified evenly over depth x group (natural sampling would make depth-4
  problems ~88% mixed). Pretraining compositions: `h_ratio` H-only, the rest contain >=1 L function.
- **Compositional split (E4).** 24 ordered adjacent pairs (6 each HH/HL/LH/LL) never occur in
  pretrain, the RL pool or E1-E3/probe/calib; every E4 problem contains at least one.
- **RL data stops at depth 5**, so E3 (depth 6-8) stays out of distribution. The RL pool
  still has depth 6-8 candidates, for calibration only. Pretraining compositions stop at depth 4.
- **Pretraining stream.** Worker w of rank r draws from `random.Random(f"{seed}-{r}-{w}")`, so
  a run is reproducible for a fixed world size and `num_workers`. Levels are not nested any
  more (v0 corpora were); they share the generator, not the examples.
- **Functions (v1).** H and L have 12 functions each, with every category on both sides
  (position 5/5, character 3/3, sorting 2/1, conditional 2/3). `max_to_front` moved from L to H.
- **Reward** = exact match of the final `A:` line; requires `<eos>`, no credit for format or
  intermediate steps. `reward(kind="partial")` gives step accuracy for the R12 ablation.
- **Pool origins.** `rl_pool` has a balanced base grid (1000 per cell) plus oversampled
  `extra` cells (4000 each of depth-2 L, depth-4 H, depth-5 H) for the matched-p datasets. Specs use
  only `base` unless they set `pool_origin`.
- p-based datasets (`matched_p_*`, `learnability`) are written to `data/datasets/{model_id}/`
  once `data/labels/{model_id}.parquet` exists (columns as in the design doc).
