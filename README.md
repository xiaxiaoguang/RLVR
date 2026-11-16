# RLVR data-difficulty experiments

**Status (see `docs/design.md` "Status"): the four v1 base models are trained, calibrated and on
Hugging Face (`rlvr-cs291k/rlvr-pretrain-v1`; results in `docs/pretrain_v1_results.md`). v2 (binary
operators, tree problems, eval v3) is implemented and goes through the pilot (stages A and B) before
the base models are retrained.** GRPO and RL-side analysis are not implemented yet.

## Environment

```bash
conda env create -f environment.yml && conda activate rlvr     # python 3.11 + pip install -e .[dev]
# exact versions instead: pip install -r requirements.lock.txt && pip install -e . --no-deps
```
Dependencies live in `pyproject.toml`; `requirements.lock.txt` pins the full resolution
(torch 2.14.1 / CUDA 13 runtime, transformers 5.20.2, trl 1.16.0, pandas 3.1.0).

```
rlvr/                 library (frozen interface after review)
  task.py             24 atomic functions (+ v2 binary operators), Problem (chain or tree), sampling,
                      format_prompt/target, parse_answer, reward, structure variables, is_skippable
  tokenizer.py        char-level tokenizer, 48 tokens (v2: 49 with `,`); encode_example() masks the prompt with -100
  registry.py         generic name -> factory Registry (build from 'name' or {name: ..., **params})
  params.py           parameter registry: every config key, default, levels, priority, doc id (P1..R13)
  atoms.py            ATOMS registry: parameterized atomic functions (rotate(k), shift(k), sort(descending), ...);
                      BINARY registry (v2): add, sub, alternate, splice, maximum, minimum
  rewards.py          REWARDS registry: binary, partial
  splits.py           pid-hash pools (pretrain 90% / rl 5% / eval 5%) + 24 held-out function pairs
  sets.py             stratified (depth [x n_bin] x group, balanced over input length) set generation;
                      cell_candidates (per-cell seeded candidates for the v2 pilot)
  pilot.py            v2 stage B: cells with the most p ~ 0.5 problems, shared-length-mix supply, go criterion
  stream.py           streamed pretraining corpus: Mixture (comp_frac by tokens, depth mix, H/L,
                      leak-free rejection) and PretrainStream (length-bucketed micro-batches)
  difficulty.py       p bins, Wilson CI, P_sig(p, G); WEIGHTS registry (uniform, learnability, signal)
  config.py, io.py    defaults <- YAML (`base:` inheritance) <- key=value overrides, validated; jsonl helpers
  model.py            LlamaForCausalLM (RoPE, 12 x 768, ~85M) build/save/load; generate() (left pad, cut after <eos>)
  evaluate.py         greedy_eval (eval.jsonl rows, E1 per function); label_pass_rates (p_hat, Wilson CI)
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
  push_hf.py          upload final checkpoints (+ calib files, optional RL-pool labels) to a Hugging Face repo
configs/              overrides only: pretrain/{weak,mid,strong,midbal,pilot}.yaml (v1), v2/{base,stage_a,mid,smoke}.yaml,
                      datasets/*.yaml
tests/                python -m pytest tests/ (test_task, test_registry, test_model, test_gate, test_tree)
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

## v2: tree problems (design.md "v2 revision", "v2 pilot")

`configs/v2/base.yaml` adds four binary operators (F24-F27: add, alternate, splice, max), trees in
the corpus (`pretrain.tree_frac` 0.5, `pretrain.tree_shape_mix`), eval v3 (`data/eval/v3/`, metadata
in `data/eval/v3/meta/`) and the pilot pool (`rl_pool.mode: pilot`). With the v1 configs every chain,
pid, eval set, RL-pool row and pretraining example is exactly as before (`tests/test_tree.py`).

```bash
python scripts/make_eval_sets.py --config configs/v2/base.yaml          # eval v3 (already in git)
# pipeline check, stage A (operators learnable?), stage B (one run per comp_frac), one node each:
CONFIG=configs/v2/smoke.yaml CALIB_ARGS="--calib-per-cell 4" bash scripts/slurm/model_pipeline.sh v2-smoke
CONFIG=configs/v2/stage_a.yaml CALIB_ARGS=--e1-only bash scripts/slurm/model_pipeline.sh v2a-atoms
CONFIG=configs/v2/mid.yaml TRAIN_ARGS=pretrain.comp_frac=0.00032 bash scripts/slurm/model_pipeline.sh v2mid-cf0.00032
# -> runs/pt-v2mid-cf0.00032-s0/calib/pilot.json: supply per chosen cell and per D/E pair, "go"
```

- **Problems.** A tree stores its operations in post-order with operand refs (`args`: >= 0 = an
  earlier step, < 0 = a leaf); `depth` is the number of operations (n_ops) for every problem. A
  problem without binary operators is a v1 chain (same representation and pid). Leaves are
  independent random strings of one length.
- **Shapes.** `Task.sample_tree(n_ops, n_bin)`: a random binary skeleton with n_bin binary nodes
  (left subtree size uniform), then each unary function on a uniformly random edge (above the root,
  a binary node or a leaf).
- **Groups.** H / L / mixed from the unary functions only; a tree without unary functions is `bin`.
- **Cells.** Eval and calib cells are (n_ops, n_bin, group); calib covers n_ops 1-8 x n_bin 0-3 (77
  cells x 200). Gate spread pools chains and trees per n_ops.
- **Pilot pool.** From the calib labels, the D/E pairs (E >= D + 2 operations) with the largest
  share of non-skippable p_hat in [0.4, 0.6]; their cells (<= 4) get 10,000 candidates each, drawn
  from a per-cell random stream so every model sees the same first N candidates of a cell.

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
