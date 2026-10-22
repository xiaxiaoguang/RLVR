# RLVR data-difficulty experiments

**Status: step 1 (task, data, datasets) done.**
Pretraining, p-labelling, GRPO and evaluation are not implemented yet.

## Environment

```bash
conda env create -f environment.yml && conda activate rlvr     # python 3.11 + pip install -e .[dev]
# exact versions instead: pip install -r requirements.lock.txt && pip install -e . --no-deps
```
Dependencies live in `pyproject.toml`; `requirements.lock.txt` pins the full resolution
(torch 2.14.1 / CUDA 13 runtime, transformers 5.20.2, trl 1.16.0, pandas 3.1.0).

```
rlvr/                 library (frozen interface after review)
  task.py             16 atomic functions, Problem, sampling, format_prompt/target, parse_answer, reward
  tokenizer.py        char-level tokenizer, 32 tokens; encode_example() masks the prompt with -100
  registry.py         generic name -> factory Registry (build from 'name' or {name: ..., **params})
  params.py           parameter registry: every config key, default, levels, priority, doc id (P1..R13)
  atoms.py            ATOMS registry: parameterized atomic functions (rotate(k), shift(k), sort(descending), ...)
  rewards.py          REWARDS registry: binary, partial
  splits.py           pid-hash pools (pretrain 90% / rl 5% / eval 5%) + 16 held-out function pairs
  sets.py             stratified (depth x group) set generation
  difficulty.py       p bins, Wilson CI, P_sig(p, G); WEIGHTS registry (uniform, learnability, signal)
  config.py, io.py    defaults <- YAML (`base:` inheritance) <- key=value overrides, validated; jsonl helpers
scripts/
  make_eval_sets.py   data/eval/v1/{E1,E2,E3,E4,probe,calib}.jsonl + data/meta/
  make_rl_pool.py     data/rl_pool/candidates.jsonl (to be labelled with p per model)
  make_corpus.py      data/pretrain/pt-{level}-s{seed}/corpus.parquet (+ stats.json)
  build_dataset.py    data/datasets/{name}.jsonl from configs/datasets/*.yaml
  report_data.py      data/REPORT.md: distributions, degeneracy, leakage checks, examples
  make_all_data.sh    regenerate everything
configs/              overrides only: pretrain/{weak,mid,strong}.yaml, datasets/*.yaml (dataset specs)
tests/                python tests/test_task.py; python tests/test_registry.py (or pytest)
```

Regenerate: `bash scripts/make_all_data.sh` (deterministic; `SEED=1` for other corpus seeds).

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
- **Compositional split (E4).** 16 ordered adjacent pairs (4 each HH/HL/LH/LL) never occur in
  pretrain, the RL pool or E1-E3/probe/calib; every E4 problem contains at least one.
- **RL data stops at depth 4**, so E3 (depth 5-6) stays out of distribution. The RL pool
  still has depth 5-6 candidates, for calibration only.
- **Pretraining budget.** Every level has the same supervised-token budget (64M target
  tokens). Compositions are unique (`comp_repeat` times each) and atoms fill the rest.
  Corpora with the same seed are nested (weak comps ⊂ mid ⊂ strong) and share identical atoms.
  `corpus.parquet` stores unique rows with a `count` column; the trainer expands and shuffles.
- **Reward** = exact match of the final `A:` line; requires `<eos>`, no credit for format or
  intermediate steps. `reward(kind="partial")` gives step accuracy for the R12 ablation.
- **Pool origins.** `rl_pool` has a balanced base grid (1200 per cell) plus oversampled
  `extra` cells (4000 x depth-2 L, 4000 x depth-4 H) for the matched-p datasets. Specs use
  only `base` unless they set `pool_origin`.
- p-based datasets (`matched_p_*`, `learnability`) are written to `data/datasets/{model_id}/`
  once `data/labels/{model_id}.parquet` exists (columns as in the design doc).
