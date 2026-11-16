# Multi-input string task (tree-v1)

Implemented 2026-11-13. This is a new task recipe, with no trained-model results yet.
The original chain-v1 recipe and historical results remain available. Use
`configs/tasks/tree_v1.yaml` (or `configs/pretrain/tree/pilot.yaml`) to select this task.

## Operator set

There are still 24 display operators, shuffled to F00–F23: 12 unary, 8 binary,
and 4 ternary. H and L each contain 6 unary, 4 binary, and 2 ternary operators.
All inputs within one problem have equal length, sampled from 6–10. Every operator
returns a string of the same length. Symbols are numbered a=0 through z=25;
arithmetic is modulo 26 and indexing is zero-based.

The new recipe replaces `reverse`, `rotl1`, `rotr1`, `rotl2`, `rotl3`,
`swap_ends`, `shift_p1`, `shift_m1`, `shift_p3`, `shift_p5`, `complement`,
and `sort_desc`. These are simple permutations, constant character maps, or
redundant variants. This selection is a design choice, not a measured ranking of
operator learnability: all original operators reached high atomic accuracy.
Their implementations remain for loading legacy configs and checkpoints.

Retained unary operators: `swap_adjacent`, `swap_halves`, `even_pos_p1`,
`odd_pos_m1`, `max_to_front`, `min_to_back`, `mirror_first_half`,
`reverse_first_half`, `riffle`, `shift_by_pos`, `sort_first_half`, `sort_asc`.

| Operator | Arity | Definition |
| --- | --- | --- |
| zip_add | 2 | out[i] = x[i] + y[i] |
| zip_sub | 2 | out[i] = x[i] - y[i] |
| zip_mul | 2 | out[i] = x[i] * y[i] |
| gather | 2 | out[i] = x[y[i] mod n] |
| gather_add | 2 | out[i] = x[y[i] mod n] + y[i] |
| scatter_add | 2 | Start with zeros; add each x[i] into out[y[i] mod n] |
| rotate_by_first | 2 | Rotate x left by y[0] mod n |
| prefix_add | 2 | out[i] = x[i] + sum(y[0:i+1]) |
| zip_add3 | 3 | out[i] = x[i] + y[i] + z[i] |
| zip_mul_add | 3 | out[i] = x[i] * y[i] + z[i] |
| gather_mix | 3 | out[i] = x[y[i] mod n] + z[i] |
| select_parity | 3 | out[i] = x[i] if z[i] is even, otherwise y[i] |

rotate_by_first replaced rotate_by_sum (rotate x left by sum(y) mod n) on 2026-11-16: in pilot
stage A the model did not learn rotate_by_sum (E1 greedy .12 after 2000 atom-only steps; design.md
decision 12). rotate_by_sum stays in `rlvr/tree_ops.py` for loading the first recipe.

These operators have declared arities of one, two, or three; arbitrary unbounded
argument counts are not supported. Addition provides a simple baseline; it may
simplify algebraically. Branching does not guarantee higher difficulty or
intermediate success probabilities. Calibrate the new task before selecting models.

## Generation and identity

Expressions are ordered trees with independent string leaves. Evaluate children
left to right in postorder; each internal node produces one reasoning step.
A completion prints each operator with its already evaluated string arguments,
then the final `A:...<eos>`. The tokenizer appends a comma to the legacy vocabulary
(49 tokens); existing 48-token checkpoint vocabularies are unchanged.

- `depth`: maximum number of operators on a root-to-leaf path.
- `n_ops`: total internal nodes / reasoning steps, capped at 10 by default.
- `n_binary`, `n_ternary`, `n_leaves`: arity counts; leaves = 1 + n_binary + 2*n_ternary.
- `max_live_values`: peak stack values under the specified postorder traversal.
- `shape_id`: ordered topology/arity signature without operators or input values.

For requested depth d, the default sampler chooses uniformly among feasible
operation counts d through min(10, 2*d), then uniformly among ordered feasible
shapes for that size and depth, then samples operators of the required arity.
Leaves share the sampled string length, with independently sampled values.
The Python `sample_problem(..., n_ops=N)` API can fix operation count.
H/L restrictions, PID-pool rejection and held-out-edge rejection further condition
the distribution. Atomic pretraining chooses operators uniformly across all 24.
Both pretraining components are mixed by supervised tokens, not example counts.

JSONL stores the full `expr`, recomputable postorder `trace`, cached answers,
structural features, schema version, and task fingerprint. Loading recomputes
answers and checks supplied metadata. The fingerprint includes the operator
specifications/semantic version, Fxx mapping seed, H/L definition, alphabet,
format/CoT and vocabulary. Different tree shapes are not identified by flattened
operator lists. Syntactic PID separation does not imply semantic-equivalence separation.

For the default shuffled mapping, F22 is `swap_adjacent`,
F23 is `zip_sub`, and F02 is `zip_add`.
This example has depth 2 but three operations:

```text
Q:F02(F22(abcdef),F23(bcdefg,aaaaaa))=
F22(abcdef)=badcfe;F23(bcdefg,aaaaaa)=bcdefg;F02(badcfe,bcdefg)=ccggkk;
A:ccggkk<eos>
```

The usual PID hash pools remain 90% pretrain / 5% RL / 5% eval. Novel combinations
hold out **(parent operator, argument slot, child operator)**, stratified by both
operators' arities and H/L groups (36 edges in this recipe). Sibling operations
are never treated as adjacent edges. E4 contains a held-out edge; pretraining,
RL candidates and all other evaluation sets exclude them.

## Run and inspect

In the `rlvr` conda environment:

```bash
bash scripts/make_tree_data.sh
# Individual generation/preview commands:
python scripts/make_eval_sets.py --config configs/tasks/tree_v1.yaml
python scripts/make_rl_pool.py --config configs/tasks/tree_v1.yaml
python scripts/preview_stream.py --config configs/tasks/tree_v1.yaml --n 5000
python scripts/report_data.py --config configs/tasks/tree_v1.yaml
# Training is a separate action; benchmark first and sweep comp_frac, including 0.
torchrun --standalone --nproc_per_node=4 scripts/pretrain.py --config configs/pretrain/tree/pilot.yaml --bench 100
```

Artifacts go under `data/tree-v1/`, `runs/tree-v1/`, and `checkpoints/tree-v1/`.
Generated data is ignored by git; configs and source are the reproducible inputs.
The data script generates eval sets, the RL pool, a 5,000-example stream preview,
structural RL datasets and a leakage/structure report. Probability-selected
specs are skipped until labels exist. Their historical `model_id` fields must be
replaced with an actual tree checkpoint ID after calibration.

The model context is 1024; eval/RL completion limit is 768. Conservative length
bounds are checked against padding and position offsets before generation or
streaming; no hard examples are silently dropped for length. Previews, training
stats, probability labels, greedy evaluation, and the data report retain
structural counts. Dataset specs can filter/match `n_ops`, `n_binary`, `n_ternary`,
`shape_id`, and `max_live_values` alongside existing fields.

Binary reward still checks only the final answer and EOS. Partial reward divides
by operation count, not tree depth. Noise injection changes one node and
recomputes its ancestors without changing independent branches. Model save/load
checks task fingerprint and vocabulary. The historical `scripts/diag_steps.py`
remains a chain-specific diagnosis tool; use tree traces and structural reports
for the new task.

First validate atomic skills and inspect per-problem success distributions. The
research criterion is enough independently verified, covariate-matched problems
across different structures—not simply lower mean accuracy. Matching depth alone
can confound operation count, branch count and token length. Cross-task chain/tree
scores are descriptive comparisons, since operators, syntax and vocabulary differ.
