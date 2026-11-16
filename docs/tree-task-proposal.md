# Code change proposal for a new data format: string computation trees (tree-v1)

Status: proposal, not yet implemented. The existing string-chain task is called chain-v1.

## 1. Problem to solve

The current models learn composition at depths 2–4 but fall off a cliff at depth 5 and above.
strong is also near-perfect on low-exposure function compositions and on held-out function
pairs, which shows that the current exposure intervention is not enough to produce the groups
matched on success rate (p) that the original design needs. weak has also seen about 27,000
composition problems, so we cannot conclude from it that the model learned composition from
atomic problems alone.

This change tests one specific hypothesis: whether having the model compute several input
branches separately and then merge the intermediate results gives richer, controllable
structural difficulty. Binary operations do not guarantee a harder task, nor do they guarantee
that p≈0.5 appears; the pilot decides both points. A full functional language, recursion, a
type system and ternary operations are left out of the first version.

## 2. Core decisions

| Item | First-version decision |
| --- | --- |
| Input and output | a–z strings; all leaves of one problem have the same length, initially still 6–10 |
| Program structure | Ordered tree with unary and binary operators; no variables, shared nodes or loops |
| Leaves | Each leaf samples its string independently; identical literal values may occur by chance |
| Evaluation order | Children left to right, post-order; one internal node is one step |
| Output length | All operators preserve input length; binary inputs must have equal length |
| Function naming | Semantics-free F00… numbering; arity is written to metadata, and the parenthesis syntax shows the number of arguments |
| Base reward | Keep final-answer exact match + <eos> required |
| Run isolation | New tokenizer, task fingerprint, data directory and model names; keep chain-v1 |
| Model initialization | Main tree-v1 experiments train from random weights; all controls use the same model architecture |

The interface can represent ternary operators, but the first-version generator only accepts
arity=1/2; ternary training is not enabled.

## 3. Operator definitions

Let a=0,…,z=25; character arithmetic is mod 26; positions start at 0 and the string length is n.
The first version validates the pipeline with 4 unary and 4 binary operators first, instead of
expanding to many functions right away.

| Operator label | Arguments | Exact definition | Role |
| --- | --- | --- | --- |
| reverse | 1 | Reverse the string | Reuses the existing implementation |
| rotl1 | 1 | Rotate left by 1 position | Reuses the existing implementation |
| shift_p1 | 1 | Add 1 to each character value, mod 26 | Reuses the existing implementation |
| shift_by_pos | 1 | Add i at position i, mod 26 | Reuses the existing implementation |
| zip_add | 2 | z[i]=(x[i]+y[i]) mod 26 | Simple merge baseline; may simplify algebraically |
| zip_sub | 2 | z[i]=(x[i]-y[i]) mod 26 | Ordered, non-commutative, but still linearizable |
| gather | 2 | z[i]=x[y[i] mod n] | y decides how x is read |
| gather_add | 2 | z[i]=(x[y[i] mod n]+y[i]) mod 26 | Depends on both index and value |

For example, zip_add(abcdef,bcdefg)=bdfhjl; gather(abcdef,fedcba)=fedcba.
At run time the Fxx mapping comes from a fixed seed; the labels in the table above are only for
code and audits.

Initially H={reverse, shift_by_pos, zip_add, gather} and the rest are L; each group has
2 unary and 2 binary operators. H/L stands for composition exposure, not the difficulty of the
operator itself. This split is only a starting point for the pilot: the causal effect of
exposure cannot be inferred directly from H/L differences within one model. We must keep an
h_ratio=0.5 control with the same operator table, and if needed repeat the pilot with the H/L
labels swapped.

## 4. What a problem looks like

The following is only a format example, with the temporary assignment F00=reverse,
F01=shift_p1, F04=zip_add:

```text
Q:F04(F00(abcdef),F01(bcdefg))=
F00(abcdef)=fedcba;F01(bcdefg)=cdefgh;F04(fedcba,cdefgh)=hhhhhh;
A:hhhhhh<eos>
```

Each CoT step writes only the already-computed string arguments and does not expand the whole
subtree; this avoids printing subexpressions over and over. The first version introduces no
register names, to limit the effect of changing the data syntax at the same time. Multiple
occurrences of the same expression are still evaluated and counted as steps separately; there
is no common-subexpression elimination.

JSONL stores the recomputable tree and audit information; it cannot store only the flat funcs:

```json
{
  "schema_version": "tree-v1",
  "task_fingerprint": "<hash-of-semantics-and-display-mapping>",
  "pid": "<hash-of-task-fingerprint-and-ordered-tree>",
  "expr": {
    "op": "F04",
    "args": [
      {"op": "F00", "args": [{"value": "abcdef"}]},
      {"op": "F01", "args": [{"value": "bcdefg"}]}
    ]
  },
  "depth": 2,
  "n_ops": 3,
  "n_binary": 1,
  "n_leaves": 2,
  "max_live_values": 2,
  "shape_id": "B(U(L),U(L))",
  "funcs": ["F00", "F01", "F04"],
  "group": "mixed",
  "str_len": 6,
  "steps": [
    {"path": "0", "op": "F00", "args": ["abcdef"], "value": "fedcba"},
    {"path": "1", "op": "F01", "args": ["bcdefg"], "value": "cdefgh"},
    {"path": "", "op": "F04", "args": ["fedcba", "cdefgh"], "value": "hhhhhh"}
  ],
  "answer": "hhhhhh"
}
```

The example's group is recomputed from the actual Fxx mapping and H/L configuration; under the
temporary mapping above it is mixed.
`from_dict` must recompute the answer, steps and structural statistics from expr and the frozen
task definition, and validate the input metadata; it does not treat the answer cached in the
file as an independent ground truth.

## 5. Record difficulty separately, so "more steps" is not mistaken for "harder trees"

| Field | Definition |
| --- | --- |
| depth | Leaf depth is 0; an internal node has 1+max(child depths) |
| n_ops | Total number of internal nodes, i.e. the number of computation steps in the reference answer |
| n_binary | Number of binary nodes; binary_frac=n_binary/n_ops |
| n_leaves | Number of leaf occurrences; a unary/binary tree must satisfy n_leaves=n_binary+1 |
| max_live_values | Maximum number of results waiting on the stack to be used, under fixed left-to-right post-order evaluation |
| shape_id | Keeps only arity and parenthesis structure, no function names or input values |
| prompt/target_tokens | Computed with the actual tokenizer, used as length covariates |

The recurrence for max_live_values is live(leaf)=1; a node with k children takes
max((j-1)+live(child_j)), with j starting at 1. It measures the stack demand under the agreed
evaluation order and does not claim to equal the model's actual internal memory.

The generator explicitly specifies a feasible (n_ops, depth, n_binary), samples from the shapes
that satisfy the constraints, then fills in operators and leaves; in the first version
n_ops<=8, so ordered shapes can be enumerated and cached.
It does not first grow long random trees recursively and then truncate them, since that would
make the shape distribution impossible to explain. Infeasible combinations are rejected with a
stated reason; each sampling cell has a maximum number of attempts and cannot loop forever.
There is no need to cover the full Cartesian product.

Comparisons are reported in three classes:

1. Same n_ops, same n_binary, same string length, different depths/shapes: checks the effect of
   dependency structure.
2. Same n_ops, same string length, different n_binary: checks the overall effect of adding
   branches; leaf count and input length change along with it and must be reported, and we
   cannot claim to have isolated the effect of arity.
3. Within one distribution, compare different RL data recipes at a fixed budget: report the
   number of rollouts, generated tokens, optimizer steps and wall time together. Equal rollouts
   do not mean equal compute.

Results across chain-v1 and tree-v1 can only be a descriptive comparison of different
benchmarks: the operator table, the number of inputs and the tokenizer all change, so the
performance gap cannot be attributed entirely to binary operations.

## 6. Generation and quality audit

Each generation first picks the atomic/composition component, then the structure cell, operator
exposure group, string length and leaves.
An atomic problem is defined as n_ops=1, including unary problems and binary problems that take
two leaves directly.
A composition problem is defined as n_ops>=2; depth>1 can no longer be used to decide "how many
operations" there are.
P4 still controls the composition share by supervised tokens; P6 is still defined as the
probability that a composition problem is H-only, and the other problems contain at least one L.
The depth=1 atomic set must test each operator one by one; binary operators must not be missing.

The quality report includes:

- Sample and supervised-token counts by operator/arity, structure cell, H/L and input length.
- Detection of clear local simplifications: reverse(reverse(x)), merging of linear shifts, etc.
  Only proven static rules are used to remove problems, and atomic problems are never removed;
  we cannot claim to have caught every simplifiable program.
- On an independent audit sample, perturb each binary argument one at a time and measure the
  fraction of cases where the output changes, to find arguments that are persistently ignored.
  The first round only records this and does not filter problems by per-sample perturbation
  results, to avoid artificially changing the distribution.
- Record the fraction of root answers equal to a leaf/child value, output character diversity,
  duplicate pids and rejection reasons.
  A chance equality is not automatically a shortcut, and such problems are not deleted wholesale.
- Save the full raw responses and report the actual number of output steps, the first wrong node
  and the fraction that emit A early.
  These are diagnostic metrics; the final reward is still computed from the answer.

## 7. Data isolation and composition hold-out

New data goes in data/tree-v1/, which keeps meta, eval/v1, rl_pool, datasets and labels
underneath.
New models use pt-tree-* names. Old data and checkpoints remain loadable.

pid is computed from the canonical JSON serialization of the ordered tree plus the task
fingerprint; children are not sorted, and an operator's arguments are not swapped even if it is
commutative. The fingerprint includes the operator semantic versions, parameters, the Fxx
mapping, the alphabet, the H/L definition, and the format and tokenizer versions. The random
seed, sampling mix and training amount are not part of a problem's identity.
Different trees that are semantically equivalent may have different pids; the first version
guarantees isolation of syntactic instances, not of semantic equivalence classes.

Keep the 90%/5%/5% split by pid hash. The composition hold-out changes to real tree edges
`(parent_op, child_slot, child_op)`:

- Only check edges whose child is itself an operator; leaves do not form operator pairs.
- Check every edge; do not check adjacent entries of the post-order funcs, since those may come
  from sibling nodes.
- Held-out edges do not enter pretraining, the RL pool or E1–E3/probe/calib; each E4 problem
  contains at least one.
- Choose held-out edges stratified by parent/child arity and H/L; check that the remaining
  connections can still generate the target shapes.
- First build candidate trees and keep a manifest of the fingerprint, held-out list and
  exclusion rules; check all outputs.

A tree-shape hold-out can be a later E5; the first version does not enable it at the same time,
so that E4 stays interpretable.

## 8. Code change list

| File | Planned change |
| --- | --- |
| New rlvr/tree_ops.py | OperatorSpec(arity, semantic_version, factory) and a multi-arity registry; unary operators adapt the existing ATOMS |
| New rlvr/tree_task.py | Leaf/Call, TreeProblem, type and length checks, interpreter, post-order trace, formatting and parsing |
| New rlvr/tree_sampling.py | Enumeration of feasible tree shapes, controlled sampling, static simplification checks and an independent quality audit |
| rlvr/config.py | A task.kind factory selects the chain/tree Task and the matching Splits; remove direct Task(cfg) calls from other scripts |
| rlvr/task.py | Keep the old Task/Problem; extract a shared protocol instead of stitching optional x/expr into one mixed object |
| rlvr/splits.py | Keep the chain rules; add TreeSplits or a separate tree_splits.py that implements tree-edge hold-out with argument positions |
| rlvr/sets.py | Support structure cells and per-operator atomic evaluation; stop treating depth as the operation count |
| rlvr/stream.py | Call the task sampling interface; P4 by tokens; collect n_ops/arity/shape statistics; tree noise is applied per node and recomputes all its ancestors |
| rlvr/tokenizer.py | Select the vocabulary through the task format; tree-v1 adds a comma, 48→49 tokens; the chain-v1 vocabulary is unchanged |
| rlvr/rewards.py | binary stays generic; partial compares along the node trace with denominator n_ops and does not reuse p.depth |
| rlvr/io.py | Versioned deserialization and metadata validation; save expr and structural features |
| rlvr/evaluate.py | Aggregate by depth/n_ops/arity/shape; E1 by root operator; optionally save raw responses and error traces |
| rlvr/model.py | Save and load model/tokenizer together and validate the task fingerprint; forbid mismatching with the old vocabulary |
| rlvr/params.py | Register the new config keys; validate structure cells and budget/version combinations |
| scripts/make_eval_sets.py, make_rl_pool.py | Call the task factory; use the new metadata and structural stratification; do not assume unary funcs/x |
| scripts/preview_stream.py, report_data.py | Add tree examples, structure distributions, length and dependency audits |
| scripts/pretrain.py | Extend logging by n_ops/arity; validate data and model fingerprints; check actual sequence lengths |
| scripts/calibrate.py | Do not instantiate the old Task directly; labels keep the structure columns, and gates are checked per configurable structure cell |
| scripts/build_dataset.py | Add structure fields to SPEC/CELL_KEYS; match on n_ops/str_len etc.; keep p-based sampling |
| scripts/sweep.py | Sweep budgets and recipes for the pilot; keep a single selection rule and all candidate results |
| New configs/tasks/tree_v1.yaml etc. | Freeze the new task, structure grid, evaluation and pilot recipes; keep separate from the old configs |

The shared task protocol provides at least sample_problem, from_dict, format_prompt,
format_target, parse_answer, reward, describe, plus interfaces for atomic enumeration and
structural statistics.
The structural constraints for tree generation use a separate SamplingSpec; the old depth API
is taken by a chain adapter.
steps is the node trace; funcs is only for exposure statistics and can no longer serve as the
full program or the list of tree edges.

## 9. Mapping between config and the original design variables

The following are proposed new/changed keys, not config that the current code already accepts.

| Key | Purpose / relation to design |
| --- | --- |
| task.kind: tree | Selects the new task; the old default stays chain |
| task.version: tree-v1 | Format and semantics version |
| task.operators | Operator specs, parameters, arity, semantic version; distinct from the old task.functions |
| task.h_group | P6; both groups must be covered within each arity |
| pretrain.structure_cells | P5 extension: feasible cells of n_ops/depth/n_binary/frac |
| pretrain.h_only_depths | Set to [] in the first tree version, to avoid an extra coupling of depth and H; an explicit change as P5b |
| pretrain.comp_frac | P4, still the composition share of supervised tokens |
| pretrain.h_ratio | P6, still the probability of H-only problems in non-forced cells |
| pretrain.steps, save_at | P9; keep the existing 4750-step half-budget starting point and keep intermediate checkpoints |
| task.str_len, alphabet, cot | P7/P8; keep the defaults but re-freeze the task fingerprint |
| structure cells in data | D2 extension; D6 adds input/output token and operation-count covariates |
| splits.heldout_edges | New list of directed held-out edges with child_slot |
| gate.spread_cells, matched_cells | Replace the depth-only check; specify the structure cells actually used in experiments |

Suggested first version: composition n_ops=2–4; RL candidates n_ops=2–6; evaluation at larger
operation counts 7–8.
Pick feasible structure cells within each range; do not lump "operation-count extrapolation" and
"depth extrapolation" together.
E3 reports separately the depths and n_ops not covered in training, and the subset where neither
is covered.
The exact structure mix is set from a preview first; do not force in depth/branch combinations
that cannot be generated.

The research meaning of P1/P2/P2b, P10/P11, D1/D3/D4/D7–D10 and R1–R13 is kept.
P3's atomic coverage extends to all arities. D5 still introduces no eval-leakage intervention;
the M/Y metrics gain structure slices.
GRPO is not implemented yet; this proposal does not merge "add the tree task" and "finish GRPO"
into one change.

## 10. Budget and evaluation gates

First generate a preview and measure the longest prompt/target and the length after padding,
then set model.max_len, eval/rl.max_new_tokens and the position offset; do not copy the old
512/256.
Verify the length upper bound for every declared feasible structure cell; silent truncation and
dropping the hardest samples by length are forbidden.
The pilot keeps the existing 85M architecture where possible; if the context must change, change
it for all tree controls together.

A fixed supervised-token budget lets us compare optimization effort, but multi-input problems
have longer prompts, so compute cost still differs; therefore also record total non-padding
tokens, supervised tokens, padding, throughput and GPU time.

There are two kinds of gates:

- Functional gates: greedy accuracy ≥98% for each unary/binary atomic operator; no cross-pool pid
  leakage, no held-out-edge leakage into training, no truncation, no version mixing.
- Research feasibility: pre-specify a shallow and a deep structure cell, each with at least 300
  problems with p∈[0.4,0.6], while matching the feasible covariates; report the distributions
  before and after matching. Matching failures are not hidden, and cells are not chosen by E3
  score.

Overall training loss<1e-4 is no longer a standalone veto condition; it is easily dominated by
the many atomic problems already learned.
Instead, report per-operator and per-structure training loss, accuracy on an independent set and
the p distribution; use calibration to judge saturation.
Binning with K=32 is noisy; recheck the final matched candidates with fresh independent samples,
e.g. 96 extra, keep both the initial and recheck results, report intervals and the difference in
p distributions, and do not treat p_hat as the true p.

## 11. Implementation order and acceptance

1. **Implement semantics and the file format.** Finish operators, AST, interpreter, tokenizer,
   fingerprint. Verify known answers, alphabet wraparound, argument order, invalid
   arity/unequal-length inputs, JSON round-trips and regression on the old task.
2. **Implement generation and isolation.** Walk all feasible structure cells on small samples;
   cross-check an independent recursive interpreter against the post-order interpreter; verify
   structural statistics, parent-child edge hold-out, cross-pool isolation, termination on
   sampling failure and length upper bounds.
3. **Wire up streaming training, evaluation and labeling.** Verify the format and label masking
   on CPU; run a short training on an available GPU to confirm the whole chain works: model
   save/load, generation, per-operator E1, raw-response diagnostics and calibration.
4. **Run the pilot, then decide the final recipe.** With the same tree operator table, compare a
   comp_frac=0 atomic control and 0.001/0.01/0.1; at least for the middle recipe, compare
   h_ratio=0.5/0.9. Debug with a short budget first, then validate candidates with a common
   4750-step budget, and pick models by a single checkpoint rule. Do not take a lower E3 score
   of some candidate as evidence of "harder, better".

Evidence that the pilot passes: atomic skills are learnable, there are enough problems with
intermediate per-problem success rates, structure cells can be matched, and raw responses can
explain the main error sources. If it is still all-or-nothing, first tell apart operators that
cannot be learned, execution-progress errors, length limits and insufficient structural
exposure, before deciding to add ternary operations or a fuller language.

The final deliverables are the new task implementation, frozen metadata, reproducible experiment
configs, a data quality report, tests, small-scale pilot results and a descriptive comparison
with chain-v1; this file only defines the implementation plan.
