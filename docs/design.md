# RLVR data-difficulty experiment design

Oct 22, 2026 · Hanzhong Cao · v1 revision Nov 2, 2026 · v2 revision Nov 16, 2026 (draft; tree-v1 task by Hanzhong Cao)

This file is the team's shared record of settings, results and decisions, kept up to date so it can be turned into
the paper. When a setting changes, a result comes in or a decision is made, update this file in the same commit and
add the decision to the [decision log](#decision-log). Round-by-round pretraining details stay in
[`pretrain_v1_results.md`](pretrain_v1_results.md); this file keeps only the summary.

Team: Yupeng Su (pretraining), Hanzhong Cao (RL post-training and laws), Jay Lesner (evaluation and analysis).

## Status (2026-11-18)

- **v1 base models:** all four candidates are trained and fully calibrated
  ([v1 results summary](#v1-results-summary)). They stay available as chain-only reference models and for building
  the RL code. The four checkpoints, with their `calib/` files and RL-pool labels, are on Hugging Face in the private
  org repo [`rlvr-cs291k/rlvr-pretrain-v1`](https://huggingface.co/rlvr-cs291k/rlvr-pretrain-v1) (one folder per run;
  uploaded 2026-11-15 with `scripts/push_hf.py`; ask Yupeng for an org invite). Backup copy on CFS:
  `/global/cfs/cdirs/m4788/yupengsu/rlvr/{checkpoints,data/labels}/`.
- **v2 task: tree-v1** (decision 10). On 2026-11-15 we decided to add multi-input operators, so problems become trees
  instead of chains ([v2 revision](#v2-revision-tree-problems)). On 2026-11-16 we adopted Hanzhong's tree-v1 recipe
  ([`tree_v1.md`](tree_v1.md): 12 unary + 8 binary + 4 ternary operators). An earlier variant on the pretrain branch
  (24 unary + 4 binary operators) passed its stage A and was dropped during stage B; its results stay in
  [`pretrain_v2_results.md`](pretrain_v2_results.md).
- **v2 base models (handed over 2026-11-18):** the [v2 pilot](#v2-pilot) round 2 is taken as the v2 pretraining
  result. `pt-tree-mix-cf0.0035-s0` (with its pilot pool: labels, candidates, `pilot.json`) and
  `pt-tree-mix-cf0.006-s0` are on Hugging Face in the private org repo
  [`rlvr-cs291k/rlvr-pretrain-v2`](https://huggingface.co/rlvr-cs291k/rlvr-pretrain-v2), with a CFS copy in
  `/global/cfs/cdirs/m4788/yupengsu/rlvr/{checkpoints,runs,data}/tree-v1/`. Both pass the gate; the go criterion is
  narrowly missed (below). Open for the team: use these models for v2 (with 405 E problems per arm, more d4-H
  candidates, or another comp_frac) or fall back to the v1 models with Jay's pool expansion.
- **RL:** GRPO (`scripts/grpo.py`) and the sweep runner (`scripts/rl_sweep.py`) are implemented and smoke-tested on
  the v1 round-1 checkpoints ([GRPO](#grpo-hanzhong)); no RL experiment of the design has been run yet.
- **Tree-v1 stage A (2026-11-16):** rotate_by_sum was not learned (E1 .12) and is replaced by rotate_by_first
  (decision 12); on the revised table **all 24 operators pass** (E1 .997), so the operator table is fixed.
- **Tree-v1 stage B (2026-11-16): no D/E pair** at comp_frac 3.2e-4 / 6.4e-4 / 1.28e-3 (best: depth 2 mean p .39,
  depth 3 .05, depth 4 .002 at 1.28e-3), so the go criterion is not met in this round. **Round 2 (2026-11-17, decision 13):** with more depth-4
  compositions (depth mix {2: .3, 3: .35, 4: .35}) both finished models pass the gate; the best D/E pair is d2-L / d4-H
  at comp_frac 3.5e-3. Its pilot pool gives **405** in-range problems for E (d4-H) from 10,000 candidates and 667–961
  for the D cells, so the go criterion (≥ 500) is **not met**
  ([`pretrain_v2_results.md`](pretrain_v2_results.md)).
- **Compute:** 39.92 of 50 NERSC GPU node-hours used as of 2026-11-18 (the rest is for RL).

## Decision log

Newest first. "Proposed by" is who raised it in the team discussion.

| # | date | question | decision | proposed by |
| --- | --- | --- | --- | --- |
| 13 | 2026-11-17 | Stage B round 1 found no D/E pair: the depth curve is too steep (mean p d2 .39, d3 .05, d4 .002 at comp_frac 1.28e-3), and raising comp_frac alone shifts the curve without flattening it (d4 near .5 means d2 near 1; depth 5 is never trained, so d3/d5 cannot pair either) | Round 2 changes the **composition depth mix** from {2: .6, 3: .35, 4: .05} to **{2: .3, 3: .35, 4: .35}** (`configs/pretrain/tree/mix.yaml`) at comp_frac 2e-3 / 3.5e-3 / 6e-3; per supervised token this gives ~9–27× the depth-4 examples of round 1's 1.28e-3 run and 0.6–1.9× the depth-2 ones. v1's way of pairing d2-L with d4-H (h_ratio .9 and H-only depth 4) is not used: it makes D and E differ in exposure as well as depth, the confound tree-v1 removed (h_ratio .5, h_only_depths [] stay). Round 2 trains and calibrates only (`--skip-pool`); the 10,000-candidate pilot pool is labelled afterwards, on one node, for a model whose calib labels have a D/E pair. | Yupeng |
| 12 | 2026-11-16 | rotate_by_sum is not learned in tree-v1 stage A (E1 .12, no progress from step 1000 to 2000) | Replace it with **rotate_by_first** (rotate x left by y[0] mod n): still a rotation set by the other operand, but it reads one symbol instead of a sum over the whole string mod n (a parity-like reduction). It keeps id F10 and its place in H, so H and L keep 6 / 4 / 2 operators. Stage A is re-run on the revised table; the first recipe's generated data is kept in `data/tree-v1-rotate_by_sum/`. | Yupeng |
| 11 | 2026-11-16 | How are the D/E cells defined in tree-v1, where depth and n_ops differ? | Fixed before stage B, applied to its data ([Matched cells](#matched-cells-v2)): a cell is (depth, group) as in the calib set; D/E pairs need E.depth ≥ D.depth + 2 within RL depths 1–5; pairs are ranked by the smaller of the two cells' calib shares of non-skippable problems with p̂ in [0.4, 0.6]. Depth is the axis because H2 is tested on E3 (depth 6–8); n_ops, n_binary, n_ternary, max_live_values and token length are reported for D and E as covariates, and input length is matched. | Yupeng |
| 10 | 2026-11-16 | Which operator set and code for the tree task? | Hanzhong's **tree-v1** ([`tree_v1.md`](tree_v1.md)): 12 unary + 8 binary + 4 ternary operators reshuffled to F00–F23, depth = dependency depth, held-out (parent, slot, child) edges, data under `data/tree-v1/`. It replaces the pretrain-branch variant (24 unary + add / alternate / splice / max), whose stage A passed and whose stage B was cancelled after 36 min (1.81 node-hours). The pilot plan (stages A and B, go criterion) is unchanged. | Hanzhong (chosen by Yupeng) |
| 9 | 2026-11-15 | Where do checkpoints go? | Hugging Face model repo, one folder per run in `save_model` format (`scripts/push_hf.py`); too large for git. | Yupeng |
| 8 | 2026-11-15 | Where do we record settings, results and decisions? | This file, in English, as the shared record for the paper. | Hanzhong |
| 7 | 2026-11-15 | Deeper test set is too hard (E3 greedy ≤ .003 for every v1 model, including comp_frac 1e-2) | Report depth-5 accuracy separately and add a step score on E3 (correct steps before the first mistake), so H2 does not "fail" only because the test is at 0. | Jay |
| 6 | 2026-11-15 | Problems that still come out right with a step skipped | Exclude them from D and E (rule in [Matched cells](#matched-cells-v2)). | Jay |
| 5 | 2026-11-15 | Not enough p ≈ 0.5 problems in the matched cells (open question 2) | **v2: add binary operators** (tree problems) and pilot them first. Expanding the v1 RL pool (Jay's proposal, ~2–2.5 node-hours) is the fallback if the pilot fails the go criterion. | Hanzhong (chosen by Yupeng) |
| 4 | 2026-11-15 | strong: accept 8.3e-4 or try 9e-4 / 9.5e-4 (open question 4) | Keep 8.3e-4 for v1. d4 sits on a plateau from 5e-4 to 8.3e-4 and jumps to .681 at 1e-3, so another round would likely overshoot; strong is not used for D vs E. | Jay |
| 3 | 2026-11-15 | midbal's matched_cells (open question 3) | Warning only. midbal practises L chains 5× more than mid, so d2-L being easy (.870) is the intended effect of P6. D vs E runs on mid. Proposed RL role of midbal: mid vs midbal on the uniform pool (A). | Jay |
| 2 | 2026-11-15 | Which depths spread checks (open question 1) | Depths 2–4 only; drop the optional E′ (depth 5, H). No v1 run gets above 6.5% signal at depth 5, against a 20% threshold. | Jay |
| 1 | 2026-11-05 | How to choose comp_frac | Sweep at the full training length; the 400-step pilot extrapolation overestimated comp_frac by ~10×. | Yupeng |

## Revision history

### v2 revision: tree problems

**Why.** v1 trained all four base models on target (weak d3 .114, mid d3 .381, strong d4-H .594), but the decoupling
datasets D and E are hard to build from chains:

1. Per-problem p is bimodal. Even in cells whose mean p is near 0.5, at most 12.5% of problems have p in [0.4, 0.6].
2. With 5,000 candidates per matched cell, mid has only 31 and 18 d4-H problems at lengths 7 and 9 (≥ 120 needed).
   Relaxing thresholds cannot fix 18.
3. Chains give one structural knob (length of the chain). The same chain structure sampled more often keeps the same
   bimodal p.

Multi-input operators (`merge(f(x), g(y))`) turn problems into trees, which separates the number of operations from
the dependency depth and from the shape. That gives more cells in which shallow and deep problems may have
overlapping p. This has to be checked with a pilot before the base models are retrained.

**What changes** (tree-v1, [`tree_v1.md`](tree_v1.md); decision 10):

| aspect | v1 (chain-v1) | v2 (tree-v1) |
| --- | --- | --- |
| problem structure | chain of unary functions | ordered expression tree, evaluated in left-to-right post-order |
| operators | 24 unary (H/L 12 each) | 24: 12 unary (12 v1 functions dropped) + 8 binary + 4 ternary, reshuffled to F00–F23; H and L each 6 / 4 / 2 |
| structure variables | depth | `depth` (operators on the longest root-to-leaf path), `n_ops` (≤ 10), `n_binary`, `n_ternary`, `n_leaves`, `max_live_values`, `shape_id`, token length |
| tokenizer | 48 tokens | 49 tokens (adds `,`) |
| inputs | one string, length 6–10 | one string per leaf; all leaves of a problem have the same length (6–10) |
| pretraining corpus | atoms + chains (depth 2–4, depth 4 H-only) | atoms (all 24 operators, uniform) + trees of depth 2–4 (`comp_depth_mix`); pilot recipe without H-only depths, `h_ratio` 0.5 |
| held-out structure (E4) | 24 parent → child function pairs | 36 (parent, argument slot, child) edges, stratified by arity and H/L |
| lengths | `max_len` 512, max generation 256 | `max_len` 1024, max generation 768 |
| data and runs | `data/`, `runs/`, `checkpoints/` | `data/tree-v1/`, `runs/tree-v1/`, `checkpoints/tree-v1/`; rows and checkpoints carry a task fingerprint |
| matched cells | d2-L vs d4-H | chosen from the pilot among structure cells |
| base models | weak / mid / strong / midbal | same four levels, retrained from random weights |

Not changed: research questions, hypotheses, model, optimiser, GRPO, decision rules.

**Superseded variant (2026-11-15).** Before decision 10 the pretrain branch implemented another v2: the 24 v1 unary
functions unchanged plus four binary operators (add, alternate, splice, max as F24–F27), `depth` = `n_ops`, eval v3.
Its stage A passed (every operator 50/50 on E1) and its stage B was cancelled. Code and data are in git history
(commits cb99f24 to c137d10); results in [`pretrain_v2_results.md`](pretrain_v2_results.md).

### v1 revision (Nov 2): larger pretraining

v0 (4 layers / 256 dims / ~3M parameters, 64M supervised tokens, ~3 minutes on one GPU) had two problems:

1. **Too easy, too little compute.** Training loss fell to ~1e-9 within one epoch, i.e. the model memorised the corpus.
   With alphabet 10 and length 5 there are only 1.6M possible atomic problems, nothing that forces generalisation.
2. **Depth cliff.** Pretraining only saw depth 2, and `pt-weak-s0` had p ≈ 0 at depths 3/4 (d3 mean 0.004, d4 mean
   0.0005); the gate failed on spread and matched_cells. The mid model had E2 = 0.333, i.e. it solved only depth 2. No
   deep problem had p ≈ 0.5, so the decoupling experiment was impossible.

| aspect | v0 | v1 |
| --- | --- | --- |
| compute target | ~3 min on one GPU | **≥ 1 h on 4×A100 per model**; 4 models |
| model | 4 layers, d 256, ~3M | **12 layers, d 768, 12 heads, FFN 2048 (SwiGLU), ~85M non-embedding parameters** |
| alphabet / length | a–j / fixed 5 | **a–z / 6–10 variable** |
| atomic functions | 16 | **24** (H/L 12 each) |
| depth range | 1–6 | **1–8** |
| pretraining composition depth | 2 only | **mainly 2–3, plus a little H-only depth 4** |
| corpus | pre-generated parquet, repeated by `count` | **streamed online, never repeated, composition set by token share** |
| strength knob | `n_comp`: 2k / 20k / 200k | **`comp_frac`** (share of supervised tokens from compositions), set by sweeps |
| training tokens | 64M | **~2.6B (incl. prompts), ~1.8B supervised** |
| base models | weak / mid / strong | **weak / mid / strong + midbal (h_ratio = 0.5)** |

## v1 results summary

Full details, including every sweep point, the d4 length diagnosis and compute per job:
[`pretrain_v1_results.md`](pretrain_v1_results.md).

85M model, 9500 steps (~1.84B supervised tokens), seed 0, pos_offset_max 128. Calibration: 200 problems per cell,
T = 1.0, k = 32; mean p̂ per depth. E3 is greedy accuracy on depths 6–8.

| model | run | comp_frac | h_ratio | d2 | d3 | d4 | d5 | E3 | gate (v1 rules) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| weak | pt-weak-cf0.000125-s0 | 1.25e-4 | 0.9 | .440 | **.114** | .011 | .001 | .000 | PASS |
| mid | pt-mid-cf0.00032-s0 | 3.2e-4 | 0.9 | .724 | **.381** | .072 | .003 | .001 | FAIL: spread (d5), matched_cells |
| strong | pt-strong-cf0.00083-s0 | 8.3e-4 | 0.9 | .919 | .759 | **.353** (d4-H .594) | .010 | .001 | PASS |
| midbal | pt-midbal-cf0.00032-s0 | 3.2e-4 | 0.5 | .906 | **.549** | .119 | .001 | .003 | FAIL: spread (d5), matched_cells |

Design targets: weak d3 0.05–0.2, mid d3 0.35–0.55, strong d4 0.4–0.6. E1 (depth 1) = 1.000 for every model. Under
the rules of decisions 2 and 3, midbal passes and mid fails only matched_cells (d4-H at lengths 7 and 9).

mid's matched cells (problems with relabelled p in [0.4, 0.6]; lengths 6 / 7 / 8 / 9 / 10): d2-L 369 (73 / 75 / 81 /
84 / 56), d4-H 326 (103 / 31 / 103 / 18 / 71).

Findings:

1. **The short pilot misleads.** 400-step pilots (4% of the tokens) extrapolated by /25 overestimated comp_frac by
   ~10×. Composition is learned late (after ~4000 of 9500 steps), so comp_frac has to be set by full-length sweeps.
2. **Very few composition examples are enough** once the atoms are learned: mid saw 8,800 composition examples in
   total. Composition transfers across functions (d4-L reached .59 with no d4-L training example in round 1).
3. **p is bimodal per problem** (see the v2 motivation above).
4. **Cliff at untrained depths.** Depth 5 is ≤ .010 and E3 ≤ .003 for every model, even comp_frac 1e-2. Composition
   does not extend to more steps than pretraining showed.
5. **d4 at length 7 is a seed-0 accident.** d4 has only ~70–200 training examples per length; seed 0 drew the fewest at
   length 7, and the models skip step 3 there. All levels share most composition examples (same seed), so they are
   paired, not independent replicates.
6. **Some problems can be solved with a step skipped.** On the RL pool, 31% of d4-H problems and 10.5% of d2-L
   problems give the same answer with one step removed (measured 2026-11-08).
7. **The reward checks only the final answer.** 1.4–2.6% of sampled completions have a wrong chain but the right
   answer.

## Project overview

In a controlled toy setting we study how the complexity make-up of RL (GRPO) training data affects the final
capability of the model, and derive predictive empirical formulas from the data.

**Main goal (as in Learning from Less):** at fixed compute, how does the structural-difficulty make-up of the RL data
(its depth distribution) affect final capability (in-distribution and deeper problems)? Does it change the capability
ceiling or the learning efficiency?

**Core difficulty:** in real LLMs, structural difficulty (how hard the problem is) and model-relative difficulty (the
model's success rate p) are always entangled. We pretrain transformers from scratch, control what the model knows
before RL, and so separate the two.

**Hypotheses** (no formula assumed; the data decides):

- H1: in-distribution gains are explained mainly by the initial success rate p (signal hypothesis).
- H2: out-of-distribution generalisation (deeper problems) additionally depends on structural difficulty.
- H3: the complexity make-up of the data mainly affects learning efficiency, not the ceiling (cf. ScaleRL).

**Method:** list all variables → measure noise → screen important variables → parameterise training curves →
derive candidate formulas → predict first, then train to verify.

**Expected outputs:**

1. A replication of the core Learning from Less result in the toy setting.
2. One or more empirical formulas "data configuration → final capability", with confidence intervals.
3. Data-selection rules derived from the formulas, compared with uniform and learnability sampling.

## Related work

Our position: in a controlled setting, carry out the future work proposed by Learning from Less, i.e. empirical laws
"data make-up → capability" for RLVR. Methods borrowed from RegMix (small-model screening), ScaleRL (curve
parameterisation) and Data Mixing Laws (parametric formulas).

| paper | key content | relation to us |
| --- | --- | --- |
| [Learning from Less](https://arxiv.org/abs/2604.18381) (MLSys 2026) | Qwen3-4B + LoRA + GRPO, Easy/Mixed × 100/200/500; on counting, 100 mixed ≈ 500 easy; one model, one seed | replication target; the authors list RLVR data laws as future work |
| [The Art of Scaling RL Compute](https://arxiv.org/abs/2510.13786) (ScaleRL, ICLR 2026) | sigmoid fit of compute vs performance, separating ceiling A and efficiency B; loss aggregation, normalisation, curricula mostly affect efficiency | curve parameterisation; tests H3 |
| [RegMix](https://arxiv.org/abs/2407.01492) | 512 models of 1M parameters + regression predict unseen data mixtures | "many small runs + regression" screening |
| [Data Mixing Laws](https://arxiv.org/abs/2403.16952) | performance is a predictable function of the data mixture, predictable before training | parametric formula form |
| [Can GRPO Transcend Its Pretraining Origin?](https://arxiv.org/abs/2510.15990) | from-scratch transformers, effect of the pretraining distribution on GRPO | precedent for controlling the model through pretraining; does not study RL data make-up |
| [Outcome-Based RL ... Only With the Right Data](https://arxiv.org/abs/2601.15158) (NeurIPS 2026) | theory: the training distribution needs enough easy samples for RL to learn extrapolating reasoning | we test whether easy samples matter for their content or because they give reward |
| [From f(x) and g(x) to f(g(x))](https://arxiv.org/abs/2509.25123) (ICLR 2026) | string function-composition task; RL teaches composing atomic skills | direct source of the task |
| [No Regrets / SFL](https://arxiv.org/abs/2408.15099) (NeurIPS 2024) | classic RL curricula: environments with mixed success work best | learnability baseline |
| [Reasoning Depth and Environment Complexity](https://arxiv.org/abs/2605.26934) | both difficulty axes need coverage; uniform mixing beats staged curricula | already answers "which difficulty axis matters"; we do not repeat it |
| [Mechanistic Role of Sample Difficulty](https://arxiv.org/abs/2605.28388) | success rate as difficulty proxy; medium difficulty is best | difficulty and p entangled, exactly what we separate |

**Do not claim "first".** The proposal says "to our knowledge"; repeat the literature search before writing up.

## Task design

The task is string-function composition: given input strings and nested functions, the model writes out the
intermediate results step by step and then the final answer. All answers are computed in Python, so the reward is
fully verifiable.

**Basic setup:**

- Alphabet: 26 letters `a`–`z`; input length **uniform in 6–10**. There are ~1.5×10¹⁴ possible input strings, so
  2.6B tokens of pretraining never see the same problem twice and the model has to generalise.
- 24 **length-preserving** unary functions, so string length does not grow with depth. Every function has a defined
  behaviour on odd lengths (e.g. "swap halves" keeps the middle character); `Task` checks this when it is built.
- Function names are meaningless ids `F00`–`F23`; the id → function mapping is shuffled with a fixed seed, so the model
  cannot guess behaviour from the name.
- Depth d = number of nested functions, 1–8. Use: pretraining 1–4 (depth 4 H-only), RL 1–5, OOD evaluation 6–8. In
  tree-v1 depth is the dependency depth of the tree (see below); a chain's depth is unchanged.

**The 24 unary functions:**

| category | from v0 | added in v1 |
| --- | --- | --- |
| position | reverse; rotate left 1; rotate right 1; rotate left 2; swap adjacent pairs; swap first and last; swap halves | rotate left 3; reverse first half only; riffle (interleave first and second half) |
| character | +1 to every character (cyclic); −1; +3; mirror alphabet (a↔z, b↔y …) | +5; add position (i-th char +i) |
| sorting | sort ascending; sort descending | sort first half ascending |
| conditional | +1 at even positions; move largest char to the front; overwrite second half with reversed first half | −1 at odd positions; move smallest char to the back |

- H/L groups of 12 each, every category on both sides, so "L is hard" cannot mean "L is all sorts".
- A function that cannot be learned to ≥ 98% at depth 1 is replaced (all 24 reach 1.000 in v1).
- "Add position" and "riffle" need both position and character information; they make single steps harder so the
  85M model does not solve every composition easily.

**Sample format** (character tokens, innermost function first):

```text
Q:F03(F11(F07(abcdefgh)))=
F07(abcdefgh)=bcdefghi;F11(bcdefghi)=ihgfedcb;F03(ihgfedcb)=jihgfedc;
A:jihgfedc<eos>
```

Length: depth 8 at length 10 has a ~54-token prompt and a ~230-token target. `max_len` is 512.

### Tree-v1 operators and trees (v2)

Full definitions, sampling and identity rules are in [`tree_v1.md`](tree_v1.md) (Hanzhong, implemented 2026-11-13).
Summary:

- **Operators.** 12 retained unary functions (swap_adjacent, swap_halves, even_pos_p1, odd_pos_m1, max_to_front,
  min_to_back, mirror_first_half, reverse_first_half, riffle, shift_by_pos, sort_first_half, sort_asc) and 12
  multi-input operators. Symbols are numbered a = 0 … z = 25, arithmetic is mod 26, indices are zero-based, n is the
  string length:

| operator | arity | definition |
| --- | --- | --- |
| zip_add | 2 | out[i] = x[i] + y[i] |
| zip_sub | 2 | out[i] = x[i] − y[i] |
| zip_mul | 2 | out[i] = x[i] · y[i] |
| gather | 2 | out[i] = x[y[i] mod n] |
| gather_add | 2 | out[i] = x[y[i] mod n] + y[i] |
| scatter_add | 2 | start with zeros; add each x[i] into out[y[i] mod n] |
| rotate_by_first | 2 | rotate x left by y[0] mod n (replaced rotate_by_sum, decision 12) |
| prefix_add | 2 | out[i] = x[i] + sum(y[0:i+1]) |
| zip_add3 | 3 | out[i] = x[i] + y[i] + z[i] |
| zip_mul_add | 3 | out[i] = x[i] · y[i] + z[i] |
| gather_mix | 3 | out[i] = x[y[i] mod n] + z[i] |
| select_parity | 3 | out[i] = x[i] if z[i] is even, else y[i] |

- **Dropped unary functions:** reverse, rotl1, rotr1, rotl2, rotl3, swap_ends, shift_p1, shift_m1, shift_p3,
  shift_p5, complement, sort_desc (simple permutations, constant character maps or redundant variants; a design
  choice, not a measured ranking: all v1 functions reached E1 1.000). Their code stays for loading chain-v1 configs
  and checkpoints.
- **Groups.** H and L each have 6 unary, 4 binary and 2 ternary operators; a problem's group (H / L / mixed) is set by
  all of its operators.
- **Leaves.** Independent random strings of one shared length (6–10).
- **Sampling.** For depth d: `n_ops` uniform over the feasible values d … min(10, 2d), then a uniformly random ordered
  shape with that size and depth, then operators of the required arities (`n_ops` can also be fixed). Group
  restrictions, pool rejection and held-out-edge rejection condition this distribution. Atoms (depth 1) use the 24
  operators uniformly.
- **Syntax and CoT.** Operands are separated by `,`. The CoT prints every operator node in post-order with its already
  evaluated arguments, then `A:`. Example (default mapping: F22 swap_adjacent, F23 zip_sub, F02 zip_add), depth 2 with
  three operations:

```text
Q:F02(F22(abcdef),F23(bcdefg,aaaaaa))=
F22(abcdef)=badcfe;F23(bcdefg,aaaaaa)=bcdefg;F02(badcfe,bcdefg)=ccggkk;
A:ccggkk<eos>
```

- **Structure variables** (stored with every problem, so comparisons stay interpretable): `depth`, `n_ops` (operator
  nodes = CoT steps), `n_binary`, `n_ternary`, `n_leaves` (= 1 + n_binary + 2 · n_ternary), `max_live_values` (peak
  number of stack values in the post-order traversal), `shape_id` (ordered topology and arity signature, without
  operators or inputs), and prompt / target token length.
- **Length budget.** `model.max_len` 1024 and a 768-token generation limit (`eval.max_new_tokens`,
  `rl.max_new_tokens`); conservative bounds over every allowed tree are checked against padding and the position
  offset before generation or streaming, so no long example is silently dropped.
- **Identity.** JSONL rows store the full expression, the post-order trace, the structure features, a schema version
  and a task fingerprint (operator specs and versions, Fxx mapping, H/L, alphabet, format, vocabulary). Loading
  recomputes the answers and checks the metadata; checkpoints carry the fingerprint and `load_model` checks it.
- **Rewards.** The binary reward checks only the final answer and `<eos>`; the partial reward divides by `n_ops`.
  Noise injection (P10) changes one node and recomputes its ancestors.

### Data structures and functions

```python
@dataclass(frozen=True)
class Problem:              # chain-v1, rlvr/task.py
    pid: str                # hash of funcs and input
    funcs: tuple[int, ...]  # innermost first
    x: str
    depth: int
    steps: tuple[str, ...]
    answer: str
    group: str              # H / L / mixed

@dataclass(frozen=True)
class TreeProblem:          # tree-v1, rlvr/tree_task.py
    pid: str                # hash of the task fingerprint and the canonical expression
    expr: Call              # Call(op, args) with Leaf(value) leaves
    funcs: tuple[int, ...]  # operators in post-order
    steps: tuple[str, ...]  # result of each operator node, in post-order
    trace: tuple[dict, ...] # per node: path, op, args, value
    answer: str
    group: str
    # depth, n_ops, n_binary, n_ternary, n_leaves, max_live_values, shape_id, task_fingerprint

make_task(cfg["task"])      # Task (task.kind chain) or TreeTask (task.kind tree)
sample_problem(depth, rng, group=None, func_pool=None, str_len=None)   # TreeTask also takes n_ops
format_prompt(p) -> str     # "Q:...=\n"
format_target(p) -> str     # steps + "A:...<eos>"
parse_answer(text) -> str | None
reward(text, p) -> float    # 1.0 if exactly right, else 0.0; no format reward
```

**Tokenizer:** character level: `a`–`z`, `F`, `0`–`9`, `( ) = ; :`, `Q`, `A`, newline, `<pad>`, `<bos>`, `<eos>` (48
tokens); tree-v1 appends `,` (49, `CharTokenizer(extra_chars=",")`), so the 48 chain-v1 token ids are unchanged.
Every checkpoint carries its own `tokenizer.json`; saving and loading check it against the task.

**Splits (no leakage):** problems are split by `pid` hash into pretraining (90%), RL (5%) and evaluation (5%) pools,
strictly disjoint. Some **function pairs** appear only in the evaluation pool, to test compositional generalisation:
of the 576 ordered parent → child pairs of unary functions, 6 per HH/HL/LH/LL are held out (24). Tree-v1 holds out 36
(parent operator, argument slot, child operator) edges, one per (parent arity, child arity, H/L, H/L) cell; sibling
operators are never an edge (`rlvr/tree_splits.py`, salt `rlvr-tree-v1`). Streaming uses rejection sampling to keep
these constraints.

## Variables and config keys

Every variable is a key in the YAML config; experiments only change configs, not code. Defaults live in
`rlvr/params.py` (`python -m rlvr.params` lists them); RL dataset make-up is set by the fields of the dataset specs
(`configs/datasets/*.yaml`). Priority: ★ core, varied on purpose; ○ if time allows; — frozen at the default.

### Pretraining (P)

| id | variable | config key | default | levels | priority |
| --- | --- | --- | --- | --- | --- |
| P1 | model size | `model.n_layer`, `model.d_model`, `model.n_head`, `model.d_ff` | 12, 768, 12, 2048 | S (6/384) for debugging; M main; L (24/1024, ~300M) optional spot check | — |
| P2 | position encoding | `model.pos` | RoPE | — | — |
| P2b | random position offset | `pretrain.pos_offset_max` | 128 | 0 / 128 | ○ |
| P3 | atomic coverage | derived: 1 − `pretrain.comp_frac` | fills the rest | — | — |
| P4 | share of composition examples (model level) | `pretrain.comp_frac` | set by sweeps (v1 mid 3.2e-4) | weak / mid / strong | ★ |
| P5 | chain depth mix | `pretrain.comp_depth_mix` | {2: 0.6, 3: 0.35, 4: 0.05} | with / without depth 4 | ○ |
| P5b | depth 4 H-only | `pretrain.h_only_depths` | [4] | — | — |
| P6 | uneven function exposure | `task.h_group`, `pretrain.h_ratio` | 12 H functions, 0.9 | 0.5 / 0.9 | ★ |
| P7 | CoT format | `task.cot` | on | — | — |
| P8 | input diversity | `task.str_len`, `task.alphabet` | [6, 7, 8, 9, 10], a–z | — | — |
| P9 | pretraining steps / checkpoints | `pretrain.save_at` | 25% / 50% / 100% | early / final | ○ |
| P10 | noisy examples | `pretrain.noise_frac` | 0 | 0 / 0.1 | ○ |
| P11 | seed | `seed` | 0 | 0 / 1 / 2 (mid only) | repeat |
| P12 | task version and operator table (v2) | `task.kind`, `task.functions`, `task.h_group` | chain; tree-v1 in `configs/tasks/tree_v1.yaml` | chain-v1 / tree-v1 | — |
| P13 | tree size limit (v2) | `task.tree_max_ops` | 10 | — | — |

### RL data (D)

| id | variable | config key | default | levels | priority |
| --- | --- | --- | --- | --- | --- |
| D1 | amount of data | spec `n` | 500 | 100 / 500 / 2000 | ★ |
| D2 | structural make-up (main goal) | spec `cells` (depth / frac; tree-v1: also the structure variables) | uniform 1–5 | single depth 1/2/3/4/5, mixed | ★ |
| D3 | model-relative difficulty | spec `cells` (p_bin) | any | each p bin, or p ≈ 0.5 | ★ |
| D4 | diversity | spec `n_unique_combos` | any | few / many | ★ |
| D5 | overlap with eval | none (eval leakage is forbidden) | 0 | fixed at 0 | — |
| D6 | string and reasoning length | — | recorded | covariate (lengths vary, must be matched) | — |
| D7 | order | `data.order` | shuffled | shuffled / shallow to deep | ○ |
| D8 | repeats per problem | set by N and total steps | — | recorded | — |
| D9 | online filter / replay | `data.sampler` (`auto` = the spec's `sampler`), `data.p_edge`, `data.explore_frac` | static | static / keep only p(t) in [1/32, 31/32] or unknown | ○ |
| D10 | labelling samples K | `label.k` | 32 | — | — |

### GRPO (R)

| id | variable | config key | default | levels | priority |
| --- | --- | --- | --- | --- | --- |
| R1 | group size G | `rl.group_size` | 8 | 4 / 8 / 32 | ★ |
| R2 | prompts per step B | `rl.prompts_per_step` | 64 | — | — |
| R3 | learning rate | `rl.lr` | 1e-5 | 3e-6 / 3e-5 | ○ |
| R4 | total rollout budget | `rl.steps` × `rl.prompts_per_step` × `rl.group_size` | fixed | — | — |
| R5 | sampling temperature | `rl.temperature` | 1.0 | 0.7 / 1.0 | ○ |
| R6 | max generation length | `rl.max_new_tokens` | 256 (tree-v1: 768) | — | — |
| R7 | divide by group std | `rl.adv_std_norm` | yes | yes / no (Dr.GRPO) | ○ |
| R8 | loss aggregation | `rl.loss_agg` | token mean | — | — |
| R9 | KL coefficient | `rl.kl_coef` | 0 | — | — |
| R10 | clip, updates per batch | `rl.clip_eps`, `rl.ppo_epochs` | 0.2, 1 | — | — |
| R11 | entropy bonus | `rl.entropy_coef` | 0 | — | — |
| R12 | reward form | `rl.reward` | binary | binary / partial | ○ |
| R13 | seed | `seed` | 0 | 0 / 1 / 2 | repeat |

### Measured only

- **Mediators (M):** initial success-rate distribution, share of groups with signal, entropy, answer length, gradient
  norm, KL to the base model, per-problem learning speed, collapse. Not causal inputs; used to explain mechanisms.
- **Outcomes (Y):** in-distribution accuracy, deeper-problem accuracy, new-combination accuracy, pass@k, step score,
  and A (ceiling), B (efficiency) and C_mid fitted to the whole training curve.

## Pretraining (`scripts/pretrain.py`)

Pretraining produces four base models: weak / mid / strong change only the composition share (P4); midbal is mid with
`h_ratio` = 0.5 (P6). All four have the same architecture, steps and token totals.

### Model

Hugging Face `LlamaForCausalLM` with a small config:

| item | value |
| --- | --- |
| layers / hidden / heads | 12 / 768 / 12 (head_dim 64) |
| FFN | SwiGLU, intermediate 2048 |
| vocabulary | 48 (tree-v1: 49) |
| max length | 512 (longest v1 sample ~290 tokens plus a random offset ≤ 128); tree-v1: 1024 |
| parameters | ~85M (non-embedding) |
| position encoding | RoPE, θ = 10000 |
| attention | `sdpa` (no flash-attn in the env; length bucketing removes the need for varlen) |

Why 12 layers: every CoT step has to find in context which function to apply next (the k-th from inside in the nested
expression) and the previous result, then apply it position by position. Depth limits how many such "locate +
compute" hops are possible; the 4-layer v0 model hit a cliff at depth 3, possibly partly for this reason. 85M
parameters on 4 A100s for 1 hour is ~2.6B tokens, ~1.5× the Chinchilla-optimal ~1.7B, so not badly undertrained and
still small enough for RL.

### Compute

| item | value |
| --- | --- |
| hardware | Perlmutter (NERSC), one node = 4×A100-40GB per model, DDP; cap 50 GPU node-hours |
| precision | bf16 autocast, fp32 parameters and optimiser state; per-layer `torch.compile` (whole-model compile needs a C++20 compiler that RHEL 8's g++ 8.5 lacks) |
| FLOPs per token | ~6N = 5.1×10⁸ (short sequences, attention negligible) |
| throughput | ~184k tokens/s per GPU (~30% MFU), ~740k on 4 GPUs, ~10–12 GB memory per GPU |
| total tokens | ~2.6B (incl. prompts; ~70% supervised); v1 mid: 2.78B, 1.83B supervised (each run's `stats.json`) |
| global batch | 192k supervised tokens per step (~270k tokens); micro-batches ≥ 6k supervised tokens, 8 accumulation steps on 4 GPUs |
| steps | 9,500 (~0.40 s/step on 4 GPUs) |
| time per model | ~70 min on one node including start-up and evaluation; full calibration ~35–50 min more |

### Training details

- Next-token prediction, cross-entropy, **loss on the answer only**: the `Q:...=\n` prompt gets label -100.
- **Length bucketing instead of packing:** every 20k examples are sorted by length and cut into micro-batches of
  ≥ 6k supervised tokens (`pretrain.micro_target_tokens`), then shuffled. Lengths within a batch are nearly equal,
  padding is ~10%. Padding is always after real tokens, so a causal mask suffices. A micro-batch stops filling once it
  reaches the budget (the tail joins the last one), and the loss is "token sum ÷ fixed budget", so every supervised
  token has the same weight at every level.
- **Random position offset (P2b):** each example starts at a random position in [0, 128], so the model sees all
  absolute positions and deeper (longer) answers do not shift the position distribution.
- AdamW (β = 0.9, 0.95), peak lr 6e-4, 2% warmup, cosine decay to 10% of peak, weight decay 0.1, gradient clip 1.0.
- Greedy evaluation on E1–E4 and the probe set every 2000 steps; checkpoints at 25% / 50% / 100% (P9). Early
  checkpoints can serve as "less trained" weak models for comparison.
- Logged per run: loss by depth (evaluation and corpus statistics also by group), throughput, MFU, tokens consumed
  per data component.

### Streamed pretraining data

v1 needs ~2.6B tokens, ~52M examples, so the corpus is **generated online** instead of looping over a fixed corpus,
reproducible and leak-free (`rlvr/stream.py`, `PretrainStream`). There is no global dedup, so an example can repeat
inside the stream; pools are kept apart by the pid hash and rejection sampling:

1. **Deterministic seeds:** each worker's RNG derives from `(seed, rank, worker_id)`. With the same GPU and worker
   counts, a config always produces the same stream. No resume (a failed ~1 h run is simply rerun).
2. **Component sampling:** each example first picks a component by token share (table below), then a depth/shape,
   functions and inputs (length uniform in 6–10).
3. **Rejection sampling for splits:** only problems whose `hash(pid)` falls in the pretraining pool are kept; any
   example with a held-out pair (tree-v1: a held-out edge) is dropped (~10% rejected).
4. **Eval dedup:** pids of E1–E4, the probe set and the calibration set are loaded at start-up and dropped (the RL
   pool is already excluded by the hash split).
5. **Online statistics:** each worker counts examples and tokens per component / depth / group / function, merged
   into `stats.json` at the end to check the actual composition.
6. **Noisy examples (P10):** one step's result is replaced by a random string and the later steps are recomputed.

**Corpus recipe (share of supervised tokens):**

| component | content | weak | mid | strong | midbal |
| --- | --- | --- | --- | --- | --- |
| atoms | one operation; v1: 24 unary functions uniform; tree-v1: the 24 operators uniform | rest | rest | rest | rest |
| chains (chain-v1, with CoT) | depth 2 : 3 : 4 = 0.6 : 0.35 : 0.05; depth 4 H-only | `comp_frac` | " | " | " |
| trees (tree-v1, with CoT) | dependency depth 2 : 3 : 4 = 0.6 : 0.35 : 0.05, `n_ops` uniform over d … min(10, 2d); pilot recipe: no H-only depth, `h_ratio` 0.5 | `comp_frac` | " | " | " |
| H/L exposure | of the other compositions, `h_ratio` are H-only, the rest contain at least one L function | 0.9 | 0.9 | 0.9 | **0.5** |

v1 comp_frac: weak 1.25e-4, mid 3.2e-4, strong 8.3e-4, midbal 3.2e-4 (set by full-length sweeps; see
[v1 results summary](#v1-results-summary)). Tree-v1 comp_frac must be swept again at the full length (decision 1):
tree compositions are longer and the operators differ.

**Why this recipe:**

- **Mainly depth 2–3:** fixes the v0 cliff. The model needs to have seen the format "write 3 steps in a row" before
  depth 3–5 problems can have 0 < p < 1.
- **A little H-only depth 4:** puts "depth 4, H" in the seen-but-not-mastered state, so p can land in 0.4–0.6, while
  "depth 2, L" lands in the middle because L has low exposure. In v1 these were the two matched cells.
- **No depth 5 in pretraining:** RL depth 5 is structurally beyond pretraining, and E3 (depth 6–8) is fully OOD.

**Confound control:** all four models have the same steps and the same supervised-token budget per step; a model
with a smaller `comp_frac` gets more atoms. Otherwise "weak" would just mean "trained less", not "fewer composition
skills". Actual totals differ slightly because micro-batches overfill; they are reported from each run's `stats.json`.

### Deliverables

1. `checkpoints/pt-{level}-s{seed}/` in `save_pretrained` format with `tokenizer.json`, `run_config.yaml`; per run
   `stats.json`; intermediate checkpoints at 25% / 50%. Uploaded to Hugging Face with `scripts/push_hf.py`, one folder
   per run, with the run's `calib/` files and optionally the RL-pool labels.
2. **Calibration heatmap:** success rate per (depth 1–8 × H/L/mixed) cell (T = 1.0, 200 problems per cell, 32 samples
   each); tree-v1 cells are (depth × group), with the structure variables of every problem.
3. **Gate check** (`scripts/calibrate.py`, rules in `rlvr/gate.py`). Checks in `gate.hard_checks` fail the gate and
   lead to retraining with another `comp_frac` or `h_ratio`; the others are warnings in `gate.json`. Hard checks: the
   first three for mid; the first two for midbal (decision 3); only the first for weak and strong, whose other checks
   are judged against the design targets by hand.
   - Depth-1 greedy accuracy ≥ 98%: overall as a point estimate, and per function (50 problems each) by its Wilson
     upper bound (≤ 2 errors in 50 pass). Tree-v1: all 24 operators.
   - Spread: at each of depths 2–4 (decision 2; tree-v1: dependency depth), ≥ 20% of calibration problems have
     0 < p̂ < 1.
   - Matched cells: each of the two cells chosen for D and E ([Matched cells](#matched-cells-v2)) has ≥ 300 problems
     with p in 0.4–0.6 (from the second labelling), with enough per input length for D and E to share one length mix
     (decision 6). v1: d2-L and d4-H, ≥ 120 per length.
   - Composition loss not near 0: `loss_d3` ≥ 1e-3 (warning). Total loss is dominated by atoms and falls below 1e-4
     once atom accuracy is ~99.7%; it reflects `comp_frac`, not whether the task is too easy.
   - Design targets (warning, `gate.target`): weak d3 mean p 0.05–0.2, mid d3 0.35–0.55, strong d4 0.4–0.6.

## v2 pilot

Goal: decide by **Sun Nov 22** whether v2 (tree-v1, decision 10) replaces v1 for the main experiments. Lessons from
v1: the pilot runs at the full training length (decision 1), and it measures the per-problem p distribution, not only
cell means.

**Stage A: can each operator be learned?**

- One node, atoms only (`comp_frac` 0: the 24 tree-v1 operators, uniform), 2000 steps (~15 min). This is also the
  pure-atom control that [`tree_v1.md`](tree_v1.md) asks for. In v1, E1 was already .996 at step 2000.
- Pass: every operator ≥ 98% greedy on its 50 E1 problems (same Wilson rule as the gate). An operator that fails is
  replaced or dropped together with Hanzhong; the ternary operators can be ablated by editing the operator table.

**Stage B: do trees give enough p ≈ 0.5 problems?**

- Three full-length runs (9500 steps) of the tree-v1 recipe, in parallel, one node each, at three comp_frac values
  set after stage A (the v1 values do not carry over).
- Each run: calibration over (depth 1–8 × group) cells with the structure variables of every problem, then a
  candidate pool for the best D/E pairs (decision 11): their cells (≤ 4) get 10,000 RL-pool candidates each, labelled
  with k = 32, and the non-skippable ones with p̂ in [0.3, 0.7] are relabelled with k = 64.
- Report per cell: mean p, share of problems with p in [0.4, 0.6] (v1: ≤ 12.5%), signal_frac, and the structure
  variables.

**Go criterion.** For one of the stage-B runs there are two cells, a simple one for D and a complex one for E (E is at
least two levels deeper than D, as in v1's 2 vs 4; decision 11), each with ≥ 500 problems with relabelled p in [0.4, 0.6]
from ≤ 10,000 candidates, after removing skippable problems and with a shared length mix. All operators also pass
stage A.

**If no-go:** keep the v1 models and use Jay's plan: expand the v1 RL pool (d4-H to ~34,000 and d2-L to ~11,000
candidates, ~2–2.5 node-hours of labelling), remove skippable problems and give D and E the same length mix.

**Implementation.** Data and training use the tree-v1 code (`configs/tasks/tree_v1.yaml`, `scripts/make_tree_data.sh`;
the task config's `pretrain` block is a 400-step pilot recipe, so full runs override `pretrain.steps`). Still to add
for the pilot (Yupeng): an E1-only calibration mode for stage A, the stage-B candidate pool and its report (D/E cells,
shared length mix, go), and the skippable check for trees. The D/E cells follow decision 11.

**Compute (node-hours):** stage A ~0.4. Stage B and the final models are estimated after stage A and a `--bench`:
tree-v1 sequences are up to ~2× longer than v1 chains, so labelling (k = 32 samples of up to 768 tokens) costs more than
v1's 35–50 min per model. The same 50-node-hour cap also covers RL (see [Risks](#risks)).

## Difficulty labels and RL datasets (`scripts/calibrate.py`, `scripts/build_dataset.py`)

Each problem has two difficulty labels: structural difficulty (depth or the v2 structure variables, function group),
given by the generator, and model-relative difficulty (success rate p), measured by sampling each base model.

### Computing p

1. For each base model, label the RL candidate pool (`data/rl_pool/candidates.jsonl`; v1: 35,000 problems, depths 1–8 ×
   H/L/mixed, 1,000 per cell plus 4,000 extra in each of d2-L, d4-H and d5-H).
2. Each problem: `generate(do_sample=True, temperature=1.0, num_return_sequences=32)`, scored with `reward()`;
   p̂ = correct / 32.
3. Also record the Wilson 95% interval. With K = 32 the standard error of p̂ is up to ~0.09; binning has to allow for
   this noise.
4. Output `data/labels/{model_id}.parquet` with columns `pid, depth, funcs, group, str_len, n_correct, k, p_hat,
   ci_lo, ci_hi, model_id` (tree-v1 adds the structure variables and the task fingerprint); relabelled problems also have `n_correct_relabel,
   k_relabel, p_hat_relabel, ci_lo_relabel, ci_hi_relabel`.
5. Cost (measured in v1): ~730 sequences/s on 4 GPUs; calibration, pool labelling and relabelling take ~35–50 min per
   model.

**Second, independent labelling (against the winner's curse):** choosing p ≈ 0.5 problems from a 32-sample p̂
systematically picks "lucky" problems: where most problems have low p, the chosen ones are truly lower, and vice
versa. A rough Beta-prior estimate for d2-L with mean 0.2 and d4-H with mean 0.8 gives true means of ~0.44 and ~0.56
for the chosen problems, with ~40% truly outside 0.4–0.6. So problems in `label.relabel_cells` whose first p̂ is in
0.3–0.7 get 64 more independent samples (`label.relabel_k`), and the matched datasets (`p_label: relabel`) select on
the second estimate. The relabelling screen is 0.3–0.7 (`label.relabel_range`, as in every v1 run_config.yaml);
`relabel_report.py` then checks the problems first selected with p̂ in 0.4–0.6: in v1 their mean relabelled p was
.47–.52 for mid and midbal, and 55–57% stayed in range. The second estimate still has sampling and selection error, so
it is not the true p. Both estimates are reported (`mean_p` and `mean_p_relabel` in `meta.json`).

**Online p during training:** p changes during RL. No extra sampling is needed: per-problem group accuracy from the
rollout log, smoothed with an EMA, gives each problem's p(t).

**Why p and G go together:** the chance that a group of G answers has both right and wrong ones (and so a gradient) is

```latex
P_{\text{sig}}(p, G) = 1 - p^{G} - (1-p)^{G}
```

For small p this is about G × p, so analyses record both p and G.

### Bins

With G = 8, p is split into 7 bins: `[0, 1/32)`, `[1/32, 0.2)`, `[0.2, 0.4)`, `[0.4, 0.6)`, `[0.6, 0.8)`,
`[0.8, 31/32)`, `[31/32, 1]`. The two end bins give almost no gradient and are counted separately.

### Dataset specs (YAML)

Each RL dataset is defined by a spec; `build_dataset.py` samples stratified by (depth × p bin × group) with a fixed
seed and writes `data/datasets/{name}.jsonl` (one `pid` per line).

```yaml
name: matched_p_deep
model_id: pt-mid-s0
n: 500
seed: 0
cells:            # share per cell, summing to 1.0
  - {depth: 4, p_bin: "0.4-0.6", group: H, frac: 1.0}
match_covariates: [str_len]   # lengths vary, so match them to the control group
```

### Matched cells (v2)

- **Choice of cells** (decision 11, fixed before the stage-B results). v1 used d2-L (D) and d4-H (E). In tree-v1 a
  cell is (depth, group), the stratification of the calib set (200 problems per cell; finer cells by `n_ops` or arity
  counts would hold only tens of calib problems each). Candidate pairs: D and E in RL depths 1–5 with
  E.depth ≥ D.depth + 2. Score of a pair: the smaller of the two cells' calib shares of non-skippable problems with
  p̂ in [0.4, 0.6]; ties go to the pair with smaller depths. The cells of the best pairs (≤ 4 cells) get the candidate
  pool, and the pair with the largest supply after relabelling (shared length mix) is used. Depth is the axis because
  H2 is tested on E3 (depth 6–8), as in v1 where depth equals the number of operations.
- **Covariates.** `n_ops`, `n_binary`, `n_ternary`, `max_live_values` and token length are reported for D and E;
  deeper cells also have more operations, as in v1. If the pool allows, a secondary contrast at equal depth and
  different `n_ops` is reported separately. The v1 pair stays as a reference if it is available.
- **Skippable problems are excluded** (decision 6). A problem is skippable if removing any one unary node, or
  replacing any multi-input node by one of its operands, gives the same answer. The share of excluded problems is
  reported per cell (v1 RL pool: 31% of d4-H, 10.5% of d2-L). Not yet implemented for tree-v1.
- **Length matching.** D and E use the same input-length mix, set by the scarcer cell, instead of a fixed 100 per
  length. Token length is reported as a covariate, because trees are longer than chains at the same n_ops.

### Datasets

| group | name | spec | purpose |
| --- | --- | --- | --- |
| replication | easy_N / mixed_N | Easy = depth 1–2; Mixed = depth 1–5 uniform; N = 100 / 500 | replicate Learning from Less |
| heatmap | depth_k | only depth k (k = 1…5), and the uniform mix | main goal: make-up × model level |
| A | uniform_pool | uniform over the whole RL pool | "all data is fine" baseline; also mid vs midbal (decision 3) |
| B | online_filter | whole pool, keep only problems with 0 < p < 1 during training | common practice |
| C | learnability | sample in proportion to p(1 − p) (SFL) | learnability baseline |
| D | matched_p_shallow | simple cell, p ∈ [0.4, 0.6] (v1: depth 2, L) | decoupling |
| E | matched_p_deep | complex cell, p ∈ [0.4, 0.6] (v1: depth 4, H) | decoupling |
| F | predicted_p | whole pool weighted by p(1 − p) with p from a cheap regressor on problem features | can selection skip labelling? (from the proposal) |
| G | curriculum | same problems as A, ordered easy → hard by p; optionally by depth | ordering by p vs by structure (from the proposal) |
| ~~E′~~ | ~~matched_p_deeper~~ | ~~depth 5, H~~ | dropped (decision 2) |

A–G all use the same total rollouts. B, C and G need dynamic sampling in the training loop.

**Optional intervention (hint prefix):** give the first two steps of a depth-4 problem as a prefix, which raises p
with roughly the same content, as a reverse control. It changes the number of steps the model has to do; say so in
the report.

## GRPO (Hanzhong)

GRPO is our own PyTorch implementation with every variant behind a config switch: `rlvr/grpo.py` (group advantages,
teacher-forced completion log-probs, micro-batched update), `rlvr/sampler.py` (static / online_filter / learnability
samplers and online p(t)), `scripts/grpo.py` (one run) and `scripts/rl_sweep.py` (checkpoints × datasets × seeds ×
grid, one run per GPU, resumable); usage in the README. A run inherits task, model, splits, paths, eval version and
length budgets from the base checkpoint's `run_config.yaml`, so the same commands work for chain-v1 and tree-v1
checkpoints. **Status:** implemented and smoke-tested on the v1 round-1 checkpoints (`tests/test_grpo.py`: advantages,
log-probs equal to generation scores, update equal to the hand-written policy gradient, micro-batch invariance,
samplers, a tiny end-to-end run). Not done yet: the TRL `GRPOTrainer` cross-check, the A/B/C_mid fits.

### One step

1. Sample B problems from the dataset (B/C sample dynamically here: B keeps only problems with online p in (0, 1), C
   weights by p(1 − p)).
2. Generate G answers per problem: left padding, `do_sample=True`, T = 1.0, stop at `<eos>`.
3. Score with `reward()` into a [B, G] reward matrix.
4. Group advantages: subtract the group mean; with `adv_std_norm=True` divide by the group std + 1e-6. All-right or
   all-wrong groups have zero advantage and are counted as "no-signal groups".
5. Log-probs only for answer tokens; prompt and padding are masked.
6. Loss = −(advantage × log-prob), token mean or sequence mean per `loss_agg`. With `ppo_epochs=1` no importance ratio
   or clip is needed.
7. Backward, gradient clip 1.0, AdamW step.

```python
for step in range(cfg.rl.steps):
    batch = sampler.sample(B)                          # static or dynamic sampling
    out = rollout(model, batch, G, cfg.rl.temperature) # tokens, masks
    r = rewards(out, batch).view(B, G)
    adv = r - r.mean(1, keepdim=True)
    if cfg.rl.adv_std_norm:
        adv = adv / (r.std(1, keepdim=True) + 1e-6)
    logp = token_logprobs(model, out)                  # [B*G, T], answer tokens only
    loss = -(adv.flatten()[:, None] * logp * out.mask).sum() / out.mask.sum()
    loss.backward(); clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad()
    logger.log_step(step, batch, out, r, adv)          # rollouts.jsonl and metrics
```

### Default hyperparameters (starting values)

| parameter | default |
| --- | --- |
| G | 8 |
| prompts per step B | 64 |
| learning rate | 1e-5, constant, 10 warmup steps |
| training steps | 500 (set by the total rollout budget) |
| max generation length | 256 (tree-v1: 768) |
| KL coefficient | 0 |
| entropy bonus | 0 |
| updates per batch | 1 |

**Time per RL run (measured on one A100, weak round-1 checkpoint, B 64 × G 8):** ~3.3 s/step (2.6 s rollout, 0.7 s
update) at ~100-token completions, ~13 GB; a full E1–E4 greedy evaluation ~55 s; 500 steps with the default evaluation
schedule ~35–40 min. Longer completions (deeper problems, tree-v1) are slower in proportion. At ~40 min, 135 runs are
~90 GPU-hours, ~22 node-hours. If it is much slower, speed up generation (vLLM or a static KV cache) before shrinking
the plan.

### Common pitfalls

- `generate` must left-pad, otherwise log-prob positions are misaligned.
- Log-probs only on answer tokens; including the prompt points the gradient in a completely wrong direction.
- Cut generation after `<eos>`; an answer without `<eos>` scores 0.
- Validate with exact `reward()` scoring, never with a training-time helper metric.
- Before each run, evaluate on a fixed set of 64 problems to confirm the right checkpoint is loaded.
- Pretraining used a random position offset; RL and inference start positions at 0. Depth-1 accuracy is 1.000 for
  every v1 model, so this is fine.

### Correctness checks

- **Smoke test:** 20 steps; mean reward should rise and all log fields be present.
- **TRL cross-check:** same model, dataset and hyperparameters with TRL `GRPOTrainer`; the two reward curves should
  roughly agree.
- **Overfitting test:** train on only 8 problems; reward should quickly approach 1.

## Evaluation and logs (Jay)

Pretrained and RL models use the same `rlvr/evaluate.py`; all figures come from offline scripts reading the logs, so a
plotting bug never needs retraining.

### Evaluation sets (fixed and versioned; chain-v1 sets are `data/eval/v2/`, tree-v1 sets `data/tree-v1/eval/v1/`)

| set | content | problems | measures |
| --- | --- | --- | --- |
| E1 atomic | depth 1, all functions (50 each); tree-v1: the 24 operators | 1200 | pretraining quality |
| E2 in-distribution | depth 2–5, H/L/mixed separately | 1800 | main RL gain |
| E3 deeper | depth 6–8 | 900 | OOD generalisation (H2) |
| E4 new combinations | contains held-out pairs (tree-v1: held-out edges), never seen in pretraining or RL | 800 | compositional generalisation |
| probe | fixed problems never used for training, all depths and groups | 600 | transfer during training |

All sets are stratified over input lengths 6–10 so results can be reported per length. In E2, depths 2–4 are covered
by pretraining and depth 5 is not, so they are reported separately. Tree-v1 rows also carry the structure variables,
and `eval.jsonl` adds one row per (depth, n_ops, n_binary, n_ternary).

**Metrics:** greedy accuracy; pass@k (k = 1, 2, 4, 8, 16, T = 1.0, unbiased estimator); format error rate; mean answer
length. Depth-5 accuracy is reported on its own (decision 7). **Step score** (decision 7): the number of correct steps
before the first mistake, also divided by n_ops; on trees, steps are counted in post-order. It is the main E3 metric,
because E3 accuracy is ~0 for every v1 model. Evaluation every 50 steps, probe every 25.

### Log format

One directory per run, `runs/{run_name}/`, named `pt-{level}-s{seed}` (pretraining) or
`rl-{model}-{dataset}-G{G}-s{seed}` (RL).

| file | content per line |
| --- | --- |
| `run_config.yaml` | full config + git commit (RL also `env`: torch version, GPU name and memory in GiB; not saved with checkpoints) |
| `stats.json` (pretraining) | examples and tokens actually consumed per component / depth / group / function |
| `rollouts.jsonl` | one row per (step, problem): `step, pid, depth, group, str_len, rewards[G], lens[G], p_online, completions[G]` (`rl.log_completions=false` drops the texts) |
| `metrics.jsonl` | `step, mean_reward, signal_frac, entropy, mean_len, grad_norm, kl_to_base, lr` (pretraining also `tok_per_s, mfu` and loss per depth; RL also `rollouts, trunc_frac, loss, reward_d{d}, signal_d{d}`, sampler state and time per step, `kl_to_base` every `rl.kl_every` steps, `gpu_mem_gb` peak GPU memory of the step on CUDA) |
| `summary.json` (RL) | written last: first and last greedy accuracy per evaluation set, training reward, time (`minutes` the whole run from the step-0 eval, `eval_minutes` the part spent in evals), peak GPU memory `gpu_mem_gb` (training steps) and `eval_gpu_mem_gb` (evals), `None` on CPU; `rl_sweep.py` uses it to see that a run finished |
| `eval.jsonl` | `step, eval_set, depth, group, greedy_acc, pass@k…, step_score, step_frac, format_err` (pass@k only with `eval.pass_k_samples` > 0; off by default); the overall row of each set (depth and group `all`) also `sec`, that set's eval wall time |
| `eval_problems.jsonl` (RL) | one row per (eval step, eval set, problem): `step, eval_set, pid, correct, n_correct, err_type, err_step` (problem fields join by `pid` from the eval set files; `rl.log_eval_problems=false` drops the file) |

### Offline analysis

1. **Curve parameterisation:** fit each run's E2/E3 accuracy against rollouts C with a ScaleRL-style sigmoid, giving
   ceiling A, efficiency B and midpoint C_mid (R0 is the starting accuracy):

```latex
R(C) = R_0 + \frac{A - R_0}{1 + (C_{\text{mid}} / C)^{B}}
```

2. **Per-problem learning speed:** from `rollouts.jsonl`, fit each problem's p(t) with a logistic curve to get its
   learning speed k_i, and test whether it depends on depth:

```latex
p_i(t) = \sigma\left(k_i\, t + \operatorname{logit}(p_{i,0})\right)
```

3. **Screening regression:** config variables in, A, B and final accuracy out; linear regression with interactions
   or a tree model; output a variable-importance ranking.
4. **Prediction check:** predict new configs with the fitted formula first, then train, and plot predicted vs actual.

## Experiment phases and timeline

Five phases, ~135 short RL runs in total, plus pretraining.

| phase | content | runs (approx.) | condition to move on |
| --- | --- | --- | --- |
| P. pretraining | v1 done; tree-v1 pilot (stages A, B) → four tree-v1 models → calibration + gate | 4 + 4 | four models pass the gate |
| 0. noise | default config, 5 seeds | 5 | seed std known, used as the minimum detectable effect |
| 1. replication + screening | Easy/Mixed × N; 6–8 ★ variables at two levels each, random or fractional design | 50 | the 2–3 variables with the largest effect and interaction with make-up |
| 2. main experiment + curves | heatmap (model level × make-up × 3 seeds); groups A–G; A, B, C_mid per run | 60 | clear patterns in A and B, complete per-problem p(t) data |
| 3. formulas | candidate functional forms compared on held-out configs; predict 4–6 new configs, then train | 20 | prediction error within seed noise |
| 4. other goals (optional) | diversity (pass@k), transfer, collapse conditions, optimal data size; G = 4/8/32; mid seeds 1/2; L-size spot check | as time allows | — |

**Schedule** (week 1 = Nov 2–8, from the proposal):

| week | pretraining (Yupeng) | RL (Hanzhong) | evaluation (Jay) |
| --- | --- | --- | --- |
| 3 (Nov 16–22) | tree-v1 adopted (decision 10); stage A and stage B on tree-v1; go/no-go on Sun Nov 22 | GRPO implemented and smoke-tested; TRL and overfitting checks; time one run | d5 accuracy and step score in `evaluate.py` |
| 4 (Nov 23–29) | four tree-v1 models (or v1 pool expansion if no-go), calibration, D/E datasets | phase 0 (noise), phase 1 | pre-RL step-level baseline on the labelling samples |
| 5 (Nov 30–Dec 6) | mid seeds 1/2 if budget allows | phase 2 | F and G, analysis |
| 6 (Dec 7–13) | write-up | phase 3 (reduced), write-up | write-up |

**Decision rules (fixed in advance, to avoid picking conclusions afterwards):**

- An effect smaller than 2× the phase-0 seed std counts as "no difference".
- H1 holds: on E2, D (simple, p ≈ 0.5) and E (complex, p ≈ 0.5) do not differ, and B or C is not worse than the other
  groups.
- H2 holds: on E3 (accuracy and step score), E is significantly better than D.
- H3 holds: when the make-up changes, B changes significantly and A does not.

**Fallback result:** even without the decoupling sets, "replication of Learning from Less in the toy setting +
heatmap of model level × make-up + A/B parameterisation" is a complete course project.

## Code and roles

The repo layout is in the README: library code in `rlvr/`, entry points in `scripts/` (Perlmutter jobs in
`scripts/slurm/`), configs in `configs/`, small results in `results/`, docs in `docs/`. `runs/`, `checkpoints/` and
`data/labels/` are on scratch and not in git; checkpoints are shared through Hugging Face.

Base models are both the objects of p labelling and the initial weights for GRPO; the evaluation code serves both
stages. Pipeline: task → streamed corpus → pretraining → calibration and gate → p labels → datasets → GRPO → evaluation
→ analysis.

| part | Yupeng (pretraining) | Hanzhong (RL and laws) | Jay (evaluation and analysis) |
| --- | --- | --- | --- |
| task, tokenizer, model (`rlvr/task.py`, `tokenizer.py`, `model.py`) | writes (chain-v1) | writes (tree-v1) | reviews |
| splits, log format | agreed together | agreed together | agreed together |
| streamed corpus, DDP training, sweeps | writes | — | — |
| calibration, gate, labelling | writes | reviews | — |
| datasets, samplers | — | writes | — |
| GRPO and TRL cross-check | — | writes | — |
| evaluation sets and `evaluate.py`, step-level metrics | reviews | reviews | writes |
| baselines F and G | — | reviews | writes |
| analysis | pretraining part | curves and formulas | step-level and RL evaluation |

**Tree-v1 code** (Hanzhong, 2026-11-13; merged into the pretrain branch on 2026-11-16): `rlvr/tree_task.py`
(`TreeTask`, `TreeProblem`), `rlvr/tree_ops.py` (operators with declared arity), `rlvr/tree_sampling.py` (shape
counting and sampling), `rlvr/tree_splits.py` (held-out edges), `rlvr/task.py` (`make_task`), `rlvr/tokenizer.py`
(`extra_chars`), `rlvr/model.py` (vocabulary and fingerprint checks), `scripts/make_tree_data.sh`. The chain-v1 code
paths and data are unchanged. `scripts/diag_steps.py` stays chain-only. Still to add for the pilot (Yupeng): stage-A
calibration mode, stage-B candidate pool and report, skippable check for trees.

## Risks

The main risk is still that simple and complex problems do not both have enough p ≈ 0.5 problems. v0 failed on this
(deep p ≈ 0); v1 got the levels on target but p stayed bimodal; v2 and its pilot target it directly.

| risk | consequence | response |
| --- | --- | --- |
| trees are just as bimodal as chains | v2 does not add matched problems | go criterion; fall back to v1 + pool expansion |
| an operator (binary or ternary) is not learned | depth-1 gate fails | stage A; replace or drop it, or ablate the ternary operators |
| operator shortcuts (zip_add simplifies algebraically, select_parity ignores one operand per position) | some steps do not matter, structure is overstated | skippable filter; report the excluded share |
| longer sequences | slower labelling and RL generation, length confounds structure | max_len 1024, generation 768; measure cost in the pilot; report token length; match lengths |
| v2 retraining delays RL | fewer phase-2/3 runs | RL code developed on v1 mid now; no-go fallback on Nov 22; phase 3 reduced |
| compute cap (50 node-hours) shared with RL | not enough for everything | estimate tree-v1 cost after stage A; RL ~22 node-hours for 135 runs at the measured speed; reuse a stage-B run as mid; check NERSC_LOG before each submission |
| all levels share seed 0's data luck | levels are paired, not independent | mid seeds 1/2 (P11); interpret level comparisons as paired |
| deeper problems stay at 0 after RL | H2 untestable on accuracy | step score on E3 (decision 7) |
| the 85M model solves every depth | no medium difficulty | smaller comp_frac; 25% checkpoint |
| some functions unlearnable | depth-1 gate fails | replace the function |
| seed noise too large | effects indistinguishable | more seeds; per-problem statistics |
| deeper answers are longer, lengths differ | structure and token count confounded | report length-normalised results once; match lengths in datasets |
| toy results do not generalise | the report is questioned | discuss in limitations; if time allows, spot-check one result on Qwen2.5-0.5B with the same task |

### Open items

- [x] GPUs: Perlmutter, 4×A100-40GB per node, 50 node-hours
- [x] Throughput measured: ~184k tokens/s per A100; 9,500 steps per model
- [x] v1 comp_frac levels, pos_offset_max 128, 24 unary functions
- [x] Fourth base model: midbal (h_ratio 0.5)
- [x] Spread depths, midbal matched cells, strong comp_frac (decisions 2–4)
- [x] Upload the v1 checkpoints to Hugging Face and put the repo name in [Status](#status-2026-11-16)
- [x] Gate rules from decisions 2–3 in code and configs (2026-11-15)
- [x] Operator set and code for the tree task: tree-v1 (decision 10)
- [ ] Check that the Hugging Face v1 checkpoints load with `load_model` after download
- [x] Tree-v1 stage A: every operator learnable (after decision 12)
- [x] D/E cell definition for tree-v1 (decision 11)
- [ ] Tree-v1 stage B: comp_frac levels, candidate pool code; skippable check for trees
- [ ] v2 go/no-go (Sun Nov 22) and the D/E cell pair
- [ ] RL role of midbal (proposed: mid vs midbal on A)
- [ ] Total RL rollout budget (sets the steps per run)
- [ ] Repeat the literature search to confirm nobody has done the decoupling
