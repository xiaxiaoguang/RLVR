# v2 pretraining results (tree problems)

Results of the v2 pilot ([`design.md` "v2 pilot"](design.md#v2-pilot)) and, after the go/no-go on Sun Nov 22, of the
v2 base models. Since 2026-11-16 the v2 task is Hanzhong's **tree-v1** ([`tree_v1.md`](tree_v1.md), design.md decision
10). An earlier variant (24 unary + 4 binary operators) was piloted first; its results are kept below as a record.

- Full job log: `/pscratch/sd/y/yupengsu/rlvr/runs/NERSC_LOG.md` (scratch). A snapshot is copied to
  `results/pretrain_v2/NERSC_LOG.md` with every results commit.
- Small per-run files (gate.json, relabel_report.json, stats.json, eval.jsonl, run_config.yaml, calib heatmap):
  `results/pretrain_v2/models/<run>/`, collected from scratch by `scripts/collect_results.sh`.

## Status (2026-11-18)

| step | run | result | status |
|---|---|---|---|
| tree-v1 stage A: atomic skills of the 24 operators | pt-tree-a-atoms-s0 | 23 of 24 pass; rotate_by_sum E1 .12 | **FAIL** on one operator; replaced by rotate_by_first (decision 12) |
| tree-v1 stage A, revised table | pt-tree-a2-atoms-s0 | all 24 pass; E1 .997 | **PASS**, operator table fixed |
| tree-v1 stage B: p ≈ 0.5 supply | pt-tree-mid-cf{0.00032,0.00064,0.00128}-s0 | no D/E pair at any comp_frac | **no-go** for this round; next step is a team decision |
| tree-v1 stage B round 2: more depth-4 compositions (decision 13) | pt-tree-mix-cf{0.0035,0.006}-s0 | gate PASS for both; best D/E pair d2-L / d4-H (cf 3.5e-3) | pilot pool of cf 3.5e-3: d4-H has **405** problems in range from 10,000 candidates (D cells 667–961): **go criterion (≥ 500) not met**; handed over as the v2 result, next step is a team decision |
| go/no-go | — | — | Sun Nov 22 |

## Tree-v1 stage A: can each operator be learned? (2026-11-16)

`configs/pretrain/tree/stage_a.yaml`: atoms only (the 24 tree-v1 operators, uniform; `comp_frac` 0), 2000 steps, then
`calibrate.py --e1-only`. Pass rule: every operator ≥ 98% greedy on its 50 E1 problems (Wilson rule). Results in
`results/pretrain_v2/models/tree-v1/pt-tree-a-atoms-s0/`.

| operator | arity | group | E1 at step 1000 | E1 at step 2000 |
|---|---|---|---|---|
| rotate_by_sum (F10) | 2 | H | .16 | **.12** (fails) |
| scatter_add (F13) | 2 | L | .02 | .96 (passes: Wilson upper bound ≥ .98) |
| prefix_add (F12) | 2 | L | .36 | .98 |
| odd_pos_m1 (F03) | 1 | L | 1.00 | .98 |
| the other 20 operators | | | ≥ .98 | 1.00 |

E1 overall .960 (H .927, L .993). rotate_by_sum (rotate x left by sum(y) mod n) shows no progress between steps 1000
and 2000; its output depends on a sum over the whole second operand mod n (6–10), a parity-like reduction that
transformers learn poorly from examples. scatter_add and prefix_add, which also aggregate over positions, were learned
late. By the design rule the failing operator is replaced before stage B: rotate_by_first (rotate x left by y[0] mod
n) keeps the rotation set by the other operand without the global sum (design.md decision 12), and stage A is re-run.
The first recipe's eval sets are kept in `data/tree-v1-rotate_by_sum/`.

**Re-run on the revised table** (pt-tree-a2-atoms-s0, same config): **PASS**, E1 .997 (H 1.000, L .993).

| operator | E1 at step 1000 | E1 at step 2000 |
|---|---|---|
| prefix_add (F12) | .32 | .96 (48/50; Wilson upper bound ≥ .98) |
| scatter_add (F13) | .72 | .96 (48/50) |
| rotate_by_first (F10) and the other 21 operators | ≥ .92 | 1.00 |

prefix_add and scatter_add, the two operators that aggregate over positions, are learned last in both runs; in
stage B (9500 steps, almost all atoms) they get ~5× more training. 15 min training; final loss 2.4e-4.

Training: 15 min on one node (0.82M tokens/s, final loss .0047). Eval sets for tree-v1 (`data/tree-v1/eval/v1/`:
E1 1200, E2 1800, E3 900, E4 800, probe 575, calib 4600) were generated on the node in 33 s.

**Composition length** (for the stage-B comp_frac levels; `Mixture` estimate): a tree-v1 composition example has 137
target tokens on average, a chain-v1 one 69, so at equal `comp_frac` tree-v1 sees half as many composition examples.
Mean `n_ops` per depth 2 / 3 / 4: 2.94 / 4.40 / 6.00.

## v2 result models (handed over 2026-11-18)

Round 2 of the pilot is the v2 pretraining result; what to do with it is left to the team.

| run | comp_frac | calib mean p d2 / d3 / d4 / d5 | gate | pilot pool | Hugging Face folder |
|---|---|---|---|---|---|
| `pt-tree-mix-cf0.0035-s0` | 3.5e-3 | .68 / .31 / .10 / .01 | PASS | d2-L 914, d2-mixed 961, d2-H 667, d4-H 405 in range | `pt-tree-mix-cf0.0035-s0/` (+ `labels.parquet`, `calib/pilot.json`, `calib/pilot_candidates.jsonl`) |
| `pt-tree-mix-cf0.006-s0` | 6e-3 | .84 / .67 / .40 / .13 | PASS | not labelled | `pt-tree-mix-cf0.006-s0/` |

- Hugging Face (private org repo, ask Yupeng for an invite):
  [`rlvr-cs291k/rlvr-pretrain-v2`](https://huggingface.co/rlvr-cs291k/rlvr-pretrain-v2). Load with
  `rlvr.model.load_model` after `snapshot_download` (the model card has the snippet); the tree-v1 task config must be
  the current one (rotate_by_first, decision 12), as the checkpoint's task fingerprint is checked.
- CFS copy: `/global/cfs/cdirs/m4788/yupengsu/rlvr/checkpoints/tree-v1/`, `.../runs/tree-v1/` (run files without
  intermediate checkpoints) and `.../data/tree-v1/labels/`.
- Small files (gate, calib heatmap, pilot report, eval logs) are also in `results/pretrain_v2/models/tree-v1/`.
- Open choices: (1) no-go as written and Jay's v1 pool expansion (~2–2.5 node-hours); (2) relax the 10,000-candidate
  cap for d4-H (~3,000 more candidates, ~0.4 node-hours, expected ~500); (3) a model at comp_frac ~4.5e-3, between the
  two, where d2-L and d4-H should both be near 5–6% in range (~3.7 node-hours); or use the 405 / 914 problems as they
  are if the RL design needs fewer than 500 per arm. Before training more tree-v1 models, cap the tokens per
  micro-batch (the cf 2e-3 run ran out of GPU memory, see round 2).

## Tree-v1 stage B round 2: more depth-4 compositions (2026-11-17)

`configs/pretrain/tree/mix.yaml` (decision 13): the round-1 recipe with the composition depth mix {2: .3, 3: .35,
4: .35} instead of {2: .6, 3: .35, 4: .05}, still h_ratio 0.5 and no H-only depth, 9500 steps, at comp_frac 2e-3,
3.5e-3 and 6e-3. Training and calibration only (`--skip-pool`); the D/E pairs below are computed from the calib labels
with `rlvr/pilot.py` (decision 11), without the 10,000-candidate pool.

Mean calib p (T = 1, k = 32, 200 problems per (depth, group) cell), and the calib share of non-skippable problems
with p̂ in [0.4, 0.6] ("rate") for the cells of the best pairs:

| comp_frac | d2 (H / L / mixed) | d3 (H / L / mixed) | d4 (H / L / mixed) | d5 (H / L / mixed) | spread signal d2 / d3 / d4 | gate |
|---|---|---|---|---|---|---|
| 2e-3 | training ran out of GPU memory at step ~7700 (no final model) | | | | | |
| 3.5e-3 | .765 / .580 / .682 | .394 / .275 / .272 | .158 / .054 / .100 | .020 / .013 / .010 | .75 / .64 / .35 | PASS |
| 6e-3 | .965 / .718 / .836 | .843 / .537 / .637 | .548 / .267 / .372 | .222 / .066 / .109 | .32 / .56 / .54 | PASS |

| comp_frac | best pairs (D / E: rates) | expected per cell from 10,000 candidates |
|---|---|---|
| 3.5e-3 | d2-L / d4-H: .100 / .055; d2-mixed / d4-H: .075 / .055; d2-H / d4-H: .050 / .055 | ~500–550 (E cell d4-H is the bottleneck) |
| 6e-3 | d3-L / d5-H: .055 / .040; d3-mixed / d5-H: .045 / .040; d2-mixed / d5-H: .035 / .040 | ~400 |

Pilot pool of pt-tree-mix-cf0.0035-s0 (`calib/pilot.json`, `calib/relabel_report.json`; 1 node, 1.93 node-h): the
cells of the best pairs get 10,000 RL-pool candidates each, labelled with k = 32; the 6,815 non-skippable ones with
p̂ in [0.3, 0.7] are relabelled with k = 64. "In range" = non-skippable with relabelled p in [0.4, 0.6].

| cell | mean p̂ | skippable | first p̂ in range | relabelled | in range | per input length 6 / 7 / 8 / 9 / 10 | mean n_ops / n_binary / n_ternary of in-range |
|---|---|---|---|---|---|---|---|
| d2-L | .614 | 9.3% | 1110 | 2093 | **914** | 176 / 169 / 175 / 181 / 213 | 3.25 / 1.02 / 1.49 |
| d2-mixed | .655 | 8.1% | 1103 | 2171 | **961** | 194 / 175 / 199 / 184 / 209 | 3.55 / 0.96 / 1.75 |
| d2-H | .751 | 7.8% | 867 | 1654 | **667** | 128 / 147 / 117 / 134 / 141 | 3.50 / 0.92 / 1.75 |
| d4-H | .147 | 20.6% | 473 | 897 | **405** | 80 / 84 / 79 / 86 / 76 | 4.29 / 1.41 / 1.95 |

- **No go:** every pair (d2-L, d2-mixed or d2-H with d4-H) can supply **405** problems per cell with one shared length
  mix, below the 500 of the go criterion. The E cell d4-H is the bottleneck: 4.05% of its candidates are in range
  (calib estimate 5.5%), so 500 would need ~12,400 candidates. The D cells have 1.3–2.4× what is needed.
- **The second labelling shows no selection bias:** for d4-H the first-selected problems have mean p̂ .498 and
  relabelled mean .492 (61.5% stay in range); for d2-L .504 and .508 (57.5%).
- **Structure covariates differ** between D and E as intended (n_ops 3.3 vs 4.3); input length is balanced by
  construction.

- **The depth curve is flatter.** Mean p falls by 2–4× from depth 2 to 4 (round 1: ~200×), with symmetric H/L
  exposure. Both models pass the gate's spread check (round 1 failed it). E1 is 1.000 for both.
- **p ≈ 0.5 problems are still scarce within a cell.** Per-problem p is bimodal: d4-H at 6e-3 has mean p .55 but only
  7.5% of its problems in [0.4, 0.6]; the shares are 3–10% in every cell (v1: ≤ 12.5%). The rates come from 200
  problems per cell (±3% standard error at 5%).
- **H is easier than L at every depth** (e.g. d4 .55 vs .27 at 6e-3) although exposure is even (h_ratio 0.5): the
  operators differ in difficulty. The v1 pair d2-L / d4-H appears again, now without uneven exposure; d2-H / d4-H is a
  same-group alternative.
- **E3 (depth 6–8) stays near 0** (greedy .002 at 3.5e-3, .007 at 6e-3), as in v1 (decision 7).
- **Run notes.** cf 2e-3 ran out of GPU memory in a training step between steps 7500 and 8000 (38 GB allocated on one
  GPU), likely the longest micro-batch: the leftover tail of each 20,000-example length bucket joins the last group, and
  this mix has many long compositions. Its node was released at once by shrinking the allocation; after the resize srun
  could not start new steps in the job, so the two calibrations ran over ssh on their nodes. Calib labelling took
  22 min per model (`label.batch_size=8`).

## Tree-v1 stage B: do trees give enough p ≈ 0.5 problems? (2026-11-16)

`configs/pretrain/tree/mid.yaml`: the tree-v1 recipe (h_ratio 0.5, no H-only depth) at the full length (9500 steps),
three runs in parallel at comp_frac 3.2e-4 (v1 mid's token share), 6.4e-4 (v1 mid's number of composition examples,
as tree compositions are 2× longer) and 1.28e-3. Then calibration and the pilot pool by decision 11. Results in
`results/pretrain_v2/models/tree-v1/`.

Mean calib p (T = 1, k = 32, 200 problems per (depth, group) cell):

| comp_frac | d1 | d2 (H / L / mixed) | d3 | d4 | d5 | spread signal d2 / d3 / d4 |
|---|---|---|---|---|---|---|
| 3.2e-4 | 1.00 | .002 (.005 / .001 / .001) | .000 | .000 | 0 | .03 / .00 / .00 |
| 6.4e-4 | 1.00 | .038 (.058 / .035 / .020) | .0003 | .0001 | 0 | .24 / .02 / .00 |
| 1.28e-3 | 1.00 | .387 (.548 / .291 / .322) | .050 | .002 | 0 | .67 / .20 / .03 |

- **No go in this round.** At 1.28e-3 the calib shares of non-skippable problems with p̂ in [0.4, 0.6] are 5.5–12.5% at
  depth 2, 2% at depth 3 and 0 at depths 4–5. No D/E pair with E.depth ≥ D.depth + 2 has a nonzero share in both
  cells (depth 1 is at p = 1), so the pilot skipped the candidate pool for every run (`calib/pilot.json`: go false).
  All three fail the spread check; E1 is 1.000 in all three.
- **Why.** A depth gap of 2 is a large step in tree-v1 (mean n_ops 2.9 at depth 2, 6.0 at depth 4): when depth 2
  reaches p ≈ 0.4, depth 4 is still near 0. In v1, d2-L and d4-H were brought together by uneven exposure (h_ratio
  0.9 and H-only depth 4); the tree-v1 pilot recipe removes that on purpose (h_ratio 0.5, no H-only depth).
  Interpolating the logits between 6.4e-4 and 1.28e-3, depth 3 reaches ~0.45 near comp_frac 2e-3 and depth 4 near
  4.5e-3, where depth 2 is close to 1.
- **Skippable problems** (decision 6) grow with depth: ~8% at depth 2, ~15% at depth 3, 19–28% at depth 4, 26–42% at
  depth 5.
- **Run notes.** Calibration of the first attempt ran out of GPU memory in the NCCL gather after the calib labelling
  (40 GB A100s; tree-v1 completions up to 768 tokens). Fixed by emptying the CUDA cache before the gather; the three
  calibrations were re-run with `label.batch_size=8` on the saved checkpoints. Calib labelling took 20–34 min per run
  (v1: 5 min); weak models are slowest because they often do not stop before 768 tokens.

## Superseded variant: 24 unary + 4 binary operators (2026-11-15 to 16)

The pretrain branch first implemented v2 as the 24 v1 unary functions plus four binary operators (add, alternate,
splice, max as F24–F27), with `depth` = `n_ops` and eval v3. It was dropped on 2026-11-16 for tree-v1 (decision 10).
Code and data are in git history (commits cb99f24 to c137d10).

| step | run | result |
|---|---|---|
| smoke run (pipeline check) | pt-v2-smoke-s0 | every stage runs on GPU, incl. the pilot pool |
| stage A: binary operators | pt-v2a-atoms-s0 | **PASS**: E1 1.000 for all 24 unary + 4 binary operators |
| stage B: p ≈ 0.5 supply | pt-v2mid-cf{0.0002,0.00032,0.0005}-s0 | cancelled after 36 min (step ~4300 of 9500), not used |

### Smoke run (2026-11-15)

`configs/v2/smoke.yaml`: the mid v2 recipe at comp_frac 0.01 for 300 steps, then `calibrate.py --calib-per-cell 4`
and a pilot pool of 40 candidates per cell. It only checks that the code runs end to end on the GPUs; its p ranges
are widened to [0, 1] so that the pilot branch runs on an undertrained model. Not an experiment.

- Training: token shares atom .990 / chain .005 / tree .005 (comp_frac 0.01, tree_frac 0.5, as configured).
- Eval v3, calibration, pilot cell choice, candidate labelling (k = 32), relabelling (k = 64), `pilot.json` and
  `relabel_report.json` all ran. `calibrate.py` exited 1 because the gate fails after 300 steps, as expected.
- 8 minutes on one node.

### Stage A: can each binary operator be learned? (2026-11-15)

`configs/v2/stage_a.yaml`: atoms only (24 unary + 4 binary, uniform), 2000 steps, then `calibrate.py --e1-only`.
Pass rule: every operator ≥ 98% greedy on its 50 E1 problems (Wilson rule, as the depth-1 gate).

| step | E1 unary H | E1 unary L | E1 binary | all (1400 problems) |
|---|---|---|---|---|
| 1000 | 1.000 | .998 | .995 | .999 |
| 2000 | 1.000 | 1.000 | 1.000 | 1.000 |

Every function and operator is at 50/50 at step 2000 (gate PASS, no function below the minimum). Training took 14
minutes; final training loss 1.8e-5.

## Compute ledger (NERSC GPU node-hours, cap 50)

| date | job | what | node-h | total |
|---|---|---|---|---|
| ≤ 2026-11-08 | — | v1 (see [`pretrain_v1_results.md`](pretrain_v1_results.md)) | 21.09 | 21.09 |
| 2026-11-15 | 61385591 | smoke run + stage A (1 node, 0:23:31) | 0.39 | 21.48 |
| 2026-11-16 | 61417711 | stage B, cancelled (3 nodes, 0:36:10) | 1.81 | 23.29 |
| 2026-11-16 | 61422984 | tree-v1 eval sets + stage A (1 node, 0:17:18) | 0.29 | 23.58 |
| 2026-11-16 | 61428426 | tree-v1 eval sets + stage A re-run (1 node, 0:17:41) | 0.29 | 23.87 |
| 2026-11-16 | 61430769 | tree-v1 stage B, 3 runs incl. calibration re-runs (3 nodes, 3:02:12) | 9.11 | 32.98 |
| 2026-11-17 | 61495943 | tree-v1 stage B round 2 (3 nodes 1:16:26, then 2 nodes to 1:52:04) | 5.01 | 37.99 |
| 2026-11-17 | 61512421 | tree-v1 pilot pool of pt-tree-mix-cf0.0035-s0 (1 node, 1:55:55) | 1.93 | 39.92 |
