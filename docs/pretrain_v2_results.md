# v2 pretraining results (tree problems)

Results of the v2 pilot ([`design.md` "v2 pilot"](design.md#v2-pilot)) and, after the go/no-go on Sun Nov 22, of the
v2 base models. Since 2026-11-16 the v2 task is Hanzhong's **tree-v1** ([`tree_v1.md`](tree_v1.md), design.md decision
10). An earlier variant (24 unary + 4 binary operators) was piloted first; its results are kept below as a record.

- Full job log: `/pscratch/sd/y/yupengsu/rlvr/runs/NERSC_LOG.md` (scratch). A snapshot is copied to
  `results/pretrain_v2/NERSC_LOG.md` with every results commit.
- Small per-run files (gate.json, relabel_report.json, stats.json, eval.jsonl, run_config.yaml, calib heatmap):
  `results/pretrain_v2/models/<run>/`, collected from scratch by `scripts/collect_results.sh`.

## Status (2026-11-16)

| step | run | result | status |
|---|---|---|---|
| tree-v1 stage A: atomic skills of the 24 operators | pt-tree-a-atoms-s0 | 23 of 24 pass; rotate_by_sum E1 .12 | **FAIL** on one operator; replaced by rotate_by_first (decision 12) |
| tree-v1 stage A, revised table | pt-tree-a2-atoms-s0 | all 24 pass; E1 .997 | **PASS**, operator table fixed |
| tree-v1 stage B: p ≈ 0.5 supply | pt-tree-mid-cf{0.00032,0.00064,0.00128}-s0 | no D/E pair at any comp_frac | **no-go** for this round; next step is a team decision |
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
