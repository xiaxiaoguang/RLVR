# v2 pretraining results (tree problems)

Results of the v2 pilot ([`design.md` "v2 pilot"](design.md#v2-pilot)) and, after the go/no-go on Sun Nov 22, of the
v2 base models. The task change itself is described in [`design.md` "v2 revision"](design.md#v2-revision-tree-problems);
the implementation choices the design left open are in
[`design.md` "Implementation (2026-11-15)"](design.md#implementation-2026-11-15).

- Full job log: `/pscratch/sd/y/yupengsu/rlvr/runs/NERSC_LOG.md` (scratch). A snapshot is copied to
  `results/pretrain_v2/NERSC_LOG.md` with every results commit.
- Small per-run files (gate.json, pilot.json, relabel_report.json, stats.json, eval.jsonl, run_config.yaml, calib
  heatmap): `results/pretrain_v2/models/<run>/`, collected from scratch by `scripts/collect_results.sh`.

## Status (2026-11-16)

| step | run | result | status |
|---|---|---|---|
| smoke run (pipeline check) | pt-v2-smoke-s0 | every stage runs on GPU, incl. the pilot pool | done |
| stage A: binary operators | pt-v2a-atoms-s0 | E1 1.000 for all 24 unary + 4 binary operators | **PASS**, operator set fixed |
| stage B: p ≈ 0.5 supply | pt-v2mid-cf{0.0002,0.00032,0.0005}-s0 | — | **on hold**: started 2026-11-16, cancelled after 36 min (see below) |
| go/no-go | — | — | Sun Nov 22 |

Operator set (fixed by stage A): **add, alternate, splice, max** (F24–F27). The backups (sub, min) are not needed.

**Stage B on hold (2026-11-16).** Master now also has a separate tree task, tree-v1 (`docs/tree_v1.md`, commit
be60eb6): 12 unary operators (12 of the 24 dropped) + 8 binary + 4 ternary, reshuffled to F00–F23. It differs from
the v2 design here (24 unary + 4 binary as F24–F27) in the operator set, the depth definition, the held-out edges and
the length limits. Stage B was cancelled so that the team can first agree on one operator set; it is re-run once
that is decided. The three cancelled runs reached step ~4300 of 9500 and are not used.

## Smoke run (2026-11-15)

`configs/v2/smoke.yaml`: the mid v2 recipe at comp_frac 0.01 for 300 steps, then `calibrate.py --calib-per-cell 4`
and a pilot pool of 40 candidates per cell. It only checks that the code runs end to end on the GPUs; its p ranges
are widened to [0, 1] so that the pilot branch runs on an undertrained model. Not an experiment.

- Training: token shares atom .990 / chain .005 / tree .005 (comp_frac 0.01, tree_frac 0.5, as configured).
- Eval v3, calibration, pilot cell choice, candidate labelling (k = 32), relabelling (k = 64), `pilot.json` and
  `relabel_report.json` all ran. `calibrate.py` exited 1 because the gate fails after 300 steps, as expected.
- 8 minutes on one node.

## Stage A: can each binary operator be learned? (2026-11-15)

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
