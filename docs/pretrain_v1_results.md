# v1 pretraining results

Training results of the v1 base models, round by round, and how comp_frac was chosen.

- Full job log: `/pscratch/sd/y/yupengsu/rlvr/runs/NERSC_LOG.md` (scratch). A snapshot is copied to
  `results/pretrain_v1/NERSC_LOG.md` with every results commit.
- Small per-run files (gate.json, relabel_report.json, stats.json, eval.jsonl, run_config.yaml, calib heatmap) and the
  diagnosis outputs: `results/pretrain_v1/`, collected from scratch by `scripts/collect_results.sh`.
- Model checkpoints and RL-pool labels (`data/labels/*.parquet`) are too large for git and are still only on scratch.
  They will be copied to `/global/cfs/cdirs/m4788/yupengsu/rlvr/` once the models are final.

## Status (2026-11-08)

| model | candidate run | comp_frac | design target | measured | gate (hard checks) | status |
|---|---|---|---|---|---|---|
| weak | pt-weak-cf0.000125-s0 | 1.25e-4 | d3 0.05–0.2 | d3 .114 | PASS | on target, ready to finalise |
| mid | pt-mid-cf0.00032-s0 | 3.2e-4 | d3 0.35–0.55 | d3 .381 | **FAIL** (spread, matched_cells) | on target; gate rules pending |
| strong | pt-strong-cf0.00083-s0 | 8.3e-4 | d4 0.4–0.6 | d4 .353 (d4-H .594) | PASS | pending: accept, or try 9e-4 / 9.5e-4 |
| midbal | pt-midbal-cf0.00032-s0 | 3.2e-4, h_ratio 0.5 | d3 0.35–0.55 | d3 .549 | **FAIL** (spread, matched_cells) | gate rules pending |

All four candidates are trained and fully calibrated: calib heatmap, 35,000-problem RL-pool labelling, and relabelling
of the matched cells.

Open questions for the team (details under "Open questions" below):

1. Should spread check only d2–d4? By design depth 5 is not in pretraining, so a spread check that includes d5
   cannot pass at any comp_frac.
2. Should the shared RL pool be expanded? matched_cells needs at least 120 problems per length; with the current
   5,000 candidates per cell, mid cannot reach that.
3. Should midbal's matched_cells become a warning?
4. strong: accept 8.3e-4, or one more round?

Not done yet: renaming the candidates to pt-{weak,mid,strong,midbal}-s0 and building the matched_p datasets. Both wait
for the questions above.

Compute: 21.09 of 50 NERSC GPU node-hours used (see "Compute" below).

## Setup

- 85.0M-parameter model; 9500 steps, ~192k supervised tokens per step (1.84B in total); ~70 min on 4×A100 including
  start-up and evaluation.
- seed 0, pos_offset_max 128, h_ratio 0.9 (midbal 0.5). Everything else uses the defaults in `rlvr/params.py`.
- Calibration: 200 calib problems per cell, T=1.0, k=32. Tables show the mean p̂ per depth.

## Pilot sweep (2026-11-04)

`scripts/sweep.py`: 400 steps, ~4% of the full training tokens. 50 calib problems per cell, k=16, pos_offset_max=128.

| comp_frac | d2 | d3 | d4 | d5 |
|---|---|---|---|---|
| 0.0003 | 0 | 0 | 0 | 0 |
| 0.001 | 0 | 0 | 0 | 0 |
| 0.003 | .007 | 0 | 0 | 0 |
| 0.01 | .112 | .003 | 0 | 0 |
| 0.03 | .461 | .096 | .002 | .001 |
| 0.1 | .813 | .516 | .103 | .010 |

Extrapolating as "pilot value / 25" (the full run has 25× the tokens) gave weak = 0.001 and strong = 0.01.

## Round 1: first full runs (2026-11-05, commit 456e41d)

| run | comp_frac | comp. examples | d2 | d3 | d4 | d5 | d2-L | d4-H | E1 | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| pt-weak-s0 | 0.001 | 26,723 | .951 | **.871** | .681 | .017 | .914 | .846 | 1.000 | PASS, warnings target, spread |
| pt-strong-s0 | 0.01 | 269,092 | .999 | .987 | **.973** | .021 | .998 | .991 | 1.000 | PASS, warnings target, spread, final_loss |

Both models overshoot their design targets by far (weak: d3 0.05–0.2; strong: d4 0.4–0.6). The gate passes because
the only hard check for weak/strong is depth1_greedy. Both runs only got the calib heatmap; RL-pool labelling was
stopped part-way, so they have no `data/labels/` file.

Greedy accuracy during training (weak, E2 set) shows that composition is learned in the second half of training:

| step | 2000 | 4000 | 6000 | 8000 | 9500 |
|---|---|---|---|---|---|
| d3-H | .04 | .21 | .77 | .93 | .99 |
| d3-L | .00 | .02 | .42 | .75 | .83 |
| d4-H | .00 | .01 | .37 | .83 | .92 |
| d4-L | .00 | .01 | .14 | .39 | .69 |

### Conclusions

- The pilot extrapolation overestimated the needed comp_frac by ~10×. In full training, once the atomic functions are
  learned, composition needs very few examples: weak saw only 149 d3-L examples and reached d3-L .81, and 1,349 d4-H
  examples for d4-H .85. d4-L reached .59 without a single training example.
- There is a cliff between trained depths (d2–d4) and untrained ones (d5 and up): d5 is ~0.02 for both models.
- Rough comp_frac estimates, from a logit-linear fit through 0.001 and 0.01 and from weak's training trajectory, give
  ranges: strong ~5e-4–7e-4, mid ~1.3e-4–5e-4, weak ~2e-5–4e-4.
- Next step chosen: a comp_frac sweep at the full training length (round 2).

### Outputs (scratch)

- `runs/pt-{weak,strong}-s0/`: metrics.jsonl, eval.jsonl, stats.json, calib/ (calib_labels.parquet, heatmap.png,
  interim gate.json).
- `checkpoints/pt-{weak,strong}-s0/`: final models. `runs/pt-*/ckpt/step-{2375,4750}/`: intermediate checkpoints.

## Round 2: comp_frac sweep at the full training length (2026-11-05, job 60880260)

Four nodes in parallel, one model per node, all with the full 9500 steps, from `configs/pretrain/mid.yaml` with a
comp_frac override. Calib heatmap only (`calibrate.py --skip-pool`), no RL-pool labelling. Runs are named
pt-full-cf<cf>-s0.

| comp_frac | run | comp. examples | d2 | d3 | d4 | d5 | d2-L | d4-H |
|---|---|---|---|---|---|---|---|---|
| 3e-5 | pt-full-cf0.00003-s0 | 871 | .023 | .001 | .000 | .000 | .008 | .000 |
| 1e-4 | pt-full-cf0.0001-s0 | 2,734 | .352 | .064 | .005 | .000 | .113 | .015 |
| 2.5e-4 | pt-full-cf0.00025-s0 | 6,867 | .683 | .308 | .036 | .005 | .496 | .093 |
| 5e-4 | pt-full-cf0.0005-s0 | 13,490 | .913 | .704 | .287 | .013 | .881 | .524 |
| 1e-3 | pt-weak-s0 (round 1) | 26,723 | .951 | .871 | .681 | .017 | .914 | .846 |
| 1e-2 | pt-strong-s0 (round 1) | 269,092 | .999 | .987 | .973 | .021 | .998 | .991 |

E1 = 1.000 for every run.

- d2-L reaches 0.5 at ~2.5e-4, but d4-H only at ~5e-4. No single comp_frac puts both matched cells near 0.5.
- Per-problem p is bimodal: even when a cell's mean p is near 0.5, at most 12.5% of its problems have p in [0.4, 0.6].
- d5's signal_frac (fraction with 0 < p̂ < 1) is at most 0.065 at every comp_frac, so mid's hard spread check
  (≥ 0.2 at every depth in d2–d5) cannot pass.
- Round-3 comp_frac by logit interpolation: weak 1.25e-4 (d3 ~0.1), mid 3.2e-4 (d3 ~0.45), strong 7e-4 (d4 ~0.5).

## Round 3: four candidates, training + full calibration (2026-11-06, job 60894875)

Four nodes in parallel, each running `scripts/slurm/model_pipeline.sh`: training, calib heatmap, RL-pool labelling
(35,000 problems, k=32), then relabelling with k=64 of the problems whose p̂ falls in [0.4, 0.6]. All four wrote
`data/labels/<run>.parquet`.

Calib mean p (200 problems per cell, k=32):

| model | comp_frac | comp. examples | d2 | d3 | d4 | d5 | d2-H | d2-L | d3-H | d3-L | d4-H | d4-L | gate (hard) | warnings |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| weak | 1.25e-4 | 3,384 | .440 | **.114** | .011 | .001 | .721 | .186 | .251 | .011 | .028 | .002 | PASS | spread, matched_cells |
| mid | 3.2e-4 | 8,800 | .724 | **.381** | .072 | .003 | .914 | .497 | .684 | .131 | .167 | .013 | **FAIL** | |
| strong | 7e-4 | 18,729 | .921 | .739 | **.336** | .004 | .981 | .863 | .897 | .559 | .568 | .167 | PASS | spread, target, matched_cells |
| midbal | 3.2e-4 | 8,790 | .906 | **.549** | .119 | .001 | .938 | .870 | .656 | .438 | .180 | .081 | **FAIL** | |

For mid and midbal, spread and matched_cells are hard checks, and both fail. The target check is within range for
every model except strong (d4 .336 < 0.4); midbal's d3 sits at the top of its range.

spread signal_frac (fraction with 0 < p̂ < 1 per depth; threshold 0.2):

| model | d2 | d3 | d4 | d5 |
|---|---|---|---|---|
| mid | .583 | .697 | .353 | **.037** |
| midbal | .402 | .802 | .498 | **.015** |

mid and midbal fail only at d5.

Matched cells: problems with relabelled p in [0.4, 0.6]; in brackets, the count per length 6 / 7 / 8 / 9 / 10.
The gate needs ≥ 300 in total and ≥ 120 per length.

| model | d2-L | d4-H |
|---|---|---|
| mid | 369 (73 / 75 / 81 / 84 / 56) | 326 (103 / **31** / 103 / **18** / 71) |
| midbal | 159 (12 / 29 / 37 / 37 / 44) | 391 (101 / 63 / 84 / 63 / 80) |
| strong 7e-4 | 168 | 509 (79 / 135 / 101 / 84 / 110) |
| weak | 210 | 36 |

Problems first selected with p̂ in [0.4, 0.6]: mean p after relabelling (in brackets, the fraction still in range):

| model | d2-L | d4-H |
|---|---|---|
| mid | .506 (.56) | .472 (.57) |
| midbal | .524 (.55) | .483 (.57) |
| weak | .505 (.62) | .422 (.53) |
| strong 7e-4 | .520 (.59) | .507 (.58) |

No model had more than 32 problems first selected in d5-H.

### Conclusions

- **weak is on target:** d3 .114, within 0.05–0.2.
- **mid meets its design target but fails both hard gate checks:**
  - spread fails only at d5, which cannot pass by design.
  - matched_cells fails on the per-length counts: d4-H has only 31 and 18 problems at lengths 7 and 9. Relaxing the
    thresholds alone is not enough; either expand the RL pool (d4-H to ~34,000 problems, d2-L to ~11,000) or change
    how the cells are matched.
- **midbal:** by design it keeps mid's comp_frac and changes only h_ratio (single-factor P6). Its d2-L is too easy
  (.870), so only 159 problems fall in [0.4, 0.6].
- **strong 7e-4:** d4 only .336. Logit interpolation between 7e-4 and 1e-3 put d4 = 0.5 at ~8.3e-4, so strong was
  retrained there (see below).

## d4 length effect diagnosis (2026-11-06, job 60914166)

Observation: d4 accuracy of mid, midbal, strong and 5e-4 is clearly lower at length 7; mid is also low at length 9. It
shows on both the RL pool and the calib set. d2 and d3 do not show it.

Script `scripts/diag_steps.py`; outputs in `results/pretrain_v1/diag-d4len/` (`analyze.txt`, per-model
`analysis.json`, `replay.json`).

- `gen`: for 400 RL-pool d4-H problems per length, plus d3-H and d4-mixed, sample k=8 at T=1 plus one greedy decode.
- `analyze`: check the reasoning chain step by step and record the first wrong step and its error type.
- `replay`: replay the pretraining data stream exactly and count composition examples per depth, group and length.

### Where it fails: step 3 of depth 4

Probability that step 3 is correct given that steps 1–2 are correct:

| model | L6 | **L7** | L8 | L9 | L10 |
|---|---|---|---|---|---|
| mid | .54 | **.19** | .59 | **.15** | .68 |
| midbal | .63 | **.38** | .64 | .29 | .71 |
| 5e-4 | .69 | **.28** | .80 | .82 | .87 |
| strong 7e-4 | .73 | **.46** | .77 | .77 | .89 |

d4-H sampled accuracy:

| model | L6 | L7 | L8 | L9 | L10 |
|---|---|---|---|---|---|
| mid | .204 | .083 | .188 | .046 | .152 |
| midbal | .217 | .122 | .152 | .114 | .181 |
| 5e-4 | .546 | .198 | .550 | .591 | .498 |
| strong 7e-4 | .542 | .338 | .565 | .561 | .560 |

- Almost all errors are wrong_func: at step 3 the model writes the outermost function, skipping function 3, i.e. it
  solves a depth-4 problem as if it had depth 3.
- d4-mixed never appears in pretraining and shows the same length-7 dip, so the problem is the step-3-of-depth-4 skill
  itself, not particular functions.
- The rate of wrong function applications (apply errors) grows smoothly with length, with no dip at length 7. d3 is
  flat across lengths.

### Cause: fewest d4 training examples at length 7 (a seed-0 sampling accident)

The replay matches `stats.json` exactly, both the total number of examples and the count in every composition cell.
d4-H training examples per length:

| model | L6 | **L7** | L8 | L9 | L10 |
|---|---|---|---|---|---|
| mid | 87 | **68** | 101 | 96 | 95 |
| midbal | 87 | **68** | 102 | 96 | 95 |
| 5e-4 | 139 | **102** | 156 | 140 | 131 |
| strong 7e-4 | 199 | **157** | 213 | 188 | 188 |

- Length 7 has the fewest examples in every model. Lengths are sampled uniformly: mid expects ~89 and got 68, about
  2.5 standard deviations low.
- All runs use seed 0. When comp_frac differs, the per-worker Python rng streams diverge briefly and then
  re-synchronise, so models at different levels share most of their composition examples (mid and midbal have almost
  identical d4 counts). The length-7 shortfall is one sampling accident of seed 0, inherited by every model.
- d4 is only 5% of composition examples, about 70–200 per length. Step 3 of depth 4 has to be learned from these few
  dozen examples, separately for each length, so it is fragile.
- Counts do not explain everything: mid has a normal number of length-9 examples but does badly there (midbal shares
  the same d4 examples). Which problems were drawn matters too.
- Not a code bug: d1 = 1.0, and d2 and d3 are flat across lengths.

### Implications and possible fixes

- With another seed the dip should move. The design's mid seed 1/2 replicates (P11) can confirm this.
- The levels share the same data luck, which is good for paired comparisons, but they are not independent replicates.
- To reduce the problem at the root: raise d4's share in comp_depth_mix (listed in the design's risk table), or sample
  composition examples balanced across lengths. Both require retraining mid and midbal.

### Side finding: the reward checks only the final answer

1.4–2.6% of sampled completions have a wrong chain but a correct final answer (e.g. a later sort erases an earlier
error) and still get reward 1. Both the p̂ labels and the RL reward check only the final answer.

## strong retrain (2026-11-06, job 60914163)

pt-strong-cf0.00083-s0: `configs/pretrain/strong.yaml`, comp_frac 8.3e-4, 22,120 composition examples (1,126 at d4);
training and full calibration.

- gate: PASS (depth1 = 1.0), warnings spread, target, matched_cells.
- calib mean p: d2 .919, d3 .759, **d4 .353**, d5 .010.
- d4 by group: d4-H .594, d4-L .161, d4-mixed .304. Overall d4 is still below 0.4.
- d4-H at lengths 6–10: .59 / .54 / .65 / .58 / .61. The length-7 dip is mostly gone.
- Matched cells: d2-L 153 (13 / 12 / 53 / 27 / 48), d4-H 428 (75 / 100 / 92 / 75 / 86).
- First-selected problems after relabelling: d2-L mean .511, still in range .53; d4-H .515 (.58); d5-H .469 (.43).

d4 against comp_frac (same config, seed 0):

| comp_frac | 5e-4 | 7e-4 | 8.3e-4 | 1e-3 |
|---|---|---|---|---|
| d4 training examples | 668 | 945 | 1,126 | 1,349 |
| d4 | .287 | .336 | .353 | **.681** |
| d4-H | .524 | .568 | .594 | .846 |
| d4-L | .108 | .167 | .161 | .586 |

d4 stays on a plateau and then jumps at 1e-3, with only 20% more d4 examples, so the earlier interpolation
("8.3e-4 → 0.5") was wrong. d4-H is already in the target range from 5e-4 to 8.3e-4; overall d4 is held down by d4-L.

## Open questions

1. **Which depths spread checks.** The design keeps depth 5 out of pretraining, and d5's signal_frac is at most 0.065
   at every comp_frac. Proposal: check only d2–d4 and drop the optional E′ (d5-H).
2. **matched_cells and the RL pool size.** With 5,000 candidates per matched cell and a bimodal p, reaching 120
   problems per length is hard.
   - Option 1: expand the shared pool (`data/rl_pool/candidates.jsonl`). New problems are appended at the end, so
     existing rows stay unchanged. Relabelling takes about 2–2.5 node-hours.
   - Option 2: change the matching or the thresholds.
3. **midbal's matched_cells.** midbal's d2-L is too easy. Make the check a warning, or give midbal its own D/E
   datasets?
4. **strong.**
   - Option 1: accept 8.3e-4. By design strong's target check is only a warning; d4-H is already on target, and d3 for
     weak / mid / strong is .114 / .381 / .759.
   - Option 2: try 9e-4 and 9.5e-4 in parallel, about 4 node-hours.

## Compute (NERSC GPU node-hours, cap 50)

| job | what | nodes × hours | total |
|---|---|---|---|
| 60836696 | interactive: benchmark + pilot sweep + labelling benchmark | 1 × 1.16 | 1.16 |
| 60839035 | pt-weak, NODE_FAIL after a DVS hang (no output) | 1 × 0.49 | 1.65 |
| 60839036 | pt-strong, NODE_FAIL after a DVS hang (no output) | 1 × 0.50 | 2.15 |
| 60876535 | interactive: pt-strong-s0 training + calib heatmap | 1 × 1.43 | 3.58 |
| 60876536 | interactive: smoke test + pt-weak-s0 training + calib heatmap | 1 × 1.43 | 5.01 |
| 60880260 | 4 nodes: round-2 sweep, comp_frac 3e-5 / 1e-4 / 2.5e-4 / 5e-4 | 4 × 1.37 | 10.51 |
| 60894875 | 4 nodes: round 3, weak / mid / strong / midbal, training + full calibration | 4 × 2.11 | 18.95 |
| 60914166 | 1 node: d4 length diagnosis | 1 × 0.16 | 19.11 |
| 60914163 | 1 node: strong 8.3e-4, training + full calibration | 1 × 1.98 | 21.09 |

## Infrastructure

- Perlmutter compute nodes reach CFS and HOME through DVS. On 2026-11-05 the conda env on CFS hung in DVS and two jobs
  ended in NODE_FAIL. The env is now rebuilt on scratch from the lock file (see `scripts/slurm/env.sh`).
- Interactive nodes: `scripts/slurm/model_pipeline.sh` runs 4-GPU training and then 4-GPU calibration on one node.
  Labelling runs at ~730 seq/s on 4 GPUs; full calibration (calib, RL pool and relabelling) takes ~35–50 min per
  model.
