# NERSC pretraining log (RLVR v1)

Running log for the v1 pretraining on Perlmutter: what has been done, what is pending, decisions.
Newest entries at the bottom of each section. Times are login-node local (`date`).

## Task (from the user, 2026-11-04)
Train the four v1 base models (weak, mid, strong, midbal) with the v1 recipe, pass the gate,
label RL-pool pass rates, build the three matched_p datasets.
1. Env from environment.yml + requirements.lock.txt; offline tests on login.
2. Interactive node (1 node, ~1.5 h): `pretrain.py --bench 100` (keep 9500 steps regardless),
   then `sweep.py` on 4 GPUs -> propose comp_frac (weak d3 ~0.1, mid d3 ~0.4-0.5, strong d4 ~0.5).
   **Wait for user confirmation before changing configs.**
3. One Slurm script per model (regular, 1 node, 2 h, 4-GPU torchrun, then calibrate --skip-pool).
   Submit weak + strong first; show heatmaps + gate.json; after confirmation submit mid + midbal.
4. Full calibration: shared QOS, 1 GPU per model, `--dependency=afterok:<train job>`.
5. Report per model: gate.json (hard pass? warnings), d2-L / d4-H counts per str_len,
   mean second-label p of problems first-selected at p_hat in [0.4, 0.6]; then build the 3 matched_p datasets.

Rules: confirm with the user before every salloc / sbatch; status checks >= 5 min apart; login node
only edits / submits / checks (offline tests are the exception). Commit scripts to `pretrain`
(no Claude attribution lines), never push master, never force-push. Budget ~8-9 node-hours,
**stop and ask above 15**. Storage: follow /global/u2/y/yupengsu/STORAGE_POLICY.md.

## Layout (per STORAGE_POLICY.md)
- Code: /global/u2/y/yupengsu/projects/RLVR (branch pretrain). Not cloned to scratch.
- Outputs: runs/, checkpoints/, data/labels/, data/pretrain/ are symlinks into /pscratch/sd/y/yupengsu/rlvr/.
- Keep: /global/cfs/cdirs/m4788/yupengsu/rlvr/{checkpoints,data,results}/ (rsync -a, then rsync -a -c -n -i to verify).
- Conda env: /pscratch/sd/y/yupengsu/conda_envs/rlvr (since 2026-11-05; the CFS env was hit by a compute-node DVS hang
  and deleted at the user's request). Common has only ~3.5 GiB left, too small for torch cu13. If purged, rebuild
  from environment.yml + requirements.lock.txt (recipe in scripts/slurm/env.sh, ~5 min with the scratch caches).
  Caches: CONDA_PKGS_DIRS / PIP_CACHE_DIR under /pscratch/sd/y/yupengsu/cache/.
- Slurm account: GPU jobs use `m4788_g` (the `m4788` association has no GPU QOS).

## Node-hour ledger
| job | what | nodes x hours (charged) | running total |
| --- | --- | --- | --- |
| 60836696 | interactive: bench + sweep + label bench | 1 x 1.16 | 1.16 |
| 60839035 | pt-weak, NODE_FAIL after DVS hang (no work done) | 1 x 0.49 | 1.65 |
| 60839036 | pt-strong, NODE_FAIL after DVS hang (no work done) | 1 x 0.50 | 2.15 |
| 60876535 | interactive: pt-strong-s0 train + calib heatmap (pool labelling stopped) | 1 x 1.43 | 3.58 |
| 60876536 | interactive: smoke test + pt-weak-s0 train + calib heatmap (pool stopped) | 1 x 1.43 | 5.01 |
| 60880260 | interactive 4 nodes: full-budget sweep cf 3e-5/1e-4/2.5e-4/5e-4, train + calib heatmap | 4 x 1.37 | 10.51 |
| 60894875 | interactive 4 nodes: weak 1.25e-4 / mid 3.2e-4 / strong 7e-4 / midbal 3.2e-4, train + full calibration | 4 x 2.11 | 18.95 |
| 60914166 | interactive 1 node: d4 length diagnosis (diag_steps.py gen + replay + analyze) | 1 x 0.16 | 19.11 |
| 60914163 | interactive 1 node: strong 8.3e-4 train + full calibration | 1 x 1.98 | 21.09 |

## Log
- 2026-11-04: read README, design doc (pretraining / pilot sweep / gate / labelling), rlvr/gate.py,
  configs/pretrain/*.yaml, scripts. Started env build (log: runs/env_setup.log).
- 2026-11-04: wrote scripts/slurm/{env.sh,pretrain_job.sh,pt-{weak,strong,mid,midbal}.sbatch,calib.sbatch},
  scripts/relabel_report.py (+ test in tests/test_gate.py). Not committed yet.
  Training job exits non-zero if training fails OR the quick gate fails a hard check, so the
  afterok full calibration is skipped (--kill-on-invalid-dep=yes) for a model that must be retrained.
  Quick calibration output copied to runs/pt-*/calib-quick/ (full calibration rewrites calib/).
- 2026-11-04: user approved the interactive salloc (1 node, 1.5 h, bench + sweep with --pos-offsets 128 0).
  Waiting for the env build (conda on CFS is slow) + offline tests before submitting.
- 2026-11-04 20:22: env ready: /global/cfs/cdirs/m4788/yupengsu/conda_envs/rlvr (6.6 GB), torch 2.14.1+cu130,
  transformers 5.20.2, trl 1.16.0, pandas 3.1.0 (log runs/env_setup.log). Use: export PATH=<env>/bin:$PATH.
- 2026-11-04: offline tests on login (CPU): 31 passed. Submitted salloc (interactive, 1 node, 1.5 h, -J rlvr-pilot, --no-shell).
- 2026-11-04 20:23: salloc granted: job 60836696 on nid001137 (interactive, ends 21:53). Run steps with
  `srun --jobid=60836696 -N1 -n1 -c 128 --cpu-bind=none --gpus-per-node=4 bash -c 'source scripts/slurm/env.sh && ...'`.
- 2026-11-04 20:25: bench (mid, 4 GPUs, 100 steps): 0.437 s/step, 671k tok/s, MFU 0.274; suggested 8239 steps for 60 min.
  Keeping 9500 steps (user) -> ~69 min training + ~5 evals; fits the 2 h walltime. Log runs/bench-mid.log, runs/pt-mid-s0/bench.json.
- 2026-11-04 20:26: started sweep: python scripts/sweep.py --gpus 0 1 2 3 --pos-offsets 128 0 (driver log runs/sweep-driver.log,
  per-run logs runs/sweep/*.log). **Remember: scancel 60836696 as soon as it finishes.**
- 2026-11-04 20:47: sweep wave 1 (po128, cf 0.0003-0.01) done. Pilot "gate FAIL" = hard check (E1 / spread) on 400-step models: expected.
  Labelling speed in the pilot calibrations: 1150 x 16 seqs in ~200 s = ~92 seq/s per GPU at 512 seqs/batch
  (~22 ms per decode step; a batch runs to 256 tokens if any sequence lacks <eos>). At that rate the quick
  calibration (4600 x 32) is ~27 min and the full one (35k x 32 + relabel) ~4-5 h on one GPU.
  Plan: after the sweep, time label.batch_size 16/32/64 (seq/s + peak memory) with /pscratch/sd/y/yupengsu/rlvr/tmp/label_bench.py
  on a pilot checkpoint, then set label.batch_size + the calib walltime.
- 2026-11-04 ~21:15: po128 pilot results (calib 50/cell, k=16; mean p_hat d2/d3/d4/d5):
  cf 0.0003: 0/0/0/0; 0.001: 0/0/0/0; 0.003: .007/0/0/0; 0.01: .112/.003/0/0; 0.03: .461/.096/.002/.001; 0.1: .813/.516/.103/.010.
  E1 0.996-0.999 everywhere. d4 never reaches 0.5 -> strong target outside swept range.
  E2 greedy d3 at step 200 -> 400 (2x tokens, cf 0.1): 0.024 -> 0.613, larger than cf 0.03 -> 0.1 at step 400
  (0.189 -> 0.613): pilot is token-limited, so the full run (25x tokens) shifts the curve strongly left.
  Labelling timings in waves 2-3: 194-240 s per 18.4k seqs (one outlier 58 s) -> ~90 seq/s per GPU.
- 2026-11-04 21:26: sweep done (all 12 OK). runs/sweep/summary.{md,csv,png}, suggestion.json:
  po128 weak 0.0304, mid 0.0827, strong null (d4 max 0.103 at cf 0.1); po0 weak 0.0312, mid 0.0988, strong null.
  po0 vs po128: no consistent difference (po128 a bit higher at cf 0.1: d3 .516 vs .454). Keep pos_offset_max=128.
  Matched-cell fractions (k=16, 50/cell) at cf 0.1 po128: d2-L mean .748 (10% in [.4,.6]), d4-H mean .231 (6%).
- 2026-11-04 21:30: label batch-size bench (pilot cf0.1-po128, same 96 shuffled calib problems, k=32):
  bs16 76 seq/s 11.8 GiB; bs24 84 seq/s 17.5 GiB; bs32 83 seq/s 23.1 GiB; bs64 OOM. -> batch size does not help;
  keep label.batch_size=16. Labelling is ~80-95 seq/s per GPU.
- 2026-11-04 21:32: scancel 60836696. Elapsed 1:09:49 -> 1.16 node-h.
  Estimates: training ~69 min + 5 evals (~1 min each) + startup ~= 76 min; quick calib (4600 x 32) ~27 min -> ~105 min of 120.
  Full calib per model: calib 147k + pool 1.12M + relabel (~0.2-0.4M) seqs at ~90 seq/s ~= 4-5 h on 1 shared GPU (~1.1 node-h).
  Projected total ~= 1.16 + 4 x 1.75 + 4 x 1.1 ~= 12.6 node-h (> the 8-9 estimate, < 15). Told the user.
- 2026-11-04 21:50: user approved comp_frac weak 0.001 / strong 0.01 (mid, midbal after the weak/strong heatmaps)
  and asked to commit first. Commit beb50f5 on pretrain (not pushed): configs + scripts/slurm + relabel_report.py.
  Quick calibration left as the user specified (full calib set, 200/cell); calib.sbatch walltime 8 h.
  Pending user answers: push to origin/pretrain? submit weak+strong training + afterok full calibrations?
- 2026-11-04 21:55: pushed beb50f5 to origin/pretrain (fast-forward, user approved). Explaining the 4 jobs; waiting for submit approval.
- 2026-11-04 21:51: user approved; submitted (commit beb50f5):
  pt-weak 60839035, pt-strong 60839036 (regular, 1 node, 2 h);
  calib-pt-weak-s0 60839037 (afterok:60839035), calib-pt-strong-s0 60839038 (afterok:60839036) (shared, 1 GPU, 8 h).
  Logs: runs/slurm/{pt-weak,pt-strong,calib-pt-weak-s0,calib-pt-strong-s0}-<jobid>.out.
  Next: when training ends, show the user runs/pt-{weak,strong}-s0/calib-quick/{heatmap.png,gate.json};
  after confirmation set mid comp_frac and submit mid + midbal (+ afterok calibrations).
- 2026-11-04 23:22: weak/strong still PENDING (Priority) after 1.5 h in the queue; no start estimate from Slurm.
- 2026-11-05 05:19: pt-weak 60839035 started on nid008521, pt-strong 60839036 at 05:20 on nid001976 (~7.5 h in the queue).
  05:23: both .out files still 0 bytes, batch step ~0 CPU; re-checking at ~05:29 (prolog or a hang before job_header?).
- 2026-11-05 05:29-05:33: diagnosis (srun --overlap into the jobs): the batch script sat in D state,
  wchan dvsipc_wait_for_response, syscall 439 = faccessat2(X_OK) = `[ -x $RLVR_ENV/bin/python ]` in env.sh
  (CFS env). A fresh `[ -x .../bin/python ]` also hung on both nodes (killed after 20 s); `ls` of other CFS
  paths worked (one took 6.6 s); HOME reads fine. Login node mounts CFS natively (gpfs) and is fine;
  compute nodes reach CFS via DVS -> one DVS server path hung.
- 2026-11-05 05:48 / 05:49: Slurm ended both jobs NODE_FAIL (29:07 / 29:51); afterok calibrations
  60839037 / 60839038 auto-cancelled. Charged ~0.49 + 0.50 node-h (NODE_FAIL may be refunded; check later).
- 2026-11-05 ~09:15: user chose: move the runtime env off CFS onto scratch. Per STORAGE_POLICY (envs are rebuilt
  from environment.yml + lock, never moved) the env is REBUILT at /pscratch/sd/y/yupengsu/conda_envs/rlvr
  (log runs/env_setup_scratch.log); CFS env kept as the reference copy. env.sh default RLVR_ENV -> scratch.
  Note: `conda` is only a shell function in interactive shells; in scripts use `module load conda` + $CONDA_EXE.
- 2026-11-05 09:22: scratch env built (5 min from caches): pip freeze identical to the CFS env, 31 offline tests pass.
- 2026-11-05 09:26: user asked to delete the CFS env and keep only the scratch one: rm -rf
  /global/cfs/cdirs/m4788/yupengsu/conda_envs/rlvr (logged in ~/FILESYSTEM_MOVES.log; CFS snapshot 7 days).
  env.sh: default RLVR_ENV = scratch env, rebuild recipe in its header. Not committed yet.
  Next: commit env.sh, then propose resubmitting pt-weak + pt-strong (+ afterok calibrations) to the user.
- 2026-11-05 09:31: user approved; resubmitted (commit 1e7f8cd, pushed to origin/pretrain):
  pt-weak 60863934, pt-strong 60863936 (regular, 1 node, 2 h);
  calib-pt-weak-s0 60863937 (afterok:60863934), calib-pt-strong-s0 60863938 (afterok:60863936) (shared, 1 GPU, 8 h).
- 2026-11-05 12:20: regular GPU queue 4372 pending (debug 14). User asked for faster turnaround; considered debug (30 min,
  2 running / 5 submitted per user; would need resumable training + calibration) vs interactive (4 h, 4 nodes, 2 jobs,
  salloc only). User chose interactive.
- 2026-11-05 ~12:50: commit 456e41d (not pushed): calibrate.py runs under torchrun (strided shards per rank, seed
  1000*seed+rank, gathered back in order; interim gate.json after the calib heatmap; --pool-first for smoke tests);
  scripts/slurm/model_pipeline.sh = 4-GPU training + 4-GPU full calibration + relabel report on one node.
  32 offline tests pass. Smoke-test checkpoint: /pscratch/sd/y/yupengsu/rlvr/tmp/smoke-ddp4 (copy of the cf0.1-po128 pilot).
  Plan (awaiting approval): 2 interactive sallocs (1 node, 4 h each) for weak + strong; hold the regular jobs
  60863934/36/37/38 when the pipelines start, cancel once training runs.
- 2026-11-05 13:35: user approved the interactive plan and said to cancel the regular jobs directly:
  scancel 60863934 60863936 60863937 60863938 (all pending, never ran).
- 2026-11-05 14:16: salloc granted at once: rlvr-strong 60876535 (nid001040), rlvr-weak 60876536 (nid003560); interactive, 1 node, 4 h.
  strong: setsid nohup srun ... bash scripts/slurm/model_pipeline.sh strong -> runs/slurm/pipe-strong-60876535.out (training started 14:16).
  weak node: 4-GPU smoke test of calibrate.py on tmp/smoke-ddp4 (--calib-per-cell 8 --pool-first 3300) ->
  runs/slurm/smoke-ddp4-60876536.out; then the weak pipeline. **scancel each allocation as soon as its pipeline ends.**
  Smoke outputs (small): runs/smoke-ddp4/calib/, data/labels/smoke-ddp4.parquet.
- 2026-11-05 14:17-14:20: smoke test OK (exit 1 = the pilot fails mid's gate, expected): calib 184x32 in 15 s, pool 3300x32 in 37 s,
  relabel 42x64 (d2-L, first p in [0.31, 0.69]) in 2 s; labels parquet in candidates order, no duplicates.
  14:20: weak pipeline started on 60876536 -> runs/slurm/pipe-weak-60876536.out. Both srun run in their own session (setsid).
  Expected: training done ~15:35 (strong) / ~15:40 (weak); interim heatmap + gate ~10 min later; full labels ~17:00.
- 2026-11-05 14:30: user: push 456e41d (and any later commits) together after the runs finish.
- 2026-11-05 15:27 / 15:32: training done (strong / weak). Calib heatmaps (k=32, 200/cell, 4 GPUs: 201 s / 258 s):
  weak cf 0.001: d2 .951 d3 .871 d4 .681 d5 .017 (d2-L .914, d4-H .846); target d3 0.05-0.2 -> far too strong.
  strong cf 0.01: d2 .999 d3 .987 d4 .973 d5 .021; target d4 0.4-0.6 -> far too strong. Both gates PASS (hard = depth1 only),
  warnings target + spread (+ final_loss for strong). The pilot /25 extrapolation overestimated comp_frac by ~10x:
  in the full run composition is learned late (weak E2 d3-H greedy: step 2000 .04, 4000 .21, 6000 .77, 9500 .99).
  Rough estimates (logit fit 1e-3..1e-2 vs weak's trajectory): strong 5e-4..7e-4, mid 1.3e-4..5e-4, weak 2e-5..4e-4.
  Labelling ran at ~730 seq/s on 4 GPUs (2x the estimate) -> full calibration ~35-40 min per model on 4 GPUs.
- 2026-11-05 15:41: user: stop everything, decide next steps together. scancel 60876535 60876536 (pool labelling
  of pt-weak-s0 / pt-strong-s0 never finished; no data/labels written). Nothing running. runs/ + checkpoints/
  pt-weak-s0 and pt-strong-s0 kept as they are (proposed rename to pt-full-cf0.001-s0 / pt-full-cf0.01-s0 not done).
- 2026-11-05 ~15:55: commit f4b1e12 (docs/pretrain_v1_results.md: pilot + round-1 key results). git push origin pretrain failed (GitHub auth; no credential helper configured, earlier pushes worked); asked the user to push. origin/pretrain = 1e7f8cd.
- 2026-11-05 15:52: user pushed from their VS Code terminal; origin/pretrain = f4b1e12 = local HEAD (master untouched at 4cc4d6a).
- 2026-11-05 16:05: user plan: full-budget sweep of comp_frac 3e-5, 1e-4, 2.5e-4, 5e-4 (9500 steps each, one node each,
  all at once, quick calibration only; full labelling only for the chosen models). New cap stated as "50 gpuhours"
  (asked whether node-hours or GPU-hours). Implementation: one 4-node interactive salloc (limit 2 jobs / 4 nodes),
  runs named pt-full-cf<cf>-s0 from configs/pretrain/mid.yaml (+ comp_frac override), calibrate --skip-pool;
  chosen runs get renamed/promoted to pt-{weak,mid,strong}-s0 later. Commit (model_pipeline.sh env options), not pushed.
- 2026-11-05 16:10: user approved; salloc 60880260 (interactive, 4 nodes, 1:45): cf 3e-5 nid001136, 1e-4 nid001145,
  2.5e-4 nid001685, 5e-4 nid003717; logs runs/slurm/pipe-full-cf<cf>-60880260.out. Background waiter scancels the
  allocation once all four print "done at". **Budget cap now 50 NERSC GPU node-hours** (user, replaces the 15 node-h stop line).
- 2026-11-05 17:21-17:28: sweep done (training ~17:21, calib heatmaps by 17:28; calibrate exit 1 everywhere = mid gate
  with matched_cells not run under --skip-pool, expected). The waiter stopped on that exit-1 traceback before its scancel;
  scancel 60880260 at 17:32 (elapsed 1:22:26 x 4 nodes = 5.50 node-h; total 10.51 of the 50 cap).
  Calib mean p (k=32, 200/cell):    d2    d3    d4    d5  | d2-L  d4-H | frac p in [.4,.6]: d2-L d4-H
    cf 3e-5   (pt-full-cf0.00003-s0) .023  .001  .000  .000 | .008  .000 | .005 .000
    cf 1e-4   (pt-full-cf0.0001-s0)  .352  .064  .005  .000 | .113  .015 | .045 .010
    cf 2.5e-4 (pt-full-cf0.00025-s0) .683  .308  .036  .005 | .496  .093 | .090 .055
    cf 5e-4   (pt-full-cf0.0005-s0)  .913  .704  .287  .013 | .881  .524 | .030 .125
    cf 1e-3   (pt-weak-s0, round 1)  .951  .871  .681  .017 | .914  .846 | .020 .040
    cf 1e-2   (pt-strong-s0, round 1).999  .987  .973  .021 | .998  .991 | .000 .005
  E1 = 1.000 everywhere. Findings: d2-L reaches 0.5 at ~2.5e-4 but d4-H only at ~5e-4 (not both at 2.5e-4);
  per-problem p is bimodal (in-range frac <= 12.5% even at cell mean 0.5); d5 signal_frac <= 0.065 at every cf, so
  mid's hard spread check (depths 2-5, >= 0.2) cannot pass. Logit interpolation: weak d3=0.1 at ~1.25e-4 (1e-4 already
  in target at .064), mid d3=0.45 at ~3.2e-4, strong d4=0.5 at ~7.3e-4. Waiting for the user's choice.
- 2026-11-06 00:48: user: "你先submit上去" -> submitted the recommended plan before the gate questions are settled.
  salloc 60894875 (interactive, 4 nodes, 3:00): model_pipeline.sh, train + full calibration (calib, RL pool, relabel):
    nid002072 pt-weak-cf0.000125-s0   (configs/pretrain/weak.yaml,   comp_frac 1.25e-4)
    nid002073 pt-mid-cf0.00032-s0     (configs/pretrain/mid.yaml,    comp_frac 3.2e-4)
    nid200428 pt-strong-cf0.0007-s0   (configs/pretrain/strong.yaml, comp_frac 7e-4)
    nid200493 pt-midbal-cf0.00032-s0  (configs/pretrain/midbal.yaml, comp_frac 3.2e-4, h_ratio 0.5)
  run_config checked (level tag, comp_frac, h_ratio, gate block of each level). Logs runs/slurm/pipe-<name>-60894875.out.
  Gate as in the configs (mid/midbal: spread over d2-5 + matched_cells hard -> FAIL expected); labels are written anyway,
  gate can be re-judged after the user decides. Round-1 pt-weak-s0 / pt-strong-s0 untouched. Waiter (scratchpad
  wait_final.sh) scancels once all four print "done at". Expected done ~02:45; ~8 node-h -> total ~18.5 of 50.
- 2026-11-06 02:30-02:53: all four done (pool 35000x32 in 1689-2670 s, relabel 576-1664 x64 in 47-124 s); waiter scancelled
  60894875 at 02:55 (elapsed 2:06:39 x 4 = 8.44 node-h; total 18.95 of 50). data/labels/<name>.parquet written for all four.
  calib mean p          d2    d3    d4    d5  | d2-L  d4-H | gate (hard)  warnings
    weak   1.25e-4    .440  .114  .011  .001 | .186  .028 | PASS          spread, matched_cells   (target d3 .05-.2: ok)
    mid    3.2e-4     .724  .381  .072  .003 | .497  .167 | FAIL          spread + matched_cells hard (target d3 .35-.55: ok)
    strong 7e-4       .921  .739  .336  .004 | .863  .568 | PASS          spread, target (d4 .336 < .4), matched_cells
    midbal 3.2e-4     .906  .549  .119  .001 | .870  .180 | FAIL          spread + matched_cells hard (target ok, at the top)
  matched cells, relabel p in [.4,.6], total (str_len 6/7/8/9/10); gate needs >= 300 and >= 120 per length:
    mid    d2-L 369 (73/75/81/84/56)   d4-H 326 (103/31/103/18/71)
    midbal d2-L 159 (12/29/37/37/44)   d4-H 391 (101/63/84/63/80)
    strong d2-L 168                    d4-H 509 (79/135/101/84/110)
    weak   d2-L 210                    d4-H 36
  Relabel of first-pass [.4,.6] selections (mean p_relabel; frac still in [.4,.6]): mid d2-L .506 (.56), d4-H .472 (.57);
  midbal .524 (.55) / .483 (.57); weak .505 (.62) / .422 (.53); strong .520 (.59) / .507 (.58). d5-H: <= 32 selected anywhere.
  strong d4 rises steeply between 7e-4 (.336) and 1e-3 (.681): d4 = 0.5 at ~8.3e-4 by logit interpolation.
  Waiting for the user: strong (accept / retrain ~8.3e-4), mid gate (spread depths, matched-cell pool size / thresholds).
- 2026-11-06 ~03:10-09:05: discussed with the user: the gate (spread incl. d5 is unsatisfiable by design; matched_cells
  per-length limited by the 5000-candidate pool), mid vs midbal (midbal keeps mid's comp_frac by design: only h_ratio
  differs). Found the d4 length effect: d4 p dips at str_len 7 in mid/midbal/strong/5e-4 (all d4 groups, pool and
  calib), at 9 only in mid; d2/d3 smooth; not explained by parity-sensitive functions. User: investigate + retrain strong.
- 2026-11-06 09:08: salloc 60914163 (interactive 1 node, 3:00, nid001336): pt-strong-cf0.00083-s0 (strong.yaml,
  comp_frac 8.3e-4 = logit interpolation of 7e-4 .336 and 1e-3 .681), train + full calibration;
  log runs/slurm/pipe-strong-cf0.00083-60914163.out.
  salloc 60914166 (interactive 1 node, 1:00, nid003176): runs/diag-d4len/run.sh = scripts/diag_steps.py
  gen (mid/strong 7e-4/midbal/5e-4, RL-pool d4-H 400/len, d3-H, d4-mixed; k=8 T=1 + greedy, one GPU each)
  + replay (exact pretraining-stream replay, comp examples per depth/group/str_len, checked vs stats.json)
  + analyze (first wrong step and error type by str_len). Waiter scancels the diag allocation when done.
- 2026-11-06 09:17: diagnosis done in 9 min (scancel 60914166 at 09:18; 0.16 node-h). Outputs runs/diag-d4len/
  (analyze.log, */analysis.json, */errors.parquet, replay/replay.json). Commit 570fefd (scripts/diag_steps.py), not pushed.
  * Replay of the pretraining streams matches stats.json exactly (examples and every comp cell) for all four runs.
  * The d4 failure is at step 3: P(step 3 ok | steps 1-2 ok) at L7 = .19 mid / .38 midbal / .28 5e-4 / .46 strong vs
    .54-.89 at the other lengths (mid also L9 .15, midbal L9 .29). Error type: wrong_func, mostly writing the outermost
    function at step 3 (skips function 3 = treats the chain as depth 3). Same dip in d4-mixed (never trained at d4).
    d3 is flat across lengths; apply errors grow with length but show no L7 dip.
  * d4-H pretraining examples per str_len 6..10: mid 87/68/101/96/95, midbal 87/68/102/96/95, 5e-4 139/102/156/140/131,
    strong 7e-4 199/157/213/188/188: L7 is the fewest everywhere (mid: 68 vs 89 expected, ~-2.5 sd). The runs share
    seed 0, and the per-worker python rng streams re-synchronise after a comp/atom decision differs, so the runs largely
    share their composition examples (mid and midbal d4 counts are identical) -> one sampling accident of seed 0,
    inherited by every level. d4 = 5% of compositions (~70-200 examples per length), so step 3 of depth 4 is learned
    per length from very few examples. Not a code bug (functions/reward fine: d1 = 1.0, d2/d3 flat).
  * Side finding: 1.4-2.6% of sampled completions get reward 1 with a wrong chain (e.g. a later sort erases the error).
- 2026-11-06 11:07: pt-strong-cf0.00083-s0 done (train 09:09-10:20, full calibration to 11:07; scancel 60914163 at 11:07;
  1.98 node-h, total 21.09). Gate PASS (hard depth1 1.0), warnings spread, target, matched_cells.
  calib mean p d2 .919 d3 .759 d4 .353 d5 .010 | d4-H .594 d4-L .161 d4-mixed .304 -> d4 still < .4.
  d4 vs comp_frac (same config, seed 0): 5e-4 .287, 7e-4 .336, 8.3e-4 .353, 1e-3 .681 (round-1 pt-weak-s0):
  plateau then a jump, so the logit interpolation (8.3e-4 -> .5) was wrong. d4-H is in range at 5e-4..8.3e-4
  (.52-.59); the overall d4 is held down by d4-L (.11-.17 -> .59 at 1e-3). L7 dip of d4-H closed at 8.3e-4
  (calib d4-H by len 6..10: .59/.54/.65/.58/.61).
  matched cells (p_relabel in [.4,.6]): d2-L 153 (13/12/53/27/48), d4-H 428 (75/100/92/75/86).
  Relabel of first-pass [.4,.6] picks: d2-L .511 (.53 still in range), d4-H .515 (.58), d5-H .469 (.43).
