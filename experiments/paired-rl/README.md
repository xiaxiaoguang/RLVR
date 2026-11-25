# Paired RL runner (record copy)

A copy of the scripts and configs that run the paired 500-step GRPO experiment on the two v2 base models
(`pt-tree-mix-cf0.0035-s0`, `pt-tree-mix-cf0.006-s0`) on the lab GTX 1080 Ti. The setup, interim results and
operational notes are in [`docs/rl.md`](../../docs/rl.md#paired-rl-500-steps-two-base-models).

The scripts are kept here as a record. They run from their original location, `runs/idea-20261121-01-paired-rl/`,
and hard-code absolute paths on the lab machine: the run directory, the code tree
`runs/idea-20261121-01-branch-integration/worktree` (commit `2f22119`), the `RLVR-rl1080` uv environment and GPU 1's
UUID. They also verify the hashes of their own files and of the tracked sources before each model starts
(`common.py`, `validate_sources`). To rerun the experiment elsewhere, make a new run directory and adjust `env.sh`,
`common.py` and the configs; do not reuse the original run directory, which refuses to start twice.

| file | role |
| --- | --- |
| `job.sh` | host entry point: runs `supervise.py --mode main` |
| `supervise.py` | runs the two models in order on one GPU, enforces the GPU-time cap, writes `main-status.json` and the accounting files |
| `worker.py` | one model: checks sources and the GPU, then calls `scripts/grpo.py:main` with the resolved config |
| `common.py` | paths, hashing, source validation, GPU checks |
| `env.sh` | environment variables (uv paths, offline mode, `PAIRED_RUN`, `PAIRED_CODE`) |
| `configs/main-*.yaml` | the two 500-step configs; they differ only in `rl.base` |
| `configs/smoke-*.yaml`, `smoke.sh` | the one-step smoke runs done before launch |
| `prepare.py`, `verify.py`, `check.sh`, `control_tests.py`, `signal_tests.py`, `freeze_delivery.py` | input preparation, CPU acceptance checks and tests of the timeout, signal and conflict handling |
| `inspect_results.py` | read-only summary of the evaluation results so far |
| `build_report.py` | writes `launch-plan.json` and the handoff documents from the saved evidence |
| `REPORT_ZH.md`, `REPRO.md`, `USER_GUIDE_ZH.md` | report (Chinese), exact commands and hashes, user guide (Chinese), as of launch |
