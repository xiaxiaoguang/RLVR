# Reproduction — paired pretrained RL execution package

Purpose: verify both delivered pretrained bases and smoke the actual integrated trainer, then hand off a sequential two-model seed0/500step host job. Tiny smoke evaluation is wiring only, never a full scientific baseline. No download/API/pretraining.

Project root: `/home/lab_admin/workspace/RLVR` (P). Run N=`P/runs/idea-20261121-01-paired-rl`. Authoritative source C=`P/runs/idea-20261121-01-branch-integration/worktree`, HEAD `2f2211953c889583642a4f23dcfd3d8d50418042`, tree `73844c0ed836bdd20528e287c43b188b39cfb13d`. Original checkout remains `2d5973bf03bc0e3b0741459d3259ad290d6e0431` on collaborator branch. No tracked production changes, commits, refs changes or pushes. All uncommitted run dependencies are frozen by N/delivery.json; all tracked source, inputs and environment locks by N/inputs.json. Imported absolute paths/hashes are in smoke/<model>/imported-source.json. N/integrated-source.tar.gz archives the integrated HEAD. Do not relocate sources silently.

Environment: reuse `RLVR-rl1080`; `/work/users/lab_admin/envs/RLVR-rl1080/uv.lock`, SHA256 `da9cbed8ab7ca6484c55505bdb4b4d7700efcc328330f823246870bc87f292b9`. No environment/lock changes. N/environment.json records Python/package versions and CPU affinity. Every shell starts with the following exports (also set by env.sh and both launchers):

```bash
cd /home/lab_admin/workspace/RLVR
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/work/users/lab_admin/uv-cache UV_LINK_MODE=hardlink UV_PYTHON_INSTALL_DIR=/work/users/lab_admin/uv-python
export HF_HOME=/data/users/lab_admin/migration/_cache/huggingface TORCH_HOME=/data/users/lab_admin/migration/_cache/torch PIP_CACHE_DIR=/data/users/lab_admin/migration/_cache/pip
source runs/idea-20261121-01-paired-rl/env.sh
```

Historical commands executed in this task, from P:

```bash
# Preparation ran CPU-only. Attempt 1 found an old relative base-YAML path absent in P;
# fixed the new preparation script to load C/configs/rl/1080ti.yaml, preserving the failure log.
CUDA_VISIBLE_DEVICES='' uv run --project /work/users/lab_admin/envs/RLVR-rl1080 --frozen python "$PAIRED_RUN/prepare.py" > "$PAIRED_RUN/prepare.log" 2>&1
CUDA_VISIBLE_DEVICES='' uv run --project /work/users/lab_admin/envs/RLVR-rl1080 --frozen python "$PAIRED_RUN/control_tests.py" > "$PAIRED_RUN/control-tests.log" 2>&1
# smoke.sh must receive the actual scheduler mask, not env.sh's CPU-check default.
# The original shell had CUDA_VISIBLE_DEVICES unset, recorded in accounting.json.
# This foreground invocation was run once; do not repeat in this evidence directory.
bash runs/idea-20261121-01-paired-rl/smoke.sh > runs/idea-20261121-01-paired-rl/smoke-supervisor.log 2>&1
```

The wrappers execute `uv run --project /work/users/lab_admin/envs/RLVR-rl1080 --frozen python N/worker.py --mode smoke|main --model MODEL` from C with `PYTHONPATH=C:C/scripts`, using only the full index1 UUID `GPU-b511889f-19f3-d5de-fdad-d3ef1f776278` after scheduler-mask, idle and cooperative-lock checks. Worker calls `scripts/grpo.py:main` directly, observing AdamW/sampler/evaluation without changing updates, labels, rewards or RNG draws. Native trainer preflight loads both bases on CPU and checks canonical inputs/evaluation leakage. CPU-only control tests exercise actual child process-group timeouts, a synthetic external-conflict exception and a real signal; no GPU work in these tests.

One-shot smoke and main claims intentionally prevent overwriting/replaying evidence. Preparation and smoke commands document the original run; to repeat GPU work requires a new authorized run directory, new frozen identities and a new ledger, not removal of the old claims. CPU acceptance is safe to repeat before main starts:

```bash
cd /home/lab_admin/workspace/RLVR
bash runs/idea-20261121-01-paired-rl/check.sh
```

Host command (lab launches after worker exit; worker did not launch/background it):

```bash
cd /home/lab_admin/workspace/RLVR
bash runs/idea-20261121-01-paired-rl/job.sh
```

Inputs: both original pretrained folders under P/checkpoints/tree-v1; actual weights under `/data/users/lab_admin/migration/RLVR/checkpoints/tree-v1/`. Each safetensors is 340128712 bytes. cf.0035 SHA256 `6476c96d6a50ce809791379f64cd1d93332ba031d7c1f5a29077b84201c13ee8`; cf.006 SHA256 `36c50e13f27feaf43f75f97ff1150add99da4459e7b118027bba0ec1e6b8a04d`. All config/tokenizer/run_config sizes/hashes, frozen pool and E1–E4/probe paths/sizes/hashes are in inputs.json. Pool 39772203 bytes, 40000 rows, SHA256 `a04a2e1a922cad3901f27ef829ef38c913d5a5839e0ae1310a304d7c9f26a731`; input-order SHA256 `a143858e5b8c93445abac3aa9ed2dd8b695a57a5b188f667b07eb8ce5de27c08`. Loader checks finite fp32 tensors, exact state keys/shapes/values and task fingerprint `195a655b63e47644f32bc96ad52ecb3cb163cb841f14ce923a27ba79c66a4ca4`. Cached official tree metadata for intended revision `2cc23a4ebdfe5a64dd4d2220e02194a1f5745736` agrees with second delivery hashes/blob IDs; no fresh remote verification or second delivery receipt is asserted.

Seeds and settings: pretrain seed0/9500steps, comp_frac .0035 vs .006, otherwise same task/architecture. New RL seed0; static p-independent four-cell pool; same 500x64 PID sequence; G8, T1, fp32, 768 completion limit. AdamW lr1e-5, 10 warmup steps, weight_decay0, gradient clip1, token loss, group std normalization, one update/batch, no entropy/KL penalties, diagnostic KL every10 steps (and first). Execution gen_batch_size1, micro_batch_tokens1024, compile=false. Eval batch2; original E1–E4 every50/probe every25 with step0/final, all frozen rows and depth/step/structure metrics. Original pass_k_samples0/pass_k[1,2,4,8,16] retained. Config differences: rl.base, pretrain.level, pretrain.comp_frac only. Smoke alone sets steps1 and uses first two rows of each eval group (10 total per evaluation), same 40000-candidate sampler and first64 PIDs.

Outputs: small status/accounting/optimizer/eval/metrics/summary evidence under N. Bulky checkpoints and rollouts under `/data/users/lab_admin/migration/RLVR/runs/idea-20261121-01-paired-rl/{smoke,main}/<model>/`, symlinked from N/{smoke,main}/output/<trainer-run-name>/{ckpt,rollouts.jsonl}. Host saves models every50steps and final. Trainer does not save optimizer/scheduler/RNG/sampler state, so checkpoints are not exact-resume states. No model copying to root disk. Before/after disk snapshots and exact sizes are N/storage-*.txt. CPU control-test throwaways under N/tmp were removed automatically.

Status/wall time and results: N/smoke-status.json, N/accounting.json, smoke/<model>/training-evidence.json and N/verification.json record the completed smoke and acceptance. N/launch-plan.json records authorized remaining host seconds and bounded max_hours. Reservations conservatively charge the entire invocation budget until verified cleanup/final actual rounded-up wall time. Running entries persist elapsed time and worst-case reservation every second; SIGKILL/power loss retains conservative reservation. Total hard cap360000s minus inherited13257s minus worker charge, with30s reserved shutdown inside the deadline; independent GNU timeout per model includes startup/eval/save/failures. Main starts both fresh bases sequentially cf.0035 then cf.006, no exact resume. Interrupted/incomplete stages stay explicitly marked, and main.summary files alone define completed individual runs. SIGTERM/INT/HUP clean owned process groups; external PIDs are never killed. Main runtime and scientific comparison are pending host launch.

Inspect (CPU only):

```bash
cd /home/lab_admin/workspace/RLVR
source runs/idea-20261121-01-paired-rl/env.sh
CUDA_VISIBLE_DEVICES='' uv run --project /work/users/lab_admin/envs/RLVR-rl1080 --frozen python "$PAIRED_RUN/inspect_results.py"
cat runs/idea-20261121-01-paired-rl/REPORT_ZH.md
cat /work/users/lab_admin/shared/IDEA-20261121-01/HANDOFF.md
du -sh /data/users/lab_admin/migration/RLVR/runs/idea-20261121-01-paired-rl
```

Limitations: one seed/order, shared pretrain seed, tiny saved-policy B1/B2 equivalence covers only fixed two E2 rows per base, no all-set numerical equivalence/speed guarantee, no new sampled pass@k, D/E go=false unchanged. Historical task15/16 failed/blocked packets stay untouched. Completion of this task means setup/smoke readiness only; final paired analysis requires later worker/manager review and only PI can close the idea.

Final documentation/signal verification commands, also from P (these were executed after smoke; they do not retrain):

```bash
cd /home/lab_admin/workspace/RLVR
source runs/idea-20261121-01-paired-rl/env.sh
CUDA_VISIBLE_DEVICES='' uv run --project /work/users/lab_admin/envs/RLVR-rl1080 --frozen python "$PAIRED_RUN/signal_tests.py" > "$PAIRED_RUN/signal-tests.log" 2>&1
bash runs/idea-20261121-01-paired-rl/check.sh > runs/idea-20261121-01-paired-rl/check.log
cp runs/idea-20261121-01-paired-rl/check.log runs/idea-20261121-01-paired-rl/verification.json
CUDA_VISIBLE_DEVICES='' uv run --project /work/users/lab_admin/envs/RLVR-rl1080 --frozen python "$PAIRED_RUN/build_report.py"
```

signal-tests.json separately records real SIGTERM, SIGINT and SIGHUP injection into the current supervisor handler and cleanup of actual CPU process groups/descendants. The earlier control-tests.json also retains its SIGALRM-based handler test. delivery.json is the unchanged runtime receipt used by both GPU smokes and the future host. Final docs, analysis helper, supplemental signal test, and small evidence identities are in deliverable-identities.json; source and runtime receipts were not retroactively rewritten.

Final prelaunch file manifest (does not rewrite delivery.json or historical evidence):

```bash
cd /home/lab_admin/workspace/RLVR
source runs/idea-20261121-01-paired-rl/env.sh
CUDA_VISIBLE_DEVICES='' uv run --project /work/users/lab_admin/envs/RLVR-rl1080 --frozen python "$PAIRED_RUN/freeze_delivery.py"
```

This is a prelaunch snapshot; accounting/status files intentionally advance once the lab launches main. Do not treat the old prelaunch digest of a live accounting file as an instruction to revert it.
