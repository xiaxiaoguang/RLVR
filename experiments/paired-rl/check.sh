#!/usr/bin/env bash
set -euo pipefail
export READ_ONLY_CHECK=1
source /home/lab_admin/workspace/RLVR/runs/idea-20261121-01-paired-rl/env.sh
export CUDA_VISIBLE_DEVICES=''
cd "$PAIRED_CODE"
exec uv run --project /work/users/lab_admin/envs/RLVR-rl1080 --frozen --no-sync python "$PAIRED_RUN/verify.py"
