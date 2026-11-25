#!/usr/bin/env bash
set -euo pipefail
export PAIRED_SCHEDULER_MASK=${CUDA_VISIBLE_DEVICES-__UNSET__}
source /home/lab_admin/workspace/RLVR/runs/idea-20261121-01-paired-rl/env.sh
export CUDA_VISIBLE_DEVICES=''
cd "$PAIRED_CODE"
exec uv run --project /work/users/lab_admin/envs/RLVR-rl1080 --frozen python "$PAIRED_RUN/supervise.py" --mode smoke
