#!/bin/bash
# A series of GRPO runs on one Perlmutter GPU node (4 A100s, one run per GPU at a time).
# Submit from the repo root; all arguments go to scripts/rl_sweep.py:
#   sbatch -J rl-repro scripts/slurm/rl_sweep.sbatch --name repro \
#       --models checkpoints/pt-weak-cf0.000125-s0 checkpoints/pt-mid-cf0.00032-s0 \
#       --datasets easy_500 mixed_500 --seeds 0 1 2
# Finished runs (summary.json) are skipped, so resubmitting the same command continues a sweep
# that hit the time limit. A default run (500 steps, B 64 x G 8) took ~35-50 min on a local A100;
# re-measure on Perlmutter before sizing -t.
#SBATCH -A m4788_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -t 04:00:00
#SBATCH -J rl-sweep
#SBATCH -o /pscratch/sd/y/yupengsu/rlvr/runs/slurm/%x-%j.out
set -euo pipefail
cd "${RLVR_ROOT:-$SLURM_SUBMIT_DIR}"
[ -f rlvr/grpo.py ] || { echo "submit from the RLVR repo root (or set RLVR_ROOT)" >&2; exit 2; }
source scripts/slurm/env.sh
job_header
python scripts/rl_sweep.py --gpus 0 1 2 3 "$@"
