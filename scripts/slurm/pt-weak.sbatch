#!/bin/bash
# Pretrain base model 'weak' (configs/pretrain/weak.yaml) on one Perlmutter GPU node, then the quick
# calibration. Submit from the repo root; key=value arguments go to pretrain.py:
#   sbatch scripts/slurm/pt-weak.sbatch
#   sbatch --dependency=afterok:<this job> --kill-on-invalid-dep=yes -J calib-pt-weak-s0 scripts/slurm/calib.sbatch pt-weak-s0
#SBATCH -A m4788_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -t 02:00:00
#SBATCH -J pt-weak
#SBATCH -o /pscratch/sd/y/yupengsu/rlvr/runs/slurm/%x-%j.out
LEVEL=weak
source "${RLVR_ROOT:-$SLURM_SUBMIT_DIR}/scripts/slurm/pretrain_job.sh"
