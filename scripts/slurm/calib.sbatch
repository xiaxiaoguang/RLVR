#!/bin/bash
# Full calibration of one base model on one shared Perlmutter GPU: E1 greedy, calib heatmap,
# RL-pool labels (+ second labelling), gate with matched_cells, then the relabel report.
# Submit from the repo root, chained to the training job; key=value arguments go to calibrate.py:
#   sbatch --dependency=afterok:<train job> --kill-on-invalid-dep=yes -J calib-pt-mid-s0 \
#       scripts/slurm/calib.sbatch pt-mid-s0
#SBATCH -A m4788_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 08:00:00
#SBATCH -J calib
#SBATCH -o /pscratch/sd/y/yupengsu/rlvr/runs/slurm/%x-%j.out
set -euo pipefail
NAME=${1:?usage: calib.sbatch pt-LEVEL-sSEED [key=value ...]}
shift
cd "${RLVR_ROOT:-$SLURM_SUBMIT_DIR}"
[ -f rlvr/gate.py ] || { echo "submit from the RLVR repo root (or set RLVR_ROOT)" >&2; exit 2; }
source scripts/slurm/env.sh
job_header
echo "== $NAME: full calibration ($*)"
rc=0
python scripts/calibrate.py "checkpoints/$NAME" "$@" || rc=$?
if [ -f "data/labels/$NAME.parquet" ]; then
  python scripts/relabel_report.py "$NAME"
fi
echo "== $NAME: done at $(date '+%F %T'), calibrate exit $rc (1 = gate FAIL or error)"
exit $rc
