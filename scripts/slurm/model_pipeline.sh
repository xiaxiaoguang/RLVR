#!/bin/bash
# Train one base model and fully calibrate it on one 4-GPU node of an allocation (e.g. an
# interactive salloc; batch jobs use pt-*.sbatch + calib.sbatch instead):
#   salloc --no-shell -A m4788_g -C gpu -q interactive -N 1 -t 04:00:00 -J rlvr-weak
#   setsid nohup srun --jobid=<id> -N1 -n1 -c 128 --cpu-bind=none --gpus-per-node=4 \
#       bash scripts/slurm/model_pipeline.sh weak > runs/slurm/pipe-weak-<id>.out 2>&1 &
# Steps: 4-GPU training (skipped if checkpoints/pt-LEVEL-sSEED exists), 4-GPU calibrate.py
# (E1, calib heatmap + interim gate.json, RL pool + second labelling, final gate.json), then
# relabel_report.py. Exit code: calibrate's (1 = gate FAIL or error); training errors exit early.
# Optional environment (comp_frac sweeps at the full budget):
#   CONFIG      training config (default configs/pretrain/LEVEL.yaml); pretrain.level=LEVEL is always set
#   TRAIN_ARGS  extra key=value overrides for pretrain.py, e.g. "pretrain.comp_frac=0.0001"
#   CALIB_ARGS  extra arguments for calibrate.py, e.g. "--skip-pool" (calib heatmap + gate only)
#   CONFIG=configs/pretrain/mid.yaml TRAIN_ARGS=pretrain.comp_frac=0.0001 CALIB_ARGS=--skip-pool \
#       bash scripts/slurm/model_pipeline.sh full-cf0.0001      # -> pt-full-cf0.0001-s0
set -euo pipefail
LEVEL=${1:?usage: model_pipeline.sh LEVEL [SEED]}
SEED=${2:-0}
cd "${RLVR_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
[ -f rlvr/gate.py ] || { echo "run from the RLVR repo (or set RLVR_ROOT)" >&2; exit 2; }
CONFIG=${CONFIG:-configs/pretrain/$LEVEL.yaml}
read -r -a TRAIN_EXTRA <<< "${TRAIN_ARGS:-}"
read -r -a CALIB_EXTRA <<< "${CALIB_ARGS:-}"
source scripts/slurm/env.sh
NAME=pt-$LEVEL-s$SEED
job_header
echo "== $NAME: pipeline start $(date '+%F %T'); config $CONFIG ${TRAIN_EXTRA[*]}; calibrate ${CALIB_EXTRA[*]}"
if [ -f "checkpoints/$NAME/run_config.yaml" ]; then
  echo "== $NAME: final checkpoint exists, skipping training"
else
  torchrun --standalone --nproc_per_node=4 scripts/pretrain.py \
    --config "$CONFIG" "pretrain.level=$LEVEL" "seed=$SEED" "${TRAIN_EXTRA[@]}"
  echo "== $NAME: training done $(date '+%F %T')"
fi
rc=0
torchrun --standalone --nproc_per_node=4 scripts/calibrate.py "checkpoints/$NAME" "${CALIB_EXTRA[@]}" || rc=$?
if [ -f "data/labels/$NAME.parquet" ]; then
  python scripts/relabel_report.py "$NAME"
fi
echo "== $NAME: done at $(date '+%F %T'), calibrate exit $rc (1 = gate FAIL or error)"
exit $rc
