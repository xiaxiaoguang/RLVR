#!/bin/bash
# Train one base model and fully calibrate it on one 4-GPU node of an allocation (e.g. an
# interactive salloc; batch jobs use pt-*.sbatch + calib.sbatch instead):
#   salloc --no-shell -A m4788_g -C gpu -q interactive -N 1 -t 04:00:00 -J rlvr-weak
#   setsid nohup srun --jobid=<id> -N1 -n1 -c 128 --cpu-bind=none --gpus-per-node=4 \
#       bash scripts/slurm/model_pipeline.sh weak > runs/slurm/pipe-weak-<id>.out 2>&1 &
# Steps: 4-GPU training (skipped if checkpoints/pt-LEVEL-sSEED exists), 4-GPU calibrate.py
# (E1, calib heatmap + interim gate.json, RL pool + second labelling, final gate.json), then
# relabel_report.py. Exit code: calibrate's (1 = gate FAIL or error); training errors exit early.
set -euo pipefail
LEVEL=${1:?usage: model_pipeline.sh LEVEL [SEED]}
SEED=${2:-0}
cd "${RLVR_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
[ -f rlvr/gate.py ] || { echo "run from the RLVR repo (or set RLVR_ROOT)" >&2; exit 2; }
source scripts/slurm/env.sh
NAME=pt-$LEVEL-s$SEED
job_header
echo "== $NAME: pipeline start $(date '+%F %T')"
if [ -f "checkpoints/$NAME/run_config.yaml" ]; then
  echo "== $NAME: final checkpoint exists, skipping training"
else
  torchrun --standalone --nproc_per_node=4 scripts/pretrain.py \
    --config "configs/pretrain/$LEVEL.yaml" "seed=$SEED"
  echo "== $NAME: training done $(date '+%F %T')"
fi
rc=0
torchrun --standalone --nproc_per_node=4 scripts/calibrate.py "checkpoints/$NAME" || rc=$?
if [ -f "data/labels/$NAME.parquet" ]; then
  python scripts/relabel_report.py "$NAME"
fi
echo "== $NAME: done at $(date '+%F %T'), calibrate exit $rc (1 = gate FAIL or error)"
exit $rc
