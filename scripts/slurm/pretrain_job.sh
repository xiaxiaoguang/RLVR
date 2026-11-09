# Body of scripts/slurm/pt-{level}.sbatch (sourced; LEVEL set by the caller, extra sbatch
# arguments are key=value overrides for pretrain.py). Steps:
#   1. 4-GPU DDP pretraining (torchrun --standalone) -> checkpoints/pt-$LEVEL-s$SEED
#   2. quick calibration: E1 greedy + calib heatmap + gate without the RL pool (--skip-pool),
#      kept in runs/pt-$LEVEL-s$SEED/calib-quick/ (the full calibration rewrites calib/)
# The job fails if training fails or the quick gate fails a hard check, so a full calibration
# submitted with --dependency=afterok on this job is not run on a model that must be retrained.
set -euo pipefail
cd "${RLVR_ROOT:-$SLURM_SUBMIT_DIR}"
[ -f rlvr/gate.py ] || { echo "submit from the RLVR repo root (or set RLVR_ROOT)" >&2; exit 2; }
source scripts/slurm/env.sh
SEED=${SEED:-0}
NAME=pt-$LEVEL-s$SEED
job_header
echo "== $NAME: pretraining (configs/pretrain/$LEVEL.yaml seed=$SEED $*)"
torchrun --standalone --nproc_per_node=4 scripts/pretrain.py \
  --config "configs/pretrain/$LEVEL.yaml" "seed=$SEED" "$@"

echo "== $NAME: quick calibration (--skip-pool) at $(date '+%F %T')"
rc=0
python scripts/calibrate.py "checkpoints/$NAME" --skip-pool || rc=$?
if [ -d "runs/$NAME/calib" ]; then
  mkdir -p "runs/$NAME/calib-quick" && cp -a "runs/$NAME/calib/." "runs/$NAME/calib-quick/"
fi
echo "== $NAME: done at $(date '+%F %T'), quick calibration exit $rc (1 = gate FAIL or error)"
exit $rc
