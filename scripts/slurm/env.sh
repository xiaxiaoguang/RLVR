# Perlmutter (NERSC) job environment, sourced by scripts/slurm/*.sbatch from the repo root.
# Paths follow the account's storage layout: env on CFS, caches on scratch; runs/, checkpoints/
# and data/labels/ are symlinks into $SCRATCH. Override RLVR_ENV to use another env.
RLVR_ENV=${RLVR_ENV:-/global/cfs/cdirs/m4788/yupengsu/conda_envs/rlvr}
[ -x "$RLVR_ENV/bin/python" ] || { echo "no python in $RLVR_ENV" >&2; exit 2; }
export PATH=$RLVR_ENV/bin:$PATH
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/pscratch/sd/y/yupengsu/cache/pip}
export HF_HOME=${HF_HOME:-/pscratch/sd/y/yupengsu/cache/huggingface}
export WANDB_DIR=${WANDB_DIR:-/pscratch/sd/y/yupengsu/rlvr/wandb}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# compile caches on node-local /tmp (default ~/.triton would write into HOME)
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_$USER
export TRITON_CACHE_DIR=/tmp/triton_$USER
export PYTHONUNBUFFERED=1

job_header() {  # what ran where, for the job log
  echo "job ${SLURM_JOB_ID:-none} on $(hostname) at $(date '+%F %T')"
  echo "commit $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- || echo ' (dirty)'), python $(command -v python)"
  nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total --format=csv,noheader || true
}
