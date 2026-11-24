#!/usr/bin/env bash
# Default is read-only preflight. --run explicitly enables training.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=/work/users/lab_admin/uv-cache
export UV_LINK_MODE=hardlink
export UV_PYTHON_INSTALL_DIR=/work/users/lab_admin/uv-python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# The older Pascal driver/allocator may not support expandable_segments.
export PYTORCH_CUDA_ALLOC_CONF=backend:native
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == "-1" ]]; then
    echo 'No assigned GPU: CUDA_VISIBLE_DEVICES must be set by the lab; do not select another card.' >&2
    exit 2
fi
mode=(--preflight)
if [[ "${1:-}" == "--run" ]]; then
    mode=()
    shift
fi
exec uv run --project /work/users/lab_admin/envs/RLVR-rl1080 --frozen python scripts/grpo.py \
    --config configs/rl/1080ti.yaml --device cuda "${mode[@]}" "$@"
