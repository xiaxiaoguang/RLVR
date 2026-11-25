#!/usr/bin/env bash
export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=/work/users/lab_admin/uv-cache UV_LINK_MODE=hardlink UV_PYTHON_INSTALL_DIR=/work/users/lab_admin/uv-python
export HF_HOME=/data/users/lab_admin/migration/_cache/huggingface TORCH_HOME=/data/users/lab_admin/migration/_cache/torch PIP_CACHE_DIR=/data/users/lab_admin/migration/_cache/pip
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 UV_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0
export PAIRED_RUN=/home/lab_admin/workspace/RLVR/runs/idea-20261121-01-paired-rl
export PAIRED_CODE=/home/lab_admin/workspace/RLVR/runs/idea-20261121-01-branch-integration/worktree
export PYTHONPATH=$PAIRED_CODE:$PAIRED_CODE/scripts
export TMPDIR=$PAIRED_RUN/tmp
if [[ ${READ_ONLY_CHECK:-0} != 1 ]]; then mkdir -p "$TMPDIR" "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR"; fi
