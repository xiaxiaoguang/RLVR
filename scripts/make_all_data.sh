#!/usr/bin/env bash
# Regenerate every data artifact from configs/ (deterministic). ~2 min.
set -euo pipefail
cd "$(dirname "$0")/.."
SEED=${SEED:-0}
python tests/test_task.py
python tests/test_registry.py
python scripts/make_eval_sets.py
python scripts/make_rl_pool.py
for lv in weak mid strong midbal; do
  python scripts/preview_stream.py --config configs/pretrain/$lv.yaml seed=$SEED
done
python scripts/build_dataset.py configs/datasets/*.yaml --skip-unlabelled
python scripts/report_data.py
