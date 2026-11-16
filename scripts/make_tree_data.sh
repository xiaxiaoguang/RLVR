#!/usr/bin/env bash
# Generate the versioned multi-input task, leaving chain-v1 data untouched.
# Run in the rlvr environment; overrides are forwarded to generation/preview/report.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${CONFIG:-configs/tasks/tree_v1.yaml}
PREVIEW_N=${PREVIEW_N:-5000}
python scripts/make_eval_sets.py --config "$CONFIG" "$@"
python scripts/make_rl_pool.py --config "$CONFIG" "$@"
python scripts/preview_stream.py --config "$CONFIG" --n "$PREVIEW_N" "$@"
# Dataset builder takes config paths, not key=value overrides; generate datasets
# separately if paths.data is overridden above.
if [ "$#" -eq 0 ]; then
  python scripts/build_dataset.py configs/datasets/*.yaml --config "$CONFIG" --skip-unlabelled
fi
python scripts/report_data.py --config "$CONFIG" "$@"
