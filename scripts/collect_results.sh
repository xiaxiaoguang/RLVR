#!/bin/bash
# Copy the small, shareable pretraining results from runs/ and checkpoints/ (symlinks into scratch,
# not in git) into results/pretrain_v1/ so they can be committed. Run from the login node; it only
# copies files (no compute). Checkpoints and data/labels/*.parquet stay out of git.
#   bash scripts/collect_results.sh
# Per run with a calib/gate.json: run_config.yaml, stats.json, eval.jsonl, calib/{gate.json,
# relabel_report.json, heatmap.png}. Also the d4 length diagnosis (runs/diag-d4len) and a snapshot
# of the job log runs/NERSC_LOG.md.
set -euo pipefail
cd "${RLVR_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT=results/pretrain_v1
mkdir -p "$OUT/models"
copy() {  # copy SRC DST if SRC exists
  [ -f "$1" ] || return 0
  mkdir -p "$(dirname "$2")"
  cp -p "$1" "$2"
}
n=0
for run in runs/pt-*-s0; do
  [ -f "$run/calib/gate.json" ] || continue
  name=$(basename "$run")
  dst=$OUT/models/$name
  if [ -f "$run/run_config.yaml" ]; then
    copy "$run/run_config.yaml" "$dst/run_config.yaml"
  else
    copy "checkpoints/$name/run_config.yaml" "$dst/run_config.yaml"
  fi
  for f in stats.json eval.jsonl calib/gate.json calib/relabel_report.json calib/heatmap.png; do
    copy "$run/$f" "$dst/$f"
  done
  n=$((n + 1))
done
diag=runs/diag-d4len
if [ -d "$diag" ]; then
  copy "$diag/analyze.log" "$OUT/diag-d4len/analyze.txt"
  copy "$diag/replay/replay.json" "$OUT/diag-d4len/replay.json"
  copy "$diag/run.sh" "$OUT/diag-d4len/run.sh"
  for d in "$diag"/*/; do
    [ -f "$d/analysis.json" ] || continue
    copy "$d/analysis.json" "$OUT/diag-d4len/$(basename "$d")/analysis.json"
    copy "$d/gen.json" "$OUT/diag-d4len/$(basename "$d")/gen.json"
  done
fi
copy runs/NERSC_LOG.md "$OUT/NERSC_LOG.md"
echo "collected $n runs into $OUT ($(du -sh --apparent-size "$OUT" | cut -f1))"
