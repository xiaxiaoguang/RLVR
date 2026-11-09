#!/bin/bash
# d4 length diagnosis (2026-11-06): step-level errors of 4 models on 4 GPUs + exact replay of their
# pretraining streams on the CPUs, then the analysis. Outputs next to this file.
set -uo pipefail
cd /global/u2/y/yupengsu/projects/RLVR
source scripts/slurm/env.sh
job_header
O=runs/diag-d4len
names=(mid-cf0.00032 strong-cf0.0007 midbal-cf0.00032 full-cf0.0005)
echo "== diag start $(date '+%F %T')"
for i in 0 1 2 3; do
  n=${names[$i]}
  CUDA_VISIBLE_DEVICES=$i python scripts/diag_steps.py gen checkpoints/pt-$n-s0 --out $O/$n > $O/gen-$n.log 2>&1 &
done
python scripts/diag_steps.py replay runs/pt-mid-cf0.00032-s0 runs/pt-strong-cf0.0007-s0 \
  runs/pt-midbal-cf0.00032-s0 runs/pt-full-cf0.0005-s0 --out $O/replay --procs 56 > $O/replay.log 2>&1 &
wait
echo "== gen + replay done $(date '+%F %T')"
python scripts/diag_steps.py analyze $O/mid-cf0.00032 $O/strong-cf0.0007 $O/midbal-cf0.00032 $O/full-cf0.0005 \
  > $O/analyze.log 2>&1
echo "== diag done at $(date '+%F %T')"
