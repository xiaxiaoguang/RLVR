#!/usr/bin/env bash
# Pretrain the four base models, two at a time on NPROC GPUs each, then calibrate.
#   SEED=0 GPUS="0 1 2 3 4 5 6 7" bash scripts/pretrain_all.sh [key=value ...]   (overrides go to pretrain.py)
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 scripts/pretrain.py --config configs/pretrain/mid.yaml --bench 100

# Round 1: weak + strong (the extremes; check their quick heatmaps before round 2 and adjust
# configs/pretrain/mid.yaml if they do not separate). Round 2: mid + midbal. Then full
# calibration (gate + RL-pool labels) of all four, one GPU each.
# Needs data/eval (scripts/make_all_data.sh). Logs: runs/pt-{level}-s{seed}.log, runs/calib-*.log
set -euo pipefail
cd "$(dirname "$0")/.."
SEED=${SEED:-0}
NPROC=${NPROC:-4}
read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 5 6 7}"
ROUNDS=${ROUNDS:-"weak,strong mid,midbal"}
mkdir -p runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

slot_gpus() {  # GPUs of slot $1 as a comma list
  local s=$(($1 * NPROC)); local IFS=,; echo "${GPUS[*]:$s:$NPROC}"
}

all_levels=()
for round in $ROUNDS; do
  IFS=, read -r -a LV <<< "$round"
  pids=()
  for i in "${!LV[@]}"; do
    lv=${LV[$i]}; name=pt-$lv-s$SEED; devs=$(slot_gpus "$i")
    [ -n "$devs" ] || { echo "not enough GPUs for $name"; exit 1; }
    echo "$name -> GPUs $devs (runs/$name.log)"
    CUDA_VISIBLE_DEVICES=$devs torchrun --standalone --nproc_per_node="$NPROC" \
      scripts/pretrain.py --config "configs/pretrain/$lv.yaml" "seed=$SEED" "$@" > "runs/$name.log" 2>&1 &
    pids+=($!)
    all_levels+=("$lv")
  done
  for p in "${pids[@]}"; do wait "$p"; done
  # Quick heatmap + gate without the RL pool, one model per slot.
  pids=()
  for i in "${!LV[@]}"; do
    name=pt-${LV[$i]}-s$SEED; dev=$(slot_gpus "$i" | cut -d, -f1)
    CUDA_VISIBLE_DEVICES=$dev python scripts/calibrate.py "checkpoints/$name" --skip-pool \
      > "runs/calib-quick-$name.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || true; done
  grep -h "gate PASS\|gate FAIL" runs/calib-quick-pt-*-s"$SEED".log || true
done

pids=()
for i in "${!all_levels[@]}"; do
  name=pt-${all_levels[$i]}-s$SEED; dev=${GPUS[$((i % ${#GPUS[@]}))]}
  echo "calibrate $name -> GPU $dev (runs/calib-$name.log)"
  CUDA_VISIBLE_DEVICES=$dev python scripts/calibrate.py "checkpoints/$name" > "runs/calib-$name.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || true; done
grep -h "gate PASS\|gate FAIL" runs/calib-pt-*-s"$SEED".log
