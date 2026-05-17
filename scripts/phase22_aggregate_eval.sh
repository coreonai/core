#!/bin/bash
# Phase 22 Stage D — aggregate eval (Phase 17 metric) on 5-seed
# checkpoints from G4 / G5 / etc batches.
#
# Usage: ./scripts/phase22_aggregate_eval.sh <batch_tag>
# Example: ./scripts/phase22_aggregate_eval.sh G5
#
# Expects checkpoints at /tmp/phase22d_<tag>_seed{0..4}/ckpts/r0_merged.r1.safetensors
# (the r=2 checkpoint, after 2 training rounds).
#
# Outputs to /tmp/phase22d_<tag>_seed{0..4}/aggregate/r2_eval.log
#
# Launches 5 aggregate evals in parallel on GPUs 0/1/5/6/7.
# Wallclock: ~50-80 min total.
set -e
BATCH=${1:?usage: $0 <batch_tag> (e.g. G4, G5)}
cd /raid/users/paul/workLLM

for i in 0 1 2 3 4; do
  case $i in
    0) gpu=5 ;; 1) gpu=6 ;; 2) gpu=7 ;; 3) gpu=1 ;; *) gpu=0 ;;
  esac
  CKPT="/tmp/phase22d_${BATCH}_seed${i}/ckpts/r0_merged.r1.safetensors"
  if [ ! -f "$CKPT" ]; then
    echo "⚠ missing $CKPT — skipping seed=$((i*100+100))"
    continue
  fi
  mkdir -p /tmp/phase22d_${BATCH}_seed${i}/aggregate
  CUDA_VISIBLE_DEVICES=$gpu ./target/release/examples/phase22_humaneval_baseline \
    --n-problems 164 --passk 10 --sequential --aggregate --max-new-tokens 200 \
    --checkpoint "$CKPT" \
    > /tmp/phase22d_${BATCH}_seed${i}/aggregate/r2_eval.log 2>&1 &
  echo "${BATCH}-agg seed=$((i*100+100)) GPU $gpu PID=$!"
done
echo "Launched 5 aggregate evals for batch=$BATCH"
echo "Wait: until [ \"\$(pgrep -af 'phase22_humaneval_baseline.*--checkpoint' | wc -l)\" -eq 0 ]; do sleep 60; done"
echo "Then collect: for i in 0 1 2 3 4; do grep 'aggregate pass@1' /tmp/phase22d_${BATCH}_seed\$i/aggregate/r2_eval.log; done"
