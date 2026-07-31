#!/bin/bash
# Phase 22 RL variance — post-hoc aggregate eval of an arm's final checkpoints
# on the pre-registered ruler (64 hard-tail problems, passk=5, aggregate).
# One GPU per checkpoint + a base control. ~20 min wallclock.
#
# Usage: eval_arm.sh <ckpt_dir>   # evals every *_final.safetensors in the dir
set -e
cd /raid/users/paul/workLLM

BIN=./target/release/examples/phase22_humaneval_baseline
CKPT_DIR=${1:?checkpoint dir}
OUT="$CKPT_DIR/eval"
mkdir -p "$OUT"

if [ "$(strings $BIN | grep -c cudarc)" -eq 0 ]; then
  echo "⚠ $BIN is a CPU build — rebuild with --features cuda." ; exit 1
fi

COMMON="--model-id Qwen2.5-Coder-7B --offset 100 --n-problems 64 --passk 5 \
  --sequential --aggregate --max-new-tokens 192"

gpu=0
# base control — re-measure the starting point under identical eval settings.
CUDA_VISIBLE_DEVICES=$gpu $BIN $COMMON > "$OUT/base.log" 2>&1 &
echo "base GPU $gpu PID=$!"
gpu=$((gpu+1))

for CKPT in "$CKPT_DIR"/*_final.safetensors; do
  [ -f "$CKPT" ] || continue
  name=$(basename "$CKPT" _final.safetensors)
  CUDA_VISIBLE_DEVICES=$gpu $BIN $COMMON --checkpoint "$CKPT" \
    > "$OUT/${name}.log" 2>&1 &
  echo "$name GPU $gpu PID=$!"
  gpu=$((gpu+1))
done

wait
echo "=== eval done. aggregate lines: ==="
grep -H "aggregate pass@" "$OUT"/*.log
