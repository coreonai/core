#!/bin/bash
# Phase 22 §6.5 — LiveCodeBench recipe run (contamination/transfer test).
#
# Generate with a trained checkpoint on the SAME slice as the base run
# (idx 640..760, F32), score with the official eval core, split pre/post cutoff.
# Compare post-cutoff pass@1 to base 0.087: does the self-improve recipe help on
# UNSEEN (post-cutoff) problems, or only pre-cutoff (contamination)?
#
# Usage: lcb_run_recipe.sh <checkpoint.safetensors> <out_dir> <gpu_base>
#   4 GPUs starting at gpu_base, 4 slices of 30.
set -e
cd /raid/users/paul/workLLM
BIN=./target/release/examples/phase22_dump_completions
CKPT=${1:?checkpoint path}
OUT=${2:?out dir}
GPU0=${3:-0}
CUTOFF="2024-09-01"
mkdir -p "$OUT"
[ -f "$CKPT" ] || { echo "⚠ missing checkpoint $CKPT"; exit 1; }

g=$GPU0
for off in 640 670 700 730; do
  CUDA_VISIBLE_DEVICES=$g $BIN \
    --benchmark livecodebench --model-id Qwen2.5-Coder-7B --dtype f32 \
    --checkpoint "$CKPT" \
    --offset $off --n-problems 30 --passk 1 --max-new-tokens 1024 \
    --dump "$OUT/slice_$off.json" > "$OUT/gen_$off.log" 2>&1 &
  echo "slice off=$off GPU $g PID=$!"
  g=$((g + 1))
done
wait
python3 -c "
import json, glob
allg = []
for f in sorted(glob.glob('$OUT/slice_*.json')): allg += json.load(open(f))
json.dump(allg, open('$OUT/gens_all.json','w'))
print(f'merged {len(allg)} -> $OUT/gens_all.json')
"
V=/raid/users/paul/workLLM/scratch-7b-sft/tools/lcb-venv/bin
for label in "overall" "PRE <${CUTOFF}" "POST >=${CUTOFF}"; do
  case "$label" in
    overall) dflag="" ;;
    PRE*)    dflag="--end-date $CUTOFF" ;;
    POST*)   dflag="--start-date $CUTOFF" ;;
  esac
  echo "=== $label ==="
  $V/python scripts/phase22_bench/lcb_score.py --gens "$OUT/gens_all.json" --release release_v5 $dflag 2>&1 | sed -n '/LCB RESULT/,/====/p'
done
echo "=== LCB_RECIPE_COMPLETE ($OUT) ==="
