#!/bin/bash
# Phase 22 §6.5 — LiveCodeBench base-7B run (F32, cutoff-spanning slice).
#
# Generate in Rust (F32 — BF16 corrupts long prompts), score with the official
# lcb_runner eval core, split pre/post the Qwen2.5-Coder-7B cutoff (2024-09-01).
# 4 GPUs x 30 problems = idx 640..760 (spans 2024-06..2024-11). F32 7B = 28GB,
# one instance per 40GB card.
set -e
cd /raid/users/paul/workLLM

BIN=./target/release/examples/phase22_dump_completions
OUT=scratch-7b-sft/lcb_base
mkdir -p "$OUT"
CUTOFF="2024-09-01"   # Qwen2.5-Coder-7B released ~2024-09; problems on/after are post-cutoff

if [ "$(strings $BIN | grep -c cudarc)" -eq 0 ]; then
  echo "⚠ $BIN is a CPU build — rebuild with --features cuda." ; exit 1
fi

# 4 parallel slices (F32).
gpu=0
for off in 640 670 700 730; do
  CUDA_VISIBLE_DEVICES=$gpu $BIN \
    --benchmark livecodebench --model-id Qwen2.5-Coder-7B --dtype f32 \
    --offset $off --n-problems 30 --passk 1 --max-new-tokens 1024 \
    --dump "$OUT/slice_$off.json" > "$OUT/gen_$off.log" 2>&1 &
  echo "slice off=$off GPU $gpu PID=$!"
  gpu=$((gpu + 1))
done
wait
echo "=== generation done; merging slices ==="
python3 -c "
import json, glob
allg = []
for f in sorted(glob.glob('$OUT/slice_*.json')):
    allg += json.load(open(f))
json.dump(allg, open('$OUT/gens_all.json','w'))
print(f'merged {len(allg)} question_ids -> $OUT/gens_all.json')
"

V=/raid/users/paul/workLLM/scratch-7b-sft/tools/lcb-venv/bin
echo "=== score: overall ==="
$V/python scripts/phase22_bench/lcb_score.py --gens "$OUT/gens_all.json" --release release_v5 2>&1 | sed -n '/LCB RESULT/,/====/p'
echo "=== score: PRE-cutoff (< $CUTOFF) ==="
$V/python scripts/phase22_bench/lcb_score.py --gens "$OUT/gens_all.json" --release release_v5 --end-date $CUTOFF 2>&1 | sed -n '/LCB RESULT/,/====/p'
echo "=== score: POST-cutoff (>= $CUTOFF) ==="
$V/python scripts/phase22_bench/lcb_score.py --gens "$OUT/gens_all.json" --release release_v5 --start-date $CUTOFF 2>&1 | sed -n '/LCB RESULT/,/====/p'
echo "=== LCB_BASE_COMPLETE ==="
