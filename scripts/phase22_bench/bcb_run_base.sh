#!/bin/bash
# Phase 22 §6.5 — BigCodeBench base-7B generation (Complete / Hard, 148 tasks).
# Generate in Rust (F32, greedy pass@1 — the leaderboard's calibrated-pass@1
# metric is greedy), one slice per GPU, then merge to a single samples JSONL
# {task_id, solution} for the official Docker harness (syncheck -> evaluate).
# Scoring is a separate CPU-bound Docker step (bcb_score.sh); this script only
# does the GPU generation.
set +e
cd /raid/users/paul/workLLM
BIN=./target/release/examples/phase22_dump_completions
JSONL=data/bigcodebench/BigCodeBench-Hard.jsonl
OUT=scratch-7b-sft/bcb_hard_base
mkdir -p "$OUT"
N=$(wc -l < "$JSONL")               # 148
PER=19                              # 8 slices cover 148
echo "=== BigCodeBench Hard base gen: $N tasks, F32 greedy, PER=$PER ==="

g=0
for off in $(seq 0 $PER $((N-1))); do
  CUDA_VISIBLE_DEVICES=$g $BIN --benchmark bigcodebench --model-id Qwen2.5-Coder-7B \
    --jsonl "$JSONL" --split complete --dtype f32 \
    --offset $off --n-problems $PER --passk 1 --max-new-tokens 1024 \
    --dump "$OUT/slice_$off.jsonl" > "$OUT/gen_$off.log" 2>&1 &
  g=$((g + 1))
done
wait
echo "=== generation done; merging slices ==="

# merge slice JSONLs into one samples file (one {task_id, solution} per line).
cat "$OUT"/slice_*.jsonl > "$OUT/bcb_hard_base_samples.jsonl"
echo "merged $(wc -l < "$OUT/bcb_hard_base_samples.jsonl") samples -> $OUT/bcb_hard_base_samples.jsonl"
# sanity: unique task_ids
python3 -c "
import json
ids=[json.loads(l)['task_id'] for l in open('$OUT/bcb_hard_base_samples.jsonl') if l.strip()]
print('samples:', len(ids), 'unique task_ids:', len(set(ids)))
"
echo "=== BCB_BASE_GEN_COMPLETE ==="
