#!/bin/bash
# Phase 22 Stage E follow-up — held-out tail eval for the n_prompts
# generalization test.
#
# Common held-out set = task 64..164 (100 problems), unseen by BOTH the
# n=16 sync run (trained task 0..16) and the n=64 run (trained task
# 0..64). Scored via `--offset 64 --n-problems 100`. Per-task seeds key
# off the absolute index, so this slice is bit-identical to that window
# of a full 0..164 run.
#
# 7 configs (base + n16-sync x3 + n64-sync x3), each on its own GPU,
# each with an isolated TMPDIR so the python verifies never collide
# (call_id counter is per-process; shared scratch dir → cross-run race).
# Pure inference — no concurrent training, so no CPU-verify stall.
set -e
cd /raid/users/paul/workLLM
BIN=./target/release/examples/phase22_humaneval_baseline
OUT=scripts/phase22_stage_e/eval
TAIL="--n-problems 100 --offset 64 --passk 10 --sequential --aggregate --max-new-tokens 200"
mkdir -p "$OUT"

launch() {  # label gpu [checkpoint-args...]
  local label=$1 gpu=$2; shift 2
  local tmp=/tmp/p22e_eval_${label}; mkdir -p "$tmp"
  CUDA_VISIBLE_DEVICES=$gpu TMPDIR=$tmp $BIN $TAIL "$@" \
    > "$OUT/tail_${label}.log" 2>&1 &
  echo "tail_${label} GPU $gpu PID=$!"
}

launch base 0
launch n16_42  2 --checkpoint scripts/phase22_stage_e/rl_sync_seed42_final.safetensors
launch n16_100 3 --checkpoint scripts/phase22_stage_e/rl_sync_seed100_final.safetensors
launch n16_200 4 --checkpoint scripts/phase22_stage_e/rl_sync_seed200_final.safetensors
launch n64_42  5 --checkpoint scripts/phase22_stage_e/rl_n64_seed42_final.safetensors
launch n64_100 6 --checkpoint scripts/phase22_stage_e/rl_n64_seed100_final.safetensors
launch n64_200 7 --checkpoint scripts/phase22_stage_e/rl_n64_seed200_final.safetensors
echo "Launched 7 held-out tail evals (task 64..164). Logs: $OUT/tail_*.log"
