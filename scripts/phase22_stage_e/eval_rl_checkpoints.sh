#!/bin/bash
# Phase 22 Stage E — signal-bearing eval (deferred in the Stage E infra commit).
#
# Evaluates the 3 RL final checkpoints (rl_seed{42,100,200}_final.safetensors,
# 50 RL steps × 16 prompts × k=4, trained on HumanEval task_id 0..16) against
# base Qwen2.5-Coder-0.5B on the FULL 164-problem benchmark.
#
# RL trained on only task 0..16, so full-164 measures generalization
# (148 held-out problems).
#
# Protocol (identical for all 4 configs):
#   --n-problems 164 --passk 10 --sequential --aggregate --max-new-tokens 200
#   → aggregate pass@1 (raw, total_passes/total_attempts) + per-prompt pass@10
#
# 4-way parallel on free GPUs 2/3/4/5 (GPU 1 busy with external vLLM).
# Wallclock ~50-80 min.
set -e
cd /raid/users/paul/workLLM
BIN=./target/release/examples/phase22_humaneval_baseline
OUT=scripts/phase22_stage_e/eval
mkdir -p "$OUT"

COMMON="--n-problems 164 --passk 10 --sequential --aggregate --max-new-tokens 200"

# base control (no --checkpoint → loads model.safetensors from snapshot)
CUDA_VISIBLE_DEVICES=2 $BIN $COMMON \
  > "$OUT/base.log" 2>&1 &
echo "base        GPU 2 PID=$!"

for spec in "42:3" "100:4" "200:5"; do
  seed=${spec%%:*}; gpu=${spec##*:}
  CUDA_VISIBLE_DEVICES=$gpu $BIN $COMMON \
    --checkpoint scripts/phase22_stage_e/rl_seed${seed}_final.safetensors \
    > "$OUT/rl_seed${seed}.log" 2>&1 &
  echo "rl_seed$seed GPU $gpu PID=$!"
done
echo "Launched 4 evals (base + 3 RL seeds). Logs in $OUT/"
