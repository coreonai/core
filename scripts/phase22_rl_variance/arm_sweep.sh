#!/bin/bash
# Phase 22 RL variance study — arm sweep. Runs the posonly REINFORCE recipe
# (the winning objective) under a chosen advantage transform, writing a
# --final-checkpoint per seed for a post-hoc aggregate eval on the SAME ruler
# the pre-registration fixes (docs/phase22-rl-variance.md).
#
# The MeanCenter arm already exists as the C4 posonly 6-seed arm
# (0.412 ± 0.103 pass@1) — do NOT re-run it; compare against it. This script
# runs the TEST arms: grpo, grpo+clip, rloo (control), and k=8 (harvest lever).
#
# Config is byte-identical to the C4 re-run except --advantage-mode /
# --advantage-clip / --k-per-prompt, so the arms are directly comparable.
#
# 2 GPUs/run (model + trainer), so 4 runs fill 8 GPUs. --sync-every 1 makes
# each step a 15 GB merge+reload => ~15 min/step => ~7.5 h for 30 steps.
#
# Usage: arm_sweep.sh <mode> <clip|-> <seeds> <k> <out> [steps]
#   arm_sweep.sh grpo -   42,100,200,300 4 scratch-7b-sft/rlvar_grpo
#   arm_sweep.sh grpo 1.0 42,100,200,300 4 scratch-7b-sft/rlvar_grpo_clip1
#   arm_sweep.sh mean -   42,100,200,300 8 scratch-7b-sft/rlvar_k8     # K=8 harvest arm
set -e
cd /raid/users/paul/workLLM

BIN=./target/release/examples/phase22_he_reinforce
MODE=${1:?mode: mean|rloo|grpo}
CLIP=${2:-'-'}
SEEDS=${3:?comma-separated seeds}
K=${4:-4}
OUT=${5:?output dir}
STEPS=${6:-30}

if [ "$(strings $BIN | grep -c cudarc)" -eq 0 ]; then
  echo "⚠ $BIN is a CPU build (0 cudarc symbols) — rebuild with --features cuda." ; exit 1
fi
mkdir -p "$OUT"

clip_arg=""
tag="$MODE"
if [ "$CLIP" != "-" ]; then clip_arg="--advantage-clip $CLIP"; tag="${MODE}clip${CLIP}"; fi
[ "$K" != "4" ] && tag="${tag}_k${K}"

IFS=, read -r -a SEED_ARR <<< "$SEEDS"
if [ "${#SEED_ARR[@]}" -gt 4 ]; then
  echo "⚠ ${#SEED_ARR[@]} seeds but only 4 runs fit on 8 GPUs — run in waves of 4." ; exit 1
fi

gpu=0
for seed in "${SEED_ARR[@]}"; do
  gpu_m=$gpu; gpu_t=$((gpu + 1)); gpu=$((gpu + 2))
  CUDA_VISIBLE_DEVICES=$gpu_m,$gpu_t $BIN \
    --model-id Qwen2.5-Coder-7B --train-bf16 --trainer-gpu 1 \
    --prompt-offset 100 --n-prompts 64 \
    --rl-steps "$STEPS" --k-per-prompt "$K" --max-new-tokens 192 \
    --pg-micro-batch-size 1 --sync-every 1 --lr 2e-4 --seed "$seed" \
    --pg-positive-only --advantage-mode "$MODE" $clip_arg \
    --sync-path /dev/shm/rlvar_${tag}_seed${seed}.safetensors \
    --final-checkpoint "$OUT/${tag}_seed${seed}_final.safetensors" \
    > "$OUT/${tag}_seed${seed}.log" 2>&1 &
  echo "$tag seed=$seed GPUs $gpu_m(model),$gpu_t(trainer) PID=$!"
done
echo
echo "Launched arm '$tag' x ${#SEED_ARR[@]} seeds, $STEPS steps. Logs: $OUT/${tag}_seed*.log"
echo "Eval when done: phase22_humaneval_baseline --checkpoint <f> --offset 100 --n-problems 64 --passk 5 --sequential --aggregate"
