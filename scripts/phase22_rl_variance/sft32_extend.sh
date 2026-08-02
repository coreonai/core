#!/bin/bash
# Phase 22 RL-variance attribution — extend SFT samples-per-prompt=32 to 6 seeds.
#
# Firms the control for "is K=8 RL's mean lift from RL or just more harvest?"
# The samples=32 hard-tail run existed at 4 seeds (0.385 pass@1) but the launch
# command did not survive. Config RECOVERED from the surviving samples=16 log
# (scratch-7b-sft/htrescue_seed42.log, same 7월-26 session/recipe):
#   rounds=3, samples-per-prompt (16→32), train-steps=100, max-new=200,
#   temp=0.8, top-k=0, lora r16/α32, lr=2e-4, wd=0, batch=1, skip 0..99, 7B.
# This script ships that command so it never gets lost again.
#
# Existing seeds 42/100/200/300 checkpoints are at hts32_out_s{seed}; this adds
# 400/500. Eval all 6 uniformly afterward (sft32_eval.sh). ~7 h/run,
# 2 GPUs/run (7B trainer-split) → 2 runs fill 4 GPUs.
set -e
cd /raid/users/paul/workLLM

BIN=./target/release/examples/phase22_he_mr_sft
SEEDS=${1:-400,500}
if [ "$(strings $BIN | grep -c cudarc)" -eq 0 ]; then
  echo "⚠ $BIN is a CPU build — rebuild with --features cuda." ; exit 1
fi
SKIP=$(seq -s, 0 99)   # hard tail = hide idx 0..99, keep 100..163

IFS=, read -r -a SEED_ARR <<< "$SEEDS"
gpu=0
for seed in "${SEED_ARR[@]}"; do
  gpu_m=$gpu; gpu_t=$((gpu + 1)); gpu=$((gpu + 2))
  CUDA_VISIBLE_DEVICES=$gpu_m,$gpu_t $BIN \
    --model-id Qwen2.5-Coder-7B --train-bf16 --trainer-gpu 1 \
    --rounds 3 --samples-per-prompt 32 --train-steps 100 \
    --max-new-tokens 200 --temperature 0.8 --top-k 0 \
    --lora-rank 16 --lora-alpha 32 --lr 0.0002 --weight-decay 0 --batch-size 1 \
    --eval-passk 5 --prompt-skip-list "$SKIP" \
    --seed "$seed" \
    --out-dir "scratch-7b-sft/hts32_out_s${seed}" \
    --scratch-dir "scratch-7b-sft/hts32_scr_s${seed}" \
    > "scratch-7b-sft/hts32_seed${seed}.log" 2>&1 &
  echo "samples=32 seed=$seed GPUs $gpu_m(model),$gpu_t(trainer) PID=$!"
done
echo
echo "Launched samples=32 x ${#SEED_ARR[@]} seeds, rounds=3. Checkpoints: hts32_out_s{seed}/r0_merged.r2.safetensors"
echo "Eval all 6 when done: bash scripts/phase22_rl_variance/sft32_eval.sh"
