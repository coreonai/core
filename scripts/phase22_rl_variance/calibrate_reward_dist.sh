#!/bin/bash
# Phase 22 RL variance study — guard #1: re-measure the POST-FIX per-prompt
# pass-count distribution the RL loop sees, to calibrate --advantage-clip.
#
# Why this is safe as a base-policy measurement: with --sync-every 0 the
# inference model never receives the adapter, so EVERY rl_step samples from
# the frozen base 7B policy. Each step prints a per-prompt pass-count
# histogram (added to phase22_he_reinforce). So `--rl-steps N` gives N
# independent base-policy draws over the 64 hard-tail problems from one
# process; running several seeds in parallel multiplies the sample.
#
# For binary rewards over k=4 samples the GRPO advantage depends ONLY on p
# (passes/k): p=1 -> passing sample +1.732, p=2 -> +1.0, p=3 -> +0.577
# (p=0,4 are zero-advantage, skipped). So the p-histogram fully determines
# the positive-advantage spread and hence the right clip.
#
# ~10-15 min/step (256 gens x 192 tok on 7B, no training-dominant sync).
set -e
cd /raid/users/paul/workLLM

BIN=./target/release/examples/phase22_he_reinforce
STEPS=${1:-2}
SEEDS=${2:-42,100,200}
OUT=${3:-scratch-7b-sft/rl_var_calib}

if ! $BIN --help 2>&1 | grep -q "advantage-mode"; then
  echo "⚠ binary predates --advantage-mode — rebuild the CUDA example first." ; exit 1
fi
if [ "$(strings $BIN | grep -c cudarc)" -eq 0 ]; then
  echo "⚠ $BIN is a CPU build (0 cudarc symbols) — rebuild with --features cuda." ; exit 1
fi
mkdir -p "$OUT"

IFS=, read -r -a SEED_ARR <<< "$SEEDS"
gpu=0
for seed in "${SEED_ARR[@]}"; do
  gpu_m=$gpu; gpu_t=$((gpu + 1)); gpu=$((gpu + 2))
  CUDA_VISIBLE_DEVICES=$gpu_m,$gpu_t $BIN \
    --model-id Qwen2.5-Coder-7B --train-bf16 --trainer-gpu 1 \
    --prompt-offset 100 --n-prompts 64 \
    --rl-steps "$STEPS" --k-per-prompt 4 --max-new-tokens 192 \
    --pg-micro-batch-size 1 --sync-every 0 --lr 2e-4 --seed "$seed" \
    --advantage-mode grpo \
    > "$OUT/calib_seed${seed}.log" 2>&1 &
  echo "seed=$seed GPUs $gpu_m(model),$gpu_t(trainer) PID=$!"
done
echo
echo "Launched ${#SEED_ARR[@]} seeds x $STEPS base-policy draws. Logs: $OUT/calib_seed*.log"
echo "Histogram lines:  grep 'pass-count hist' $OUT/calib_seed*.log"
