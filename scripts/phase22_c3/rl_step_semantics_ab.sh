#!/bin/bash
# Phase 22 follow-up C3 — does the RL collapse come from the *number* of
# optimizer updates, not from the adapter sync?
#
# Stage E finding (docs/phase22-7b-results.md): REINFORCE on the 7B hard
# tail is healthy (~15/256 passes) for steps 0-3, then the first adapter
# sync craters it to 0/256 forever. Read as "RL + adapter-sync is unstable".
#
# Re-reading the training step says otherwise. `--pg-micro-batch-size 1`
# was introduced as a *memory* knob, but `train_qwen_lora_pg_step` issued
# one AdamW update per micro-batch — so 64 prompts x k=4 = **256 optimizer
# updates per RL step**, ~1024 before the first `--sync-every 4` sync.
# (An SFT round uses 30.) AdamW normalises by gradient magnitude, so the
# numerically tiny PG loss is not a small step. The sync did not break the
# policy; it was the first time the sampler *saw* the already-broken one,
# because before that the sampler ran on frozen base weights.
#
# The C3 fix accumulates the micro-batch gradients into ONE update per RL
# step and drops RLOO zero-advantage samples (~94% of the hard-tail batch).
#
# A/B, 2 seeds per arm, `--sync-every 1` so any damage shows up at the very
# next step:
#   FIXED  (default)                          -> expect pass to hold ~15/256
#   LEGACY (--pg-legacy-updates + keep-zero)  -> expect collapse at step 1
#
# The collapse signature to watch in the log is `comp_len` falling off a
# cliff (mode-collapsed policy emits EOS immediately) together with
# `pass = 0` and a much shorter `elapsed_step`.
#
# 4 runs x 2 GPUs (model + trainer) = all 8 cards. ~14 min/step + ~1 min
# sync => ~1.5 h for 6 steps.
set -e
cd /raid/users/paul/workLLM

BIN=./target/release/examples/phase22_he_reinforce
OUT=scratch-7b-sft/c3_ab
STEPS=${1:-6}
# Optional 2nd arg: run only one arm ("fixed" or "legacy"). Used to relaunch
# an arm without disturbing the other one's GPUs.
ONLY_ARM=${2:-}

if ! ./target/release/examples/phase22_he_reinforce --help 2>&1 | grep -q "pg-legacy-updates"; then
  echo "⚠ binary predates the C3 flags — rebuild:"
  echo "  CUDA_HOME=/usr/local/cuda-12.5 PATH=/usr/local/cuda-12.5/bin:\$PATH \\"
  echo "    cargo build -p llm-actors --example phase22_he_reinforce --features cuda --release"
  exit 1
fi
mkdir -p $OUT

# arm:seed:model_gpu:trainer_gpu
for spec in "fixed:42:0:1" "fixed:100:2:3" "legacy:42:4:5" "legacy:100:6:7"; do
  IFS=: read -r arm seed gpu_m gpu_t <<< "$spec"
  if [ -n "$ONLY_ARM" ] && [ "$arm" != "$ONLY_ARM" ]; then continue; fi
  extra=""
  if [ "$arm" = "legacy" ]; then
    extra="--pg-legacy-updates --pg-keep-zero-advantage"
  fi
  CUDA_VISIBLE_DEVICES=$gpu_m,$gpu_t $BIN \
    --model-id Qwen2.5-Coder-7B --train-bf16 --trainer-gpu 1 \
    --prompt-offset 100 --n-prompts 64 \
    --rl-steps "$STEPS" --k-per-prompt 4 --max-new-tokens 192 \
    --pg-micro-batch-size 1 --sync-every 1 --lr 2e-4 --seed "$seed" \
    $extra \
    --sync-path /dev/shm/phase22c3_${arm}_seed${seed}.safetensors \
    > $OUT/${arm}_seed${seed}.log 2>&1 &
  echo "$arm seed=$seed GPUs $gpu_m(model),$gpu_t(trainer) PID=$!"
done

echo
echo "Launched ${ONLY_ARM:-both arms} x 2 seeds, $STEPS steps each. Logs: $OUT/*.log"
echo "Watch: grep rl_step $OUT/*.log"
