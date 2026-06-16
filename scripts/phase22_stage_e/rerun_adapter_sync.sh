#!/bin/bash
# Phase 22 Stage E follow-up — adapter-sync re-experiment.
#
# The original signal-bearing run (rl_seed{42,100,200}_final.safetensors)
# was OFF-POLICY: --sync-every 0, so QwenModelActor sampled on base
# weights all 50 steps while the trainer's LoRA drifted → merged policy
# collapsed (3/3 seeds, mean pass@1 0.032 vs base 0.218).
#
# This re-run is IDENTICAL except --sync-every 1 (bake LoRA into base +
# reload sampler EVERY step → fully on-policy). Tests whether the
# off-policy approximation was the cause of the collapse.
#
# Params match the off-policy run (documented signal-bearing recipe):
#   --rl-steps 50 --n-prompts 16 --k-per-prompt 4 --max-new-tokens 64
#   --pg-micro-batch-size 4   + per-seed distinct sync/final paths.
#
# 3 seeds on GPUs 2/3/6. Per-step ~65s + ~30s sync ≈ 95s → ~80 min/seed.
set -e
cd /raid/users/paul/workLLM
BIN=./target/release/examples/phase22_he_reinforce
OUT=scripts/phase22_stage_e

for spec in "42:2" "100:3" "200:6"; do
  seed=${spec%%:*}; gpu=${spec##*:}
  CUDA_VISIBLE_DEVICES=$gpu $BIN \
    --rl-steps 50 --n-prompts 16 --k-per-prompt 4 --max-new-tokens 64 \
    --pg-micro-batch-size 4 --sync-every 1 --seed $seed \
    --sync-path /dev/shm/phase22e_sync_seed${seed}.safetensors \
    --final-checkpoint $OUT/rl_sync_seed${seed}_final.safetensors \
    > $OUT/log_sync_seed${seed}.txt 2>&1 &
  echo "sync seed=$seed GPU $gpu PID=$!"
done
echo "Launched 3 adapter-sync RL runs (--sync-every 1). Logs: $OUT/log_sync_seed*.txt"
