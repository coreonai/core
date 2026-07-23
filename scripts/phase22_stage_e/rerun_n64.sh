#!/bin/bash
# Phase 22 Stage E follow-up — larger n_prompts generalization test.
#
# The adapter-sync win (n_prompts=16) was on-distribution-concentrated:
# the 16 training prompts hit ~0.81, held-out 148 improved only modestly.
# Hypothesis: training on MORE prompts broadens the held-out gain.
#
# IDENTICAL to rerun_adapter_sync.sh except --n-prompts 64 (task 0..64).
# Common held-out tail vs the n=16 run = task 64..164 (100 problems,
# unseen by BOTH), eval'd via `--offset 64 --n-problems 100`.
#
# 3 seeds on GPUs 2/3/6. n=64 → 256 gens/step (~4x n=16) → ~290s/step +
# sync ≈ ~4h/seed. Peak GPU mem unchanged (samples micro-batched at 4).
#
# SCRATCH ISOLATION: HumanEvalDomain writes `solution_{call_id}.py` to
# `$TMPDIR/workllm-phase22e-humaneval` and runs python3 on it. The
# call_id counter is per-PROCESS (starts at 0 each run), so concurrent
# runs sharing one TMPDIR collide on identical filenames → one run
# executes another's code → corrupted reward. Each seed gets its own
# TMPDIR so the scratch dirs are disjoint and verifies are race-free.
# Run training ALONE (no concurrent evals) to avoid python-verify CPU
# contention that previously stalled step 0 to ~25 min.
set -e
cd /raid/users/paul/workLLM
BIN=./target/release/examples/phase22_he_reinforce
OUT=scripts/phase22_stage_e

for spec in "42:2" "100:3" "200:6"; do
  seed=${spec%%:*}; gpu=${spec##*:}
  TMP=/tmp/p22e_n64_s${seed}; mkdir -p $TMP
  CUDA_VISIBLE_DEVICES=$gpu TMPDIR=$TMP $BIN \
    --rl-steps 50 --n-prompts 64 --k-per-prompt 4 --max-new-tokens 64 \
    --pg-micro-batch-size 4 --sync-every 1 --seed $seed \
    --sync-path /dev/shm/phase22e_n64_sync_seed${seed}.safetensors \
    --final-checkpoint $OUT/rl_n64_seed${seed}_final.safetensors \
    > $OUT/log_n64_seed${seed}.txt 2>&1 &
  echo "n64 seed=$seed GPU $gpu TMPDIR=$TMP PID=$!"
done
echo "Launched 3 n_prompts=64 adapter-sync RL runs (TMPDIR-isolated). Logs: $OUT/log_n64_seed*.txt"
