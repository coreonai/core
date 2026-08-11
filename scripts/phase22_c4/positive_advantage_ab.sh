#!/bin/bash
# Phase 22 follow-up C4 — bound the REINFORCE objective.
#
# C3 (docs/phase22-c3-rl-step-semantics.md) found the 7B hard-tail RL
# collapse is NOT adapter sync. `pg_sample_loss` computes `mean_ce * reward`,
# so a negative-advantage sample is gradient ASCENT on cross-entropy — no
# upper bound — and under RLOO k=4 that is ~75% of surviving samples. C3
# fixed a 256x optimizer-update amplifier but left the objective unbounded;
# 2/2 legacy seeds still collapsed to 0/256 at 1024-1280 cumulative updates.
#
# C4 removes the unbounded term: `--pg-positive-only` keeps only the
# completions that PASSED the verifier, so the loss is `reward * CE >= 0`.
# That is reward-weighted SFT on verified-correct completions — rejection-
# sampling FT / RAFT, the same family as the Phase 22 SFT recipe worth
# +0.254 on this hard tail.
#
# Two questions, one run:
#   1. Does positive-only LEARN?  C3's 6-step run showed survival only; every
#      reading sat inside the noise floor (16.4/256, sigma 4.5, 2sigma band
#      7.3-25.4, measured from 16 readings of the frozen base policy).
#      30 steps = 30 updates, comparable to an SFT round's 30.
#   2. Does full-advantage (C3 fixed arm) eventually RUN AWAY at 1 update
#      per step, or did the fix actually cure it? C3 could not tell: 6
#      updates is ~200x below the dose that kills the legacy arm.
#
# Per-step pass counts are too noisy (sigma 4.5) to settle question 1 on
# their own, so every run also writes a --final-checkpoint for a proper
# post-hoc aggregate eval on the same 64 hard-tail problems.
#
# 4 runs x 2 GPUs = all 8 cards, ~15 min/step => ~7.5 h for 30 steps.
set -e
cd /raid/users/paul/workLLM

BIN=./target/release/examples/phase22_he_reinforce
STEPS=${1:-30}
ONLY_ARM=${2:-}
# Seeds for THIS batch. Only 4 runs fit on 8 GPUs (2 cards each), so a
# 4-seed x 2-arm design is two batches: "42,100" then "200,300".
SEEDS=${3:-42,100}
# Output dir. Never reuse one across runs — the final checkpoints are 15 GB
# each and the logs are the only record of the trajectory.
OUT=${4:-scratch-7b-sft/c4_posadv}

if ! $BIN --help 2>&1 | grep -q "pg-positive-only"; then
  echo "⚠ binary predates the C4 flag — rebuild:"
  echo "  CUDA_HOME=/usr/local/cuda-12.5 PATH=/usr/local/cuda-12.5/bin:\$PATH \\"
  echo "    cargo build -p llm-actors --example phase22_he_reinforce --features cuda --release"
  exit 1
fi
mkdir -p $OUT

# --- tmpfs hygiene -----------------------------------------------------------
# Each run rewrites a ~15 GB merged checkpoint to /dev/shm every step
# (--sync-every 1). These are pure scratch: nothing reads them after the run
# exits. Leaving them behind filled tmpfs to 100% (33 files, 541 GB of a 504 GB
# fs) and killed a run mid-batch with "No space left on device" — at step 0,
# after the model was already loaded. Two defences:
#   1. a pre-flight free-space check, so the batch fails loudly up front
#      instead of a subset of runs dying at a random step;
#   2. each run removes its own sync file when the binary exits (subshell, so
#      it fires on success AND failure without needing to `wait`).
shm_guard() { # $1 = number of runs about to start
  local need_gb=$(( $1 * 16 ))
  local free_gb=$(df -BG --output=avail /dev/shm | tail -1 | tr -dc '0-9')
  if [ "$free_gb" -lt "$need_gb" ]; then
    echo "⚠ /dev/shm has ${free_gb}G free, need ~${need_gb}G for $1 run(s)."
    echo "  Stale sync files from finished runs are the usual cause:"
    echo "    ls -la /dev/shm/*.safetensors"
    echo "  Remove the ones whose runs have exited, then retry."
    exit 1
  fi
}

IFS=, read -r -a SEED_ARR <<< "$SEEDS"
shm_guard "$(( ${#SEED_ARR[@]} * 2 ))"
gpu=0
for arm in posonly fulladv; do
  for seed in "${SEED_ARR[@]}"; do
    gpu_m=$gpu
    gpu_t=$((gpu + 1))
    gpu=$((gpu + 2))
    if [ -n "$ONLY_ARM" ] && [ "$arm" != "$ONLY_ARM" ]; then continue; fi
    extra=""
    if [ "$arm" = "posonly" ]; then
      extra="--pg-positive-only"
    fi
    SYNC=/dev/shm/phase22c4_${arm}_seed${seed}.safetensors
    ( CUDA_VISIBLE_DEVICES=$gpu_m,$gpu_t $BIN \
      --model-id Qwen2.5-Coder-7B --train-bf16 --trainer-gpu 1 \
      --prompt-offset 100 --n-prompts 64 \
      --rl-steps "$STEPS" --k-per-prompt 4 --max-new-tokens 192 \
      --pg-micro-batch-size 1 --sync-every 1 --lr 2e-4 --seed "$seed" \
      $extra \
      --sync-path "$SYNC" \
      --final-checkpoint $OUT/${arm}_seed${seed}_final.safetensors \
      > $OUT/${arm}_seed${seed}.log 2>&1
      rm -f "$SYNC" ) &
    echo "$arm seed=$seed GPUs $gpu_m(model),$gpu_t(trainer) PID=$!"
  done
done
echo
echo "Launched ${ONLY_ARM:-both arms} x 2 seeds, $STEPS steps. Logs: $OUT/*.log"
echo "Table:  ./scripts/phase22_c3/summarize_ab.sh $OUT"
echo "Noise floor: base 16.4/256, sigma 4.5, 2sigma band 7.3-25.4"
