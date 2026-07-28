#!/bin/bash
# Phase 22 follow-up C4 — post-hoc aggregate eval of the RL final checkpoints.
#
# Per-step pass counts in the RL log are too noisy to settle whether the
# policy actually improved: the noise floor on this substrate is
# base 16.4/256, sigma 4.5 (2sigma band 7.3-25.4), measured from 16 readings
# of the SAME frozen base policy. This script instead scores each run's
# --final-checkpoint on the same 64 hard-tail problems at passk=5 — the
# metric the SFT experiments used, so the numbers are directly comparable:
#
#   SFT hard tail (samples=16, rounds=3): pass@5 0.246 -> 0.500 (+0.254)
#
# 5 evals (4 checkpoints + the untrained base as a control) on 5 GPUs,
# 64 problems x passk=5 = 320 generations each, ~20 min wallclock.
set -e
cd /raid/users/paul/workLLM

BIN=./target/release/examples/phase22_humaneval_baseline
CKPT_DIR=${1:-scratch-7b-sft/c4_posadv}
OUT=$CKPT_DIR/eval
mkdir -p "$OUT"

COMMON="--model-id Qwen2.5-Coder-7B --offset 100 --n-problems 64 --passk 5 \
  --sequential --aggregate --max-new-tokens 192"

gpu=0
# The base control re-measures the starting point under the same eval
# settings rather than trusting the 0.246 from the SFT runs.
CUDA_VISIBLE_DEVICES=$gpu $BIN $COMMON > "$OUT/base.log" 2>&1 &
echo "base (no checkpoint) GPU $gpu PID=$!"
gpu=$((gpu+1))

for arm in posonly fulladv; do
  for seed in 42 100; do
    CKPT="$CKPT_DIR/${arm}_seed${seed}_final.safetensors"
    if [ ! -f "$CKPT" ]; then
      echo "⚠ missing $CKPT — skipping"
      continue
    fi
    CUDA_VISIBLE_DEVICES=$gpu $BIN $COMMON --checkpoint "$CKPT" \
      > "$OUT/${arm}_seed${seed}.log" 2>&1 &
    echo "$arm seed=$seed GPU $gpu PID=$!"
    gpu=$((gpu+1))
  done
done

echo
echo "Launched. Results:"
echo "  grep -h 'aggregate pass@\\|per-prompt pass@' $OUT/*.log"
