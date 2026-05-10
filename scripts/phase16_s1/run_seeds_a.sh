#!/bin/bash
# Phase 16 S1 — samples=6 substrate, seeds 0/1/2 on GPU 0.
# Reuses Phase 15 S1 self_improve.py with --samples 6.
set -e
cd /raid/users/paul/workLLM
GPU=${1:-0}
for s in 0 1 2; do
  echo "=== p16s1 seed=$s gpu=$GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
    scripts/phase15_s1/self_improve.py \
    --seed $s --rounds 1 --samples 6 --train-steps 200 --max-new-tokens 200 \
    --out scripts/phase16_s1/run_s6_seed${s}.json \
    2>&1 | tail -8
  echo "=== p16s1 seed=$s done ==="
done
echo "=== seeds_a done ==="
