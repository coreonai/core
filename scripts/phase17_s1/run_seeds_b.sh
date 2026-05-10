#!/bin/bash
# Phase 17 S1 — multi-round SFT, seeds 3/4.
set -e
cd /raid/users/paul/workLLM
GPU=${1:-1}
for s in 3 4; do
  echo "=== p17s1 seed=$s gpu=$GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
    scripts/phase15_s1/self_improve.py \
    --seed $s --rounds 2 --samples 6 --train-steps 200 --max-new-tokens 200 \
    --out scripts/phase17_s1/run_r2s6_seed${s}.json 2>&1 | tail -10
  echo "=== p17s1 seed=$s done ==="
done
echo "=== seeds_b done ==="
