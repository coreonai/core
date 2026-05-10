#!/bin/bash
# Phase 17 S7a — samples=10 SFT, seeds 3/4.
set -e
cd /raid/users/paul/workLLM
GPU=${1:-6}
for s in 3 4; do
  echo "=== p17s7a s10 seed=$s gpu=$GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
    scripts/phase15_s1/self_improve.py \
    --seed $s --rounds 1 --samples 10 --train-steps 200 --max-new-tokens 200 \
    --out scripts/phase17_s7/run_s10_seed${s}.json 2>&1 | tail -10
  echo "=== p17s7a s10 seed=$s done ==="
done
echo "=== s10_b done ==="
