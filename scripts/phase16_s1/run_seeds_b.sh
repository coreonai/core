#!/bin/bash
# Phase 16 S1 — samples=6 substrate, seeds 3/4 on GPU 1.
set -e
cd /raid/users/paul/workLLM
GPU=${1:-1}
for s in 3 4; do
  echo "=== p16s1 seed=$s gpu=$GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
    scripts/phase15_s1/self_improve.py \
    --seed $s --rounds 1 --samples 6 --train-steps 200 --max-new-tokens 200 \
    --out scripts/phase16_s1/run_s6_seed${s}.json \
    2>&1 | tail -8
  echo "=== p16s1 seed=$s done ==="
done
echo "=== seeds_b done ==="
