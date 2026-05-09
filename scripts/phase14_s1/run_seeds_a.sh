#!/bin/bash
# Phase 14 S1: seeds 0, 1, 2 on GPU 0
set -e
cd /raid/users/paul/workLLM
for s in 0 1 2; do
  echo "=== p14s1 seed=$s gpu=0 ==="
  CUDA_VISIBLE_DEVICES=0 /tmp/p14_env/bin/python scripts/phase14_s1/self_improve.py \
    --seed $s --rounds 3 --samples 8 --train-steps 60 2>&1 | tail -8
  echo "=== p14s1 seed=$s done ==="
done
echo "=== seeds_a done ==="
