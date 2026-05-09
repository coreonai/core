#!/bin/bash
# Phase 14 S1: seeds 3, 4 on GPU 1
set -e
cd /raid/users/paul/workLLM
for s in 3 4; do
  echo "=== p14s1 seed=$s gpu=1 ==="
  CUDA_VISIBLE_DEVICES=1 /tmp/p14_env/bin/python scripts/phase14_s1/self_improve.py \
    --seed $s --rounds 3 --samples 8 --train-steps 60 2>&1 | tail -8
  echo "=== p14s1 seed=$s done ==="
done
echo "=== seeds_b done ==="
