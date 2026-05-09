#!/bin/bash
# Phase 15 S1: HumanEval substrate, seeds 3/4 on GPU 3
set -e
cd /raid/users/paul/workLLM
for s in 3 4; do
  echo "=== p15s1 seed=$s gpu=3 ==="
  CUDA_VISIBLE_DEVICES=3 /tmp/p14_env/bin/python scripts/phase15_s1/self_improve.py \
    --seed $s --rounds 1 --samples 3 --max-new-tokens 200 --train-steps 200 \
    2>&1 | tail -8
  echo "=== p15s1 seed=$s done ==="
done
echo "=== seeds_b done ==="
