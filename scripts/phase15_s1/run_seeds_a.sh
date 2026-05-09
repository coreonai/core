#!/bin/bash
# Phase 15 S1: HumanEval substrate, seeds 0/1/2 on GPU 2
set -e
cd /raid/users/paul/workLLM
for s in 0 1 2; do
  echo "=== p15s1 seed=$s gpu=2 ==="
  CUDA_VISIBLE_DEVICES=2 /tmp/p14_env/bin/python scripts/phase15_s1/self_improve.py \
    --seed $s --rounds 1 --samples 3 --max-new-tokens 200 --train-steps 200 \
    2>&1 | tail -8
  echo "=== p15s1 seed=$s done ==="
done
echo "=== seeds_a done ==="
