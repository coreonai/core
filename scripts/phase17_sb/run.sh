#!/bin/bash
# Phase 17 SB — Multi-round MBPP, 3 seeds.
set -e
cd /raid/users/paul/workLLM
GPU=${1:-3}
for s in 0 1 3; do
  echo "=== p17sb mr_mbpp seed=$s gpu=$GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
    scripts/phase17_sb/run_mr_mbpp.py \
    --seed $s --rounds 2 --samples 6 --train-steps 200 --max-new-tokens 200 \
    2>&1 | tail -10
  echo "=== p17sb mr_mbpp seed=$s done ==="
done
echo "=== sb done ==="
