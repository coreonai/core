#!/bin/bash
# Phase 15 S3b — HumanEval init-axis: vary --init-seed, fix --harvest-seed=0.
set -e
cd /raid/users/paul/workLLM
GPU=${1:-0}
for i in 0 1 2 3 4; do
  echo "=== p15s3b he init=$i harvest=0 gpu=$GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
    scripts/phase15_s3/decompose_seeds_humaneval.py \
    --init-seed $i --harvest-seed 0 \
    --rounds 1 --samples 3 --train-steps 200 --max-new-tokens 200 \
    2>&1 | tail -8
  echo "=== p15s3b he init=$i done ==="
done
echo "=== he_init_axis done ==="
