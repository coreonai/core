#!/bin/bash
# Phase 15 S3a — init-axis only: vary --init-seed, fix --harvest-seed=0.
# 5 inits × 1 harvest = 5 runs. Default GPU passable as $1.
set -e
cd /raid/users/paul/workLLM
GPU=${1:-2}
for i in 0 1 2 3 4; do
  echo "=== p15s3 init=$i harvest=0 gpu=$GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
    scripts/phase15_s3/decompose_seeds.py \
    --init-seed $i --harvest-seed 0 \
    --rounds 3 --samples 8 --train-steps 60 2>&1 | tail -8
  echo "=== p15s3 init=$i done ==="
done
echo "=== init_axis done ==="
