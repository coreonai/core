#!/bin/bash
# Phase 15 S3a — harvest-axis only: fix --init-seed=0, vary --harvest-seed.
# 1 init × 5 harvests = 5 runs.
set -e
cd /raid/users/paul/workLLM
GPU=${1:-3}
for h in 0 1 2 3 4; do
  echo "=== p15s3 init=0 harvest=$h gpu=$GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
    scripts/phase15_s3/decompose_seeds.py \
    --init-seed 0 --harvest-seed $h \
    --rounds 3 --samples 8 --train-steps 60 2>&1 | tail -8
  echo "=== p15s3 harvest=$h done ==="
done
echo "=== harvest_axis done ==="
