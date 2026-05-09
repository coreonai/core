#!/bin/bash
# Phase 15 S3b — HumanEval harvest-axis: fix --init-seed=0, vary --harvest-seed.
set -e
cd /raid/users/paul/workLLM
GPU=${1:-1}
for h in 0 1 2 3 4; do
  echo "=== p15s3b he init=0 harvest=$h gpu=$GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
    scripts/phase15_s3/decompose_seeds_humaneval.py \
    --init-seed 0 --harvest-seed $h \
    --rounds 1 --samples 3 --train-steps 200 --max-new-tokens 200 \
    2>&1 | tail -8
  echo "=== p15s3b he harvest=$h done ==="
done
echo "=== he_harvest_axis done ==="
