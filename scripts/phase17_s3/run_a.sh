#!/bin/bash
# Phase 17 S3 — MBPP-100 substrate, seeds 0/1/2.
set -e
cd /raid/users/paul/workLLM
GPU=${1:-5}
for s in 0 1 2; do
  echo "=== p17s3 mbpp seed=$s gpu=$GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
    scripts/phase17_s3/run_mbpp.py \
    --seed $s --rounds 1 --samples 6 --train-steps 200 --max-new-tokens 200 \
    2>&1 | tail -10
  echo "=== p17s3 mbpp seed=$s done ==="
done
echo "=== mbpp_a done ==="
