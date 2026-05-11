#!/bin/bash
# Phase 18 S3 — Complete MBPP multi-round 5-seed (seeds 2, 4 still missing).
# SB had seeds 0, 1, 3 done; this finishes the set for clean σ.
set -e
cd /raid/users/paul/workLLM
GPU=${1:-5}
for s in 2 4; do
  echo "=== p18s3 mbpp-mr seed=$s gpu=$GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
    scripts/phase17_sb/run_mr_mbpp.py \
    --seed $s --rounds 2 --samples 6 --train-steps 200 --max-new-tokens 200 \
    --out scripts/phase18_s3/run_mr_mbpp_seed${s}.json 2>&1 | tail -10
  echo "=== p18s3 mbpp-mr seed=$s done ==="
done
echo "=== p18s3 done ==="
