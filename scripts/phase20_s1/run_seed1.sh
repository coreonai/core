#!/bin/bash
# Phase 20 S1 — rounds=6, seed 1 (Phase 19 record holder at r=5: 0.620).
set -e
cd /raid/users/paul/workLLM
GPU=${1:-1}
echo "=== p20s1 r6 seed=1 gpu=$GPU ==="
CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
  scripts/phase15_s1/self_improve.py \
  --seed 1 --rounds 6 --samples 6 --train-steps 200 --max-new-tokens 200 \
  --out scripts/phase20_s1/run_r6_seed1.json 2>&1 | tail -16
echo "=== p20s1 r6 seed=1 done ==="
