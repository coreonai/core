#!/bin/bash
# Phase 20 S1 — rounds=6, seed 2.
set -e
cd /raid/users/paul/workLLM
GPU=${1:-5}
echo "=== p20s1 r6 seed=2 gpu=$GPU ==="
CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
  scripts/phase15_s1/self_improve.py \
  --seed 2 --rounds 6 --samples 6 --train-steps 200 --max-new-tokens 200 \
  --out scripts/phase20_s1/run_r6_seed2.json 2>&1 | tail -16
echo "=== p20s1 r6 seed=2 done ==="
