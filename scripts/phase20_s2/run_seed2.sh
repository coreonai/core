#!/bin/bash
# Phase 20 S2 — rounds=5 MBPP, seed 2.
set -e
cd /raid/users/paul/workLLM
GPU=${1:-5}
echo "=== p20s2 mbpp_r5 seed=2 gpu=$GPU ==="
CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
  scripts/phase17_sb/run_mr_mbpp.py \
  --seed 2 --rounds 5 --samples 6 --train-steps 200 --max-new-tokens 200 \
  --out scripts/phase20_s2/run_mbpp_r5_seed2.json 2>&1 | tail -14
echo "=== p20s2 mbpp_r5 seed=2 done ==="
