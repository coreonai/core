#!/bin/bash
# Phase 20 S2 — rounds=5 SFT at MBPP, seed 0.
# Phase 18 S3 MBPP r=3 mean 0.457; HumanEval r=5 mean 0.556.
# Question: does MBPP saturation curve parallel HumanEval?
set -e
cd /raid/users/paul/workLLM
GPU=${1:-0}
echo "=== p20s2 mbpp_r5 seed=0 gpu=$GPU ==="
CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
  scripts/phase17_sb/run_mr_mbpp.py \
  --seed 0 --rounds 5 --samples 6 --train-steps 200 --max-new-tokens 200 \
  --out scripts/phase20_s2/run_mbpp_r5_seed0.json 2>&1 | tail -14
echo "=== p20s2 mbpp_r5 seed=0 done ==="
