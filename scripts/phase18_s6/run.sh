#!/bin/bash
# Phase 18 S6 — rounds=4 single seed (compounding saturation test).
# P18 S2 rounds=3 added +0.05~+0.06 over rounds=2. Does rounds=4 plateau,
# continue, or collapse?
set -e
cd /raid/users/paul/workLLM
GPU=${1:-7}
echo "=== p18s6 r4 seed=0 gpu=$GPU ==="
CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
  scripts/phase15_s1/self_improve.py \
  --seed 0 --rounds 4 --samples 6 --train-steps 200 --max-new-tokens 200 \
  --out scripts/phase18_s6/run_r4_seed0.json 2>&1 | tail -15
echo "=== p18s6 done ==="
