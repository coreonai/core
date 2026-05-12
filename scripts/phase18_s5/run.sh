#!/bin/bash
# Phase 18 S5 — MR-SFT + pass@k eval at MBPP (single seed PoC).
set -e
cd /raid/users/paul/workLLM
GPU=${1:-7}
echo "=== p18s5 mr-passk-mbpp gpu=$GPU ==="
CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
  scripts/phase18_s5/run_mr_passk_mbpp.py \
  --seed 0 --rounds 2 --samples 6 --passk-k 10 --train-steps 200 --max-new-tokens 200 \
  2>&1 | tail -25
echo "=== p18s5 done ==="
