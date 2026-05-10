#!/bin/bash
# Phase 17 S9 — pass@k at base Qwen on MBPP-100 (cross-substrate validation of S6).
set -e
cd /raid/users/paul/workLLM
GPU=${1:-3}
echo "=== p17s9 passk-mbpp gpu=$GPU ==="
CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
  scripts/phase17_s9/run_passk_mbpp.py \
  --seed 0 --k 10 --max-new-tokens 200 \
  2>&1 | tail -25
echo "=== passk_mbpp done ==="
