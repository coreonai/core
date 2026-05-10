#!/bin/bash
# Phase 17 S7b — SFT train + pass@k eval, single seed.
set -e
cd /raid/users/paul/workLLM
GPU=${1:-7}
echo "=== p17s7b sft+passk gpu=$GPU ==="
CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
  scripts/phase17_s7/run_sft_then_passk.py \
  --seed 0 --samples 6 --passk-k 10 --train-steps 200 --max-new-tokens 200 \
  2>&1 | tail -25
echo "=== s7b done ==="
