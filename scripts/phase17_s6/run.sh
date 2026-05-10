#!/bin/bash
# Phase 17 S6 — pass@k inference baseline at HumanEval-164.
# Single-seed (no training, just inference).
set -e
cd /raid/users/paul/workLLM
GPU=${1:-7}
echo "=== p17s6 passk gpu=$GPU ==="
CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
  scripts/phase17_s6/run_passk.py \
  --seed 0 --k 10 --max-new-tokens 200 \
  2>&1 | tail -25
echo "=== passk done ==="
