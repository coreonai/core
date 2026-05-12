#!/bin/bash
# Phase 18 S7 — Sa replication (HumanEval MR+pass@k seed 1).
# Sa was 1 seed (seed 0). Add seed 1 to strengthen "MR preserves pass@k" finding.
set -e
mkdir -p scripts/phase18_s7
cd /raid/users/paul/workLLM
GPU=${1:-7}
echo "=== p18s7 sa-replicate seed=1 gpu=$GPU ==="
CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
  scripts/phase17_sa/run_mr_passk.py \
  --seed 1 --rounds 2 --samples 6 --passk-k 10 --train-steps 200 --max-new-tokens 200 \
  --out scripts/phase18_s7/run_mr_passk_seed1.json 2>&1 | tail -25
echo "=== p18s7 done ==="
