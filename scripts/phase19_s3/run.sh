#!/bin/bash
# Phase 19 S3 — rounds=3 + pass@k eval at HumanEval (single seed PoC).
# Sa rounds=2 + passk: pass@1=0.404, pass@10=0.604 (Phase 17).
# Question: does rounds=3 maintain pass@10 or start collapse?
set -e
mkdir -p scripts/phase19_s3
cd /raid/users/paul/workLLM
GPU=${1:-7}
echo "=== p19s3 r3+passk seed=0 gpu=$GPU ==="
CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
  scripts/phase17_sa/run_mr_passk.py \
  --seed 0 --rounds 3 --samples 6 --passk-k 10 --train-steps 200 --max-new-tokens 200 \
  --out scripts/phase19_s3/run_r3_passk_seed0.json 2>&1 | tail -25
echo "=== p19s3 done ==="
