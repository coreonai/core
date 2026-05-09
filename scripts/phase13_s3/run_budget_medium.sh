#!/bin/bash
# Phase 13 S3 (b): S3-budget medium — 10-challenge + 5K pretrain steps
# Test if 10M model can learn 10-challenge surface with 3.3× pretrain budget
# 5 seeds across GPUs 2 (seeds 0,1,2) + 3 (seeds 3,4) for parallelism
set -e
cd /raid/users/paul/workLLM
GPU=${1:-2}
SEEDS=${2:-"0 1 2 3 4"}
for s in $SEEDS; do
  seed_ckpt="checkpoints/p13s3b_medium_seed${s}.safetensors"
  round_ckpt="checkpoints/p13s3b_medium_s${s}"
  echo "=== budget_medium seed=$s gpu=$GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU ./target/release/examples/self_improve_rust \
    --rounds 4 \
    --pretrain-steps 5000 \
    --pretrain-examples 1500 \
    --gen-n 24 --eval-n 24 \
    --round-train-steps 400 \
    --round-ckpt "$round_ckpt" \
    --seed-ckpt "$seed_ckpt" \
    --n-layer 8 --n-head 8 --n-embd 256 --n-kv-head 4 \
    --optimizer adam 2>&1 | tail -8
  echo "=== budget_medium seed=$s done ==="
done
echo "=== budget_medium gpu=$GPU done ==="
