#!/bin/bash
# Phase 13 S3 (a): S3-isolate tiny — 3-challenge fixed via --challenge-mask 0,1,2
# Defaults n_layer=4 n_embd=128 n_head=4 (~1M)
# 5 seeds on GPU 0
set -e
cd /raid/users/paul/workLLM
for s in 0 1 2 3 4; do
  seed_ckpt="checkpoints/p13s3a_tiny_seed${s}.safetensors"
  round_ckpt="checkpoints/p13s3a_tiny_s${s}"
  echo "=== isolate_tiny seed=$s ==="
  CUDA_VISIBLE_DEVICES=0 ./target/release/examples/self_improve_rust \
    --rounds 4 \
    --pretrain-steps 1500 \
    --gen-n 24 --eval-n 24 \
    --round-train-steps 400 \
    --round-ckpt "$round_ckpt" \
    --seed-ckpt "$seed_ckpt" \
    --challenge-mask 0,1,2 \
    --optimizer adam 2>&1 | tail -8
  echo "=== isolate_tiny seed=$s done ==="
done
echo "=== isolate_tiny done ==="
