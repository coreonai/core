#!/bin/bash
# Phase 13 S3 (a): S3-isolate medium — 3-challenge fixed + 10M scale
# 5 seeds on GPU 1
set -e
cd /raid/users/paul/workLLM
for s in 0 1 2 3 4; do
  seed_ckpt="checkpoints/p13s3a_medium_seed${s}.safetensors"
  round_ckpt="checkpoints/p13s3a_medium_s${s}"
  echo "=== isolate_medium seed=$s ==="
  CUDA_VISIBLE_DEVICES=1 ./target/release/examples/self_improve_rust \
    --rounds 4 \
    --pretrain-steps 1500 \
    --gen-n 24 --eval-n 24 \
    --round-train-steps 400 \
    --round-ckpt "$round_ckpt" \
    --seed-ckpt "$seed_ckpt" \
    --challenge-mask 0,1,2 \
    --n-layer 8 --n-head 8 --n-embd 256 --n-kv-head 4 \
    --optimizer adam 2>&1 | tail -8
  echo "=== isolate_medium seed=$s done ==="
done
echo "=== isolate_medium done ==="
