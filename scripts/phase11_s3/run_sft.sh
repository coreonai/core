#!/bin/bash
set -e
cd /raid/users/paul/workLLM
echo "=== Phase 11 S3 — K9 SFT baseline (4 rounds) ==="
CUDA_VISIBLE_DEVICES=0 ./target/release/examples/self_improve_rust \
  --rounds 4 \
  --pretrain-steps 1500 \
  --gen-n 24 --eval-n 24 \
  --round-train-steps 400 \
  --round-ckpt checkpoints/p11s3_sft \
  --seed-ckpt checkpoints/rust_seed.safetensors 2>&1 | tail -60
echo "=== p11s3_sft done ==="
