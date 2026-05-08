#!/bin/bash
set -e
cd /raid/users/paul/workLLM
echo "=== Phase 12 S1 — K9 AdamW (control, 4 rounds) ==="
# Use a fresh seed so AdamW & Muon both pretrain — direct comparison
CUDA_VISIBLE_DEVICES=0 ./target/release/examples/self_improve_rust \
  --rounds 4 \
  --pretrain-steps 1500 \
  --gen-n 24 --eval-n 24 \
  --round-train-steps 400 \
  --round-ckpt checkpoints/p12s1_adam \
  --seed-ckpt checkpoints/p12s1_adam_seed.safetensors \
  --optimizer adam 2>&1 | tail -25
echo "=== p12s1_adam done ==="
