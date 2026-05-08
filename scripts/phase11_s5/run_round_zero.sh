#!/bin/bash
# Phase 11 S5: round-0-only DPO on GPU 1 (single run).
set -e
cd /raid/users/paul/workLLM

echo "=== K9 round-0-only DPO (β=0.1 r0, SFT r1+, 4 rounds) ==="
CUDA_VISIBLE_DEVICES=1 ./target/release/examples/self_improve_rust \
  --rounds 4 \
  --pretrain-steps 1500 \
  --gen-n 24 --eval-n 24 \
  --round-train-steps 400 \
  --round-ckpt checkpoints/p11s5_r0_only \
  --seed-ckpt checkpoints/rust_seed.safetensors \
  --dpo-beta 0.1 \
  --dpo-reference-from checkpoints/rust_seed.safetensors \
  --dpo-round-zero-only \
  --dpo-max-pairs-per-prompt 4 2>&1 | tail -20
echo "=== p11s5_r0_only done ==="
