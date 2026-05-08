#!/bin/bash
# Phase 11 S4 — rolling reference DPO (β=0.1)
set -e
cd /raid/users/paul/workLLM
echo "=== K9 DPO β=0.1 rolling reference (4 rounds) ==="
CUDA_VISIBLE_DEVICES=0 ./target/release/examples/self_improve_rust \
  --rounds 4 \
  --pretrain-steps 1500 \
  --gen-n 24 --eval-n 24 \
  --round-train-steps 400 \
  --round-ckpt checkpoints/p11s4_rolling \
  --seed-ckpt checkpoints/rust_seed.safetensors \
  --dpo-beta 0.1 \
  --dpo-reference-from checkpoints/rust_seed.safetensors \
  --dpo-rolling-reference \
  --dpo-max-pairs-per-prompt 4 2>&1 | tail -60
echo "=== p11s4_rolling done ==="
