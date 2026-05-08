#!/bin/bash
set -e
cd /raid/users/paul/workLLM
echo "=== Phase 12 S1 — K9 Muon (4 rounds) ==="
CUDA_VISIBLE_DEVICES=1 ./target/release/examples/self_improve_rust \
  --rounds 4 \
  --pretrain-steps 1500 \
  --gen-n 24 --eval-n 24 \
  --round-train-steps 400 \
  --round-ckpt checkpoints/p12s1_muon \
  --seed-ckpt checkpoints/p12s1_muon_seed.safetensors \
  --optimizer muon 2>&1 | tail -25
echo "=== p12s1_muon done ==="
