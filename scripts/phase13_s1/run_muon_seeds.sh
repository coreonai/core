#!/bin/bash
# Phase 13 S1 A2 — Muon × 5 seeds on GPU 1, sequential.
# Uses the SAME seed checkpoints as the AdamW run on GPU 0
# (each is independently re-pretrained with its --optimizer flag,
# so seed_ckpt collisions are avoided by using distinct paths
# below).
set -e
cd /raid/users/paul/workLLM
for s in 0 1 2 3 4; do
  seed_ckpt="checkpoints/p13s1_muon_seed${s}.safetensors"
  round_ckpt="checkpoints/p13s1_muon_s${s}"
  echo "=== Muon seed=$s ==="
  CUDA_VISIBLE_DEVICES=1 ./target/release/examples/self_improve_rust \
    --rounds 4 \
    --pretrain-steps 1500 \
    --gen-n 24 --eval-n 24 \
    --round-train-steps 400 \
    --round-ckpt "$round_ckpt" \
    --seed-ckpt "$seed_ckpt" \
    --optimizer muon 2>&1 | tail -8
  echo "=== muon seed=$s done ==="
  echo
done
echo "=== muon_seeds done ==="
