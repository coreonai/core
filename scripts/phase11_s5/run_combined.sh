#!/bin/bash
# Phase 11 S5: combined fix — smaller β + SFT anchor.
# Tests whether stacking the two most promising S4/S5 fixes
# compounds (S4 β=0.05 r1 had 33% gen spike but eval=0;
# adding hybrid SFT might translate the gen spike to eval).
set -e
cd /raid/users/paul/workLLM
echo "=== K9 combined fix β=0.05 α=0.5 (4 rounds) ==="
CUDA_VISIBLE_DEVICES=1 ./target/release/examples/self_improve_rust \
  --rounds 4 \
  --pretrain-steps 1500 \
  --gen-n 24 --eval-n 24 \
  --round-train-steps 400 \
  --round-ckpt checkpoints/p11s5_combined \
  --seed-ckpt checkpoints/rust_seed.safetensors \
  --dpo-beta 0.05 \
  --dpo-reference-from checkpoints/rust_seed.safetensors \
  --dpo-sft-anchor-weight 0.5 \
  --dpo-max-pairs-per-prompt 4 2>&1 | tail -20
echo "=== p11s5_combined done ==="
