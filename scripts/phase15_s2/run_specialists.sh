#!/bin/bash
# Phase 15 S2 — train k=3 SFT specialists, one per routing subset.
# Single-seed (99) for now — we vary student seeds, not specialist seeds.
# All 3 sequential on a single GPU (passable as $1, default 2).
set -e
cd /raid/users/paul/workLLM
GPU=${1:-2}
for subset in strings numbers collections; do
  echo "=== p15s2 specialist subset=$subset gpu=$GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
    scripts/phase15_s2/train_specialist.py \
    --subset $subset --seed 99 \
    --rounds 2 --samples 4 --train-steps 200 --max-new-tokens 200 \
    --out-adapter checkpoints/phase15_s2/specialist_$subset \
    2>&1 | tail -15
  echo "=== p15s2 specialist subset=$subset done ==="
done
echo "=== specialists done ==="
