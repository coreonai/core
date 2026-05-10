#!/bin/bash
# Phase 16 S3 — Muon at LoRA r=64 α=128, seeds 0/1/2 on GPU 5.
# Reuses Phase 15 S4 run_muon.py with --lora-r 64 --lora-alpha 128.
set -e
cd /raid/users/paul/workLLM
GPU=${1:-5}
for s in 0 1 2; do
  echo "=== p16s3 muon r64 seed=$s gpu=$GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
    scripts/phase15_s4/run_muon.py \
    --seed $s --optimizer muon --rounds 1 --samples 3 \
    --train-steps 200 --max-new-tokens 200 \
    --lora-r 64 --lora-alpha 128 \
    --out scripts/phase16_s3/run_muon_r64_seed${s}.json \
    2>&1 | tail -8
  echo "=== p16s3 muon r64 seed=$s done ==="
done
echo "=== muon_r64_a done ==="
