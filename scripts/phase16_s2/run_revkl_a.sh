#!/bin/bash
# Phase 16 S2 — Reverse-KL OPD, seeds 0/1/2 on GPU 2.
# Reuses Phase 15 S2 self_improve_opd.py + checkpoints/phase15_s2/ specialists.
set -e
cd /raid/users/paul/workLLM
GPU=${1:-2}
for s in 0 1 2; do
  echo "=== p16s2 revkl seed=$s gpu=$GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
    scripts/phase15_s2/self_improve_opd.py \
    --seed $s --rounds 1 --samples 3 --train-steps 200 --max-new-tokens 200 \
    --opd-temperature 2.0 --kl-direction reverse \
    --specialists-dir checkpoints/phase15_s2 \
    --out scripts/phase16_s2/run_revkl_seed${s}.json \
    2>&1 | tail -8
  echo "=== p16s2 revkl seed=$s done ==="
done
echo "=== revkl_a done ==="
