#!/bin/bash
# Phase 16 S4 — Hybrid OPD+SFT single-seed runner.
# Usage: bash run_hybrid_seed.sh <gpu> <seed> [alpha] [kl_direction]
set -e
cd /raid/users/paul/workLLM
GPU=$1
SEED=$2
ALPHA=${3:-0.3}
KL=${4:-reverse}
echo "=== p16s4 hybrid α=$ALPHA kl=$KL seed=$SEED gpu=$GPU ==="
CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
  scripts/phase16_s4/self_improve_hybrid_opd.py \
  --seed $SEED --sft-alpha $ALPHA --kl-direction $KL \
  --rounds 1 --samples 3 --train-steps 200 --max-new-tokens 200 \
  --opd-temperature 2.0 \
  --specialists-dir checkpoints/phase15_s2 \
  2>&1 | tail -10
echo "=== p16s4 hybrid seed=$SEED done ==="
