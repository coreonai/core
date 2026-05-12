#!/bin/bash
# Phase 18 S4 — Multi-round Reverse-KL OPD at HumanEval, seeds 0/1/2.
# Tests Risk #20 for OPD: did rounds=1 hide OPD's positive value?
# Phase 15 S2 / Phase 16 S2 / Phase 16 S4 all retracted OPD variants at rounds=1.
# Multi-round AdamW SFT lifted to 0.404 (Phase 17 S1). Does multi-round
# OPD now match or exceed AdamW MR?
set -e
cd /raid/users/paul/workLLM
GPU=${1:-5}
for s in 0 1 2; do
  echo "=== p18s4 mr-revkl seed=$s gpu=$GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
    scripts/phase15_s2/self_improve_opd.py \
    --seed $s --rounds 2 --samples 6 --train-steps 200 --max-new-tokens 200 \
    --opd-temperature 2.0 --kl-direction reverse \
    --specialists-dir checkpoints/phase15_s2 \
    --out scripts/phase18_s4/run_mr_revkl_seed${s}.json 2>&1 | tail -12
  echo "=== p18s4 mr-revkl seed=$s done ==="
done
echo "=== mr_revkl_a done ==="
