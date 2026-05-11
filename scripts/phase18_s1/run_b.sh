#!/bin/bash
# Phase 18 S1 — Multi-round Muon at HumanEval, seeds 3/4.
set -e
cd /raid/users/paul/workLLM
GPU=${1:-1}
for s in 3 4; do
  echo "=== p18s1 mr-muon seed=$s gpu=$GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU /tmp/p14_env/bin/python \
    scripts/phase15_s4/run_muon.py \
    --seed $s --optimizer muon --rounds 2 --samples 6 \
    --train-steps 200 --max-new-tokens 200 \
    --out scripts/phase18_s1/run_mr_muon_seed${s}.json 2>&1 | tail -10
  echo "=== p18s1 mr-muon seed=$s done ==="
done
echo "=== mr_muon_b done ==="
