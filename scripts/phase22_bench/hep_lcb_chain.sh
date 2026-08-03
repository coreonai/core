#!/bin/bash
# Phase 22 §6.5 — chain: wait for full-set SFT training -> generate LCB (F32,
# parallel) -> score (sequential, to avoid codegen_metrics contention) ->
# pre/post-cutoff split. Compare to base post-cutoff 0.087 and K=8 RL 0.054.
set +e
cd /raid/users/paul/workLLM
BIN=./target/release/examples/phase22_dump_completions
V=/raid/users/paul/workLLM/scratch-7b-sft/tools/lcb-venv/bin
CUT=2024-09-01
SEEDS=(42 200)

# 1) wait for training (both r2 checkpoints + no trainer procs). Bound ~8h.
for i in $(seq 1 240); do
  have=0
  for s in "${SEEDS[@]}"; do [ -f "scratch-7b-sft/hep_out_s${s}/r0_merged.r2.safetensors" ] && have=$((have+1)); done
  alive=$(pgrep -af phase22_he_mr_sft | grep -v pgrep | wc -l)
  if [ "$have" -ge 2 ] && [ "$alive" -eq 0 ]; then echo "TRAIN_DONE have=$have"; break; fi
  if [ "$alive" -eq 0 ] && [ "$i" -gt 5 ]; then echo "PROCS_EXITED have=$have"; break; fi
  sleep 120
done

# 2) generate LCB for both seeds in parallel (8 GPUs, F32).
gpubase=0
for s in "${SEEDS[@]}"; do
  CKPT="scratch-7b-sft/hep_out_s${s}/r0_merged.r2.safetensors"
  OUT="scratch-7b-sft/lcb_hep_s${s}"; mkdir -p "$OUT"
  [ -f "$CKPT" ] || { echo "⚠ missing $CKPT"; continue; }
  g=$gpubase
  for off in 640 670 700 730; do
    CUDA_VISIBLE_DEVICES=$g $BIN --benchmark livecodebench --model-id Qwen2.5-Coder-7B \
      --dtype f32 --checkpoint "$CKPT" --offset $off --n-problems 30 --passk 1 \
      --max-new-tokens 1024 --dump "$OUT/slice_$off.json" > "$OUT/gen_$off.log" 2>&1 &
    g=$((g + 1))
  done
  gpubase=$((gpubase + 4))
done
wait
echo "=== generation done ==="

# 3) merge + score sequentially.
for s in "${SEEDS[@]}"; do
  OUT="scratch-7b-sft/lcb_hep_s${s}"
  python3 -c "
import json, glob
allg=[]
for f in sorted(glob.glob('$OUT/slice_*.json')): allg+=json.load(open(f))
json.dump(allg, open('$OUT/gens_all.json','w')); print('merged',len(allg),'-> $OUT')
"
  echo "############ full-set SFT seed $s ############"
  for w in "overall:" "pre:--end-date $CUT" "post:--start-date $CUT"; do
    name=${w%%:*}; flag=${w#*:}
    echo "--- $name ---"
    $V/python scripts/phase22_bench/lcb_score.py --gens "$OUT/gens_all.json" --release release_v5 $flag 2>/dev/null | grep -aE "matched problems|per-problem mean"
  done
done
echo "=== HEP_LCB_CHAIN_COMPLETE ==="
