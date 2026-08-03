#!/bin/bash
# Phase 22 §6.5 — LiveCodeBench transfer test at the CORRECT metric.
#
# The self-improve gain lives at AGGREGATE pass@1 (temp 0.8 sampling), not
# greedy: HumanEval-64 base 0.656 -> full-set SFT 0.756 (+0.10), while greedy
# 0.488 -> 0.439 (SFT sharpens the distribution). All earlier LCB runs were
# greedy (passk 1) and cannot detect the recipe's transfer. This re-runs base +
# full-set SFT on the LCB slice at passk 5 (temp 0.8) and reports the codegen
# aggregate pass@1 (= total_passes/total_attempts), split pre/post cutoff.
# F32 (long prompts). ~1.5 h on 8 GPUs (base 0-3, hep 4-7).
set +e
cd /raid/users/paul/workLLM
BIN=./target/release/examples/phase22_dump_completions
V=/raid/users/paul/workLLM/scratch-7b-sft/tools/lcb-venv/bin
CUT=2024-09-01

gen() { # $1=out $2=gpubase $3=extra(checkpoint flag or empty)
  local OUT=$1 g=$2 extra=$3
  mkdir -p "$OUT"
  for off in 640 670 700 730; do
    CUDA_VISIBLE_DEVICES=$g $BIN --benchmark livecodebench --model-id Qwen2.5-Coder-7B \
      --dtype f32 $extra --offset $off --n-problems 30 --passk 5 --max-new-tokens 768 \
      --dump "$OUT/slice_$off.json" > "$OUT/gen_$off.log" 2>&1 &
    g=$((g + 1))
  done
}
echo "=== generating base (GPU0-3) + full-set SFT (GPU4-7) at passk 5 ==="
gen scratch-7b-sft/lcb_agg_base 0 ""
gen scratch-7b-sft/lcb_agg_hep  4 "--checkpoint scratch-7b-sft/hep_out_s42/r0_merged.r2.safetensors"
wait
echo "=== generation done; scoring (aggregate pass@1) ==="
for name in base hep; do
  OUT=scratch-7b-sft/lcb_agg_$name
  python3 -c "
import json, glob
allg=[]
for f in sorted(glob.glob('$OUT/slice_*.json')): allg+=json.load(open(f))
json.dump(allg, open('$OUT/gens_all.json','w')); print('$name merged',len(allg))
"
  echo "############ $name (aggregate pass@1) ############"
  for w in "overall:" "pre:--end-date $CUT" "post:--start-date $CUT"; do
    n=${w%%:*}; flag=${w#*:}
    echo "--- $n ---"
    $V/python scripts/phase22_bench/lcb_score.py --gens "$OUT/gens_all.json" --release release_v5 $flag 2>/dev/null | grep -aE "matched problems|codegen_metrics"
  done
done
echo "=== LCB_AGG_COMPLETE ==="
