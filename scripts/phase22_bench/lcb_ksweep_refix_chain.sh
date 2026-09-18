#!/bin/bash
# Phase 22 K-sweep re-measurement, after the F32-rotary fix.
#
# Question: the rotary bug degraded BF16 TRAINING past token 256, and the
# K-sweep arms trained at prompt ~150 + completion up to 192. The LCB
# evaluation itself ran F32 and was clean, and the degradation was common to
# every arm, so the slope should survive — but "should" is not measured. This
# re-runs the two arms that bracket the most exposed region (K=2 and K=8) with
# correct code and asks whether the published gap reproduces:
#
#   published, 6 seeds, buggy training:  K=2 0.084 -> K=8 0.111  (+0.027)
#
# Fresh seeds (901-903), not the original six: reusing seeds whose values are
# already known turns a replication into a confirmation.
#
# Three seeds, not six. That detects the gap collapsing; it will not resolve a
# small shift. This is a check on an existing claim, not a new claim.
#
# Usage: lcb_ksweep_refix_chain.sh [gpu_list]
#   gpu_list: comma-separated, default "1,2,3,4" (GPU 0 is often held).
set +e
cd /raid/users/paul/workLLM
BIN=./target/release/examples/phase22_dump_completions
V=/raid/users/paul/workLLM/scratch-7b-sft/tools/lcb-venv/bin
CUT=2024-09-01
ARMS=(k2 k8)
SEEDS=(901 902 903)
OFFSETS=(640 670 700 730)          # 4 x 30 = 120 problems, the published slice
IFS=, read -r -a GPUS <<< "${1:-1,2,3,4}"

if [ "$(strings $BIN | grep -c cudarc)" -eq 0 ]; then
  echo "⚠ $BIN is a CPU build — rebuild with --features cuda."; exit 1
fi
if [ "${#GPUS[@]}" -lt "${#OFFSETS[@]}" ]; then
  echo "⚠ need ${#OFFSETS[@]} GPUs for the slice fan-out, got ${#GPUS[@]}."; exit 1
fi

# 1) generate, one checkpoint at a time, slices fanned across the GPUs.
for arm in "${ARMS[@]}"; do
  for s in "${SEEDS[@]}"; do
    CKPT="scratch-7b-sft/p22_ksweep_refix/mean_${arm}_seed${s}_final.safetensors"
    OUT="scratch-7b-sft/lcb_refix_${arm}_s${s}"; mkdir -p "$OUT"
    [ -f "$CKPT" ] || { echo "⚠ missing $CKPT — skipping"; continue; }
    if [ -f "$OUT/gens_all.json" ]; then echo "have $OUT/gens_all.json, skip gen"; continue; fi
    gi=0
    for off in "${OFFSETS[@]}"; do
      CUDA_VISIBLE_DEVICES=${GPUS[$gi]} $BIN --benchmark livecodebench \
        --model-id Qwen2.5-Coder-7B --dtype f32 --checkpoint "$CKPT" \
        --offset $off --n-problems 30 --passk 1 --max-new-tokens 1024 \
        --dump "$OUT/slice_$off.json" > "$OUT/gen_$off.log" 2>&1 &
      gi=$((gi + 1))
    done
    wait
    # Completeness gate. A slice that OOM'd leaves no file, and merging
    # silently produces 90 problems instead of 120 — which scores as a
    # plausible number nobody questions. This repo shipped that mistake
    # once; refuse to score instead.
    missing=0
    for off in "${OFFSETS[@]}"; do [ -f "$OUT/slice_$off.json" ] || missing=$((missing+1)); done
    if [ "$missing" -gt 0 ]; then
      echo "⚠ $OUT: $missing/${#OFFSETS[@]} slices missing — NOT scoring this seed"
      continue
    fi
    python3 - "$OUT" <<'PY'
import json, glob, sys
out = sys.argv[1]
allg = []
for f in sorted(glob.glob(f"{out}/slice_*.json")):
    allg += json.load(open(f))
if len(allg) != 120:
    print(f"⚠ {out}: merged {len(allg)} generations, expected 120 — NOT writing")
    sys.exit(1)
json.dump(allg, open(f"{out}/gens_all.json", "w"))
print(f"merged {len(allg)} -> {out}")
PY
  done
done

# 2) score sequentially — codegen_metrics forks heavily and contends.
echo
echo "########## K-sweep re-measurement, F32 rotary ##########"
for arm in "${ARMS[@]}"; do
  for s in "${SEEDS[@]}"; do
    OUT="scratch-7b-sft/lcb_refix_${arm}_s${s}"
    [ -f "$OUT/gens_all.json" ] || { echo "$arm seed $s: no gens_all.json"; continue; }
    echo "--- $arm seed $s ---"
    for w in "overall:" "post:--start-date $CUT"; do
      name=${w%%:*}; flag=${w#*:}
      printf "  %-5s " "$name"
      $V/python scripts/phase22_bench/lcb_score.py --gens "$OUT/gens_all.json" \
        --release release_v5 $flag 2>/dev/null \
        | grep -aE "matched problems|per-problem mean" | tr '\n' ' '
      echo
    done
  done
done
echo "=== LCB_KSWEEP_REFIX_COMPLETE ==="
