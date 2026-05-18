#!/bin/bash
# Phase 22 Stage D — 5-seed aggregate eval summary.
#
# Reads /tmp/phase22d_<BATCH>_seed{0..4}/aggregate/r2_eval.log and
# produces a mean ± σ table for `aggregate pass@1 (raw)` and
# `per-prompt pass@10`. Use after running
# `./scripts/phase22_aggregate_eval.sh <BATCH>` to completion.
#
# Usage: ./scripts/phase22_aggregate_summary.sh G4
set -e
BATCH=${1:?usage: $0 <batch_tag>}

echo "=== $BATCH aggregate eval summary ==="
declare -a AGG_PASS1
declare -a PP_PASS10
for i in 0 1 2 3 4; do
  LOG="/tmp/phase22d_${BATCH}_seed${i}/aggregate/r2_eval.log"
  if [ ! -f "$LOG" ]; then
    echo "  seed=$((i*100+100)): no log file"
    continue
  fi
  a=$(grep -oP 'aggregate pass@1 \(raw, all samples\) = \K[0-9.]+' "$LOG" | tail -1)
  p=$(grep -oP 'per-prompt pass@10 = \K[0-9.]+' "$LOG" | tail -1)
  if [ -z "$a" ]; then
    echo "  seed=$((i*100+100)): no aggregate result (run incomplete or failed)"
    continue
  fi
  AGG_PASS1+=("$a")
  PP_PASS10+=("$p")
  printf "  seed=%-4d  agg pass@1 = %s   per-prompt pass@10 = %s\n" $((i*100+100)) "$a" "$p"
done

n=${#AGG_PASS1[@]}
if [ "$n" -eq 0 ]; then
  echo "no results — nothing to summarize"
  exit 1
fi

# Join arrays with commas for Python.
AGG_CSV=$(IFS=,; echo "${AGG_PASS1[*]}")
PP_CSV=$(IFS=,; echo "${PP_PASS10[*]}")
echo ""
echo "  N = $n seeds"
python3 - <<PY
import math
xs = [$AGG_CSV]
mean = sum(xs)/len(xs)
var = sum((x - mean)**2 for x in xs) / (len(xs) - 1) if len(xs) > 1 else 0.0
sd = math.sqrt(var)
print(f"  aggregate pass@1:")
print(f"    mean = {mean:.4f}")
print(f"    σ    = {sd:.4f}")
print(f"    min  = {min(xs):.4f}")
print(f"    max  = {max(xs):.4f}")

ys = [$PP_CSV]
ymean = sum(ys)/len(ys)
yvar = sum((y - ymean)**2 for y in ys) / (len(ys) - 1) if len(ys) > 1 else 0.0
ysd = math.sqrt(yvar)
print(f"  per-prompt pass@10:")
print(f"    mean = {ymean:.4f}")
print(f"    σ    = {ysd:.4f}")

print()
print(f"  references:")
print(f"    base Qwen aggregate pass@1   = 0.222 (Phase 22 Stage B, n=32×k=10)")
print(f"    Phase 17 S1 r=2 SFT mean     = 0.404 ± 0.013 (5 seeds, gen-n=164×k=10)")
print(f"    Phase 17 S6 base pass@10     = 0.524 (full 164)")
PY
