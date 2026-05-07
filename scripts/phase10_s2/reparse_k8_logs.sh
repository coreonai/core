#!/bin/bash
# Phase 10 S2: rebuild results.tsv from the saved per-checkpoint
# critic_baseline_korean logs (`scripts/phase10_s2/log_<name>.txt`),
# applying the corrected selection-sweep parser.

set -e
cd /raid/users/paul/workLLM
OUT_TSV=scripts/phase10_s2/results.tsv

declare -a NAMES=(baseline_lam0 lam001 lam003 lam01_k8 lam03 lam01_k2 lam01_k4 ema099)

echo -e "name\tpass_rate\tmean_auc\tsum_auc\tF2_lift\tF4_lift\tF8_lift\tF16_lift" > "$OUT_TSV"

for name in "${NAMES[@]}"; do
  log="scripts/phase10_s2/log_${name}.txt"
  if [ ! -f "$log" ]; then
    echo "[skip] $name — log missing: $log" >&2
    continue
  fi
  # pass_rate from the tracing line — strip ANSI escape sequences first
  # (tracing-subscriber emits "[3mkey[0m[2m=[0mvalue" which breaks naive
  # regex). The sed expression matches ESC + [ + parameters + final letter.
  pass_rate=$(sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' "$log" \
              | grep -oE 'pass_rate=[0-9.]+' | tail -1 | sed 's/.*=//')
  mean_auc=$(grep -E 'LogitCritic mean:' "$log" | tail -1 | awk '{print $3}')
  sum_auc=$(grep -E 'LogitCritic sum:'   "$log" | tail -1 | awk '{print $3}')
  # Selection sweep: leading whitespace stripped by awk → $1=F, $2=random, $3=critic
  f2=$(awk  '/^  2 [[:space:]]+[0-9]/  { if ($2+0 > 0) print $3/$2 }' "$log" | head -1)
  f4=$(awk  '/^  4 [[:space:]]+[0-9]/  { if ($2+0 > 0) print $3/$2 }' "$log" | head -1)
  f8=$(awk  '/^  8 [[:space:]]+[0-9]/  { if ($2+0 > 0) print $3/$2 }' "$log" | head -1)
  f16=$(awk '/^  16 [[:space:]]+[0-9]/ { if ($2+0 > 0) print $3/$2 }' "$log" | head -1)
  echo -e "$name\t$pass_rate\t$mean_auc\t$sum_auc\t$f2\t$f4\t$f8\t$f16" >> "$OUT_TSV"
done

echo
echo "=== summary ==="
column -t -s $'\t' "$OUT_TSV"
