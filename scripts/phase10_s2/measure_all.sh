#!/bin/bash
# Phase 10 S2: run critic_baseline_korean against every checkpoint and
# extract the headline metrics into a single TSV summary.
#
# Each checkpoint is measured with --n-prompts 30 --samples-per-prompt 20
# (matches Phase 10 S1 setup, ~3 min per run on A100).

set -e
cd /raid/users/paul/workLLM
EXAMPLE=./target/release/examples/critic_baseline_korean
OUT_TSV=scripts/phase10_s2/results.tsv

declare -a NAMES=(
  "baseline_lam0"
  "lam001"
  "lam003"
  "lam01_k8"
  "lam03"
  "lam01_k2"
  "lam01_k4"
  "ema099"
)
declare -a CKPTS=(
  "checkpoints/p10s1_baseline.safetensors"
  "checkpoints/p10s2_lam001.safetensors"
  "checkpoints/p10s2_lam003.safetensors"
  "checkpoints/p10s1_jepa01.safetensors"
  "checkpoints/p10s2_lam03.safetensors"
  "checkpoints/p10s2_k2.safetensors"
  "checkpoints/p10s2_k4.safetensors"
  "checkpoints/p10s2_ema099.safetensors"
)

echo -e "name\tpass_rate\tmean_auc\tsum_auc\tF2_lift\tF4_lift\tF8_lift\tF16_lift" > "$OUT_TSV"

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  ckpt="${CKPTS[$i]}"
  if [ ! -f "$ckpt" ]; then
    echo "[skip] $name — checkpoint missing: $ckpt" >&2
    continue
  fi
  echo "[run]  $name ($ckpt)"
  log="scripts/phase10_s2/log_${name}.txt"
  CUDA_VISIBLE_DEVICES=0 $EXAMPLE --init "$ckpt" --tokenizer data/kowiki/kowiki_bpe.json \
    --n-prompts 30 --samples-per-prompt 20 > "$log" 2>&1
  pass_rate=$(grep -oE 'pass_rate=[0-9.]+' "$log" | tail -1 | sed 's/.*=//')
  mean_auc=$(grep -E 'LogitCritic mean:' "$log" | tail -1 | awk '{print $3}')
  sum_auc=$(grep -E 'LogitCritic sum:'  "$log" | tail -1 | awk '{print $3}')
  # Selection sweep parsing — F lines look like "  4    0.020   0.011   -0.009   0.55×"
  # awk strips leading whitespace, so $1=F, $2=random, $3=critic, $4=Δ
  f2=$(awk  '/^  2 [[:space:]]+[0-9]/  { if ($2+0 > 0) print $3/$2 }' "$log" | head -1)
  f4=$(awk  '/^  4 [[:space:]]+[0-9]/  { if ($2+0 > 0) print $3/$2 }' "$log" | head -1)
  f8=$(awk  '/^  8 [[:space:]]+[0-9]/  { if ($2+0 > 0) print $3/$2 }' "$log" | head -1)
  f16=$(awk '/^  16 [[:space:]]+[0-9]/ { if ($2+0 > 0) print $3/$2 }' "$log" | head -1)
  echo -e "$name\t$pass_rate\t$mean_auc\t$sum_auc\t$f2\t$f4\t$f8\t$f16" >> "$OUT_TSV"
done

echo
echo "=== summary ==="
column -t -s $'\t' "$OUT_TSV"
