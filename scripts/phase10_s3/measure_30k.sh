#!/bin/bash
# Phase 10 S3: measure 3 K8 30K-step checkpoints —
# baseline (no JEPA), λ=0.3 k=8 (S2 winner A), λ=0.1 k=2 (S2 winner B).
set -e
cd /raid/users/paul/workLLM
EXAMPLE=./target/release/examples/critic_baseline_korean

declare -a NAMES=(baseline_30k lam03_30k k2_30k)
declare -a CKPTS=(
  "checkpoints/kowiki_50m_30k.safetensors"
  "checkpoints/p10s3_30k_lam03.safetensors"
  "checkpoints/p10s3_30k_k2.safetensors"
)

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  ckpt="${CKPTS[$i]}"
  if [ ! -f "$ckpt" ]; then
    echo "[skip] $name — checkpoint missing: $ckpt" >&2
    continue
  fi
  echo "[run]  $name ($ckpt)"
  log="scripts/phase10_s3/log_${name}.txt"
  CUDA_VISIBLE_DEVICES=0 $EXAMPLE --init "$ckpt" --tokenizer data/kowiki/kowiki_bpe.json \
    --n-prompts 30 --samples-per-prompt 20 > "$log" 2>&1
done
echo "S3_MEASURE_DONE"

# Reuse the S2 reparser
OUT_TSV=scripts/phase10_s3/results.tsv
echo -e "name\tpass_rate\tmean_auc\tsum_auc\tF2_lift\tF4_lift\tF8_lift\tF16_lift" > "$OUT_TSV"
for name in "${NAMES[@]}"; do
  log="scripts/phase10_s3/log_${name}.txt"
  [ -f "$log" ] || continue
  pass=$(sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' "$log" | grep -oE 'pass_rate=[0-9.]+' | tail -1 | sed 's/.*=//')
  mean=$(grep 'LogitCritic mean:' "$log" | tail -1 | awk '{print $3}')
  sum=$(grep 'LogitCritic sum:'  "$log" | tail -1 | awk '{print $3}')
  f2=$(awk  '/^  2 [[:space:]]+[0-9]/  { if ($2+0 > 0) print $3/$2 }' "$log" | head -1)
  f4=$(awk  '/^  4 [[:space:]]+[0-9]/  { if ($2+0 > 0) print $3/$2 }' "$log" | head -1)
  f8=$(awk  '/^  8 [[:space:]]+[0-9]/  { if ($2+0 > 0) print $3/$2 }' "$log" | head -1)
  f16=$(awk '/^  16 [[:space:]]+[0-9]/ { if ($2+0 > 0) print $3/$2 }' "$log" | head -1)
  echo -e "$name\t$pass\t$mean\t$sum\t$f2\t$f4\t$f8\t$f16" >> "$OUT_TSV"
done
echo
column -t -s $'\t' "$OUT_TSV"
