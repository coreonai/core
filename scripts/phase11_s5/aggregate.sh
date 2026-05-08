#!/bin/bash
# Phase 11 S5: aggregate all DPO variants (S3 + S4 + S5) into one TSV.
set -e
cd /raid/users/paul/workLLM
OUT=scripts/phase11_s5/results.tsv
echo -e "variant\tround\tgen_correct\tgen_total\tgen_pct\teval_before\teval_after\tdelta" > "$OUT"

extract_block() {
  # For multi-block logs (β sweep, hybrid sweep), tag matches a header
  # like "=== K9 ... β=$tag" or "=== K9 ... α=$tag". Matches the first
  # occurrence.
  local logfile="$1"
  local tag="$2"
  awk -v t="$tag" '
    BEGIN { keep = 0; in_hist = 0 }
    $0 ~ ("[βα]=" t " ") { keep = 1 }
    keep && /=== history ===/ { in_hist = 1; next }
    keep && in_hist && /^=== / { in_hist = 0; keep = 0 }
    keep && in_hist && /^round /
  ' "$logfile"
}

extract_single() {
  awk '
    BEGIN { in_hist = 0 }
    /=== history ===/ { in_hist = 1; next }
    in_hist && /^round /
  ' "$1"
}

parse_history() {
  local variant="$1"
  local block="$2"
  echo "$block" | while read -r line; do
    [ -z "$line" ] && continue
    rd=$(echo "$line" | grep -oE 'round [0-9]+' | awk '{print $2}')
    gen_n=$(echo "$line" | grep -oE 'gen=[0-9]+/[0-9]+' | head -1 | sed 's/gen=//')
    gen_correct=$(echo "$gen_n" | cut -d/ -f1)
    gen_total=$(echo "$gen_n" | cut -d/ -f2)
    gen_pct=$(echo "$line" | grep -oE 'gen=[0-9]+/[0-9]+ \([0-9.]+%' | sed 's/.*(//' | sed 's/%//')
    eval_before=$(echo "$line" | grep -oE 'before=[0-9]+/[0-9]+' | sed 's/before=//' | cut -d/ -f1)
    eval_after=$(echo "$line" | grep -oE 'after=[0-9]+/[0-9]+' | sed 's/after=//' | cut -d/ -f1)
    delta=$(echo "$line" | grep -oE 'Δ=[+-]?[0-9]+' | sed 's/Δ=//')
    echo -e "$variant\t$rd\t$gen_correct\t$gen_total\t$gen_pct\t$eval_before\t$eval_after\t$delta" >> "$OUT"
  done
}

# Inherit S3/S4 baselines + variants
parse_history "sft"             "$(extract_single /tmp/p11s3_sft.log)"
parse_history "dpo_b01_frozen"  "$(extract_single /tmp/p11s3_dpo.log)"
parse_history "dpo_b01_rolling" "$(extract_single /tmp/p11s4_rolling.log)"
parse_history "dpo_b001_frozen" "$(extract_block /tmp/p11s4_beta_sweep.log 0.01)"
parse_history "dpo_b003_frozen" "$(extract_block /tmp/p11s4_beta_sweep.log 0.03)"
parse_history "dpo_b005_frozen" "$(extract_block /tmp/p11s4_beta_sweep.log 0.05)"

# S5 variants
parse_history "hybrid_a03"      "$(extract_block /tmp/p11s5_hybrid.log 0.3)"
parse_history "hybrid_a05"      "$(extract_block /tmp/p11s5_hybrid.log 0.5)"
parse_history "hybrid_a07"      "$(extract_block /tmp/p11s5_hybrid.log 0.7)"
parse_history "round_zero_only" "$(extract_single /tmp/p11s5_round_zero.log)"
parse_history "combined_b005_a05" "$(extract_single /tmp/p11s5_combined.log)"

echo
echo "=== summary ==="
column -t -s $'\t' "$OUT"
