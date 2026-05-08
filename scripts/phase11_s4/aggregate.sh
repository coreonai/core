#!/bin/bash
# Phase 11 S4: extract round-by-round metrics from each variant's
# training log into a single TSV for the doc table.
set -e
cd /raid/users/paul/workLLM
OUT=scripts/phase11_s4/results.tsv
echo -e "variant\tround\tgen_correct\tgen_total\tgen_pct\teval_before\teval_after\tdelta" > "$OUT"

declare -a SOURCES=(
  "sft|/tmp/p11s3_sft.log"
  "dpo_b01_frozen|/tmp/p11s3_dpo.log"
  "dpo_b01_rolling|/tmp/p11s4_rolling.log"
  "dpo_b001_frozen|/tmp/p11s4_beta_sweep.log"
  "dpo_b003_frozen|/tmp/p11s4_beta_sweep.log"
  "dpo_b005_frozen|/tmp/p11s4_beta_sweep.log"
)

# The β-sweep log holds 3 separate "=== history ===" blocks. Pull
# them by tag.
extract_block() {
  local logfile="$1"
  local tag="$2"
  awk -v t="$tag" '
    BEGIN { keep = 0 }
    $0 ~ ("=== K9 DPO β=" t " ") { keep = 1 }
    keep && /=== history ===/ { in_hist = 1; next }
    keep && in_hist && /^=== / { in_hist = 0; keep = 0 }
    keep && in_hist && /^round /
  ' "$logfile"
}

extract_single() {
  local logfile="$1"
  awk '
    BEGIN { in_hist = 0 }
    /=== history ===/ { in_hist = 1; next }
    in_hist && /^round /
  ' "$logfile"
}

parse_history() {
  local variant="$1"
  local block="$2"
  echo "$block" | while read -r line; do
    [ -z "$line" ] && continue
    # round 0: gen=0/24 (0.0%)  eval before=0/24 after=6/24  Δ=+6
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

# SFT baseline
parse_history "sft" "$(extract_single /tmp/p11s3_sft.log)"
# Phase 11 S3 DPO frozen β=0.1
parse_history "dpo_b01_frozen" "$(extract_single /tmp/p11s3_dpo.log)"
# Phase 11 S4 rolling β=0.1
parse_history "dpo_b01_rolling" "$(extract_single /tmp/p11s4_rolling.log)"
# β sweep
parse_history "dpo_b001_frozen" "$(extract_block /tmp/p11s4_beta_sweep.log 0.01)"
parse_history "dpo_b003_frozen" "$(extract_block /tmp/p11s4_beta_sweep.log 0.03)"
parse_history "dpo_b005_frozen" "$(extract_block /tmp/p11s4_beta_sweep.log 0.05)"

echo
echo "=== summary ==="
column -t -s $'\t' "$OUT"
