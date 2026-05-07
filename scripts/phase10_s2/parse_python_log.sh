#!/bin/bash
# Phase 10 S2: parse the Python-domain pretrain+harvest log
# (`/tmp/p10s2_ema_python.log`) and write a TSV of per-run metrics.
#
# Strategy: split into per-run blocks at "=== Python ..." headers, and
# within each block grep:
#   - `samples: N (M correct,` → pass = M / N
#   - `LogitCritic mean: X` / `LogitCritic sum:  X`
#   - selection-sweep rows (lines starting with "  F  random  critic ...")

set -e
LOG=${1:-/tmp/p10s2_ema_python.log}
OUT=${2:-scripts/phase10_s2/results_python.tsv}

[ -f "$LOG" ] || { echo "missing log: $LOG" >&2; exit 1; }

echo -e "name\tpass_rate\tmean_auc\tsum_auc\tF2_lift\tF4_lift\tF8_lift\tF16_lift" > "$OUT"

awk -v out="$OUT" '
function emit() {
  if (name != "") {
    pass = (n_total > 0) ? (n_correct + 0.0) / n_total : ""
    printf("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", name, pass, mean, sum, f2, f4, f8, f16) >> out
  }
}
function reset(label) {
  name = label
  n_correct = 0; n_total = 0
  mean = ""; sum = ""
  f2 = ""; f4 = ""; f8 = ""; f16 = ""
}
/^=== Python/ {
  emit()
  label = $0
  sub(/^=== Python /, "py_", label)
  sub(/ ===.*/, "", label)
  gsub(/[^a-zA-Z0-9._=]/, "_", label)
  reset(label)
  next
}
# "samples:           90 (32 correct, 58 incorrect)"
/^samples:[[:space:]]+[0-9]+[[:space:]]+\([0-9]+ correct/ {
  n_total = $2
  # The "(32" capture: $3 starts with "(" so strip the paren
  c = $3; gsub(/[()]/, "", c)
  n_correct = c + 0
}
/LogitCritic mean:/ { mean = $3 }
/LogitCritic sum:/  { sum  = $3 }
/^  2 [[:space:]]+[0-9]/ { if ($2+0 > 0) f2 = $3/$2 }
/^  4 [[:space:]]+[0-9]/ { if ($2+0 > 0) f4 = $3/$2 }
/^  8 [[:space:]]+[0-9]/ { if ($2+0 > 0) f8 = $3/$2 }
/^  16 [[:space:]]+[0-9]/ { if ($2+0 > 0) f16 = $3/$2 }
END { emit() }
' "$LOG"

echo "wrote $OUT:"
column -t -s $'\t' "$OUT"
