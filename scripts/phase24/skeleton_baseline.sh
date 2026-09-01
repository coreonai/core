#!/usr/bin/env bash
# Phase 24 — cargo-level baseline for F0–F5 skeletons.
# reference feature (default): should be all green
# --features student: stubs should fail (documents headroom)
set -u
cd "$(dirname "$0")/../../scratch-pekko-harvest"
OUT="${1:-../scratch-7b-sft/p24_skeleton_baseline}"
mkdir -p "$OUT"
echo "OUT=$OUT"
echo "=== reference (default) ===" | tee "$OUT/run.log"
if cargo test --workspace -- --test-threads=1 > "$OUT/reference.log" 2>&1; then
  echo "reference: PASS" | tee -a "$OUT/run.log"
else
  echo "reference: FAIL" | tee -a "$OUT/run.log"
fi
grep -E "test result:" "$OUT/reference.log" | tee -a "$OUT/run.log"

echo "=== student feature per crate ===" | tee -a "$OUT/run.log"
for crate in f0_expr f1_tool f2_domain f3_message f4_repair f5_supervisor; do
  if cargo test -p "$crate" --features student -- --test-threads=1 > "$OUT/student_${crate}.log" 2>&1; then
    echo "$crate student: PASS (unexpected — stubs should fail)" | tee -a "$OUT/run.log"
  else
    # count failed tests if any
    fails=$(grep -cE "^test result: FAILED" "$OUT/student_${crate}.log" || true)
    echo "$crate student: FAIL (expected) — see student_${crate}.log" | tee -a "$OUT/run.log"
  fi
  grep -E "test result:" "$OUT/student_${crate}.log" | tee -a "$OUT/run.log" || true
done
echo "=== SKELETON_BASELINE_DONE ===" | tee -a "$OUT/run.log"
