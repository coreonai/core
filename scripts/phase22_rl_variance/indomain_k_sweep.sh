#!/bin/bash
# Phase 22 — C-2 variance attack, step 1: is harvest width the variance lever?
#
# The RL variance study concluded "RL can't be made low-variance here" after
# testing two levers (GRPO objective — worse; K=8 harvest — sigma 0.076, failed
# the pre-registered <=0.056 gate). But it stopped at K=8. Two things since then
# suggest the lever wasn't exhausted:
#   - Phase 15 S3b: SFT's variance is harvest-dominated (sigma_harvest 0.050 vs
#     sigma_init 0.009), and the seed controls the harvest draw.
#   - BigCodeBench at K=16: sigma was 58% of K=8's (0.0095 vs 0.0163) with a
#     flat mean — harvest width tightened spread on an axis where it bought no
#     mean at all.
#
# If more harvest makes each seed's draw converge, in-domain sigma should fall
# with K. This needs NO training: K=2/8/16/32 checkpoints already exist (6 seeds
# each) and K=4 exists as the C4 posonly arm (8 seeds). Only K=2/16/32 lack an
# in-domain eval, so 18 evals answer it.
#
# Ruler is the one every other in-domain number uses: HumanEval hard tail
# (--offset 100 --n-problems 64), passk 5, temp 0.8, sequential + aggregate.
#
# Usage: indomain_k_sweep.sh [arms]     default "k2,k16,k32"
set +e
cd /raid/users/paul/workLLM
BIN=./target/release/examples/phase22_humaneval_baseline
SEEDS=(42 100 200 300 400 500)
IFS=, read -r -a ARMS <<< "${1:-k2,k16,k32}"

[ "$(strings $BIN | grep -c cudarc)" -eq 0 ] && { echo "⚠ $BIN is a CPU build"; exit 1; }

# Pick cards with room for the 7B eval; external jobs turn up on this box.
usable() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
           | awk -F', ' '$2 < 6000 {print $1}'; }

for arm in "${ARMS[@]}"; do
  DIR="scratch-7b-sft/rlvar_${arm}"
  TAG="mean_${arm}"
  OUT="$DIR/eval"; mkdir -p "$OUT"
  echo "=== $arm — in-domain eval (hard tail, passk 5) ==="
  mapfile -t G < <(usable)
  [ "${#G[@]}" -lt 2 ] && { echo "⚠ only ${#G[@]} free GPUs — skipping $arm"; continue; }
  i=0
  for s in "${SEEDS[@]}"; do
    CKPT="$DIR/${TAG}_seed${s}_final.safetensors"
    [ -f "$CKPT" ] || { echo "  ⚠ missing $CKPT"; continue; }
    g=${G[$((i % ${#G[@]}))]}; i=$((i+1))
    CUDA_VISIBLE_DEVICES=$g $BIN --model-id Qwen2.5-Coder-7B \
      --offset 100 --n-problems 64 --passk 5 --sequential --aggregate \
      --max-new-tokens 192 --checkpoint "$CKPT" \
      > "$OUT/${TAG}_seed${s}.log" 2>&1 &
  done
  wait
  for s in "${SEEDS[@]}"; do
    f="$OUT/${TAG}_seed${s}.log"
    printf '  seed%-5s ' "$s"
    sed -r 's/\x1b\[[0-9;]*m//g' "$f" 2>/dev/null \
      | grep -oE "per-prompt pass@5 = [0-9.]+|aggregate pass@1 \(raw, all samples\) = [0-9.]+" \
      | tr '\n' ' '
    echo
  done
done

echo
echo "=== IN-DOMAIN sigma vs K (hard tail, 6 seeds each) ==="
python3 - <<'PY'
import re, os, glob, statistics as st
def grab(pat):
    out={}
    for f in glob.glob(pat):
        s=re.sub(r'\x1b\[[0-9;]*m','',open(f, errors='ignore').read())
        m5=re.search(r'per-prompt pass@5 = ([0-9.]+)', s)
        m1=re.search(r'aggregate pass@1 \(raw, all samples\) = ([0-9.]+)', s)
        if m5 and m1:
            sd=re.search(r'seed(\d+)', os.path.basename(f))
            out[int(sd.group(1))]=(float(m5.group(1)), float(m1.group(1)))
    return out
arms=[("K=2","scratch-7b-sft/rlvar_k2/eval/mean_k2_seed*.log"),
      ("K=4","scratch-7b-sft/c4_*/eval/posonly_seed*.log"),
      ("K=8","scratch-7b-sft/rlvar_k8/eval/mean_k8_seed*.log"),
      ("K=16","scratch-7b-sft/rlvar_k16/eval/mean_k16_seed*.log"),
      ("K=32","scratch-7b-sft/rlvar_k32/eval/mean_k32_seed*.log")]
print("%-5s %-4s %-22s %-22s" % ("K","n","pass@5 mean ± sigma","pass@1 mean ± sigma"))
for name,pat in arms:
    d=grab(pat)
    if len(d)<2: print("%-5s %-4d (insufficient)"%(name,len(d))); continue
    p5=[v[0] for v in d.values()]; p1=[v[1] for v in d.values()]
    print("%-5s %-4d %.4f ± %.4f       %.4f ± %.4f" %
          (name,len(d),st.mean(p5),st.stdev(p5),st.mean(p1),st.stdev(p1)))
print("\nSFT reference (same ruler, 4 seeds): pass@5 0.566 ± 0.020")
print("pre-registered variance gate was pass@1 sigma <= 0.056")
PY
echo "=== INDOMAIN_K_SWEEP_COMPLETE ==="
