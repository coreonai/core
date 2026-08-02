#!/bin/bash
# Eval all 6 SFT samples=32 r2 checkpoints on the pre-registered ruler
# (64 hard-tail, offset 100, passk=5, aggregate) + base, and print 6-seed
# stats. The existing-4 subset (42/100/200/300) doubles as a config-validation
# check: it should reproduce the doc's ~0.385 pass@1 if the recovered config
# matches the original samples=32 run.
set -e
cd /raid/users/paul/workLLM
BIN=./target/release/examples/phase22_humaneval_baseline
OUT=scratch-7b-sft/hts32_eval
mkdir -p "$OUT"
COMMON="--model-id Qwen2.5-Coder-7B --offset 100 --n-problems 64 --passk 5 --sequential --aggregate --max-new-tokens 192"

gpu=0
CUDA_VISIBLE_DEVICES=$gpu $BIN $COMMON > "$OUT/base.log" 2>&1 &
echo "base GPU $gpu PID=$!"; gpu=$((gpu+1))
for s in 42 100 200 300 400 500; do
  CKPT="scratch-7b-sft/hts32_out_s${s}/r0_merged.r2.safetensors"
  [ -f "$CKPT" ] || { echo "⚠ missing $CKPT"; continue; }
  CUDA_VISIBLE_DEVICES=$gpu $BIN $COMMON --checkpoint "$CKPT" > "$OUT/seed${s}.log" 2>&1 &
  echo "seed $s GPU $gpu PID=$!"; gpu=$((gpu+1))
done
wait

python3 - "$OUT" <<'PY'
import sys, os, re, glob, statistics as st
out = sys.argv[1]
def grab(p, pat):
    try: t=open(p).read()
    except OSError: return None
    m=re.search(pat, t); return float(m.group(1)) if m else None
P1=r"aggregate pass@1 \(raw, all samples\) = ([0-9.]+)"; P5=r"per-prompt pass@5 = ([0-9.]+)"
b1=grab(f"{out}/base.log",P1); b5=grab(f"{out}/base.log",P5)
print("\n=== SFT samples=32 — 6-seed eval (64 hard-tail, passk=5) ===")
print(f"base:  pass@1={b1}  pass@5={b5}")
rows=[]
for s in [42,100,200,300,400,500]:
    p1=grab(f"{out}/seed{s}.log",P1); p5=grab(f"{out}/seed{s}.log",P5)
    if p1 is None: continue
    rows.append((s,p1,p5)); print(f"  seed {s}: pass@1={p1:.4f}  pass@5={p5:.4f}")
v1=[p for _,p,_ in rows]; v5=[p for _,_,p in rows]
def ms(v): return (st.mean(v), st.stdev(v)) if len(v)>1 else (v[0],0.0)
if len(v1)>=2:
    m1,s1=ms(v1); m5,s5=ms(v5)
    print(f"\n6-seed pass@1 mean={m1:.4f} sigma={s1:.4f}   pass@5 mean={m5:.4f} sigma={s5:.4f}")
    old=[p for s,p,_ in rows if s in (42,100,200,300)]
    if len(old)==4:
        om,osd=ms(old)
        print(f"config-validation (existing 42/100/200/300): pass@1 mean={om:.4f} sigma={osd:.4f}  (doc: 0.385) "
              f"-> {'MATCH' if abs(om-0.385)<0.05 else 'DIVERGES — recovered config differs'}")
    print(f"\nvs K=8 posonly RL (6 seeds): pass@1 0.538 ± 0.076")
    gap = 0.538 - m1
    print(f"ATTRIBUTION: SFT-32 = {m1:.3f} vs RL = 0.538  -> gap +{gap:.3f} "
          f"({'RL wins — win is the RL regime, not just harvest' if gap>0.05 else 'SFT-32 approaches RL — harvest may explain it'})")
PY
echo "=== SFT32_EVAL_COMPLETE ==="
