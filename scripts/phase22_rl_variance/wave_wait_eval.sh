#!/bin/bash
# Phase 22 RL variance — wait for an arm's training runs to finish, eval every
# --final-checkpoint on the pre-registered ruler, and print the criteria
# verdict (pass@1 sigma <= 0.056 AND no mean regression vs SFT 0.364).
#
# Usage: wave_wait_eval.sh <ckpt_dir> <expected_seeds>
set -e
cd /raid/users/paul/workLLM
CKPT_DIR=${1:?ckpt dir}
NSEEDS=${2:-4}
GLOB="$CKPT_DIR/*_final.safetensors"

# 1) wait for training: all checkpoints present AND no trainer procs alive.
for i in $(seq 1 720); do   # up to 24h (2 min/poll)
  have=$(ls $GLOB 2>/dev/null | wc -l)
  alive=$(pgrep -af phase22_he_reinforce | grep -v pgrep | wc -l)
  if [ "$have" -ge "$NSEEDS" ] && [ "$alive" -eq 0 ]; then echo "TRAIN_DONE have=$have"; break; fi
  if [ "$alive" -eq 0 ] && [ "$i" -gt 3 ]; then echo "PROCS_EXITED have=$have (some may have crashed)"; break; fi
  sleep 120
done

# 2) eval (blocks until all evals finish).
echo "=== launching eval ==="
bash scripts/phase22_rl_variance/eval_arm.sh "$CKPT_DIR" || true

# 3) summarize + criteria verdict.
OUT="$CKPT_DIR/eval"
python3 - "$OUT" <<'PY'
import sys, glob, os, re, statistics as st
out = sys.argv[1]
def grab(path, pat):
    try:
        t = open(path).read()
    except OSError:
        return None
    m = re.search(pat, t)
    return float(m.group(1)) if m else None
P1 = r"aggregate pass@1 \(raw, all samples\) = ([0-9.]+)"
P5 = r"per-prompt pass@5 = ([0-9.]+)"
base1 = grab(os.path.join(out,"base.log"), P1)
base5 = grab(os.path.join(out,"base.log"), P5)
rows=[]
for f in sorted(glob.glob(os.path.join(out,"*.log"))):
    name=os.path.basename(f)[:-4]
    if name=="base": continue
    p1=grab(f,P1); p5=grab(f,P5)
    if p1 is None and p5 is None: continue
    rows.append((name,p1,p5))
print("\n=== RL VARIANCE ARM EVAL (pre-registered ruler: 64 hard-tail, passk=5) ===")
print(f"base:  pass@1={base1}  pass@5={base5}")
for n,p1,p5 in rows:
    d1 = f"{p1-base1:+.4f}" if (p1 is not None and base1 is not None) else "?"
    d5 = f"{p5-base5:+.4f}" if (p5 is not None and base5 is not None) else "?"
    print(f"  {n:32s} pass@1={p1}  (Δ{d1})   pass@5={p5}  (Δ{d5})")
v1=[p1 for _,p1,_ in rows if p1 is not None]
v5=[p5 for _,_,p5 in rows if p5 is not None]
if len(v1)>=2:
    m1=st.mean(v1); s1=st.pstdev(v1) if len(v1)<2 else st.stdev(v1)
    m5=st.mean(v5); s5=st.stdev(v5)
    print(f"\narm pass@1 mean={m1:.4f}  sigma={s1:.4f}   (n={len(v1)})")
    print(f"arm pass@5 mean={m5:.4f}  sigma={s5:.4f}")
    # LOCKED criteria (docs/phase22-rl-variance.md)
    SFT_P1_SIG=0.037; SFT_P1_MEAN=0.364
    sig_ok = s1 <= 1.5*SFT_P1_SIG
    mean_ok = m1 >= SFT_P1_MEAN - SFT_P1_SIG
    print(f"\nCRITERIA (locked):")
    print(f"  variance: pass@1 sigma {s1:.4f} <= 0.056 ?  -> {'PASS' if sig_ok else 'FAIL'}"
          f"  (MeanCenter RL was 0.103)")
    print(f"  no mean regression: pass@1 mean {m1:.4f} >= 0.327 ?  -> {'PASS' if mean_ok else 'FAIL'}")
    print(f"  VERDICT: {'ARM QUALIFIES' if (sig_ok and mean_ok) else 'ARM DOES NOT QUALIFY'}")
else:
    print("\n⚠ <2 seeds evaluated — check for crashed runs.")
PY
echo "=== WAVE_EVAL_COMPLETE ==="
