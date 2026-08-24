---
title: "Phase 22 C-1 — pre-registration: does bounding the PG objective beat not bounding it, in-domain?"
date: "2026-08-25"
status: "LOCKED — committed before any C-1 data exists"
---

# Why this exists

`docs/phase22-c4-c5-rl-vs-sft.md` reports that positive-advantage-only
(`--pg-positive-only`) beats full-advantage RLOO in-domain by **+0.124 pass@1,
8/8 seeds, paired t = 3.62, p = 0.0086**.

That number came from **optional stopping**. The sample was extended after
looking at n=4 (p ≈ 0.10) and again at n=6 (p ≈ 0.057), so the nominal p at
n=8 is optimistic and the result is recorded there as *"strong, not
established."*

This document locks a confirmatory design **before** the confirmatory data
exists. It is committed first precisely so that it cannot be adjusted after
seeing the outcome.

# Locked design

| Item | Value |
|---|---|
| **Seeds** | 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100 |
| **n (pairs)** | **12** |
| **Arms** | `posonly` (`--pg-positive-only`) vs `fulladv` (flag omitted) |
| **Pairing** | Both arms share each seed — the comparison is paired |
| **Config** | Byte-identical to the C4 re-run otherwise |
| **Primary endpoint** | In-domain hard-tail **aggregate pass@1** |
| **Test** | Paired **two-sided** t-test, α = 0.05 |
| **Interim looks** | **None.** One analysis, at n = 12 |

Training command (per run), unchanged from C4:

```
--model-id Qwen2.5-Coder-7B --train-bf16 --trainer-gpu 1
--prompt-offset 100 --n-prompts 64
--rl-steps 30 --k-per-prompt 4 --max-new-tokens 192
--pg-micro-batch-size 1 --sync-every 1 --lr 2e-4
[--pg-positive-only for the posonly arm]
```

Evaluation (per checkpoint), the same ruler every in-domain number uses:

```
phase22_humaneval_baseline --model-id Qwen2.5-Coder-7B
  --offset 100 --n-problems 64 --passk 5 --sequential --aggregate
  --max-new-tokens 192 --checkpoint <final>
```

Seeds 1000–2100 have **never been used** in this project; 42/100/200/300/400/
500/600/700 are all spent on the exploratory arms.

# Decision rule (fixed in advance)

- **ESTABLISHED** if `p < 0.05` **and** the mean paired difference is positive.
- **NOT ESTABLISHED** otherwise.

No other outcome is a win. In particular: a p between 0.05 and 0.10 is *not
established*, regardless of sign consistency, and will not be re-described as
"directional" to rescue it.

## Secondary, descriptive only — never decisive

- pass@5 on the same checkpoints (the n=8 exploration found this **null**,
  p = 0.17, so the claim is already scoped to pass@1)
- sign count (how many of 12 favour posonly)
- per-arm means and σ

# Power, stated honestly

Exploratory effect size was dz = 0.124 / 0.097 = **1.28**. Exploratory
estimates are inflated by winner's curse, so the realistic planning value is
lower:

| n | power @ dz 1.28 | @ dz 0.80 | @ dz 0.60 |
|---|---|---|---|
| 8 | 0.90 | 0.46 | 0.25 |
| **12** | **0.99** | **0.72** | **0.45** |
| 16 | 1.00 | 0.86 | 0.61 |

n = 12 was chosen as the point where a *null* is still informative at a
realistically deflated effect (72% power at dz = 0.80), against a cost of
~45 GPU-hours. n = 8 was rejected: at dz = 0.80 it is a coin flip, so a null
would have meant nothing.

**Pre-committed caveat**: if the result is null, it rules out effects around
dz ≥ 0.8 with reasonable confidence but **not** a true effect near dz = 0.6
(45% power). That limitation will be stated in the write-up rather than
quietly dropped.

# Handling of failures

- A run that dies (OOM, disk, external GPU contention) is **re-run once** with
  the identical command. The pair is the unit of analysis.
- If a pair cannot be completed after one retry, it is **excluded**, and both
  the exclusion and the reason are reported. n drops accordingly and the
  achieved power is recomputed.
- Slice/eval completeness is enforced by the existing guards; a checkpoint
  whose eval is incomplete is not scored.

# What this cannot settle

- **Transfer.** Bounding the objective was measured null out-of-domain
  (LCB, t = −0.52). This replication is in-domain only and does not revisit
  that.
- **Other K.** Pinned at K=4 to match the exploratory arms. The harvest sweep
  is a separate axis.
- **Other substrates.** HumanEval hard tail only.

# Pre-registered analysis script

`scripts/phase22_c1/analyze.py` — written and committed with this document,
before any data exists. It takes the eval logs, computes the paired test, and
prints the verdict against the rule above. It will not be edited after the
run; if it has a bug, the fix will be a separate commit visible in history.
