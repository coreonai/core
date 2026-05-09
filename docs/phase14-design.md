# Phase 14 design — Stage C: Qwen + HumanEval (real-substrate algorithmic comparison)

Phase 13 closed with: **K9 1M is smoke-test infrastructure, not
measurement substrate**. Cross-batch σ ≈ 6-7/24 swallows most
algorithmic deltas (Muon, DPO variants, hybrid α). Phase 14 moves
algorithmic comparisons to the Phase 9 path: external Qwen
2.5-Coder-0.5B + PEFT LoRA + HumanEval-style real-world coding
problems.

## Why Stage C now

Phase 9 S4 already validated the substrate: Qwen2.5-Coder-0.5B on 6
challenges gave **sum-AUC 0.702** and Phase 9 S5 closed-loop
self-improve hit **+33pp in 1 round** (39.8% → 72.7%). These are
robust signals far above K9 1M's noise floor — algorithmic deltas
that were noise at K9 1M may show up clearly here.

The substrate already exists in `scripts/phase9_s5/self_improve.py`
(harvest + critic-rerank + LoRA-FT). Phase 14 extends it with:
1. **Bigger problem set** (11 → 30-50 HumanEval-style)
2. **Variance bound at Qwen substrate** (5 seeds × SFT baseline)
3. **Algorithmic variants** that K9 couldn't distinguish (Muon for
   LoRA, DPO, hybrid SFT+DPO, OPD)

## 4-stage sub-plan

| stage | scope | cost | answers |
|--:|---|---:|---|
| C1 | Problem-set expansion + variance bound | 1-2 hours | Substrate noise floor at Qwen |
| C2 | Muon for LoRA at Qwen scale | 2-3 hours | Does Muon beat AdamW for LoRA adapters? |
| C3 | DPO variants on Qwen self-improve | 3-4 hours | Phase 11 hybrid α=0.3 / round-0-only at real scale |
| C4 | OPD vs SFT on Qwen multi-task | half day | Multi-teacher distillation (Phase 12 S2-loss) at real scale |

C1 first (cheapest, validates substrate). Then C2/C3 in priority
order based on which K9 result we most want to verify.

---

## C1 — Problem-set expansion + variance bound

### Scope

1. Add 20-30 HumanEval-style problems to
   `scripts/phase9_s5/self_improve.py::CHALLENGES`. Each problem:
   - Function signature + docstring
   - Single-line return (compatible with Phase 9 S5's line-truncation)
   - 1-3 asserts as suffix
   - Solvable by 0.5B base model in some fraction of cases (warm-start)
2. Run 5-seed × SFT baseline (`--rounds 3 --samples 8`,
   different `--lr` seeds for LoRA init randomness)
3. Compute mean ± std of pass-rate per round, final pass-rate
4. Compare to K9 1M's noise floor

### Falsifier

Cross-seed σ at Qwen substrate should be **substantially smaller**
than K9 1M's σ ≈ 0.08 on mean_gen-pass. If Qwen σ ≥ K9 σ,
this substrate is no better and we'd need to escalate to bigger
benchmarks (MBPP, BIG-bench).

### Decision gate

- σ < 0.05 on final pass-rate → substrate qualified, proceed C2
- σ ≥ 0.05 → larger benchmark or longer training needed before
  algorithmic comparison

### Cost

- 5 seeds × ~30 min/run on A100 = 150 min. Two GPUs parallel = 75
  min. Single GPU acceptable.

---

## C2 — Muon for LoRA adapters at Qwen scale

### Hypothesis

Phase 13 S1 showed Muon at K9 1M is noise-equivalent to AdamW. But
Muon's strength is on dense 2D weight matrices — exactly what LoRA
adapters are. Phase 9 S5 LoRA: r=16, q_proj+v_proj ≈ 1M trainable.

Concretely: at Qwen 0.5B with LoRA r=16, the trainable matrices are
~1024×16 = 16K params per adapter, 256 adapters = 1M total. These
are *exactly* the kind of small dense matrices Muon's NS works on.

### Implementation

`scripts/phase14_s2/self_improve_muon.py` extends S5's script:
- `--optimizer muon|adam` flag
- For Muon: orthogonalize gradients on LoRA delta_A / delta_B
  matrices via Newton-Schulz before applying update
- Re-implements (or imports from) the nanogpt-rs Muon for Python

### Falsifier

C1 baseline (5-seed SFT/AdamW): final pass-rate `p_adam ± σ_adam`
C2 measurement (5-seed SFT/Muon): final pass-rate `p_muon ± σ_muon`
- Muon win iff `p_muon - p_adam > 2 max(σ_adam, σ_muon)`
- Phase 12 S1 retracted "+78% gen at K9 1M" → C2 is the true test

### Cost

5 seeds × Muon variant ~30 min = 75 min on 2 GPUs parallel.

---

## C3 — DPO variants on Qwen self-improve

Phase 11 5-session matrix on K9 found:
- Pure DPO collapses by r1
- Hybrid α=0.3 hits r1 = 18/24 (75%) but doesn't sustain
- Round-0-only matches SFT 1 round earlier
- All variants tied to SFT final 11/24

All single-run K9 1M observations. Phase 14 C3 re-measures the most
interesting variants (hybrid α=0.3, round-0-only) with 5-seed
variance at Qwen substrate.

Implementation in Python (since Qwen is HF model):
- DPO loss: PEFT's TRL library has DPO trainer; can wrap to
  reproduce hybrid (α-mixed CE + DPO) and round-0-only patterns
- Or implement DPO loss in pure PyTorch (~50 LOC)

### Falsifier

If Phase 11 results were K9-1M noise:
- Pure DPO collapse may NOT reproduce at Qwen → "noise floor was
  hiding signal"
- Or: pure DPO collapse DOES reproduce → Phase 11 finding was real,
  not noise

Either resolution closes risk #14 for this specific question.

### Cost

3 variants × 5 seeds × 3 rounds = 45 runs. ~6-8 hours total.
Significant but bounded.

---

## C4 — OPD vs SFT on Qwen multi-task

OPD trainer (`train_opd`) was Phase 12 S3 deferred. Reviving it for
Qwen:
- 3 specialists: Qwen-Coder (code), Qwen-Math (math), Qwen-Instruct
- 1 student: Qwen 0.5B base, distill from 3 teachers
- Eval: HumanEval (code) + GSM8K subset (math) + simple
  instructions

This is C2/C3 but at the multi-task level — DeepSeek V4's actual
RL replacement claim.

Cost: ~half day (multi-model orchestration).

---

## Phase 14 first session scope

**Phase 14 S1 = C1**: HumanEval-style problem expansion + 5-seed
variance bound on Qwen substrate. Validates that Qwen + HumanEval
is a lower-noise measurement substrate before investing in C2-C4.

Concrete first-session deliverables:
1. `scripts/phase14_s1/problems.py` — 25-30 HumanEval-style problems
2. `scripts/phase14_s1/self_improve.py` — Phase 9 S5 fork with
   --seed CLI for fresh LoRA init randomness
3. 5-seed × SFT measurement
4. `scripts/phase14_s1/analyze.py` — variance + comparison to K9
5. `docs/phase14-s1-substrate.md` — variance bound conclusion

---

## What this is NOT

- **Not a "proper" HumanEval evaluation** — we use single-line
  HumanEval-style problems, not the full 164 test-suite-based
  HumanEval. C5+ candidate.
- **Not multi-GPU training** — single A100 fp16 throughout.
- **Not in-house production** — uses HF Qwen via Python venv (the
  Phase 9 path). Switching back to in-house Rust + Candle is a
  separate axis (deferred Stage D).
- **Not full algorithmic re-validation** — only the most
  Phase-13-noise-bound claims (Muon, DPO collapse, hybrid α=0.3)
  are re-measured here.

---

## Risk register additions (anticipated)

After Phase 14 measurements complete, candidate risks:

- **#16 (anticipated)**: Cross-substrate transfer of algorithmic
  results. K9 1M results may not predict Qwen 0.5B results — both
  directions (algorithm wins at K9 but loses at Qwen, or vice
  versa). Don't assume substrate-neutrality.
- **#17 (anticipated)**: HumanEval-style single-line completions
  may saturate. Phase 9 S5 already showed 8/11 problems hit 100%
  by round 1. Need cold-start problems in the set for headroom.

---

## See also

- `docs/phase13-design.md` — original 4-stage scale-up plan
- `docs/phase13-s3-isolate-budget.md` — Phase 13 closure rationale
- `scripts/phase9_s5/` — substrate inherited from Phase 9
- Phase 9 S4 / S5 memory entries — Qwen + LoRA + critic-rerank
  precedent
- DeepSeek V4 sources (Notion: workLLM Phase 10 S3 + 11 S1–S5 page)
- `nanogpt-rs/src/{muon,opd,dpo}.rs` — algorithmic implementations
  awaiting real-substrate validation
