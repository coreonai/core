# Phase 15 Design — harder substrate, multi-teacher OPD, cross-axis variance

Phase 14 Stage C closed with 3-of-3 retractions (Muon, hybrid DPO,
round-0 DPO) at the Qwen + 25-problem substrate. The substrate
qualified at σ=0.011 but saturated 21/25 problems under SFT,
leaving no headroom for Phase 14 C4 (OPD multi-teacher) and capping
algorithmic-comparison signal at very small Δ.

Phase 15 graduates the project to a harder substrate and tackles the
two questions Phase 14 couldn't: does multi-teacher OPD beat
SFT-on-union, and is σ=0.011 a tight bound or a lower bound that
hides cross-axis variance.

## Stage S1 — HumanEval substrate qualification

**Goal**: SFT baseline 50-70% mean, σ ≤ 0.03, ≤ 50% saturated.

**Setup**:
- HumanEval (164 multi-line Python problems, canonical from
  openai/human-eval)
- Qwen2.5-Coder-0.5B + LoRA r=16 α=32 q_proj+v_proj
- 1 LoRA-FT round (r0 + final-1) × 3 samples × 200 train-steps
- 5 seeds, AdamW lr=2e-4
- Verifier: subprocess + actual HumanEval test suite

**Decision gate**:
- Both σ ≤ 0.03 AND ≤ 50% saturated → ROBUST → proceed to S2
- σ ok but saturation > 50% → upgrade to MBPP / BigCodeBench
- σ > 0.03 → more samples / longer training

**Why HumanEval**:
- 164 multi-line problems (vs 25 single-line at Phase 14) → more
  signal per seed, smaller per-problem variance
- Test suites are real (assert-based check functions), not toy
  argmax
- Standard benchmark — results comparable to literature
- Qwen-Coder-0.5B reported pass@1 ≈ 50-55% → exactly the headroom
  we want

## Stage S2 — multi-teacher OPD

**Goal**: Test whether OPD on disjoint specialist teachers beats
SFT-on-union baseline (the Phase 14 C4 deferred test).

**Setup** (contingent on S1 qualification):
- Split 164 problems into k=3 disjoint subsets by skill axis
  (string-manipulation / numeric / list-comprehension); ~55
  problems each
- Train k SFT specialists, each on its subset only
- Train unified-SFT baseline on union (all 164)
- Train unified-OPD student: own rollouts vs frozen specialist
  teacher logits, weighted by routing
- 5 seeds × {SFT-union, OPD-multi-teacher} = 10 runs

**Decision gate**: 2σ threshold from S1's σ measurement (e.g. if
σ_S1=0.025 → Δ > 0.05 = robust win)

**Risks**:
- k=3 may be too small for OPD's specialization signal — DeepSeek
  V4 used many more teachers
- Routing without a learned router degenerates to averaging
  teachers, which is just label-smoothing

## Stage S3 — cross-axis variance audit

Phase 14 S1 acknowledged σ=0.011 is a *lower bound* — LoRA-init RNG
and harvest sampling RNG are entangled per seed. Real noise is
larger. Phase 15 S3 measures separately:

1. **Temperature axis**: 5 seeds × {T=0.6, 0.8, 1.0} = 15 runs at
   fixed model checkpoint
2. **Checkpoint axis**: 5 seeds at 3 distinct base checkpoints (Qwen
   0.5B, 1.5B-Coder, OLMo-1B) → 15 runs

Output: σ decomposition into init-RNG / harvest-RNG / temperature
/ checkpoint contributions. Updates the significance threshold for
all future comparisons.

## Stage S4 — TBD

Reserve. Either revisit one of Phase 14's retracted claims at the
new substrate (e.g. Muon at HumanEval scale, since LoRA-saturation
no longer applies), or attempt a new technique from a 2026 paper
not yet in the codebase.

## Cumulative scoreboard target

After S1+S2+S3, the project should have:
- A robust 100+-problem substrate with measurable headroom
- One genuine algorithmic positive (or honest null) on multi-teacher
  OPD
- Calibrated significance thresholds across multiple noise axes
- Phase 12 S2's "trainer deferred" debt fully paid down

## Success criteria

- S1 substrate qualified with σ ≤ 0.03 AND saturation ≤ 50%
- S2 produces a 2σ-robust verdict (positive or negative) on OPD
  multi-teacher
- S3 quantifies the gap between σ_init-RNG and σ_total

## See also

- `docs/phase14-stage-c-closeout.md` — Phase 14 closeout +
  motivation for harder substrate
- `docs/phase14-design.md` — Stage C plan template
- `nanogpt-rs/src/opd.rs`, `scripts/phase14_c4/opd.py` — OPD loss
  implementations awaiting full trainer integration
