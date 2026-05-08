# Phase 7 design — Shape C operational guide

Phase 7 is the **consolidation phase** for Phase 6 Shape C. Phase 6
established that an LM's own log-prob can act as a free critic on
RustCodeDomain (AUC 0.727, 4.8× compounding lift in self-improve
loop). Phase 7 Sessions 1+2 stress-tested that finding on
ArithmeticDomain and overturned the initial transfer claim.

This doc operationalizes the resulting refined understanding so a
future session (or a future contributor) can apply Shape C to a new
domain without re-deriving the gotchas.

## Three-tier guidance

### Tier 1 — Architectural fact

`LogitCritic` returns `score_of(prompt, completion) → f32`. Two
length-normalization regimes exist and they behave very differently:

| Variant | Computation | Domain fit |
|---------|------------|------------|
| **mean** (default) | `sum_t log P(c_t \| ...) / len(c)` | length-uniform domains |
| **sum** | `sum_t log P(c_t \| ...)` | length-varying domains |

Switch via `critic.normalize_by_length: bool`.

### Tier 2 — When each works

| Domain type | Mean | Sum | Evidence |
|-------------|:---:|:---:|----------|
| Length-uniform (K9 slot-fill, all completions ~5 chars) | ✓ | ✓ | Phase 6 S2 (mean AUC 0.727) |
| Length-varying (Arithmetic 1–4 chars, Korean variable) | ✗ | conditional | Phase 7 S1+S2 |

The mean variant's failure mode on length-varying domains is the
**short-bias**: empty / 1-token completions get artificially high
mean log-prob because there's no opportunity for low-confidence
tokens to drag the average down. At F=16 oversample this poisons
argmax selection (lift 0.04× on Arithmetic).

The sum variant retains the length signal — long bad completions
get strongly negative sum log-prob, short bad completions get
weakly negative. With **enough pretraining** to develop confidence
calibration, sum log-prob ranks correct above incorrect.

### Tier 3 — When sum works (the empirical gate)

Phase 7 S2 swept pretrain steps on Arithmetic and tracked AUC:

| Pretrain | Pass rate | Sum AUC |
|---------:|----------:|--------:|
|     800  |     7.6%  |   0.545 |
|    2000  |     8.6%  |   0.581 |
|    5000  |     9.8%  | **0.632** PASS |
|   10000  |     9.9%  |   0.658 |

**Pass rate is roughly constant across the sweep** (the model never
learns to *do* arithmetic at this architecture/data scale). What
changes is the model's softmax distribution sharpness — its
confidence calibration improves with more pretraining even when
its accuracy doesn't.

So the Phase 7 S1 acceptance gate ("≥ 2× chance pass rate") was
**wrong**. The correct gate is the AUC measurement itself:

> **Apply Shape C iff sum-AUC ≥ 0.6 on a held-out harvest.**

Pass rate is informative but not deciding. A model can be at chance
*and* well-calibrated, in which case Shape C still works.

## Recommended workflow for a new domain

When someone wants to deploy Shape C in a new self-improve loop on
verifier-V and domain-D:

1. **Quick smoke** — Pretrain a small base model on D's training
   corpus. Run an extension of `examples/critic_baseline_arithmetic.rs`
   pointed at V to harvest ~1000 (prompt, completion, verdict) tuples.
2. **Compute both AUCs** — mean and sum variants.
3. **Decision tree (updated by Phase 8 S1 anti-calibration finding):**
   ```
   if sum_AUC < 0.4:
       model is anti-calibrated (mode collapse / undertrained).
       Fix the model first (more pretrain, different data, different
       arch). Don't deploy any Shape C variant.
   elif 0.4 ≤ sum_AUC < 0.5:
       no signal in either direction. Shape C wastes compute.
       Either train more or skip this domain.
   elif 0.5 ≤ sum_AUC < 0.6:
       marginal — small lift possible but unstable. Train more
       and retest before committing to integration.
   elif sum_AUC ≥ 0.6:
       deploy Shape C with sum variant.
   ```
   For length-uniform domains where mean ≈ sum, the same bands
   apply to mean-AUC.
4. **Choose F** — based on Phase 6 Shape C S3 sweep, F=4 is
   typically optimal. F=16 risks top-tail outlier poisoning (the
   single-completion modes the model loves but verifier rejects).
5. **Integration smoke** — run `self_improve_rust --critic-oversample F`
   for ≥ 4 rounds. Acceptance: ≥ 1.5× mean gen-pass-rate vs F=1
   baseline at ≤ 30% wall-clock penalty.

## Risks already characterized

(From Phase 6 design doc, with Phase 7 updates.)

1. ~~"Critic learns to predict cargo's syntactic checks but not its
   semantics."~~ → **Updated**: It's deeper than syntax. The critic
   is the LM's own logits; it ranks completions the way the LM's
   training distribution prefers them. If the verifier's verdict
   correlates with that distribution (K9 RustCode: yes; Arithmetic
   under-trained: no), Shape C works. The diagnostic is sum-AUC.

2. ~~"Critic over-fits on the (small) training set."~~ → Holdout
   AUC measurement covers this.

3. ~~"Critic's added compute eats the wall-clock savings."~~ →
   Confirmed bounded: F=4 adds ~25% wall-clock per round at K9
   scale; saved cargo budget is whatever the cargo cost scales to.
   On real Rust projects (cargo ~seconds) the win compounds.

4. **NEW: Top-tail outlier poisoning at high F.** Phase 6 S3
   showed AUC ≥ 0.7 doesn't guarantee argmax-correctness — at F=16
   a small set of high-prob-but-verifier-rejected completions
   dominates every cohort's argmax. Mitigation: F ≤ 8 in production.

5. **NEW: Length-varying domains.** Phase 7 S1+S2 showed mean log-
   prob's short-bias breaks Shape C on Arithmetic. Mitigation:
   sum variant + sum-AUC gate.

6. **NEW: Calibration gate, not accuracy gate.** Phase 7 S1's
   initial framing ("≥ 2× chance") was overruled by S2 (sum-AUC
   crosses 0.6 while pass rate stays at chance). Calibration is
   what gets amplified, not accuracy.

7. **NEW: Anti-calibration on undertrained models** (Phase 8 S1
   measurement on KoWiki 30K + KoreanCompletionDomain). Sum-AUC
   can land *below 0.5* when the model has collapsed onto a
   degenerate mode (e.g. K8 30K loves emitting `\n\n\n\n...`,
   which the heuristic verifier specifically rejects). Don't
   assume sum-AUC ≥ 0.5 — measure both directions. If sum-AUC <
   0.4, the model's own preferences are *anti-correlated* with
   the verifier and Shape C would actively hurt. Treat this as a
   sign the model needs much more training before any Shape C
   deployment, not a sign that you should "invert the critic"
   (which is unstable as the model improves and the inversion
   point shifts).

   Empirical band:
   - sum-AUC ≥ 0.6 → deploy Shape C
   - 0.5 ≤ sum-AUC < 0.6 → marginal; train more, retest
   - 0.4 ≤ sum-AUC < 0.5 → no signal; Shape C wastes compute
   - sum-AUC < 0.4 → anti-calibration; model is broken (mode
     collapse, undertrained, distribution mismatch). Fix the
     model first.

8. **NEW: High AUC ≠ high selection lift** (Phase 8 S2 measurement
   on PythonCodeDomain). When the base model's harvest pass rate is
   already high (≥ ~30%), the random-pick baseline is strong and
   critic-rerank's *relative* lift compresses, even with very high
   AUC. K9 RustCode: pass 19%, AUC 0.727, F=4 lift 1.22×. Python:
   pass 35.6%, AUC 0.848, F=4 lift 1.00×. Higher AUC but no
   improvement in expected pass-per-cargo-call.

9. **NEW: More pretrain can WORSEN calibration on multi-epoch
   regimes** (Phase 9 S2 measurement on K8 100K vs 30K Korean).
   Phase 7 S2's headline ("calibration improves with pretrain
   even when accuracy plateaus") was measured on char-level
   Arithmetic with a small finite (a,b) corpus — multi-epoch
   reinforced the (a,b)→a+b distribution. Phase 9 S2 measured
   the opposite direction on BPE Korean: K8 100K (val_loss 7.44,
   ~6 epochs over 21M unique tokens) has sum-AUC **0.307**,
   *worse* than K8 30K's 0.363. The same model on the same data
   with more steps developed *deeper* anti-calibration (mode
   collapse onto `\n` repetition strengthened).

   Why Phase 7 S2 vs Phase 9 S2 disagree: data uniqueness × epochs.
   - Arithmetic: 100 unique pairs, multi-epoch is *necessary*
     to memorize the table. Calibration tracks accuracy.
   - KoWiki: 21M unique tokens, multi-epoch is *over-fitting*
     into the empirical token-frequency mode (heavily-weighted
     `\n` and common particles). Calibration drifts away from
     verifier-aligned distribution.

   Updated rule: **measure, don't assume.** Pretrain budget is
   not always net-positive for calibration. Re-measure sum-AUC
   at intervals during training; if it drops, the run is
   over-fitting in a way that hurts Shape C even though val_loss
   is stable.

   Mechanism: at high pass rate, random sampling already finds
   correct candidates often. The critic can only beat random if
   the ranking is *qualitative* enough at the top tail to avoid
   the outliers (high-prob short completions like `""` or `"1"`).
   Python's high AUC averages over the bulk of the distribution
   but the very top is poisoned, so argmax at F=4+ sees outliers
   often.

   Empirical sweet spot for selection lift: **pass rate ≈ 15–25%**.
   - Below ~10%: critic might not even be calibrated (Phase 7 S2,
     Phase 8 S1).
   - 15–25%: AUC and selection align; Shape C delivers expected lift.
   - Above ~30%: AUC may pass but selection rerank is no-op or
     small. Either accept Shape C as redundant at this pass rate,
     or deploy *only at F=2* (Phase 8 S2 saw 1.08× lift at F=2,
     dropping to 1.00× by F=4).

   This means the Phase 7 sum-AUC ≥ 0.6 gate should be combined
   with a **selection-sweep smoke** as the actual deployment
   decision: only integrate Shape C if F=2 or F=4 lift ≥ 1.10×
   on a held-out sample. AUC alone misses the outlier-ceiling case.

10. **NEW: External-scale validation (Phase 9 S4 measurement on
    Qwen2.5-Coder-0.5B / 1.5B).** The decision tree carries to a
    real HF model with its own BPE: 0.5B-Coder lands sum-AUC 0.702,
    F=8 lift **1.95×** (strongest in the matrix). 1.5B-Coder on the
    same six challenges drops to sum-AUC 0.474, F≥2 lift below 1.0.

    This is the same direction as risk #9: a *bigger* / more-trained
    model can be **worse** for Shape C because priors over-fit to
    common patterns (`s = 0`, `return 1`) at the expense of rare
    verifier-aligned completions (`"hello"`, `5`). The mean-vs-sum
    split also holds — mean-AUC is at chance (0.502) on
    length-varying slot completions; only sum captures the signal.

    Operational implication: **always smoke-test a candidate model
    before deploying Shape C, regardless of nominal scale.** A
    smaller, less-confident base model can be a better Shape C
    target than a larger fine-tuned one. See
    `scripts/phase9_s4/` for the harvest+analyze scripts that
    reproduce the measurement on any HF model.

11. **NEW: Cold-start dominates the per-challenge fate** (Phase 9
    S5 measurement). End-to-end self-improve loop on
    Qwen2.5-Coder-0.5B + LoRA on 11 challenges (6 S4 slot-fill +
    5 HumanEval-style function bodies) saturated in **1 round**:
    pass rate 39.8% → 72.7% (+33 pp). 8 of 11 challenges hit 100%.
    The 3 that never improved had **0 verifier-passed samples in
    round 0**; LoRA on the other 8's pairs did not transfer.

    Implication: per-challenge self-improvability is gated by
    whether the base model produces *any* passing seed at round 0.
    Below the seed threshold, more rounds, more LoRA capacity, and
    larger critic budgets all do nothing — the loop is starved.

    Mitigations (when round-0 pass rate per challenge is 0):
    - Curriculum: introduce an easier variant of the same task
      first (e.g., `assert f() == 5` with `def f(): return ` is
      reachable; `assert f() == 14` with `def f(): return 2 * (`
      requires the rare `7)` token, was unreachable here).
    - Few-shot: prepend one solved example to the prompt.
    - Bootstrap injection: add one hand-written ground-truth
      (prompt, completion) into round-0 training set.
    - Skip Shape C for that challenge — the loop cannot fix it.

    See `scripts/phase9_s5/` for the loop, run.json results, and
    per-challenge breakdown.

12. **NEW: JEPA-style aux losses interact non-trivially with
    Shape-C calibration; the interaction is HP- and domain-
    sensitive** (Phase 10 S1 single-point + S2 sweeps on K8 and
    PythonCodeDomain).

    S1 single-point (λ=0.1, k=8 on K8): top-1 mass −33%
    (0.146 → 0.097), pass rate +50%, but **sum-AUC 0.421 → 0.238**.
    Read in isolation, this looked like a blanket "diversity ≠
    calibration" rule.

    S2 sweeps overturned the blanket reading:
    - **K8 λ sweep at k=8**: sum-AUC is U-shaped — 0.421 (baseline)
      → 0.342 (λ=0.01) → 0.291 (λ=0.03) → **0.238 (λ=0.1, worst)** →
      **0.433 (λ=0.3, recovered, slight win over baseline)**.
    - **K8 k sweep at λ=0.1**: shorter k recovers calibration —
      k=8 → 0.238, k=4 → 0.396, **k=2 → 0.432** (recovered).
    - **K8 EMA target encoder (decay=0.99) at λ=0.1, k=8**:
      sum-AUC 0.292 — gives the strongest mode-collapse mitigation
      (top1 0.049, lowest in matrix) but does NOT recover
      calibration. EMA-vs-self difference is on diversity, not
      verifier alignment.
    - **PythonCodeDomain at λ ∈ {0, 0.03, 0.1}**: ALL sum-AUCs
      ≈ 0.86 (PASS gate). λ=0.1 even posts the highest F=4
      selection lift (1.05×). The K8 anti-cal pathology does not
      transfer.

    Mechanism (refined): JEPA's latent objective rewards
    distinctive hidden states. Whether that distinctiveness is
    orthogonal-to or antagonistic-with verifier-aligned confidence
    depends on (a) λ — too low gives noisy gradients that hurt
    without enough push to be a useful regularizer; too high
    becomes a strong regularizer that doesn't fight CE; (b) k —
    short hops keep the predictor's job close to next-token
    coherence, long hops force semantic abstraction that doesn't
    track verifier verdicts; (c) domain — K8's `\n`-mode pathology
    makes the model especially eager to follow JEPA away from
    verifier-aligned tokens, while Python's verifier-tight slot-
    fill resists that drift.

    Operational: don't deploy JEPA from a single (λ, k) point.
    Sweep at least 3 points across λ and k, plot sum-AUC, prefer
    either tail of the U. EMA target encoder is *not* a free
    upgrade for calibration. See `docs/phase10-s2-jepa.md` for
    the full sweep tables, the practical recipe, and the
    reproduction commands.

    **Phase 10 S3 update — recovery is not scale-stable.** Re-running
    S2's two winners (λ=0.3 k=8, λ=0.1 k=2) at K8 **30K steps**
    instead of 5K shows both fall *below* the baseline 30K
    sum-AUC of 0.363:

      baseline 30K     0.363
      λ=0.3, k=8 30K   0.289   (Δ −0.074)
      λ=0.1, k=2 30K   0.330   (Δ −0.033)

    JEPA's latent distinctiveness becomes a stronger force at
    longer training, and calibration loses again. **S2's 5K
    "recovery" was transient.** Practical implication added to
    operational rule: don't decide JEPA hyperparameters on a 5K
    sweep — measure at ≥ 50% of the target training budget. For
    K8/Korean BPE pretrain, JEPA stays off as the default. See
    `docs/phase10-s3-jepa-longrun.md` for the full S3 result.

13. **NEW: DPO multi-round dynamics differ from single-round**
    (Phase 11 S3 measurement on K9 RustCode). At round 0, DPO
    β=0.1 with reference frozen at seed posted gen-pass **41.7%**
    vs SFT's **0.0%** — a +41.7pp lift, the strongest single-round
    signal in the project. But round 1 catastrophically collapsed:
    eval went from 7/24 to **0/24**, and rounds 2-3 stayed at
    0/24/0/24. Sample inspection showed mode collapse onto
    repetitive `-` tokens.

    Mechanism candidates:
    - β=0.1 too aggressive at 1M scale.
    - Frozen reference at seed: as policy drifts, `(π − π_ref)`
      grows unboundedly, amplifying the implicit reward gradient.
    - 400 train steps × β=0.1 = excessive effective drift budget
      per round.

    Operational: DPO is **not a drop-in SFT replacement** at this
    scale without further hyperparameter / reference work. New
    deployments must measure ≥ 4 rounds before declaring
    success on round-0 numbers. Candidate fixes for Phase 11 S4:
    β sweep (0.01-0.05), rolling reference (snapshot per round),
    fewer train steps per round, hybrid SFT+DPO loss.

    See `docs/phase11-s3-dpo-vs-sft.md` for the full result and
    follow-up plan.

    **Phase 11 S4 update — collapse is robust.** β sweep at
    {0.01, 0.03, 0.05, 0.1} *and* rolling reference (snapshot per
    round) all show the same round-1 catastrophic collapse:
    eval-after lands at 0/24 every time. β=0.01 eventually
    recovers to baseline by round 3 (final 11/24 = SFT) but adds
    no net benefit; β ∈ {0.03, 0.1} stay at 0; β=0.05 partially
    recovers to 7/24. **Hyperparameter tuning alone does not save
    pure DPO at 1M scale.** Updated mechanism hypothesis: the
    rejected pile (24 noisy incorrect completions per round) is
    mostly *noise*, not *informative wrongs*; DPO's negative
    gradient pushes the policy off the eval distribution. Phase
    11 S5 will test (a) hybrid SFT+DPO loss and (b) round-0-only
    DPO ("DPO seed boost"). See `docs/phase11-s4-dpo-fixes.md`
    for the full sweep.

    **Phase 11 S5 update — structural fixes work for collapse,
    but no DPO variant beats SFT's final eval.** 11-variant
    matrix (S3 + S4 + S5) on K9: every variant final eval ≤ SFT's
    11/24. Three positive findings inside that ceiling:
    - **Hybrid α=0.3** at β=0.1 hits **r1 eval 18/24 (75%)** —
      the project's single-round eval record, +7 over SFT max.
      Drops to 11 by r2 (no sustain), but useful for
      best-of-rounds checkpoint selection.
    - **Round-0-only DPO** reaches eval 11/24 at r0 (vs SFT
      reaching 11 at r2) — 1 round faster to baseline, useful
      for compute efficiency.
    - **Hybrid α ≥ 0.3** prevents collapse: every variant
      recovers to 11/24 by r3.

    Operational rule: **don't deploy pure DPO multi-round at this
    scale.** Use hybrid α=0.3 if you want the single-round signal
    spike; round-0-only if you want compute savings; SFT
    otherwise. K9's 21 distinct (prompt, slot) pairs may be too
    few for fine-grained DPO signal — re-measure on richer
    domains (HumanEval) before declaring DPO universally
    inferior. See `docs/phase11-s5-hybrid-dpo.md` for the full
    matrix.

## What "Phase 7 done" means

Phase 7 is a consolidation phase, not an implementation phase. Its
deliverables are:

- ✓ Sum-variant of LogitCritic (already in `critic.rs` via
  `normalize_by_length: bool`) — exposed cleanly with
  `LogitCritic::sum_scoring()` constructor (Session 3 below).
- ✓ AUC sweep methodology proven on Arithmetic
  (`critic_baseline_arithmetic.rs`).
- ✓ Decision tree above (this doc).
- ✓ Risk register update.

Future sessions can pick up Phase 7's loose ends:

- **Phase 7 S3**: ergonomic critic constructor + module doc update.
- **Phase 7 S4 (deferred)**: dedicated critic head training, only
  if a future domain fails both mean and sum AUC gates AND we
  still want Shape C there.
- **Phase 8**: domain expansion — apply Shape C to PythonPytestDomain
  or a Korean-completion variant where the AUC gate passes.

## Non-goals for Phase 7

- **No additional training infrastructure.** Phase 7 is pure
  measurement + documentation. Any new training (e.g. critic head)
  is Phase 8+.
- **No ensemble / consensus revival.** Phase 5's negative result
  stands at toy scale. Revisit at 50M+.
- **No specialist routing revival.** Phase 6 S1 negative stands at
  compute parity.

## What this consolidation buys us

The "honest negatives + falsifier tests" workflow is now an asset
the project can reuse:

1. **Cheap experiments overturn assumptions.** Phase 7 S2 took 2
   minutes on GPU and overturned a claim that would have shaped
   Phase 8+ design.
2. **Acceptance gates are explicit and falsifiable.** Sum-AUC ≥
   0.6, F ≤ 8, +25% wall-clock cap. Future contributors can
   measure rather than guess.
3. **Memory entries capture mechanism, not just outcomes.** The
   project can remember why something failed, not just that it
   failed — so when the same shape comes up at a different scale,
   the past failure mode informs the new design.

This pattern is independently more valuable than any specific
mechanism we've shipped today. It's why the day produced 24
commits with 4 honest negatives and 1 strong positive without any
single experiment dominating.
