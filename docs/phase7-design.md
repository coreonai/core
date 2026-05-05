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
3. **Decision tree:**
   ```
   if sum_AUC ≥ 0.6:
       deploy Shape C with sum variant
   elif mean_AUC ≥ 0.6 and completions are length-uniform:
       deploy with mean variant
   elif sum_AUC < 0.6 and pass_rate < 2× chance:
       train base model more (calibration may follow accuracy)
   else:
       Shape C doesn't fit this domain — try Shape B (specialist)
       or Shape D (TBD: train dedicated critic head)
   ```
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
