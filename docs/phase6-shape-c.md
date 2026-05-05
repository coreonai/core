# Phase 6 Shape C — Adversarial generator/critic

Phase 5 (consensus ensemble) and Phase 6 Session 1 (specialist routing)
both produced honest negative results: **multi-actor structure isn't
magic; the task distribution has to genuinely benefit from splitting,
and K9's three challenges share too much structure for either consensus
or specialization to add value at compute parity**.

Shape C bets on a different mechanism: **a learned critic acts as a
cheap pre-filter for the expensive cargo verifier**, letting the
self-improve loop afford more candidates per round without
proportionally more cargo invocations. If a critic's accuracy on a
held-out (prompt, completion, cargo_verdict) set is meaningfully
above 50%, it's already useful as a pre-filter — and unlike Shape B
specialization, this isn't "splitting compute" so it doesn't pay the
data-fragmentation tax.

## Concrete plan

### Session 1 (this doc) — scaffolding

- `llm-actors/src/critic.rs`:
  - `pub trait Critic: Send + Sync { fn score(&self, prompt: &str, completion: &str) -> f32 }`
  - `pub struct AlwaysCorrectCritic` (returns 1.0 — equivalent to
    "no filter", current self-improve behavior).
  - `pub struct RandomCritic { seed: u64 }` for baseline.
- 2-3 unit tests on the trait API.

No model training, no integration. Sessions 2+ build the actual
learned implementations.

### Session 2 — Logit-baseline critic

Use the existing K9 generator's own `lm_head` as the critic.
Concretely: for a candidate `(prompt, completion)`, compute
`-1/T · sum_t log P(completion_t | prompt + completion[:t])` (the
model's own per-token negative log-likelihood, normalized by length).
Lower NLL = model is more confident in this completion.

Implementation:
- Add a method `GPT::sequence_log_prob(&self, prompt_ids: &[u32],
  completion_ids: &[u32]) -> f32` that returns `sum_t log P(...)`.
- Wrap as `LogitCritic { model: ActorRef<ModelActor> }` implementing
  the `Critic` trait.

Measurement: collect (gen attempt, cargo verdict) pairs from a K9
v5 round, sort attempts by `LogitCritic.score`, compute the AUC of
the binary classification. Threshold-free metric — no need to pick
a cutoff; just see if rankings correlate with cargo.

Acceptance criterion: **AUC ≥ 0.6**. Below that the model's own
logits don't carry enough signal to filter usefully.

### Session 3 — Integration into self-improve

If Session 2's AUC is acceptable:
- Add `--critic-threshold` to `self_improve_rust`. Each round:
  1. Generate `gen_n × oversample_factor` candidates per prompt.
  2. Score with critic; keep top-`gen_n` per prompt.
  3. Run cargo on the survivors only.
- Compare: same gen-pass at lower wall-clock, OR more candidates
  fitting in the same wall-clock.

Acceptance: same gen-pass-rate at ≤ 80% the wall-clock, or
gen-pass-rate ≥ 1.2× at the same wall-clock.

### Session 4 — Trained critic head (only if Session 2 fails)

If `LogitCritic` doesn't work (AUC < 0.6), the model's free signal
is insufficient and we need to actually train a critic head:

- Add a small classification head to the GPT — pool the final
  layer's hidden states, project to a scalar via a 2-layer MLP,
  sigmoid for P(correct).
- Train the head only (base frozen) on (prompt, completion,
  cargo_verdict) tuples harvested from a few rounds of K9.
- Repeat the AUC measurement.

This is a much bigger lift; only do it if the cheap version fails.

## What we already have that helps

- **Cargo-verified labels.** Every (prompt, completion) the
  VerifierActor sees gets a `Verdict::Correct` or `Incorrect`.
  That's gold-standard supervision, free.
- **CuratorActor's `rejected_incorrect` count.** The `Add` flow
  already separates kept from rejected items. We can extend the
  curator (or the verifier) to log the rejected ones for the
  critic's negative training data.
- **K9 v5 r=32 α=64 retrains in 1.5 min.** Iterating on critic
  approaches is cheap — kick off a baseline run, harvest the
  trajectories, train a critic, replay. Keep sessions tight.

## Risks

1. **Critic learns to predict cargo's syntactic checks but not its
   semantics.** If most cargo failures are syntax errors, a critic
   would learn "is this even close to valid Rust?" — useful but
   not the deep semantic check we'd want for harder challenges.
   Mitigation: log per-failure category from cargo and check that
   the critic's errors aren't all of the syntax kind.

2. **Critic over-fits on the (small) training set.** With 4–10
   rounds × 24 candidates × 3 challenges, that's at most ~1000
   labeled examples. A linear critic on top of model embeddings
   might memorize and not generalize. Mitigation: hold out 30%
   for AUC measurement, don't train end-to-end with the LM.

3. **Critic's added compute eats the wall-clock savings.** Running
   the critic forward on every candidate has cost too. The savings
   only materialize if the critic is *meaningfully* cheaper than
   cargo. Cargo is ~100ms/run; the critic's forward pass on a
   ~50-char sequence through a 1M-param transformer is ~1ms. Two
   orders of magnitude headroom — should be fine.

4. **Adversarial co-evolution failure modes** (the original
   "Shape C" framing in `docs/phase5-design.md`). If we ever escalate
   to *training* the generator AGAINST the critic (rather than
   using the critic as a passive filter), generators can hack
   critics — produce trajectories the critic accepts but cargo
   rejects. Mitigation: keep cargo as the *ultimate* judge in the
   self-improve curator; the critic is only ever a pre-filter, not
   a label source.

## What "Shape C done" means

For the project as a whole:
- `Critic` trait + 2 implementations landed (Sessions 1+2).
- `examples/critic_baseline.rs` measures AUC on harvested K9 data
  (Session 2).
- `--critic-threshold` integration shows ≥ 1.2× wall-clock-adjusted
  gen-pass improvement OR clean negative result (Session 3).
- Memory entry capturing the comparison vs Phase 5/6 Session 1 baselines.

If positive: the codebase has a way to spend cargo more efficiently
in self-improve. If negative: K9's slot space is too narrow for
even a learned critic to add information beyond what the LM's own
logits already encode (Session 2's AUC test catches this); the
self-improve loop is doing as well as it can at this scale and
further gains require qualitative changes (different challenges,
larger model, or different verifier).

## Non-goals

- **Reinforcement learning of any kind.** No PPO, no policy
  gradients. The critic informs which candidates to send to cargo;
  it does not directly shape the generator's loss.
- **Adversarial co-training.** As above — generator's loss is
  always against cargo via the curator, never against the critic.
- **Critic for non-Rust domains.** ArithmeticDomain and Korean
  already have heuristic verifiers; the critic only earns its
  keep where the verifier is expensive (cargo). Other domains
  could pick this up later but aren't in scope for Phase 6.
