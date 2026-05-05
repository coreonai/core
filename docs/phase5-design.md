# Phase 5 design — multi-actor agentic interaction

This is a planning document. Nothing here is built yet. Phases 1–4
plus the Phase 1 epilogue ship single-agent self-improvement on top
of the pekko-rust actor framework — `ModelActor` × 1 owns the
weights and `GeneratorActor` / `VerifierActor` / `CuratorActor` /
`TrainerActor` / `EvaluatorActor` orchestrate around it. Phase 5's
job is to make the **plurality of `ModelActor`s** the load-bearing
design choice. If a future session boils Phase 5 down to "spawn N
copies of the existing loop and concat the buffers," that's not
actually using the actor model.

## Why multi-actor

The infrastructure has been a single-agent vehicle so far:

- Phase 3's NAS spawns many models in parallel, but only to **rank**
  them. Variants don't talk to each other.
- Phase 4's `AgenticGeneratorActor` runs an agent loop, but the agent
  itself is one model talking to a tool registry — the "multi-actor"
  is just (model, tool executor).
- The K9 self-improve loop is verifier-as-environment: model emits,
  cargo grades, model trains. Single feedback loop.

What's load-bearing about pekko-rust isn't **scaling** the same
computation to many GPUs (Python + multiprocessing handles that
fine). It's **independent failure domains** and **typed message
protocols between long-lived stateful peers**. Phase 5 should
demonstrate something that's awkward without those — multiple
models with different histories, weights, and policies, exchanging
typed messages, none of them trusted, and none of them in lockstep.

## Three candidate shapes (pick one)

### Shape A — N-actor ensemble consensus on a verifier

`N` `ModelActor`s, each with its own `VarMap` and (optionally) its
own architecture from a Phase-3 NAS variant. Per round:

1. **Sample**: each model independently generates `K` trajectories
   for the same prompt set.
2. **Verify**: cargo (`RustCodeDomain`) judges each trajectory.
3. **Score by consensus**: a trajectory's curator weight is the
   **product** of (cargo says correct) × (number of models that
   produced this exact slot) ÷ N. Single-model lucky guesses get
   downweighted; trajectories N − 1 of N agree on get heavy weight.
4. **Train**: all `N` models train on the same shared curator
   (consensus-weighted priority sampling).
5. **Eval**: each model evaluated independently; report per-model
   eval pass-rate so we see whether consensus training keeps the
   ensemble homogenized or whether some models drift.

What's interesting:
- Failure mode of single-agent K9 (greedy collapses onto one slot)
  is broken — different models will collapse onto different slots,
  and the consensus filter discards the per-model fixed points
  that the others didn't independently arrive at.
- The actor framework earns its keep here: each `ModelActor` owns
  its varmap, has its own checkpoint path, can `ReloadCheckpoint`
  independently. There's no shared mutable state.
- Need a new `ConsensusScorerActor` (or extend `VerifierActor`)
  that takes `Vec<(Trajectory, Verdict)>` from N generators and
  emits weighted `VerifiedTrajectory`s.

What we'd measure:
- Does ensemble eval pass-rate exceed best-individual?
  (Naïve baseline: pick the strongest model and ignore the rest.)
- Does the gen-phase 37.5% rate generalize *across N models*?
  We'd hope ensemble gen-pass climbs noticeably above any single
  model's solo run.
- Does consensus prevent the round-1 transient we saw in K9 v3?

### Shape B — Specialist routing

One **generalist** (large) `ModelActor` plus `M` **specialist**
(small) `ModelActor`s. Each specialist is trained on one
`Domain` only; the generalist learns to route prompts.

- Per prompt: generalist emits a "delegate to specialist X" header
  → that specialist generates the actual completion.
- Verifier judges; both generalist (for routing accuracy) and the
  picked specialist (for completion quality) get reward.
- Curator stores `(prompt, chosen_specialist, completion)` triples;
  trainer updates routing-only on generalist, content-only on
  specialist.

What's interesting:
- The actor framework's typed message protocol naturally encodes
  the routing decision (`RouteMessage { to: ActorRef<...> }`).
- Could compose with Phase 3 NAS: each specialist is a different
  evolved architecture, picked for its `Domain`-specific fitness.
- Different from Shape A: parallelism is over **domains**, not over
  **per-prompt sampling**.

What we'd measure:
- Does specialization beat a single generalist on the same total
  parameter budget?
- Routing accuracy: how often does the generalist pick the
  specialist that the verifier ends up rewarding?
- Capacity: does adding the M+1th specialist help, or saturate?

### Shape C — Adversarial generator/critic co-evolution

Two `ModelActor`s: a **generator** trained to produce trajectories
the **critic** can't reliably reject, and a critic trained to score
trajectories that match a held-out cargo-verified label set.

- Generator's reward = +1 if cargo accepts AND critic also accepts.
- Critic's reward = matching cargo's verdict on a held-out batch.
- Both train alternately. Curriculum: critic gets a head start so
  generator has something to push against.

What's interesting:
- The verifier is no longer a fixed external program — it's a
  learning peer. Cargo stays in the loop as ground truth, but the
  critic is what the generator interacts with most often (cargo is
  expensive; critic is cheap).
- Failure modes worth watching: generator finds a slot the critic
  accepts but cargo rejects (adversarial example); critic learns to
  reject everything (collapse).

What we'd measure:
- Critic accuracy vs cargo on held-out trajectories.
- Generator's gen-pass-rate as critic improves.
- Does the generator's pass rate plateau higher with a learned
  critic than with cargo alone? (i.e. does the cheap critic let us
  spend more compute on training?)

## Recommendation: start with Shape A

Shape A composes cleanest with what we already have:

| Existing piece | Shape A reuse |
|----------------|---------------|
| `ModelActor` (Phase 1) | spawn `N` of them, one varmap each |
| `RustCodeDomain` (Phase 2.5) | unchanged — same cargo verifier |
| `JoinSet` parallel dispatch (Phase 3 evolution) | run N generators concurrently |
| `CuratorActor` (Phase 2) | extend with consensus-weighted priority |
| `TrainerActor` (Phase 2) | unchanged — train each model on shared corpus |

Shape B requires designing a new typed routing message and per-actor
training schedules — at least a week of work. Shape C requires a
critic architecture decision (regression head? separate small
transformer?) and risks the standard adversarial-collapse failure
modes; not a smoke-friendly first pass.

## Concrete next steps for Shape A (~3–5 sessions of work)

### Session 1 — Plumbing

- New module `llm-actors/src/ensemble.rs`:
  - `EnsembleConfig { models: Vec<GPTConfig>, init_paths: Vec<PathBuf> }`
  - `EnsembleActors { models: Vec<ActorRef<ModelActor>>, ... }`
  - Helper: `ensemble_generate(prompts, samples_per_model)`
    returns `Vec<Vec<Trajectory>>` indexed by [model][sample].
- Tests: 2-model ensemble using two different `nano_*` presets,
  smoke that they produce independent trajectories.

### Session 2 — Consensus curator

- `CuratorActor::Add` already accepts `VerifiedTrajectory`. Add a
  new variant `CuratorMessage::AddEnsemble` that takes
  `Vec<(Trajectory, Verdict, model_id)>` and:
  - Groups by exact-string `(prompt, completion)`.
  - Filters to those where ≥ floor(N/2) models produced the same
    pair AND cargo says correct.
  - Sets weight = (matching models / N) × cargo correctness × 1.0.
- Test: ensemble of 3, 2 of 3 produce `"hello"` for the
  string_len prompt → accepted with weight 2/3.

### Session 3 — Training round

- `examples/self_improve_ensemble_rust.rs`:
  - Pretrain N models from same seed corpus (deterministic per
    `--seed-offset`).
  - Run 3–4 rounds of (parallel-generate × consensus-curate ×
    train-each-on-shared-buffer × eval-each).
  - Print per-round per-model eval pass rate, ensemble-weighted
    pass rate, training loss.

### Session 4 — Measurement

Compare against K9 v4 baseline (single model, same total compute):

- For ensemble of N=3 small models: total params = 3 × 1M = 3M.
- Single-model baseline: 3M-param model trained for 3× the steps.
- Apples-to-apples: same compute budget either way. Does the
  ensemble's stochastic gen-pass rate exceed the baseline's?

If yes — Phase 5 unlocks something. If no — Phase 5 is
infrastructure plumbing without a payoff at this scale, which is
also a real result worth memory-entrying.

### Session 5 — Stretch

- Heterogeneous ensemble: models from Phase 3 NAS (different
  architectures, not just different seeds). Test whether
  architectural diversity in the ensemble gives more independent
  failure modes than seed diversity alone.
- LoRA-only fine-tune per model after a shared pretrain — keeps
  individual checkpoints small, ensemble sharing cheap.

## Risks / open questions

1. **Determinism of "same slot" matching.** Char-level
   tokenization makes string-equality the right consensus key. For
   BPE+Korean (where multiple token sequences can decode to the
   same string), need to normalize to the decoded text before the
   set-intersection.

2. **Ensemble size sweet spot.** At N=2 consensus is a single
   match — barely a filter. At N=10 the floor(N/2) threshold is
   stricter, but each model trains on far less data per round
   (most trajectories get filtered). Sweep N ∈ {2, 3, 5, 7} in
   session 4.

3. **Per-model checkpoint costs.** Each `ModelActor` owns a
   `VarMap` + a `.safetensors` per round. N=5 → 5× the disk per
   round. With LoRA-only continual fine-tune (Phase 4 already
   supports this) we can keep each per-round checkpoint at ~MB.

4. **Synchronization point.** All N generators must finish before
   the consensus step. With heterogeneous models that's a
   straggler problem. Phase 3's `JoinSet` round-robin handles it
   — adapt.

5. **Cargo bottleneck.** RustCodeDomain serializes cargo
   invocations behind a `Mutex<()>` in `write_program`. With N
   models each emitting K trajectories per round, that's `N × K`
   sequential cargo runs. Either parallelize (separate scratch dir
   per model) or move to `cargo check` (faster, no binary
   execution). Both feasible; the scratch-dir-per-model approach
   removes the mutex entirely.

## Non-goals for Phase 5

- **Distillation between ensemble members.** Could be Phase 6;
  conflates the consensus learning signal with knowledge
  transfer. Keep them separate.
- **Larger models** (50M+ Korean-style). Phase 5 is about
  multi-actor *infrastructure*; the toy slot domain is a feature,
  not a limitation. The Phase 1 epilogue's 50M KoWiki model can
  later be plugged in as one ensemble member.
- **Production HTTP serving of ensembles.** The axum `serve_inference`
  example today exposes one `InferenceServerActor`. Ensemble
  serving — query routing, fallback, weighted-voting at
  inference — is a separate concern from training-time ensemble
  consensus. Defer.

## What "Phase 5 done" means

- 1 new module (`ensemble.rs`), 1 new example
  (`self_improve_ensemble_rust.rs`), 1 new curator message variant.
- Smoke run of N=3 produces a per-round table comparable to K9 v4.
- Memory entry comparing ensemble gen-pass-rate to the
  single-model 37.5% baseline.
- README example 5c with 5–10 lines on what to expect.

If session 4's measurement is **negative** (ensemble doesn't beat
single-model at same compute), that's a **valid stopping point** —
phase 5 is then a clean negative result documenting "multi-actor
consensus alone isn't enough; need specialization or adversarial
co-evolution," which directs Phase 6 toward Shape B or C.
