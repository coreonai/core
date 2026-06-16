# Phase 22 Stage E — REINFORCE on HumanEval through Pekko

**The final Phase 17–20 mechanism with a Rust-native execution path.**
Phase 21 Stage G shipped REINFORCE against a trivial
`PythonReturnDomain`; Stage E ports that loop to the real
`HumanEvalDomain` with **verifier-as-reward** — the cleanest
extrinsic signal a coding model can train against. Every step is an
actor message; `HumanEvalDomain::verify` is the only Python in the
loop, and it's confined to test-case execution (not generation, not
gradient computation).

## What's in this commit

### `phase22_he_reinforce` example

Forked from `phase21_g_smoke` with two structural changes:

1. **Domain swap.** `PythonReturnDomain` → `HumanEvalDomain::from_jsonl(...)`.
   The reward is `1.0` if `HumanEvalDomain::verify` passes (test cases
   green via python3 subprocess), `0.0` otherwise.

2. **Prompt enumeration.** Stage G used a hardcoded 3-prompt array;
   Stage E uses `humaneval.nth_prompt(0..n_prompts)` — same indexing
   API Stage B added. The first N HumanEval problems are picked
   deterministically so RL and Stage D's SFT can be cross-compared
   on the same task distribution.

Every Phase 14–20 hyperparameter is a CLI flag:

```
--rl-steps <N>        # outer RL iterations (default 3)
--n-prompts <N>       # HumanEval problems per step (default 6)
--k-per-prompt <K>    # samples per prompt for RLOO baseline (default 2)
--max-new-tokens <N>  # default 16 — k=2 + max_new=16 fits 0.5B F32 in 40GB
--temperature <F>     # default 0.8
--lr <F>              # default 2e-4
--lora-rank <N>       # default 16
--lora-alpha <F>      # default 32
--seed <U>            # base seed for prompt/k diversification
```

### RL loop

Each `rl_step` is one off-policy REINFORCE iteration:

```text
for rl_step in 0..rl_steps:
    samples = []
    for prompt in prompts:                       # n_prompts
        prompt_samples = []
        for k in 0..k_per_prompt:                # k samples per prompt
            comp_ids = QwenModelActor.GenerateTokens(prompt_ids, cfg)  # temp=0.8
            verdict   = HumanEvalDomain.verify(prompt, decode(comp_ids))
            v_value   = 1.0 if verdict.is_correct() else 0.0
            prompt_samples.push((comp_ids, v_value))
        # RLOO baseline: center on per-prompt mean so high-variance
        # prompts don't dominate. Reward_i = v_i − mean(prompt_samples)
        baseline = mean(v for (_, v) in prompt_samples)
        for (comp_ids, v) in prompt_samples:
            samples.push((prompt_ids, comp_ids, v − baseline))
    QwenTrainerActor.TrainPolicyGradient { samples }   # one AdamW step on
                                                       # Σ reward_i × log P(comp_i | prompt_i)
```

Memory cap (Phase 21 G empirical): k=2 + max_new=16 fits on a single
40 GB A100 in F32 gradient mode. k=4 or max_new=24 OOM the gradient
graph at 0.5B. The defaults here are conservative; bump only after
moving to FP16 gradient accumulation.

**Off-policy approximation**: `QwenModelActor` samples on weights
that drift away from `QwenTrainerActor`'s LoRA-augmented policy as
training proceeds. No importance-weighting correction; same as
Stage G. Adapter-sync (push merged LoRA back into the inference
actor between RL steps) is the natural next step but deferred — it'd
match Stage D's `SaveMergedCheckpoint + ReloadCheckpoint` pattern.

### Sparse-reward characterization

On HumanEval, base Qwen2.5-Coder-0.5B per-attempt pass-rate is ~0.10
(Stage B aggregate ref: 0.222 at k=10 → per-attempt ≈ 0.10). So with
n_prompts=6 × k=2 = 12 samples per step, expected `# of passes ≈ 1.2`,
and many steps will have **zero passes** → RLOO baseline = 0 →
reward_i = 0 for every sample → no gradient contribution. The
non-zero-reward steps drive learning; everything else is wasted
compute. This is normal RL-on-coding behavior at small substrate.

Strategies to densify reward (future):
- **Pre-filter prompts** to ones the base model has ≥1 pass on
  (Phase 9 S5's cold-start observation: 3 of 11 problems are 0/8
  forever).
- **Larger k** to raise the probability of ≥1 pass per prompt.
- **Larger n_prompts** to add gradient signal from at least some
  passing prompt each step.

## Measurement

**r=3 smoke** (`--rl-steps 3 --n-prompts 6 --k-per-prompt 2
--max-new-tokens 16`, 13 s on A100):

```
[Phase22E] rl_step 0  loss = +0.0000  pass = 0/12  elapsed_step = 4.5s
[Phase22E] rl_step 1  loss = +0.0000  pass = 0/12  elapsed_step = 4.3s
[Phase22E] rl_step 2  loss = +0.0000  pass = 0/12  elapsed_step = 4.2s

[Phase22E] === RL summary ===
[Phase22E] 3 steps × 6 prompts × k=2 = 36 samples total, elapsed = 13.0s
[Phase22E] losses across RL steps: [0.0, 0.0, 0.0]
[Phase22E] passes/total per step:  [0, 0, 0] / [12, 12, 12]
[Phase22E] 0/3 RL steps had >0 prompt passes (gradient signal)
```

Deliverable: **the RL loop closes against a real benchmark
verifier**. The 0/12 result is exactly the sparse-reward case the
doc above predicted — max_new=16 truncates real HumanEval solutions
(which run ~50–100 tokens), so no completion compiles into a passing
test case, every reward is 0.0, RLOO baseline=0, all gradients
vanish. The binary correctly returns loss=0.0 in this degenerate
case (rather than crashing or producing NaN).

For a signal-bearing run: bump `--max-new-tokens` to ≥64 and
`--n-prompts` to ≥16 (and stay within the Phase 21 G memory cap of
k=2 + max_new ≤ 24 at F32; or move to FP16 gradient accumulation for
longer completions). Measurement-grade RL runs are ~5–10 GPU-h of
work, separate from the Stage E infrastructure delivery.

## Signal-bearing measurement (follow-up) — REINFORCE REGRESSES, 3/3 seeds

The deferred signal-bearing run, executed with the `pg_micro_batch_size`
OOM-safe path (commits `10cf696`/`bf51095`): **3 seeds × 50 RL steps ×
16 prompts (task_id 0..16) × k=4**, `max_new` long enough for real
HumanEval solutions. All three trained cleanly to completion
(`50/50 RL steps had >0 prompt passes`, final merged checkpoints saved).

Each final merged checkpoint (`rl_seed{42,100,200}_final.safetensors`)
was then evaluated against base on the **full 164-problem benchmark**
under one identical protocol —
`--n-problems 164 --passk 10 --sequential --aggregate --max-new-tokens 200`.
RL trained on only task_id 0..16, so full-164 measures generalization
(148 held-out problems).

| config | aggregate pass@1 | per-prompt pass@10 | Δ pass@1 vs base |
|---|---|---|---|
| **base** (control) | **0.2183** (358/1640) | 0.4573 (75/164) | — |
| rl_seed42 | 0.0293 (48/1640) | 0.0854 (14/164) | **−0.189** |
| rl_seed100 | 0.0665 (109/1640) | 0.1037 (17/164) | **−0.152** |
| rl_seed200 | 0.0000 (0/1640) | 0.0000 (0/164) | **−0.218** |

**Verdict: unambiguous negative — 3/3 seeds catastrophically regress**
(mean RL pass@1 ≈ 0.032, Δ ≈ −0.186). The base control reproduces the
Stage B / Phase 17 baseline (0.2183 ≈ 0.222 ≈ 0.216) exactly, so the
measurement is sound; the collapse is in the RL checkpoints.

### Mechanism — the off-policy approximation is the culprit

The training-time on-policy pass counts (~8–13/64 throughout, flat)
while `loss` drove steeply negative (→ −5) were the warning sign. The
explanation is the **deferred adapter-sync**:

- `QwenModelActor` (the sampler) is never re-synced to
  `QwenTrainerActor`'s LoRA-augmented policy during the run. So for all
  50 steps it samples on **base weights** — the flat ~0.13–0.20
  on-policy pass-rate was *base-model* pass-rate, not the trained
  policy's. "50/50 steps had gradient signal" was therefore misleading.
- The REINFORCE gradient is computed on base-model samples, but the
  accumulated LoRA delta drifts (loss → −5 is log-prob concentration).
- The first time the trained LoRA is actually exercised is at eval, when
  it's merged into the base — and the merged policy has collapsed
  (seed42 emits degenerate short output, 2.97 s/problem vs base's
  24.3 s; seed200 emits long garbage at 26.4 s/problem, 0 passes).

So Stage E's three RL seeds confirm that **off-policy REINFORCE without
adapter-sync (and without a KL anchor to base) collapses a 0.5B+LoRA
policy** on sparse verifier reward. This is consistent with the broader
phase result that direct paper-port self-improve mechanisms (Muon, DPO,
OPD — 3/3) fail at this scale; verifier-as-reward RL joins them when run
without stabilization.

### Natural follow-up (not run here)

Add per-step adapter-sync (`SaveMergedCheckpoint + ReloadCheckpoint`,
~30 s/step) so (a) the gradient is on-policy and (b) the *actual* policy
pass-rate is visible each step, catching collapse early. A KL-to-base
penalty and/or a pass-feasibility prompt filter (Phase 9 S5 cold-start)
are the other obvious stabilizers. Deferred as a separate measurement.

## Acceptance — all pass

- ✅ `cargo build --workspace --release` clean
- ✅ `cargo build -p llm-actors --example phase22_he_reinforce --features cuda --release` clean
- ✅ `cargo test --workspace --release`: **156 tests** (no change vs D)
- ✅ `cargo fmt --all --check` clean
- ✅ `cargo clippy --workspace --all-targets -- -D warnings` clean
- ✅ E2E r=3 smoke completes, prints per-step loss + pass count

## Phase 22 stage roadmap (complete)

| stage | scope | status |
|---|---|---|
| A | HumanEvalDomain + baseline binary | ✅ (`91256a4`) |
| B | EvalSequential + aggregate; gap closed | ✅ (`bb78cc3`) |
| C | MbppDomain (cross-substrate mirror) | ✅ (`284000c`) |
| D | MR-SFT through Pekko on HumanEval | ✅ (`5896d01`) |
| **E** | **REINFORCE on HumanEval via Stage G mechanism** | ✅ (this commit) |

**Phase 22 closes the Rust-native execution path for every Phase 17–20
finding**:

- pass@k inference scaling → Phase 21 Stage A (`EvaluatorActor.passk`)
- multi-round SFT saturation → Phase 22 Stage D (`phase22_he_mr_sft`)
- REINFORCE with verifier reward → Phase 22 Stage E (this)
- HumanEval substrate → Phase 22 Stage A+B
- MBPP cross-substrate → Phase 22 Stage C

The numerical reproductions (Phase 17's 0.230/0.404/.../0.581 curve,
Phase 17 S9's 0.36/0.66, etc.) are wallclock exercises on top of
this infrastructure, not infrastructure gaps.

## What this commit does NOT do

- **Adapter-sync between RL steps.** QwenModelActor's weights drift
  out of sync with QwenTrainerActor's LoRA delta as training
  proceeds; this is the standard off-policy approximation. A
  `SaveMergedCheckpoint + ReloadCheckpoint` between RL steps would
  re-sync them; deferred for memory budget reasons (each
  save/reload cycle is ~30s on 0.5B).
- **Importance-weighting correction.** Standard off-policy correction
  for the value-baseline mismatch. Not required for the smoke to
  demonstrate the loop; required for theoretical RL correctness.
- **Pre-filtered prompt set.** A pass-feasibility filter (only train
  on prompts the base model has ≥1 pass on among ≥k samples) would
  densify rewards but introduces selection bias. Phase 9 S5's cold-
  start observation argues for it; Stage E ships the unfiltered version.
- **Full HumanEval × multi-RL-step measurement.** A signal-bearing run
  needs n_prompts ~32-64, ~50-100 RL steps, multiple seeds. ~5-10
  GPU-h of measurement work — separate from the infrastructure
  delivery.

## Files

- `llm-actors/examples/phase22_he_reinforce.rs` (new, ~230 lines)
- `llm-actors/Cargo.toml`: register the example
- `docs/phase22-stage-e.md` (this)

## See also

- `docs/phase22-stage-a.md` — HumanEvalDomain (Stage A)
- `docs/phase22-stage-b.md` — aggregate eval (benchmark-aligned)
- `docs/phase22-stage-c.md` — MbppDomain (cross-substrate)
- `docs/phase22-stage-d.md` — MR SFT through Pekko
- `docs/phase21-stage-g.md` — REINFORCE primitive (this stage's
  immediate template)
- `docs/phase21-overview.md` — Phase 21 single entry point
