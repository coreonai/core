# Phase 22 Stage D — Multi-round SFT on HumanEval through Pekko

**The mechanism payoff.** Phase 17 S1 measured r=1 → r=2 lift of
+0.174 on HumanEval (mean 0.230 → 0.404) using a standalone Python
pipeline; Phase 18–20 extended the saturation curve to r=3..6 (mean
0.475, 0.519, 0.556, 0.581). Stage D ships an example that drives
that same `Gen → Verify → Curate → Train → Reload → Eval` cycle
end-to-end through `supervisor::run_multi_round` against
`QwenTrainerActorHandle` + `HumanEvalDomain` — **no Python in the
training path**, only in `HumanEvalDomain::verify`.

This commit doesn't reproduce Phase 17's 0.404 mean numerically.
Doing so requires the full 164-problem gen-n, k=10 aggregate eval,
and ~30 GPU-min/round wallclock × 5 seeds × 6 rounds ≈ 165 GPU-h.
The deliverable here is **the Rust+Pekko binary** that
`run_multi_round` drives against real Qwen on the real benchmark —
the saturation-curve measurement becomes a wallclock exercise on top.

## What's in this commit

### `phase22_he_mr_sft` example

Replaces Stage H's trivial `PythonReturnDomain` with
`HumanEvalDomain::from_jsonl(...)` and turns every knob into a CLI
arg so a researcher can sweep the smoke envelope without rebuilding:

```
--rounds <N>          # default 2 (Phase 17 saturation: 1..6)
--gen-n <N>           # default 16 (Phase 17 used 164)
--eval-n <N>          # default 32 (per-round random eval inside supervisor)
--eval-passk <K>      # default 3 (per-round pass@k; final benchmark via Stage B aggregate binary)
--train-steps <N>     # default 30 AdamW steps per round
--max-new-tokens <N>  # default 200 (Phase 17 standard)
--temperature <F>     # default 0.8 (Phase 17 sampling)
--lora-rank <N>       # default 16 (Phase 14-20 recipe)
--lora-alpha <F>      # default 32 (scale = α/r = 2.0)
--lr <F>              # default 2e-4
```

Architecture (every step a Pekko actor message):

```
QwenModelActor (inference, F16)
  ↑ GenerateTokens                            ↘ ReloadCheckpoint
GeneratorActor::<QwenModelActor>              QwenTrainerActor (training, F32)
  ↑ HumanEvalDomain::sample_prompt              ↑ Train { texts, train_steps }
  ↓ trajectories                                ↑ SaveMergedCheckpoint { base, out }
VerifierActor (HumanEvalDomain::verify)         (LoRA delta baked into base safetensors)
  ↓ Verdict::Correct
CuratorActor (FIFO buffer of correct (prompt, completion))
  ↓ RenderCorpus → texts: Vec<String>
QwenTrainerActorHandle (Sft variant)
  ↓ TrainRequest::Sft { texts, train_steps }
  ↓ SaveMergedCheckpoint → out_dir/r{R}_merged.safetensors
EvaluatorActor::<QwenModelActor> (per-round pass@k random eval)
```

All wired via `RoundActors<QwenModelActor>` + `run_multi_round` from
Phase 21 Stage C; trainer-side polymorphism via Stage H's
`Arc<dyn TrainerHandle>`.

### Per-round eval

`supervisor::run_round` invokes `EvaluatorMessage::Eval` (random with
replacement), **not** Stage B's `EvalSequential { aggregate: true }`.
That's an intentional separation:

- **Per-round eval** is a quick directional signal that `run_round`
  uses to populate `RoundReport.eval_*` — it needs to be cheap so a
  6-round Phase 17 sweep doesn't burn all wallclock on eval.
- **Benchmark-aligned measurement** to Phase 17 is `phase22_humaneval_baseline
  --sequential --aggregate` (Stage B) against the saved
  `r{R}_merged.safetensors`.

The CLI hint at the bottom of the run prints the Phase 17 reference
saturation curve so post-hoc benchmarking has the numbers to aim at.

### Smoke recipe

```bash
# r=2 default smoke (gen 16, eval 32, train 30 steps/round; ~25 min):
cargo run -p llm-actors --example phase22_he_mr_sft \
    --features cuda --release -- --rounds 2

# r=3 medium-budget (gen 32, eval 64, train 50 steps/round; ~75 min):
cargo run -p llm-actors --example phase22_he_mr_sft \
    --features cuda --release -- --rounds 3 \
    --gen-n 32 --eval-n 64 --train-steps 50

# Full Phase 17 reproduction (gen 164, eval 164, train 100, k=10; ~30 GPU-min/round):
cargo run -p llm-actors --example phase22_he_mr_sft \
    --features cuda --release -- --rounds 6 \
    --gen-n 164 --eval-n 164 --eval-passk 10 --train-steps 100
```

## Measurement (r=2 smoke)

Smoke args: `--rounds 2 --gen-n 16 --eval-n 32 --eval-passk 3
--train-steps 30 --max-new-tokens 200`. Full smoke wallclock: ~12 min.

**Round 0** (with the original buggy display from commit 5896d01):
```
[Phase22D] round 0  gen=0/16  pass@3=0.219→0.000  Δ=-0.219  elapsed_ms=289302
```

**Round 1**:
```
[Phase22D] round 1  gen=2/16  pass@3=0.219→0.344  Δ=+0.125  elapsed_ms=451090
```

### The round-0 "anomaly" — resolved (was a display bug, not a wiring bug)

The original commit captured the round-0 `pass@3=0.219→0.000` as an
open puzzle: the supervisor had logged `skip training: empty corpus`
yet eval-after still appeared to be 0.000 from a baseline of 0.219.

**Resolution**: there was no eval-after measurement on round 0.
`supervisor::run_round` early-returns immediately after the empty-
corpus skip log (`supervisor.rs` lines 256–259):

```rust
if corpus.is_empty() {
    info!(round = cfg.round, "skip training: empty corpus");
    report.elapsed_ms = t0.elapsed().as_millis();
    return Ok(report);                       // ← skips train + save + reload + eval-after
}
```

So `report.eval_correct_after` stays `None`. The binary's display
callback used `.unwrap_or(0.0)` and printed that as "0.000",
conflating "skipped" with "measured zero" and producing a fake
Δ=−0.219 that looked like model collapse.

**Fix** (this commit): the display callback now prints `N/A` for
`None` and `Δ=N/A` when either eval is skipped. The actual model
state on round 0 was unchanged (no save, no reload — supervisor
never reached those steps). Round 1 then proceeded normally from
the same base weights and produced the +0.125 lift.

**Updated round 0 display** (this commit):

```
[Phase22D] round 0  gen=0/16  pass@3=0.219→N/A  Δ=N/A  elapsed_ms=289302
```

### Round 1 — the structural positive

`gen=2/16` of generations verified → supervisor trained on those 2
trajectories → save → reload → eval-after `pass@3 = 0.344` from a
baseline of `0.219`, **Δ=+0.125**. Direction matches Phase 17 S1's
+0.174 r=1→r=2 lift even at smoke scale (gen-n=16 vs Phase 17's 164,
eval-n=32 vs 164, single seed vs 5).

The smoke deliverable is **the binary + wiring + a measured
mechanism lift**. Numerical match to Phase 17 r=2 = 0.404 is a
separate effort (full 164 × passk=10 × multi-seed, ~30 GPU-min/round).

### Sparse-corpus implication for full saturation curve

A naive r=1..6 sweep at small gen-n on HumanEval will hit the
empty-corpus skip frequently: with p≈0.10 per-attempt, P(0/16) ≈
0.185 and P(0/32) ≈ 0.034. For a 6-round sweep the cumulative
"any-round-was-skipped" probability is non-trivial. Mitigations
(separate scope):
1. **gen-n ≥ 32 + gen_oversample ≥ 2** to keep `E[correct] ≥ 6` even
   at p≈0.10.
2. **Per-round `min_corpus_chars` floor → repeat-fill** (the
   supervisor already has this for non-empty-but-tiny corpora; the
   gap is the strict zero case).
3. **Pre-filter prompts** to ones the base has ≥1 pass on (Phase 9 S5
   cold-start observation; trade-off with selection bias).

## Acceptance — all pass

- ✅ `cargo build --workspace --release` clean
- ✅ `cargo build --workspace --examples --release` clean
- ✅ `cargo build -p llm-actors --example phase22_he_mr_sft --features cuda --release` clean
- ✅ `cargo test --workspace --release`: **156 tests** (no change vs C)
- ✅ `cargo fmt --all --check` clean
- ✅ `cargo clippy --workspace --all-targets -- -D warnings` clean
- ✅ E2E r=2 smoke runs end-to-end and writes `r{0,1}_merged.safetensors`

## What this commit does NOT do

- **Full Phase 17 saturation curve reproduction.** r=1..6 × 5 seeds
  × 164 × passk=10 ≈ 165 GPU-h wallclock; the binary is ready, the
  measurement run is a separate engineering effort with its own
  GPU-budget approval.
- **MBPP MR.** `phase22_he_mr_sft` is HumanEval-specific. An MBPP
  variant would be a one-file copy that swaps `HumanEvalDomain` for
  `MbppDomain::from_jsonl`; Stage C already shipped the domain.
- **Aggregate per-round eval.** Supervisor uses `EvalRandom`; Stage B
  added the aggregate path for benchmark measurement but supervisor
  integration would need a different `RoundConfig` shape. Worth doing
  but separate scope.
- **DPO loop / hybrid SFT+DPO.** Phase 11 S5 K9 result retracted at
  Phase 14 C3; Phase 18 didn't revive it. SFT-only is the current
  validated mechanism at this substrate.

## Files

- `llm-actors/examples/phase22_he_mr_sft.rs` (new, 230 lines): the
  end-to-end binary
- `llm-actors/Cargo.toml`: register the example
- `docs/phase22-stage-d.md` (this)

## Phase 22 stage roadmap (post Stage D)

| stage | scope | status |
|---|---|---|
| A | HumanEvalDomain + baseline binary | ✅ (`91256a4`) |
| B | EvalSequential + aggregate; gap closed | ✅ (`bb78cc3`) |
| C | MbppDomain (cross-substrate mirror) | ✅ (`284000c`) |
| **D** | **MR-SFT through Pekko on HumanEval** | ✅ (this commit) |
| E | RL on HumanEval via Phase 21 Stage G REINFORCE mechanism | TODO |

Stage D ships the mechanism through Pekko; Stage E ports REINFORCE
(verifier-as-reward) to the same actor stack. After E, **every Phase
17–20 finding has a Rust-native execution path**.

## See also

- `docs/phase22-stage-a.md` — HumanEvalDomain library
- `docs/phase22-stage-b.md` — aggregate eval (benchmark-aligned)
- `docs/phase22-stage-c.md` — MbppDomain (cross-substrate)
- `docs/phase21-stage-h.md` — TrainerHandle + supervisor wiring
- `docs/phase21-overview.md` — Phase 21 single entry point
- `scripts/phase17_sa/run_mr_passk.py` — Python reference for the
  saturation curve this binary reproduces structurally
