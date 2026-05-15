# Phase 21 Stage G — RL with pass@k-style reward

Closes the last deferred item on the Phase 21 roadmap. Stage G ships
**REINFORCE policy-gradient training infrastructure** on the Pekko
actor stack, with rewards derived from the same verifier the eval
pipeline uses. Optimizes directly against an inference-time objective
("does the generated completion verify?") instead of the SFT proxy
("match this teacher demonstration").

## What's in this commit

### `qwen2_lora::train_qwen_lora_pg_step`

```rust
pub fn train_qwen_lora_pg_step(
    model: &mut ModelForCausalLM,
    optimizer: &mut AdamW,
    device: &Device,
    samples: &[(Vec<u32>, Vec<u32>, f32)],  // (prompt_ids, completion_ids, reward)
) -> Result<f32>
```

For each `(prompt_ids, completion_ids, reward)`:
1. Concat `[prompt | completion]` → `(1, P+C)`.
2. `forward_train` → logits `(1, P+C, V)`.
3. Slice positions `P−1 .. P−1+C` — those are the logits used to
   predict each completion token.
4. `mean_ce = cross_entropy(pred_logits, completion_ids)` ≈ `−mean log P`.
5. Sample loss contribution: `reward * mean_ce`.

Aggregate loss = mean of per-sample contributions. Minimizing it
ascends `reward_i * log P(comp_i | prompt_i)` — the REINFORCE
objective. The bwd-supporting `log_softmax → nll → cross_entropy`
path means gradient flows back to the LoRA Vars via the
Stage F-fixed forward chain (rope_slow / ops::softmax / rms_norm_slow).

### `QwenTrainerMessage::TrainPolicyGradient`

```rust
TrainPolicyGradient {
    samples: Vec<(Vec<u32>, Vec<u32>, f32)>,
    reply: oneshot::Sender<Result<f32>>,
}
```

One message = one AdamW step on the trainer's LoRA Vars. Use repeated
sends for multi-step RL.

### `phase21_g_smoke` — RL loop end-to-end on Pekko

```
Spawn 2 actors: QwenModelActor (F16 inference) + QwenTrainerActor (F32 LoRA)

For each RL step:
  For each prompt:
    For k samples:
      QwenModelActor :: ModelMessage::GenerateTokens(prompt, cfg)
                                → completion ids
      Domain::verify(prompt, decode(comp_ids))
                                → verdict ∈ {0, 1}
    RLOO baseline = mean(verdicts_for_prompt)
    reward_i = verdict_i − baseline

  QwenTrainerActor :: TrainPolicyGradient { samples: (p, c, r)... }
                                → one AdamW step
```

Output (3 RL steps × 3 prompts × k=2):
```
[Phase21G] 2 actors spawned (model + trainer)
[Phase21G] rl_step 0  loss = +0.0183  pass@1-of-k = 5/6  per-prompt = [2, 2, 1]
[Phase21G] rl_step 1  loss = -0.0467  pass@1-of-k = 5/6  per-prompt = [2, 1, 2]
[Phase21G] rl_step 2  loss = -0.0093  pass@1-of-k = 5/6  per-prompt = [1, 2, 2]

phase21_g_smoke: PASS
```

Loss fluctuates around 0 because the RLOO baseline equals the
per-prompt verdict mean — most samples on this trivial domain
verify, so baseline ≈ 1.0 and rewards collapse to 0. The
**infrastructure works end-to-end** — that's the Stage G deliverable.
Real RL gains require a non-saturated domain (HumanEval / MBPP) where
baseline < 1.

## Memory note

Initial smoke OOMed at `k_per_prompt=4, max_new_tokens=24` —
12 samples × ~40 tokens of forward+backward activations in F32 on
0.5B Qwen blew past the GPU memory budget. Reduced to
`k_per_prompt=2, max_new_tokens=16` (6 samples × ~30 tokens) and it
fits. For larger batches the trainer should accumulate gradient
step-by-step rather than concatenating the whole batch's loss graph.
Deferred.

## Acceptance — all pass

- ✅ `cargo build --workspace --release` clean
- ✅ `cargo build --workspace --examples --release` clean
- ✅ `cargo test --workspace --release`: **145 tests** (no change)
- ✅ `cargo fmt --all --check` clean
- ✅ `cargo clippy --workspace --all-targets -- -D warnings` clean
- ✅ E2E `phase21_g_smoke` PASS — 3 RL steps via
  `TrainPolicyGradient`, each followed by a fresh sampling round

## What this commit does NOT cover

- **On-policy correctness**. Samples come from `QwenModelActor` (the
  inference actor with no LoRA updates) while gradient updates apply
  to `QwenTrainerActor`'s LoRA. Their policies diverge as training
  proceeds. Importance-weight correction (PPO-style) is the textbook
  fix; deferred.
- **Adapter sync after training**. To make subsequent sampling
  reflect the trained policy, call `SaveMergedCheckpoint` →
  `ModelMessage::ReloadCheckpoint` (the Stage E.next.next +
  Stage H mechanism). The smoke skips this because the deliverable
  is the RL step itself.
- **Real benchmark signal**. Trivial domain (`contains "return"`)
  saturates → RL signal degenerates. HumanEval / MBPP needed.
- **PPO / GRPO / RLHF KL term**. Pure REINFORCE only.
- **Reward shaping beyond verdict**. The reward IS the verdict
  (1 if verifies, 0 if not), baseline-subtracted. Phase 17 S6's
  pass@k connection: if you sample k completions and reward each
  by its verdict, the model's expected pass@1 grows toward pass@k.

## Phase 21 stage roadmap — fully closed

| stage | scope | status |
|---|---|---|
| A | Pass@k actor infra | ✅ (`7a5d18b`) |
| C | `run_multi_round` helper | ✅ (`f09d97d`) |
| D | Candle-native Qwen2 inference | ✅ (`acfdc5d`) |
| F | Candle-native Qwen2 LoRA training standalone | ✅ (`8367d2a`) |
| B | Substrate scale-up + pass@k lift | ✅ (`8b925a1`) |
| E | Generic Evaluator/Generator | ✅ (`3ead388`) |
| E.next | QwenTrainerActor | ✅ (`ec00809`) |
| E.next.next | Multi-actor multi-round demo + merge | ✅ (`8105007`) |
| H | TrainerHandle trait + supervisor wiring | ✅ (`b534679`) |
| **G** | **RL with pass@k reward** | ✅ (this commit) |

All 10 stages closed. The Pekko bridge is the **complete** Phase 17-20
Python recipe substrate plus an RL extension on top.

## Files

- `llm-actors/src/qwen2_lora.rs` — `train_qwen_lora_pg_step` helper
- `llm-actors/src/qwen_trainer_actor.rs` — `TrainPolicyGradient` message
- `llm-actors/examples/phase21_g_smoke.rs` — E2E RL loop (new)
- `llm-actors/Cargo.toml` — example registration
- `docs/phase21-stage-g.md` (this)

## See also

- `docs/phase21-stage-h.md` — full supervisor pipeline against Qwen
- `docs/phase21-stage-f.md` — Candle 0.10 no_bwd gotcha + SFT step
- `docs/phase20-deployment-recipe.md` — Pareto front this RL extension
  could push past
