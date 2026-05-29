# Phase 22 — Final consolidation (Stage A → Phase-17 reproduction on Pekko)

End-of-Phase-22 narrative summary. Captures the full arc from substrate
setup (Stage A-E) through the catastrophic regression diagnosis
(A-batch → G1 → G2), the recipe-divergence stack (G4-G9), the decisive
fix (G9 completion truncation), and the saturation-curve + cross-substrate
validation that confirms the Rust/Pekko self-evolving loop reproduces
Phase 17 on both HumanEval and MBPP.

## TL;DR — RESOLVED

**The Rust/Pekko self-evolving MR-SFT loop reproduces Phase 17 on both
benchmarks.** Full recipe = `completion-mask + cosine LR + batch=4 padded
SFT + fresh-AdamW/round + non-cumulative harvest + completion truncation
+ top_k=0 harvest`. With it (truncated aggregate, 5-seed):

| | base | r=2 | r=5 / plateau | Phase 17 ref |
|---|---|---|---|---|
| HumanEval | 0.218 | **0.436** | ~0.50 (r≈4-5) | r2 0.404, plateau ~0.55-0.58 |
| MBPP-100 | 0.201 | **0.447** | ~0.50 (r≈3) | r2 0.453, r5 0.541 |

r=2 matches Phase 17 on both; the saturation-curve SHAPE (diminishing
returns → plateau) reproduces; the high-round plateau sits ~0.05 below
Phase 17's. The −0.20 regression that opened Stage D was a STACK of
recipe divergences resolved in order; completion truncation (G9) was the
decisive one.

## Phase 22's purpose

Migrate Phase 17-20's HumanEval/MBPP recipe from Python (HF
Transformers + PEFT) onto Pekko-Rust (Candle-native Qwen2 + LoRA).
The benchmark target: reproduce Phase 17 S1's r=2 HumanEval pass@1 =
0.404 ± 0.013 through `supervisor::run_multi_round` on actor stack.
**Achieved** (G9, r=2 0.436 truncated 5-seed), plus the MBPP
cross-substrate (r=2 0.447 ≈ Phase 17 SB 0.453) and saturation curves
to r=5 on both.

## Stage matrix

| Stage | Scope | Commit | Status |
|---|---|---|---|
| A | HumanEvalDomain (164 problems) | `91256a4` | ✅ |
| B | EvalSequential + aggregate (Phase 17 metric reproduces ≈ within 1σ) | `bb78cc3` | ✅ |
| C | MbppDomain (cross-substrate) | `284000c` | ✅ |
| D | Multi-round SFT through Pekko | `5896d01` + G4-G9 | ✅ **Phase 17 reproduced** (G9 truncation `aaf0594`; HE r=2 0.436, MBPP r=2 0.447; saturation curves to r=5 on both) |
| E | REINFORCE on HumanEval | `eb6da62` | ✅ wiring shipped |

## The Stage D regression saga

| Batch | Recipe | Aggregate r=2 pass@1 | Diagnosis |
|---|---|---|---|
| A-batch | train-steps=100, gen-n=164 | 0.116 ± 0.037 | HALF base 0.222 (catastrophic) |
| G1 | train-steps=30 | 0.154 ± 0.032 | partial recovery, still below base |
| G2 MBPP | train-steps=30 | (per-round) 0.244 < base 0.313 | substrate-divergent damage |
| G4 | train-steps=30 + **mask-prompt** | 0.117 ± 0.048 | mask alone insufficient (≈ A-batch) |
| G5 | train-steps=30 + mask + **cosine LR** | 0.152 ± 0.063 | recipe-detail ports net-zero vs G1 |
| G6 | samples=6 systematic + batch=1 + ts=200 | 0.108 (1-seed) | data volume FALSIFIED; isolation: single round craters base 0.222→0.085 = step-level batch=1 overfitting |
| G7 | + **batch=4** padded SFT | 0.173 ± 0.009 (4-seed) | collapse FIXED (σ 0.063→0.009, loss blowup gone) but still < base |
| G8 | + **fresh AdamW/round** + **non-cumulative buffer** | 0.2037 (1-seed) | round-level flags ≈ no effect (G8 seed800 0.204 ≈ G7 seed700 0.209); loop lands ≈base, gap to 0.404 unclosed |

## Infrastructure shipped this session

### Recipe fixes (close the Phase 17 gap)

| Component | Commit |
|---|---|
| Prompt-mask completion-only loss (`train_qwen_lora_step_masked` + `cross_entropy_with_prompt_mask` + `TrainSftPairs` actor message + `RenderPairs` curator message + `TrainRequest::Sft.sft_pairs` + `RoundConfig.sft_mask_prompt`) | `bc90db5`, `3405173` |
| Cosine LR schedule with 10% warmup | `c7a7aed` |
| `batch_size > 1` + padding | (deferred — wallclock opt, not correctness) |

### Stage D follow-up infrastructure (independent wins)

| Component | Commit |
|---|---|
| Display fix for empty-corpus round (anomaly resolved) | `76d0f0e` |
| Sparse-corpus mitigation (`--gen-n` default 16→32) | `835393f` |
| `--scratch-dir` + `--prompt-skip-list` | `d2d0aa4` |
| `--seed` CLI flag (multi-seed RNG fan-out) | `1736393` |
| FilteredDomain wrapper + 4 unit tests | `b9be505` |
| MBPP variant binary `phase22_mbpp_mr_sft` | `2753241` |
| `--checkpoint` flag for `phase22_humaneval_baseline` | `d861a36` |
| QwenModelActor::ScoreLogProb implementation | `e787f79` |
| QwenModelActor::LossOn implementation (C1) | `7d2c7c0` |
| REINFORCE `--sync-every` adapter sync (C2) | `905adfd` |
| tmpfs hint on `--out-dir` (C3) | `7d2c7c0` |
| Thread-safe Domain.verify + VerifierActor parallel pool (#3) | `38025bd` |
| `run_multi_round` eval-before dedup (#1) | `cb6a9b7` |
| EvaluatorActor aggregate-mode pipelined verify (#2) | `d9834a6` |
| CPU fail-fast guard + CLAUDE.md gotcha #8 | `e030fce` |
| CLAUDE.md gotcha #9 (recipe byte-compare) + E bundle | `10799cb` |

### Tests: 145 (pre-Phase-22) → **167** (post-fix)

- +4 HumanEval Stage A
- +7 MBPP Stage C
- +4 FilteredDomain
- +1 parallel-verify regression
- +4 prompt-mask helper
- +2 cosine LR schedule

## Final measurement table (G4 → G9)

The G5-G8 aggregates were measured **without** eval-truncation (the
pre-G9 baseline binary). G9 added completion truncation at both train
and eval; the G9 rows + base-trunc are measured **with** eval-truncation
(matching Phase 17). So the honest comparison is the paired
**base-trunc vs G9** at the bottom — both use the same metric Phase 17
used.

| Configuration | Mean r=2 pass@1 | σ | n | eval-trunc | Note |
|---|---|---|---|---|---|
| Base Qwen (Stage B) | 0.222 | — | — | no | original anchor |
| A-batch (pre-fix) | 0.116 | 0.037 | 5 | no | catastrophic regression |
| G4 (mask-only) | 0.117 | 0.048 | 5 | no | mask alone insufficient |
| G5 (mask + cosine) | 0.152 | 0.063 | 5 | no | net-zero vs G1 |
| G6 (samples=6, batch=1, ts=200) | 0.108 | — | 1 | no | data volume falsified |
| G7 (+ batch=4) | 0.170 | 0.011 | 5 | no | collapse fixed, σ 7× tighter, still < base |
| G8 (+ fresh-opt + non-cumul) | 0.204 | — | 1 | no | round-level flags ~no effect |
| **base (truncated)** | **0.218** | — | — | yes | matches Phase 17 base 0.216 |
| **G9 (+ truncation) 1-seed** | **0.438** | — | 1 | yes | matches/exceeds Phase 17 |
| **G9 (+ truncation) 5-seed** | **0.4356** | **0.016** | 5 | yes | **reproduces Phase 17** |
| Phase 17 reference (S1) | 0.404 | 0.013 | 5 | yes | (target) |

**Conclusion (Phase 22 Stage D — RESOLVED):** the full Phase-17 recipe
is `{completion-mask, cosine LR, batch=4, fresh AdamW/round,
non-cumulative harvest, **completion truncation**}`. The −0.20 gap that
survived G4-G8 was the missing truncation: `build_program` fed the RAW
model completion, and neither generator nor evaluator truncated it
(`stop_char = None` for multi-line HumanEval). Raw completions carried
trailing test scaffolding (`def test_`, `print(`, `if __name__`), leaked
`<|fim_middle|>` tokens, and cut-off-mid-statement tails — which (a)
failed the verifier (harvest yield ~80 instead of ~200 pairs/round) and
(b) failed at eval. Porting Phase 17's `truncate_completion`
(`Domain::truncate_completion` + `truncate_python_completion`, commit
`aaf0594`) fixed both:

- **base (truncated) = 0.218** ≈ Phase 17 base **0.216**
- **G9 r=2 (5-seed) = 0.4356 ± 0.016** ≈ Phase 17 r=2 **0.404 ± 0.013**

The +0.218 lift (0.218 → 0.436, a clean doubling) tracks Phase 17's
+0.188. **The Rust/Pekko self-evolving MR-SFT loop now reproduces
(slightly exceeds) the Phase 17 Python reference — the full
self-evolving loop is end-to-end native in Rust + Pekko.** (The LoRA
merge math was byte-compared and proven consistent — it was NOT the
bug; the divergence was data preprocessing on both sides of the
training loop.)

### Recipe resolution order (the −0.20 gap was a stack)

mask (`bc90db5`) → cosine LR (`c7a7aed`) → batch=4 (`59aab8d`) →
fresh-opt + non-cumul (`e69de7e`, ~neutral on their own) →
**truncation (`aaf0594`, decisive)**.

### Saturation curve (G9 recipe, `--rounds 3`, truncated aggregate)

| round | pass@1 | Δ vs prev | n | Phase 17 ref |
|---|---|---|---|---|
| base | 0.218 | — | — | 0.216 |
| r=2 | 0.436 ± 0.016 | +0.218 | 5 | 0.404 |
| r=3 | 0.4645 ± 0.026 | +0.029 | 5 | 0.475 |
| r=4 | 0.481 ± 0.028 | +0.017 | 5 | 0.519 |
| r=5 | 0.477 ± 0.040 | −0.004 | 5 | 0.556 |

The increment shrinks monotonically then flattens (+0.218 → +0.029 →
+0.017 → −0.004): **PLATEAU confirmed at r≈4-5, ~0.48** (r=5 0.477 ≈
r=4 0.481, within noise). Two findings: (1) the saturation SHAPE
(diminishing returns → plateau) reproduces Phase 17. (2) BUT the
absolute plateau is ~0.07-0.10 LOWER than Phase 17 at high rounds —
r=2 matched closely (0.436 vs 0.404) but the gap widens by r=5 (0.477
vs 0.556). Our loop saturates earlier/lower; the exact high-round
harvest dynamics (diversity / non-cumulative buffer / temperature)
weren't byte-compared and are the candidate cause for future work.

**Buffer ablation (r=5, 3-seed paired):** cumulative buffer (drop
`--reset-curator-each-round`) vs non-cumulative — cumulative is WORSE
(mean 0.454 vs 0.486, Δ=−0.032, 3/3 seeds negative). The buffer choice
is RULED OUT as the high-round-gap cause; non-cumulative (the G9 /
Phase 17 default) is correct. Mechanism: cumulative trains a
1024-capped FIFO mix of stale + fresh pairs at <1 epoch/round, vs
non-cumulative concentrating fresh high-quality pairs at ~2
epochs/round. Remaining gap suspects: harvest diversity / sampling
temperature.

**Harvest-temperature ablation (r=5, 3-seed paired):** gen temp
0.8 → 1.0 (eval fixed at 0.8) — no lift, mean 0.469 vs 0.486 (Δ=−0.017,
mixed signs, within noise; harvest sizes comparable so not confounded).
Temperature RULED OUT — and note it already matched Phase 17 (both 0.8).

**High-round gap — status (Stage D closeout):** both buffer
(cumulative −0.032) and harvest temperature (−0.017) are ruled out as
the cause of our ~0.48 plateau vs Phase 17's ~0.55. The one remaining
concrete sampling divergence is **top_k**: our harvest uses
`top_k=Some(40)`; Phase 17 uses none (`do_sample, temperature, top_p`
only).

**top_k harvest ablation (r=5):** added a `--top-k` flag (harvest only;
eval kept at 40 so the metric stays comparable). `--top-k 0` (match
Phase 17's no-top_k sampling) vs 40, **5-seed paired**: top_k=0 mean
**0.502 ± 0.046** vs 0.476 ± 0.039, Δ=**+0.026** (3/5 seeds positive).
Weak but consistent — not significant (paired t≈1.2) but the only knob
with a consistent improvement direction, and it matches Phase 17's
actual sampling. Narrows the gap from −0.080 to −0.054.

All three ablations on the high-round gap:

| knob | r=5 Δ vs baseline | verdict |
|---|---|---|
| cumulative buffer | −0.032 (3/3 neg) | ruled out (worse) |
| harvest temp 1.0 | −0.017 (mixed) | ruled out (no lift) |
| harvest top_k=0 | +0.026 (3/5 +, 5-seed) | weak positive — best lever |

top_k removal (matching Phase 17) is the best lever found and explains
roughly half the ~0.05-0.08 high-round gap; the remainder is
small-factor / plateau seed variance. **Recommended recipe refinement:
`--top-k 0`.** It is now the default (commit `e647540`).

**Full top_k=0 saturation curve (5-seed, re-measured):**

| round | top_k=0 | top_k=40 | Δ | Phase 17 |
|---|---|---|---|---|
| r=2 | 0.440 ± 0.017 | 0.436 | +0.004 | 0.404 |
| r=3 | 0.490 ± 0.030 | 0.4645 | +0.026 | 0.475 |
| r=4 | 0.509 ± 0.026 | 0.481 | +0.028 | 0.519 |
| r=5 | 0.502 ± 0.046 | 0.476 | +0.026 | 0.556 |

The full curve turns the weak single-round signal into a strong,
consistent one: r=2 unchanged (+0.004) but r=3/r=4/r=5 ALL lift by
+0.026-0.028 — the "top_k matters only at high rounds" pattern. With
top_k=0, **r=4 = 0.509 ≈ Phase 17's 0.519** (essentially matched) and
r=3 0.490 slightly exceeds Phase 17's 0.475. Residual gap concentrated
at r=5 (−0.054; Phase 17 keeps climbing to r=6 0.581, our plateau sits
~0.51). top_k=0 reproduces Phase 17's curve through r=4.

### MBPP cross-substrate (top_k=0 recipe) — Phase 17 SB reproduced

Ported the full recipe flags to `phase22_mbpp_mr_sft` (`53af878`) +
`--checkpoint` to `phase22_mbpp_baseline`. MBPP-100 rounds=2 5-seed with
the full top_k=0 recipe:

| | r=2 pass@1 (truncated, n=100×k=10) |
|---|---|
| base (truncated) | 0.201 |
| MBPP r=2 (5-seed) | **0.447 ± 0.027** |
| Phase 17 SB r=2 (ref) | 0.453 ± 0.016 |

**MBPP r=2 0.447 ≈ Phase 17 SB 0.453** (Δ=−0.006, within noise; lift
base 0.201 → 0.447, +0.246). The full top_k=0 recipe validated on
HumanEval **reproduces Phase 17 on MBPP too** — substrate-agnostic.

**MBPP saturation curve (top_k=0, 5-seed):**

| round | MBPP | Δ | Phase 17/20 |
|---|---|---|---|
| base | 0.201 | — | — |
| r=2 | 0.447 ± 0.027 | +0.246 | 0.453 |
| r=3 | 0.487 ± 0.012 | +0.040 | 0.457 |
| r=4 | 0.488 ± 0.013 | +0.001 | — |
| r=5 | 0.499 ± 0.014 | +0.011 | 0.541 |

MBPP r=2 ≈ Phase 17 and r=3 EXCEEDS it (0.487 vs 0.457). MBPP plateaus
EARLIER than HE (r≈3 ~0.49 vs HE r≈4-5 ~0.51) and is much TIGHTER
(σ ~0.013 vs HE 0.026-0.046). r=5 0.499 sits −0.042 below Phase 20's
0.541 — the SAME shape as HE (low/mid rounds match-or-exceed, high-round
plateau ~0.05 lower). The Rust/Pekko loop reproduces Phase 17 on BOTH
HumanEval and MBPP, curve shape included.

### High-round plateau gap — four ablations (final)

The residual ~0.05 high-round gap (our plateau ~0.51 vs Phase 17
~0.55-0.58) was investigated with four paired r=5 ablations:

| knob | r=5 Δ vs baseline | verdict |
|---|---|---|
| cumulative buffer | −0.032 | ruled out (worse) |
| harvest temp 1.0 | −0.017 | ruled out |
| harvest top_k=0 | +0.026 (5-seed) | weak positive — adopted (best lever) |
| AdamW weight_decay 0.01 | +0.008 (3-seed) | ruled out |

weight_decay (Phase 17's AdamW default 0.01 vs our 0.0) was the last
concrete divergence; at wd=0.01 r=5 = 0.519 vs top_k=0 0.512 (+0.008,
mixed, within noise). After adopting top_k=0, the residual gap is NOT
explained by any single remaining knob — the only untested divergence
is training dtype (Phase 17 fp16 vs our F32, but F32 is more precise so
an unlikely cause). **Conclusion: the residual high-round gap is
small-factor / inherent plateau variance, not one dominant divergence.**

## Stage D — closed

The Rust/Pekko self-evolving MR-SFT loop reproduces Phase 17 end-to-end:
base 0.218 → r=2 0.436 (≈ Phase 17 0.404), saturation curve climbs with
diminishing returns to a plateau at r≈4-5 ~0.48, matching Phase 17's
shape (which plateaus ~0.55-0.58). Recipe: completion-mask + cosine LR
+ batch=4 padded SFT + fresh-AdamW/round + non-cumulative harvest +
**completion truncation** (the decisive fix). Two ablations (buffer,
harvest temperature) confirm the design; the residual high-round gap
(top_k) is documented for a future session.

(Infra: seeds 400/500 OOM'd repeatedly at batch=4 round-2/3 training
on contended shared GPUs — never on aggregate eval, which is lighter.
Workaround that held: run batch=4 TRAINING only on fully clean GPUs;
aggregate eval can use contended GPUs. All 5 seeds completed this way.)

## Recipe recommendation (final)

After G4 + G5 land. Skeleton:

```bash
# Recommended Phase 22 Stage D recipe for production:
cargo run -p llm-actors --example phase22_he_mr_sft \
    --features cuda --release -- \
    --seed N \
    --rounds 2 \
    --gen-n 164 --eval-n 32 --eval-passk 3 \
    --train-steps 30 \
    --max-new-tokens 200 \
    --out-dir /dev/shm/phase22_he_mr_sft_seed_N \
    --scratch-dir /dev/shm/phase22_he_scratch_seed_N
```

Defaults guarantee:
- `sft_mask_prompt = true` (RoundConfig default) → completion-only CE
- Cosine LR schedule with 10% warmup → applied in TrainSftPairs path
- VerifierActor parallel pool (8 workers) → faster batch verify
- Aggregate-mode verify pipelining in EvaluatorActor → faster aggregate eval
- `--out-dir /dev/shm/...` → tmpfs cuts ~10-15s/round disk I/O

## Open items (Phase 23 candidates)

- D7: batch_size > 1 with padding (Phase 17 used batch=4). Wallclock
  optimization; not correctness-critical now that mask + cosine land.
- Adapter-sync between RL steps (Stage E C2 shipped the flag,
  measurement work remains).
- LoRA rank-32 ablation at the fixed recipe.
- Full Phase 17 saturation curve (r=1..6 × 5 seeds × full 164 ×
  passk=10) — true numerical reproduction at scale.
- Pre-filtered prompt set via FilteredDomain (Phase 9 S5 cold-start
  mitigation).

## See also

- `docs/phase22-overview.md` — Phase 22 single entry point
- `docs/phase22-stage-d-A-batch-gen-n-164.md` — A-batch result
- `docs/phase22-stage-d-train-steps-ablation.md` — G1 result
- `docs/phase22-stage-d-G2-mbpp-cross-substrate.md` — G2 result
- `memory/phase22_stage_d_root_cause_fix.md` — discovery writeup
- Notion: workLLM — Phase 22 Stage D ROOT-CAUSE 발견 (Phase 17 복구)
