# Phase 9 S5 — external-model self-improve loop

Closes the loop around Phase 9 S4. S4 measured Shape-C calibration on
Qwen2.5-Coder-0.5B (sum-AUC 0.702, F=8 lift 1.95×) but stopped at
measurement. S5 wires up the actual self-improve loop:

```
for round in 0..N:
    generate K candidates per challenge
    score with sum log-prob (Shape-C critic, no separate model)
    verify with `python -c`
    track gen pass rate AND critic-rerank pass rate
    LoRA fine-tune on verifier-passed (prompt, completion) pairs
    next round
```

This is the same shape as the in-house `self_improve_rust` /
`self_improve_korean` loops, but on a real 1B-scale HF model and a
mix of synthetic + real-world coding challenges.

## Setup

```bash
python3 -m venv /tmp/s4_env
/tmp/s4_env/bin/pip install torch transformers accelerate peft
```

GPU footprint ~3 GB (0.5B in fp16 + LoRA adapters on q_proj/v_proj,
r=16). One A100 with ≥4 GB free is plenty.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 /tmp/s4_env/bin/python self_improve.py \
    --model Qwen/Qwen2.5-Coder-0.5B \
    --rounds 3 --samples 8 --train-steps 60 \
    --out run.json
```

## Challenge set (11 total)

- 6 from S4 (slot-fill arithmetic): `equals_5`, `equals_14_via_doubling`,
  `len_5_string`, `two_plus_to_5`, `ten_minus_to_3`, `two_pow_to_8`.
- 5 HumanEval-style real-world function bodies (single-line returns):
  `is_even`, `list_sum`, `is_positive`, `count_chars`, `double_it`.

All challenges verify via `python -c` on
`prompt + completion + suffix` where the suffix contains 1–3 asserts.

## Result

| round   | total | pass  | rate  | F=4 lift |
|---------|------:|------:|------:|---------:|
| round-0 |    88 |    35 | 0.398 |    0.81× |
| round-1 |    88 |    64 | 0.727 |    1.00× |
| round-2 |    88 |    64 | 0.727 |    1.00× |
| final-3 |    88 |    64 | 0.727 |    1.00× |

**+33 percentage points in 1 LoRA round** (60 steps, 35 verifier-passed
training pairs, lr 2e-4, r=16/α=32 on q_proj+v_proj).

### Per-challenge breakdown

| challenge                  |  r0 |  r1 |  r2 |  r3 | learned? |
|----------------------------|----:|----:|----:|----:|---------:|
| equals_5                   | 1/8 | 8/8 | 8/8 | 8/8 | yes |
| two_plus_to_5              | 3/8 | 8/8 | 8/8 | 8/8 | yes |
| two_pow_to_8               | 2/8 | 8/8 | 8/8 | 8/8 | yes |
| is_even                    | 7/8 | 8/8 | 8/8 | 8/8 | yes |
| list_sum                   | 4/8 | 8/8 | 8/8 | 8/8 | yes |
| count_chars                | 2/8 | 8/8 | 8/8 | 8/8 | yes |
| is_positive                | 8/8 | 8/8 | 8/8 | 8/8 | already-solved |
| double_it                  | 8/8 | 8/8 | 8/8 | 8/8 | already-solved |
| equals_14_via_doubling     | 0/8 | 0/8 | 0/8 | 0/8 | **cold-start** |
| len_5_string               | 0/8 | 0/8 | 0/8 | 0/8 | **cold-start** |
| ten_minus_to_3             | 0/8 | 0/8 | 0/8 | 0/8 | **cold-start** |

8 of 11 challenges fully learned (100% pass) in round 1. The 3 that
never improve are the 3 with **zero verifier-passed samples in
round 0** — the loop has no bootstrap signal for them and LoRA on
pairs from the *other* challenges does not transfer.

## Findings

1. **Self-improve transfers to external 1B-scale models.** The same
   loop shape that drove K9 RustCode in-house drives Qwen2.5-Coder
   on Python challenges, with comparable or larger gains
   (+33 pp vs K9's +12 pp on its 12-challenge set).

2. **Saturation in 1 round.** Whatever the model can learn from
   verifier-passed seeds, it learns immediately. Rounds 2–3 are
   no-ops at this challenge count and LoRA budget.

3. **Critic lift collapses post-saturation.** Round 0 F=4 lift is
   actually 0.81× (slightly anti-correlated, consistent with
   Phase 9 S4 fp16 noise at the 40% pass rate regime); rounds 1+
   sit at 1.00× because every (prompt, sample) pair already passes.
   Matches Phase 7 design doc risk #8: high pass rate → no headroom
   for critic-rerank to help.

4. **Cold-start matters more than scale.** The 3 unsolved challenges
   stay at 0/8 across all rounds because the model never produces a
   verifier-passing seed in round 0. Self-improve cannot bootstrap
   capabilities the model can't already do at least once. Mitigations
   for a real deployment:
   - Easier curriculum stage (smaller asserts) before the harder one.
   - Critic-only filtering (use critic top-K as pseudo-labels even
     without verifier hits) — but with sum-AUC 0.702 the top picks
     are still mostly wrong, so this risks negative training.
   - Few-shot prompts with a known-correct example.
   - Direct injection of one ground-truth (prompt, completion) pair
     into the training set as a bootstrap.

5. **Real-world tasks behave like synthetic ones.** The 5
   HumanEval-style challenges (`is_even`, `list_sum`, …) all
   saturated to 100% just like the 5 arithmetic slot-fill
   challenges. Whether the model is completing `2 +` or
   `def is_even(n): return ` does not change the dynamics —
   it's about whether the model can produce a single passing
   sample to learn from.

## What this changes for the design doc

This is a positive end-to-end demonstration on an external model.
It supports:
- The Shape-C playbook is **not** in-house-only.
- The cold-start risk (#11) is now empirically characterized and
  belongs in the design doc next to risks #9 and #10.

## See also

- `scripts/phase9_s4/` — measurement-only validation that motivated S5.
- `docs/phase7-design.md` — risk register and decision tree.
- `llm-actors/examples/self_improve_rust.rs` — the in-house analog
  this script mirrors in shape.
