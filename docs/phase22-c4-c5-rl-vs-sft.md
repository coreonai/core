---
title: "Phase 22 follow-up C4/C5 — RL matches SFT once the reward is scored correctly"
date: "2026-07-30"
---

# TL;DR

Three results, in increasing order of consequence:

- **C4 (re-run, 4 seeds/arm, correct reward) — RL matches SFT.**
  Positive-advantage-only gives **+0.152 pass@5 / +0.218 pass@1** over base
  on the 7B hard tail; multi-round SFT on the same ruler gives +0.145 /
  +0.203. Full-advantage RL gives +0.145 / +0.161. **But RL's spread is
  3–4× wider** (pass@5 σ 0.068–0.077 vs SFT's 0.020), so SFT is still the
  better bet in practice even at an equal mean.
  ⚠ The **first** C4 run concluded "RL loses to SFT, roughly half the gain"
  (+0.070 / +0.083). That run was scored with the C5 reward bug active and
  is **superseded** — see "C4, first attempt" below. It is kept in this
  document because the way it failed is the point.
- **C5 — completion truncation was skipped in two places, and both mattered.**
  (a) The RL loop verified raw completions; every other consumer first calls
  `domain.truncate_completion`. Same base policy, identical sampling: RL saw
  pass@1 ≈ 0.06, the evaluator 0.172 — a **3× gap**. This starved the reward
  signal and penalised long completions for being long rather than wrong.
  (b) `FilteredDomain` never delegated `truncate_completion`, so wrapping a
  code domain in it turned truncation **off** — and every hard-tail
  experiment uses `--prompt-skip-list`. That is what produced the
  mis-measured base behind the corrected SFT claim.
- **The `+0.254` hard-tail SFT headline is inflated; on one consistent ruler
  it is `+0.145`.** The SFT-era number paired a **mis-measured base**
  (0.246) with a fine r2 measurement (0.500). Re-scoring the *same* saved r2
  checkpoints through the eval path this study uses gives 0.566, while the
  base measures 0.422 — so the gain is +0.145 pass@5 (+0.203 pass@1), not
  +0.254.

C5 did not merely add noise — it **inverted a conclusion**. The first C4 run
produced a clean-looking double dissociation between the arms and a "RL is
half of SFT" verdict; both evaporated once the reward was scored the same way
the eval scores it. Both attempts are written up below rather than the second
one replacing the first, because the failure mode is the reusable lesson:
an in-loop metric that was never checked against the eval it will be compared
to can look like a result and point the wrong way.

# C5: the reward bug

`evaluator_actor.rs` (and `generator_actor.rs`) do:

```rust
let completion = self.domain.truncate_completion(&completion);
let v = self.domain.verify(&prompt, &completion);
```

`phase22_he_reinforce.rs` did:

```rust
let comp_text = tk.decode(&comp_ids)?;
let verdict = humaneval.verify(prompt, &comp_text);   // no truncation
```

`truncate_python_completion` cuts at the first top-level `def`/`class`/
`import`/`from`/`if __name__`/`print(` after some body content, and at any
`<|` special-token marker — the standard HumanEval post-processing that
Phase 17's Python recipe also applies. Without it, "correct function +
trailing `def helper():`" is scored **wrong**.

Measured, same 7B base policy, same sampling (temp 0.8 / top_k 40 /
top_p 0.95 / max_new 192), HumanEval idx 100–163:

| scorer | pass@1 |
|--------|--------|
| RL loop (raw) | ~0.06 |
| evaluator (truncated) | 0.172 |

Two consequences, and the second is the dangerous one:

1. **Reward starvation.** A 3× thinner reward signal, in the exact regime
   (sparse-reward hard tail) where the study lives.
2. **A spurious length gradient.** Longer completions have more opportunity
   to emit a trailing top-level statement, so length was punished as if it
   were incorrectness. This is what made C4's `fulladv` arm *look* like it
   was collapsing — see below.

**Fix**: truncate before verifying, and train on the token prefix that
survives truncation. Truncating only the reward would leave a
positive-advantage step reinforcing the trailing junk it was rewarded in
spite of. `truncated_token_prefix` binary-searches the token prefix whose
decode covers the truncated text (~8 decodes/sample) instead of re-encoding
it, so the trainer sees tokens the model actually sampled.

# C4, first attempt (superseded — reward bug active)

`scripts/phase22_c4/positive_advantage_ab.sh` — 7B, HumanEval idx 100–163,
k=4, max_new=192, `--pg-micro-batch-size 1`, `--sync-every 1`, lr=2e-4,
**30 RL steps** (30 optimizer updates under the C3 fix, comparable to an SFT
round's 30), 2 arms × 2 seeds, 4 runs × 2 GPUs.

- `posonly` = `--pg-positive-only`: keep only verifier-passing completions →
  loss `reward * CE >= 0`, bounded below. Reward-weighted SFT on
  verified-correct completions, i.e. rejection-sampling FT / RAFT.
- `fulladv` = the C3 fixed configuration: RLOO advantages, ~75% of surviving
  samples negative, objective unbounded.

**⚠ Both arms ran with the C5 bug present.** The fix landed after these runs.

## What the training-time metric said (and why it was wrong)

| arm | first 10 steps | last 10 steps |
|-----|----------------|---------------|
| posonly s42 | 15.9 | **30.0** |
| posonly s100 | 23.5 | **27.6** |
| fulladv s42 | 13.7 | **7.3** |
| fulladv s100 | 19.2 | **4.3** |

(pass count out of 256, base ≈ 16.4, σ = 4.5 for a single reading.)

A clean double dissociation — 2/2 seeds up, 2/2 seeds down — and `fulladv`'s
`comp_len` drifting 133 → 152 and 137 → 158, the same "never emits EOS"
runaway signature C3 documented. It looked decisive.

**It did not survive the eval.** `fulladv_seed100`, the run whose training
metric cratered hardest (1–4 passes out of 256 at the end), scored the
**highest pass@5 of any checkpoint** (0.609). Its completions had grown
longer, and under C5's un-truncated scoring, longer meant "wrong". The
double dissociation was substantially an artifact of the reward bug.

## What the eval said

Post-hoc, all checkpoints on one ruler
(`phase22_humaneval_baseline --offset 100 --n-problems 64 --passk 5
--sequential --aggregate`):

| | pass@5 | pass@1 | Δ pass@5 | Δ pass@1 |
|---|---|---|---|---|
| base | 0.4219 | 0.1719 | — | — |
| **SFT samples=16 r2** (4 seeds) | **0.566 ± 0.020** | **0.364 ± 0.037** | **+0.145** | **+0.203** |
| C4 posonly (2 seeds) | 0.492 | 0.255 | +0.070 | +0.083 |
| C4 fulladv (2 seeds) | 0.508 | 0.236 | +0.086 | +0.064 |

Per-seed: posonly 0.500 / 0.484; fulladv 0.406 / 0.609.

**Verdict against the pre-registered rules:**

| rule | outcome |
|------|---------|
| posonly learns | **pass** — 2/2 seeds above base on both metrics |
| posonly ≥ fulladv | **fail** — means indistinguishable (0.492 vs 0.508); `fulladv` merely has a huge spread (0.406, 0.609) |
| is the unbounded objective destructive | **not answered** — the evidence was the training metric, which C5 invalidates |

(Re-run verdict: posonly learns **pass**; posonly ≥ fulladv **4/4 seeds
paired, p ≈ 0.10 — directional, not established**; unbounded objective
destructive **no — 8/8 runs rise and `fulladv` never collapses**.)

So the reading at the time was: bounding the objective did not demonstrably
help, and both RL arms land at roughly half of SFT's gain. **That conclusion
was wrong** — the reward bug depressed both arms. See the re-run below.

# C4, re-run (C5 fixed, 4 seeds per arm)

Identical configuration — the only change is the C5 fix, so the two batches
are directly comparable. 4 seeds × 2 arms, run as two batches of 4 (only 4
runs fit on 8 GPUs at 2 cards each):
`positive_advantage_ab.sh 30 "" 42,100 <dir>` then `... 200,300 <dir>`.

The fix is immediately visible at step 0, before any training: 41/256 passes
versus 16/256 in the first attempt, on the same policy and the same
completions. That is the ~3× reward-density gain, and it means **the old
noise floor (16.4/256, σ 4.5) does not apply to these runs** — it was
measured with un-truncated scoring.

Training-metric trajectories (mean passes/256 over the first and last 10 of
30 steps):

| seed | posonly | fulladv |
|------|---------|---------|
| 42 | 41.6 → 101.6 | 40.5 → 99.5 |
| 100 | 35.5 → 55.6 | 38.4 → 49.3 |
| 200 | 39.5 → 88.1 | 37.1 → 85.1 |
| 300 | 39.4 → 70.8 | 39.6 → 57.9 |

**8/8 runs rise, and `fulladv` never collapses.** In the first attempt 2/2
`fulladv` seeds fell (to 7.3 and 4.3) with `comp_len` drifting toward the
ceiling. With the reward scored correctly that reverses completely. The
"unbounded objective runs away" narrative — built across C3 and C4 — was an
artifact of the length penalty, and there is now **no evidence for it**.

## Eval (same ruler, 64 hard-tail problems, passk=5)

| | pass@5 | pass@1 | Δ pass@5 | Δ pass@1 |
|---|---|---|---|---|
| base | 0.4219 | 0.1719 | — | — |
| **posonly** (4 seeds) | **0.574 ± 0.068** | **0.390 ± 0.100** | **+0.152** | **+0.218** |
| **fulladv** (4 seeds) | **0.566 ± 0.077** | **0.333 ± 0.137** | **+0.145** | **+0.161** |
| SFT samples=16 r2 (4 seeds) | 0.566 ± 0.020 | 0.364 ± 0.037 | +0.145 | +0.203 |
| first-attempt C4 (2 seeds, buggy) | 0.492 / 0.508 | 0.255 / 0.236 | +0.070 / +0.086 | +0.083 / +0.064 |

Per-seed pass@5 — posonly 0.641 / 0.516 / 0.625 / 0.516; fulladv 0.641 /
0.500 / 0.625 / 0.500.

**Three findings:**

1. **RL matches SFT on the mean.** posonly edges it on both metrics (+0.008
   pass@5, +0.026 pass@1). "RL is the weak axis" no longer holds at this
   configuration — that conclusion was an artifact of the reward bug.
2. **RL is far less reliable.** pass@5 σ is 0.068–0.077 against SFT's 0.020,
   a 3–4× spread. At equal means, SFT is the better deployment choice.
   Seeds 42/200 land near 0.63 in *both* arms and seeds 100/300 near 0.51 in
   *both* — **the seed dominates the arm**, which is also why the arms are
   hard to separate.
3. **Bounding the objective shows a consistent but inconclusive edge.** The
   arms share seeds, so the comparison pairs. posonly ≥ fulladv in **4/4
   seeds on both metrics**; the paired pass@1 difference is
   **+0.057 (diffs +0.016 / +0.063 / +0.025 / +0.125, paired t = 2.30,
   df = 3, p ≈ 0.10)**. Sign-consistent across every seed, but not
   significant at n = 4. This is the first of three measurements to show a
   direction at all; it is not yet a claim.

# The correction to the SFT hard-tail claim

`docs/phase22-7b-results.md` reports the samples=16 hard-tail run as
**pass@5 0.246 → 0.500, +0.254 (4/4 seeds, ~3.7σ)** — "the first robust 7B
self-improve win". Comparing C4 against that number failed a sanity check:
our base measured **0.422**, not 0.246, for the same model on the same 64
problems.

The saved r2 checkpoints (`scratch-7b-sft/htr_out_s{42,100,200,300}`) made
this decidable. Re-scoring them through this study's eval path:

| | SFT-era path | this study's path |
|---|---|---|
| base | **0.246** | **0.422** |
| r2 (4 seeds) | 0.500 | 0.566 ± 0.020 |
| **gain** | **+0.254** | **+0.145** |

**The r2 endpoint reproduces; the base does not.** The gain was inflated by
pairing a low base measurement with a sound r2 measurement.

## Verifying the base independently

The base number now carries the whole correction, so it was re-measured on a
different draw. `EvalSequential` derives generation seeds as
`prompt_idx * passk + k` and **ignores `--seed`**, so re-running is
deterministic — but changing `passk` changes the draw:

| | pass@1 @ passk=5 (320 samples) | pass@1 @ passk=10 (640 samples) | pass@10 |
|---|---|---|---|
| base | 0.1719 | **0.1609** (103/640) | 0.5313 |
| SFT r2 seed 42 | 0.4188 | **0.4359** (279/640) | 0.6719 |

Two independent draws agree (se ≈ 0.015), and hard-tail pass@10 = 0.531 sits
below the published full-set pass@10 = 0.713, which is the right direction.
The base measurement holds; the SFT-era 0.246 is the outlier.

## The mechanism: `FilteredDomain` silently disabled truncation

The initial guess — that the two paths differed only in random-vs-sequential
prompt selection — was wrong. The cause is a third instance of the same bug
as C5, this time in the library:

**`FilteredDomain` implements `Domain` but never overrode
`truncate_completion`**, so it inherited the trait's identity default. Only
`HumanEvalDomain` and `MbppDomain` override that method, so wrapping either
in the filter switched truncation **off** at every generate/verify site —
and every hard-tail experiment runs through `--prompt-skip-list`.

Measured, same 7B base, same 64 problems, same sampling:

| | pass@5 | pass@1 |
|---|---|---|
| unfiltered (truncation on) | 0.4219 | 0.1719 |
| **filtered (truncation off)** | **0.1562** | **0.0437** |

This also explains why the discrepancy was *asymmetric* — r2 reproduced far
better than base (0.500 vs 0.566). SFT teaches the model to stop emitting
trailing top-level statements, so an un-truncated scorer penalises a trained
checkpoint less than it penalises the base. Penalising the base harder than
the endpoint is exactly what inflates a measured gain.

The module doc already warned: *"use it for training convenience, not for
benchmark-aligned eval."* The hard-tail eval did precisely that, and nothing
in the type system objected — a defaulted trait method is invisible at the
call site.

**Honest limit**: this is the dominant cause but does not fully account for
the SFT-era 0.246, which sits *between* 0.156 and 0.422. The residual is
`EvaluatorMessage::Eval` (random, with-replacement) versus `EvalSequential`,
or a `max_new_tokens` difference. Those runs were launched ad hoc and **no
command line survives**, so the remainder is not reconstructible.

Fixed by delegating (plus `score`, which no domain overrides today — closed
so the next one to do so isn't silently ignored the same way).

**SFT still wins, and still comfortably.** +0.145 pass@5 / +0.203 pass@1 over
base, 4 seeds, σ = 0.020. The correction changes the magnitude of the
headline, not its direction.

# Lessons

1. **An in-loop metric must be checked against the eval it will be compared
   to, before it is used to steer anything.** C5 was one missing function
   call, and it produced a training signal that looked like a clean
   scientific result and pointed the opposite way from the truth.
2. **When a comparison to a prior claim fails a sanity check, re-measure the
   prior claim rather than explaining the difference away.** The base
   discrepancy could easily have been written off as seed noise; it was a
   1.75× inflation of a headline result.
3. **Keep the launch command.** The SFT-era mechanism is unrecoverable purely
   because the runs were ad hoc. Every batch in this follow-up ships a script.
4. **CLAUDE.md gotcha #8 bit three times in one session.** A plain
   `cargo build --examples` / `cargo test` (which also builds examples)
   silently replaces the CUDA example binaries with CPU ones. The
   `PHASE22_ALLOW_CPU` guard caught it each time at zero GPU cost, but the
   trap is: verifying a binary's *timestamp* is unchanged does not verify it
   was ever a CUDA build. Check for `cudarc` symbols instead:
   `strings target/release/examples/<name> | grep -c cudarc`.
5. **Absolute-threshold alerting misses slow failure.** The C4 collapse
   monitor fired on `pass == 0` or `comp_len >= 190`; `fulladv` degraded to
   1–10 passes with `comp_len` 158 and never tripped it. Trend-based alerts
   are the right shape for runaway detection.
6. **A defaulted trait method is a silent-failure surface.** Three separate
   truncation bugs in one study, all invisible at the call site: two missing
   calls and one wrapper inheriting the default. `Domain::truncate_completion`
   has a sensible identity default, which is exactly what let a *code* domain
   end up scoring raw completions with nothing to notice. A wrapper should
   delegate every defaulted method mechanically, not case by case.
7. **Wait on artifacts, not on process names.** Two orchestration deadlocks
   here came from `pgrep -f <name>` matching the waiting shell itself,
   because the same shell's command line also contained the binary path. The
   `[p]attern` trick only protects the literal it is written in. The second
   deadlock idled 8 GPUs for 3 hours. Poll for the output file instead —
   `until [ -f checkpoint ]` cannot match itself — and bound every wait.

# Where next

- **Settle posonly vs fulladv.** 4/4 seeds favour posonly at pass@1 (+0.057)
  but p ≈ 0.10. The paired design is efficient, so ~8 seeds/arm would resolve
  it — worth it, because it is the difference between "bound the objective"
  being a real recipe rule and a coin flip.
- **Attack the variance, not the mean.** RL already matches SFT's mean; its
  problem is σ 0.068 vs 0.020. The seed dominates the arm (42/200 ≈ 0.63,
  100/300 ≈ 0.51 in *both* arms), so the lever is whatever the seed controls
  — prompt order and the harvest draw — not the objective. Phase 15 S3b
  reached the same conclusion for SFT: harvest, not init, is the noise axis.
- **Re-check the other filtered experiments.** `FilteredDomain` disabled
  truncation for every `--prompt-skip-list` run, and `MbppDomain` overrides
  `truncate_completion` too — so the MBPP filtered results carry the same
  defect. Any absolute number from a filtered run needs re-measuring; gains
  measured base-to-endpoint within that ruler are inflated, not merely
  shifted.
- **Verify the two eval paths now agree.** The post-fix filtered measurement
  still needs a GPU (the first attempt used a stale binary and is invalid).
  Expect ~0.42 to match the unfiltered path; anything else means the
  delegation fix is not the whole story.
- **n = 4 is a direction, not a headline.** This repo retracted Phase 12 S1,
  14 C2/C3, and 15 S2 for under-powered claims. The one durable statement
  here is negative and robust: **the unbounded-objective runaway does not
  exist** (8/8 runs rise).
