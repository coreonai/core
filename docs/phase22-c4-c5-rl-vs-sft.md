---
title: "Phase 22 follow-up C4/C5 — bounded RL, a reward-measurement bug, and a correction to the SFT hard-tail claim"
date: "2026-07-29"
---

# TL;DR

Three results, in increasing order of consequence:

- **C4 — positive-advantage-only RL works, but loses to SFT.** Bounding the
  objective (train only on verifier-passing completions, so the loss is
  `reward * CE >= 0`) gives **+0.070 pass@5 / +0.083 pass@1** over base on
  the 7B hard tail. Multi-round SFT on the same ruler gives **+0.145 /
  +0.203** — roughly double, with 4 seeds instead of 2 and much tighter
  spread. RL remains the weak axis.
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

The C4 experiment itself was nearly wrecked by C5: the training-time metric
showed a dramatic double dissociation between the arms that **did not
survive** a correctly-scored eval. That is written up below rather than
buried, because the failure mode — trusting an in-loop metric that was never
checked against the eval it is compared to — is the reusable lesson.

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

# C4: the experiment

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

So the honest reading: **bounding the objective did not demonstrably help**,
and the arm that was supposed to be broken was not measurably broken at
eval. Both RL arms land at roughly half of SFT's gain.

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

- **Re-run C4 with the C5 fix.** The reward is now ~3× denser and the
  spurious length gradient is gone; both arms deserve a clean measurement,
  and the "is the unbounded objective destructive" question is still open.
- **Fix the eval-path split.** Two paths give 0.246 and 0.422 for the same
  base. Pick one as canonical, and re-state any hard-tail number that was
  measured on the other.
- **n = 2 is not a claim.** Any C4 result that survives the re-run needs 4–5
  seeds before it goes in a headline — this repo has retracted Phase 12 S1,
  14 C2/C3, and 15 S2 for exactly that.
