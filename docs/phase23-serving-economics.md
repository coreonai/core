# Phase 23 — serving economics of the tool-using 7B

What it costs to serve the model, measured rather than assumed, and what the
two dtype failures actually are. All numbers are one A100, Qwen2.5-Coder-7B +
merged LoRA, `phase23_serve_7b`.

## Where the time goes

Latency is linear in generated tokens, with prefill nearly free at these
prompt lengths:

| `max_new_tokens` | latency |
|---|---|
| 8 | 272 ms |
| 32 | 839 ms |
| 64 | 1655 ms |
| 128 | 3311 ms |

That is **~25.9 ms/token** of decode, ≈39 tok/s. Fitting the four points
leaves only tens of ms for prefill. So decode is the entire bill, and batching
or prefix caching were not the first thing to fix.

## The first fix was free: stop decoding at the stop sequence

`generate_autoregressive` stopped only on EOS or `max_new_tokens`, while the
agentic loop cut each chunk at a stop sequence **after** generation returned.
A 20-token tool call under `max_new_tokens = 64` therefore paid for 44 tokens
that were thrown away — ~1.1 s per loop step, two or three steps per answer.

`QwenModelActor::with_stop_sequences` ends the forward passes instead of the
string:

| tool query | before | after |
|---|---|---|
| divisors of 720 | 6797 ms | **1944 ms** |
| coprime count to 45 | 6797 ms | **1934 ms** |
| sum of primes below 500 | — | **2042 ms** |

**3.5×**, with `grounded` still true — nothing traded for it.

One detail that would have been a correctness bug: the check decodes the whole
completion each step rather than appending `decode(&[tok])` per token.
Byte-level BPE splits multi-byte characters across tokens, so per-token
decoding produces replacement characters and can both miss a real stop and
invent one. A full decode of <200 tokens is microseconds against a 26 ms
forward pass, so being clever here buys nothing and risks a wrong answer.

## The dtype question, and why it dominates everything else

F32 at 7B is 28 GB — one replica per 40 GB card — and decode is
memory-bandwidth-bound, so F32 costs roughly 2× on *both* axes against BF16.
The repo ran F32 because of two recorded corruptions. Those turn out to be
**two different failures with different fixes**, which the earlier notes
conflated.

Same prompt, greedy, ten-family checkpoint:

```
F32    (python print(sum(1 for i in range(1,720+1) if 720%i==0)))    correct
BF16   (python print(sum(1 for d in range(1,720+1) if 720%d==0)))    correct
F16    ::::: : : : : : \tThe: \tThe                                  garbage
```

BF16 differs from F32 only in a tie-broken loop variable. **BF16 is fine
here.** F16 emits token id 152063 — the last id in a 152064 vocab — which is
the signature of logits overflowing to infinity: F16's maximum is 65504, and
`argmax` over an all-`inf` row returns the final index. BF16 shares F32's
exponent range and cannot fail this way.

So **the F16 failure is dynamic range, not rotary precision, and F32 rotary
would not fix it.**

### BF16 fails by position, which is the rotary signature

Sweeping prompt length with the same question appended, F32 alongside:

| prompt | F32 | BF16 |
|---|---|---|
| ~40 tokens | clean | clean |
| ~150 | clean | clean |
| ~300 | clean | `A: 24 … 24 divisorsisors` |
| ~600 | clean | `A: 11 … 1111111111111111` |
| ~1000 | clean | `divisors of of 00, 0,` |

F32 is clean to 1000 tokens. BF16 breaks around **300**, and degrades further
with length — the token-doubling the earlier note described (`"return
return"`, `"== =="`). Getting worse with position is what a position encoding
losing precision looks like: rotary angles grow with index, and an 8-bit
mantissa runs out of resolution in sin/cos, blurring relative position.

Note this also corrects the earlier threshold. The note said "past ~500
tokens"; measured, it is already broken at 300.

## The cause was one line, and the training path had it too

Upstream candle builds the rotary table in the model dtype:

```rust
let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
    .to_dtype(dtype)?               // position indices, in BF16
```

`max_position_embeddings` is 32768, and **BF16's 8-bit mantissa represents
integers exactly only to 256.** Past that, distinct positions collapse onto
the same value — 300 and 301 get identical rotary angles — so the model cannot
tell adjacent tokens apart. Token doubling is what that looks like from
outside, and 256 is why the break is at 300 rather than 500.

`llm-actors/src/qwen2_f32rope.rs` vendors the model with positions,
frequencies and sin/cos computed in F32, casting the tables to the model dtype
only afterwards: sin/cos are bounded in [-1, 1] and survive it, the index does
not. Everything else is upstream, so re-diff on a candle upgrade. `Config` is
re-exported rather than duplicated.

Same sweep, after:

| prompt | BF16 before | **BF16 after** | F32 |
|---|---|---|---|
| ~40 | clean | clean | clean |
| ~150 | clean | clean | clean |
| ~300 | `24 divisorsisors` | **clean** | clean |
| ~600 | `1111111111…` | **clean** | clean |
| ~1000 | `of of 00, 0,` | **clean** | clean |

Character-identical to F32 at every length.

Measured gain:

| | F32 | BF16 |
|---|---|---|
| GPU resident | 31153 MiB | **14993 MiB** |
| tool query | 1773 ms | **1155 ms** (1.54×) |
| `grounded` | true | true |

1.54× rather than 2× because prefill, tool execution and loop overhead do not
scale with weight bandwidth. (The earlier per-token figures are not comparable
after the stop-sequence fix: generation now ends at the call boundary instead
of running to `max_new_tokens`.)

**The LoRA training path had the identical bug** — `qwen2_lora.rs` inherited
the same three lines and trains in BF16. Phase 23's sequences are 40–150
tokens and never reached it, but Phase 22 trained HumanEval/MBPP at prompt
~150 plus completion up to 192, which crosses 256. Those runs had degraded
position information over the tail of every sequence.

### How much it moved Phase 22 — measured, and the reservation

The affected set is narrower than "everything before the fix". Position
rounding starts at 256 in BF16 but at 2048 in F16 (10-bit mantissa), and no
Phase 14–20 sequence is that long, so the F16 0.5B work is untouched. What is
affected is every **BF16** run — which, note, includes `phase22_humaneval_
baseline` even at 0.5B, since it selects BF16 on CUDA.

A/B'd in one binary via `QWEN_LEGACY_ROTARY=1`, which reproduces upstream's
cast. Rebuilding an older commit would change more than the rotary and could
not attribute a difference to this line. Qwen2.5-Coder-0.5B, HumanEval 164,
greedy, BF16:

| config | legacy (buggy) | fixed | Δ |
|---|---|---|---|
| parallel | 0.2683 (44/164) | 0.2866 (47/164) | +0.018 |
| **canonical (`--sequential`)** | **0.2622** (43/164) | **0.2683** (44/164) | +0.006 |
| published reference | 0.280 ± 0.10 | 0.280 ± 0.10 | — |

Both are greedy, so these differences are deterministic rather than noise. The
fix moves the canonical number *toward* the published figure, which is the
direction a correct fix should move it.

The magnitude is an order below the effects Phase 22 rests on — `+0.2070` for
`--pg-positive-only` (12 seeds, p<0.0001) and `+0.0148` per harvest doubling
(6/6 seeds, t=3.68). The bug's bias is ~3% of the former. And Phase 22's claims
are *differences between arms* measured with the same code on both sides, so a
common bias largely cancels in a paired comparison; it would only survive if
the bug acted differently per arm, and position rounding depends on sequence
length, which does not differ much across arms.

**The reservation, stated plainly.** This is one base measurement. Nothing in
the K sweep, the pre-registered `posonly` replication, or RL-vs-SFT was
re-run — that is 6–12 seeds across several arms, weeks of A100 time. What can
be said is that the bug's size is an order below the main effect sizes and its
sign is consistent, so those conclusions are unlikely to invert. That is not
the same as confirmed.

The most exposed numbers are the **K=4−K=2 and K=8−K=4 steps**. Neither was
significant to begin with (t≈1.5) and a single step is +0.0148, the same order
as the bug's bias of up to +0.018. CLAUDE.md already says only the trend is
significant, not the steps, so the conclusion does not change — but anyone
quoting an individual step should know both facts.

### The exposed region was then re-measured, and the slope holds

K=2 and K=8 re-trained with the fix on **fresh seeds** (901–903; reusing the
original six would turn a replication into a confirmation), then evaluated on
the published LCB slice — 120 problems, F32 generation, `lcb_score.py`.

| | published (6 seeds, buggy training) | re-measured (3 seeds, fixed) |
|---|---|---|
| K=2 | 0.084 | **0.0688** ± 0.0349 |
| K=8 | 0.111 | **0.1123** ± 0.0226 |
| **gap** | **+0.027** | **+0.0435** |

Paired by seed:

```
901   0.1087 -> 0.1304   +0.0217
902   0.0435 -> 0.1196   +0.0761
903   0.0543 -> 0.0870   +0.0327
                mean     +0.0435    3/3 same sign, t=2.62, df=2
```

All six seeds matched 120 problems; the completeness gate passed.

**K=8 reproduces almost exactly** — 0.1123 against a published 0.111, on
different seeds with corrected code. That is the strongest single result here.
K=2 came out lower (0.0688 vs 0.084), but at σ=0.0349 the published value sits
within 0.5σ of the new mean, so it is not distinguishable from noise. The gap
did not shrink; it widened, which is the opposite of what "the bug manufactured
the slope" predicts.

**t=2.62 at df=2 is p≈0.12 — not significant.** The 5% critical value at df=2
is 4.30, so three seeds could not have produced significance regardless; that
was stated before the run, not after. What this establishes is that the gap did
not disappear, not a new estimate of its size.

So the slope over the most exposed region is **not an artifact of the rotary
bug**. Still not re-measured: the individual K=4−K=2 and K=8−K=4 steps (only
the endpoints were run), the pre-registered `posonly` replication, and
RL-vs-SFT.

## What this licenses

- **The current product shape can run BF16 today.** Tool-use prompts are
  40–150 tokens and that band is clean. Memory 28 GB → 15 GB (two replicas per
  card) and roughly 2× decode.
- **But not as a blanket default.** Above ~300 tokens BF16 is wrong in a way
  no error reports — it returns fluent text. Serving BF16 requires an enforced
  input-length ceiling, with anything longer refused or routed to an F32
  replica.
- **The ceiling is now removed** — F32 rotary landed, BF16 matches F32 to 1000
  tokens, and serving runs BF16 at half the memory. The input-length gate
  discussed above is no longer needed.

## Batching: two shapes, one of which cannot be exact

General cross-request batching needs padding masks, continuous batching and a
scheduler. It was not the place to start, because the GPU time does not go to
serving — it goes to harvest, 3920 generations per self-improve round. Harvest
has a shape that makes batching easy, and the repair turns have one that makes
it hard.

### Same prompt, K samples — exact

Harvest and pass@k both draw K samples from one prompt, so every sequence has
identical length: no padding, no mask beyond the causal one. Agreement was
checked before speed, and exactly: each sample `i` is seeded `seed + i` in both
paths, so batch sample `i` must equal the `i`-th sequential draw token for
token. `phase23_batch_check` fails on a single mismatch.

| K | sequential | batched | speedup | identical |
|---|---|---|---|---|
| 4 | 6386 ms | 3147 ms | 2.03× | 4/4 |
| 8 | 12617 ms | 5363 ms | 2.35× | 8/8 |
| 16 | 24847 ms | 8618 ms | 2.88× | 16/16 |
| 32 | 48658 ms | 15312 ms | 3.18× | 32/32 |

End to end on a self-improve round (490 prompts × K=8, two rounds): 7.07 h →
4.35 h, **1.62×**. Less than the microbenchmark because the repair turns were
still sequential — 94 of round 0's 138 minutes.

### Different prompts — padding works, and is not bit-identical

The repair turns each carry their own failed call, so their prompts differ in
content *and length*. Batching them needs left padding plus a mask.

That was built and it runs, but it cannot be exact, and the reason is not a
bug:

| | identical |
|---|---|
| 4 prompts, different lengths (padded) | 3/4 — one diverged at token ~70 |
| 4 prompts, different content, **equal** lengths (unpadded) | 4/4 |

Left padding puts a sequence's real tokens at positions `p..max_len` instead of
`0..len`. RoPE is relative, so that shift is a no-op in exact arithmetic — but
not in floating point, where cos/sin at different absolute positions round
differently. The error is tiny and compounds until a near-tie flips. Right
padding is worse: generated tokens would sit at different offsets per row,
which one shared KV cache cannot express.

One real bug surfaced on the way, worth recording because its symptom pointed
elsewhere. Masking a pad **row** entirely leaves a softmax over all `-inf`,
which is NaN; a real query then multiplies that NaN value by a zero weight, and
`0 * NaN` is NaN. Every padded sequence returned empty output while the single
unpadded one was correct. Pad rows must be allowed to attend to earlier pads —
their outputs are never read, and as keys they are masked out of every real
query anyway.

### Length grouping: exact, and 2.06×

Since equal lengths are exact and padding is not, the repairs are grouped by
exact token length and each group is batched. The distribution makes this work
— measured over 256 repairs on families 3 and 5:

```
lengths 37..70, 14 distinct
top three lengths cover 236/256
92% of repairs sit in a group of 8 or more
```

The failed calls are near-identical templates, so their lengths collapse onto a
few values.

A/B on one round, 32 prompts × K=8:

| | time | repaired | accepted | harvest |
|---|---|---|---|---|
| grouped | **619 s** | 3 | 2 | 3/256 |
| sequential | 1273 s | 3 | 2 | 3/256 |

**2.06×, every counter identical.**

### Two phases: draw everything, then repair everything

Grouping within a prompt caps a group at K, which wastes most of what the
length distribution offers — 236 of 256 repairs share three lengths, but split
across 32 prompts those become a handful each. A repair depends only on its own
draw, so they can be collected across the whole run.

| | time | repair batches | repaired | accepted | harvest |
|---|---|---|---|---|---|
| sequential | 1273 s | — | 3 | 2 | 3/256 |
| grouped within a prompt | 619 s | — | 3 | 2 | 3/256 |
| **two-phase, run-wide** | **531 s** | **11** | 3 | 2 | 3/256 |

**2.40×**, every counter identical. The diagnostic line is the evidence:
`repair_candidates=256 distinct_lengths=13 repair_batches=11` — 256 sequential
generations became 11 batched calls.

Two things this needed:

- **A batch cap.** A run-wide length group can hold hundreds; batching all of
  them at once is an OOM rather than a speedup. `max_repair_batch` defaults to
  32, the largest size measured (3.18×), and the curve is flat past it because
  decode is bandwidth-bound on the weights.
- **Emission in draw order.** The curator's dedup depends on the order it sees
  trajectories in, so reordering would change which duplicate survives and
  therefore the harvest itself.

### On a full self-improve round

Same configuration all three times — 490 prompts × K=8, two rounds,
`--harvest-repair`:

| | round 0 generation | whole run | r0 harvest | r0 accepted |
|---|---|---|---|---|
| sequential | 220.9 min | 7.07 h | 1979 | 376 |
| same-prompt batching | 138.0 min | 4.35 h | 1979 | 376 |
| **two-phase** | **89.0 min** | **3.04 h** | **1979** | **376** |

**2.48× on generation, 2.33× end to end**, and the harvest reproduces for the
third time — which is the number that matters, since it is what training
consumes.

Grouping at this scale:

```
round 0   repair_candidates 1949   distinct_lengths 52   repair_batches 88
round 1   repair_candidates 1127   distinct_lengths 34   repair_batches 56
```

1949 sequential generations became 88 batched calls.

**A correction to the earlier distribution claim.** "Top three lengths cover
236/256" was measured on families 3 and 5 alone. Across all ten the lengths
spread much further — 13 distinct became 52 — because each family has its own
code template. The grouping still pays (1949 → 88), but the tight concentration
was a property of that two-family sample, not of the domain.

### What cannot be compared across these runs

`pass@1` could not, until the cause was found. The three runs read 0.542 /
0.542 / **0.531** at eval-*before* — same checkpoint, same configuration,
before any training.

**The cause was Python hash randomisation in the verifier.** The interpreter
randomises string hashing per process, so a snippet that iterates a set or dict
of strings returns a different answer each run — no error, just a different
number:

```
list(set(['a','b','c','d','e']))[0]   ->  a / e / b   on three runs
```

Fixed by pinning `PYTHONHASHSEED=0`. Six consecutive runs now give 52/96, and
the problem that used to flip (`idx=86`) verifies Correct every time.

A second fact fell out of fixing it: setting the variable had no effect at
first, because **`-I` implies `-E`, which discards `PYTHON*` variables** —
including that one. The interpreter now runs `-s`. Dropping `-E` costs nothing
because bwrap already starts from a cleared environment, and `-s` still keeps
the user site directory out. That combination — an isolation flag quietly
cancelling a determinism flag — is the kind of thing only a test catches, so
`same_snippet_gives_the_same_answer` and `the_flipping_snippet_never_flips` now
pin it.

### Two wrong explanations, and how they were killed

Both were inference dressed as diagnosis, and both cost a documentation
correction:

1. **"Batching changes the order the model actor consumes its RNG."** Killed by
   holding batching constant and running twice: 51/96 then 52/96.
2. **"CUDA kernels reduce with atomics, so logits differ."** Killed by
   `phase23_determinism`: five draws inside one loaded model are token
   identical, and two separate processes print the same token digest
   (`6559cc909121f31d`) at both temperature 0 and 0.8.

The instrumentation that actually settled it logs a digest and verdict per
problem. It showed exactly one problem differing out of 96, with a
**byte-identical completion and an inverted verdict** — which rules out
generation entirely and points at the verifier, where the answer was.

One self-inflicted detour worth recording: after the fix, a re-check still
showed 51/52 and was reported as "the fix did not work". It had run the
*previous* binary — built 18:22, fix built 19:50. Timestamps on the artifact
would have caught it, the same discipline CLAUDE.md gotcha #8 already demands
for CUDA builds.

### What this means for the three runs

The pass rates in the table above (0.927 / 0.865 / 0.833) were measured under
the nondeterministic verifier and cannot be compared with each other. Harvest
counts were and remain identical at 1979, which is what training consumes.
Future comparisons on this axis can use pass rates, now that the verifier is
reproducible.

## Reproducing

```bash
phase23_serve_7b --checkpoint <merged.safetensors> --dtype f32|bf16|f16

# decode cost
for n in 8 32 64 128; do curl -s localhost:8080/inference \
  -H 'content-type: application/json' \
  -d "{\"prompt\":\"Q: how many divisors does 720 have?\n\",\"max_new_tokens\":$n}"; done

# the dtype split — a short prompt separates F16 from BF16
# the length sweep — a long DIVERSE prompt separates BF16 from F32.
# Repetitive filler does not reproduce it.
```
