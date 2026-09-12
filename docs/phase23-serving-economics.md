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

## Still open

- **No batching.** Concurrent requests serialise; `GenerateTokens` takes one
  prompt.
- **The agentic loop re-prefills.** Each step regenerates against the whole
  running buffer. Cheap at 150 tokens, not at 1000.
- **Load time.** Minutes per replica, no warm pool.
- **Response shape is inconsistent**: the non-tool path returns the completion
  alone, the agentic path returns prompt+completion as `completion`. Caught
  while writing a probe against it, which is how long it takes to bite.

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
