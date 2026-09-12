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

## What this licenses

- **The current product shape can run BF16 today.** Tool-use prompts are
  40–150 tokens and that band is clean. Memory 28 GB → 15 GB (two replicas per
  card) and roughly 2× decode.
- **But not as a blanket default.** Above ~300 tokens BF16 is wrong in a way
  no error reports — it returns fluent text. Serving BF16 requires an enforced
  input-length ceiling, with anything longer refused or routed to an F32
  replica.
- **The real fix is F32 rotary with BF16 weights**, which removes the ceiling.
  The position-dependence measured above is the evidence that it targets the
  right operation.

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
