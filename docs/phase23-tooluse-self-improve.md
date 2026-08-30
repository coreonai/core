# Phase 23 — tool-use self-improve on the 7B

The self-improve loop, pointed at a capability the model did not have, with
a verifier that costs nothing. Everything here is measured on
Qwen2.5-Coder-7B (base, not Instruct) + LoRA r=16 α=32, F32 inference /
BF16 training, driven entirely through the Pekko actor stack.

## The gap this was aimed at

Two prior measurements set it up:

- The tool-call format transfers to task families the model has never seen
  (12/12 dispatchable calls emitted), but **task-solving does not** — 4/12
  correct, the same as the base model. It learned the grammar and the
  discipline of using the tool's result, not how to solve new problems.
- What the format SFT actually bought was **grounding**, not format. The base
  model with unresolved few-shot examples already emits 12/12 calls and gets
  10/12 correct results — then writes `A: 20826` for a tool that returned
  `17575`. Under `--sabotage 1` only 3/12 of its answers track the tool,
  against 12/12 after SFT.

So: it can call the tool and will believe it. It cannot write correct code
for an unfamiliar problem. That is a learning problem with a free verifier.

## The domain, and why verification is free

`ToolUsePythonDomain` — eight task families parameterised by `n`, disjoint
from both the SFT families (saturated, no headroom) and the
`phase23_python_tool_7b --novel` set (kept clean as a transfer probe).

Every prior domain paid for verification: `RustCodeDomain` shells out to
`cargo`, `HumanEvalDomain` runs a test harness. Here the answer is an integer
computed in Rust when the question was generated, and the candidate is
executed by the very tool the model is learning to call. **784 completions
verify in 6.9 seconds.**

Reward hacking is guarded where it is unambiguous and *counted* where it is
not. A bare `print(<literal>)` is rejected outright. The tighter rule is
wrong: "the snippet must mention `n`" rejects correct solutions — the model
writes `range(1, 38)` for n=37, which mentions neither operand of the
question. Across every run, the weak hardcode signal fired 0 times.

## The starting point is bimodal, and the zeros are not a reasoning gap

Per-family pass@1 of the format-SFT'd model, 32 values of `n` each:

| family | pass@1 |
|---|---|
| sum of cubes | 1.000 |
| sum of divisors | 1.000 |
| multiples of 7 | 1.000 |
| **trailing zeros of n!** | **0.000** |
| digit sum of n³ | 1.000 |
| **coprime count (φ)** | **0.000** |
| **largest prime factor** | **0.000** |
| sum of primes below n | 0.938 |

0/32 across 32 different `n` is systematic, not noise. And the failures are
not wrong algorithms:

```
(python print(sum(1 for i in range(1,13+1) if math.gcd(i,13)==1)))
  → NameError: name 'math' is not defined
```

That is a correct Euler-φ computation. The tool is fine — `import math; ...`
works — the model simply assumes a preloaded namespace. **It is a tool
contract gap, not a reasoning gap.**

## Sampling cannot reach it

The decision was to leave the tool alone and let the loop learn the contract.
That requires the harvest to find *something*, and it cannot:

- **0 of 576 sampled snippets contained an import**, at temperature 0.8.
- pass@16 stays 0/12 on two of the three failing families.

A turn-1-only harvest is therefore empty and the loop cannot start. This is
the cold-start failure this codebase has hit before.

The information exists in exactly one place: the tool's own error message.
Hence `Domain::repair_prompt` (splice the error where the result would have
gone, the same shape `agentic_generator_actor` produces) and
`GeneratorActor::with_repair_failures` (retry once, harvest the fix only if
it verifies). Measured 4/96 repaired against 0/24 first turns.

What the model learns from the error is **not** "add the import" — it wrote
one 0/96 times in the repair turn too. It drops `math` and writes the
arithmetic directly:

```
1st: print(math.factorial(18))                    → NameError
2nd: print(sum([18//5**i for i in range(1,10)]))  → 3, correct
```

It learns the tool's contract, which is the more general lesson.

## Run 1 — narrow harvest (families 3+5 only)

98 prompts, K=8, 3 rounds, self-repair on.

| round | harvest | of which repaired | first-turn | pass@1 |
|---|---|---|---|---|
| 0 | 5/784 | 5 | **0** | 0.000 → 0.266 |
| 1 | 192/784 | 15 | 177 | 0.266 → 0.750 |
| 2 | 567/784 | 38 | 529 | 0.750 → 0.844 |

The first-turn column is the point. In round 0 *every* harvested example came
from the repair path; turn-1 yield was zero, so without repair the corpus is
empty and the loop never starts. By round 1 the model solves 177 first try.
**The repair turn is a bootstrap ladder, not a permanent crutch.**

### …and it contaminated its own harvest

Completions came out as:

```
A: 2
(python print(sum([12//5**i for i in range(1,10)])))
```

The answer is stated **before** the tool runs. Measured 17/98 on the round-0
checkpoint and **80/98 (82%)** on round 2, from ~0 at the start.

The cause is the harvest, not the model. A repaired completion is generated
in a two-turn context where an answer line is natural, and it was paired
verbatim with the ORIGINAL prompt. Five such pairs seeded 17%; from round 1
on, the loop harvested its own contaminated first-turn output and amplified
it. The verifier cannot see this — the executor only ever receives the call —
so every other number stayed clean while the format degraded.

It also destroys the property `--sabotage` exists to establish: an answer
written before the tool ran cannot have come from the tool.

**Fix:** `truncate_completion` reduces a completion to the first call,
dropping text on *both* sides. The detector was moved to the raw text in the
same change — truncating the prefix would otherwise make the metric report 0
by construction and blind it to the artifact it exists to catch.

## Run 2 — same, under the fix

| round | harvest | repaired | pass@1 |
|---|---|---|---|
| 0 | 5/784 | 5 | 0.000 → **1.000** |
| 1 | 782/784 | 0 | 1.000 → 1.000 |
| 2 | 784/784 | 0 | 1.000 → 1.000 |

Clean pairs saturate both target families in **one** round, where the
contaminated pairs took three rounds to reach 0.837. The `A: <guess>` prefix
was not a scratchpad; it was noise diluting the signal.

**Ruler check.** `truncate_completion` also runs on the eval path, so the fix
could have moved the measurement rather than the model. Re-scoring the
contaminated run's round-2 checkpoint under the new binary gives 0.837 /
family 3 1.000 / family 5 0.673 / 80-of-98 contaminated — identical to its
pre-fix numbers. The fix changed the training pairs, not the ruler.

Family 5's learned solution is correct, and multi-line:

```
(python import math
print(sum(1 for i in range(1,12+1) if math.gcd(i,12)==1)))
```

Verified by hand: 4, want 4. The loop learned a **different strategy per
family** — import for φ, math-free Legendre for trailing zeros — starting
from a model that wrote an import 0 times in 576 samples.

That also refuted this repo's own docs: `python_tool` claimed a snippet had
to be one line. The real constraint is only that no internal `)` may be
followed by a newline. The model read the grammar more carefully than the
comment did.

### The cost

Two measurements, same ruler, loop-before vs loop-after:

| | before | after (narrow) |
|---|---|---|
| **transfer** — emits a dispatchable call | 12/12 | **4/12** |
| **transfer** — computes the right answer | 4/12 | **1/12** |
| **retention** — five unharvested families | ~0.988 | **0.806** |

Not a null, a regression. Family 1 halved (1.000 → 0.500). The mechanism is
visible in the failures: the model reaches for the multi-line import idiom
everywhere, and on an unfamiliar prompt it stops at `(python import math\n`
without ever closing the call. 8 of 12 novel problems produced no parseable
call at all — which is why dispatch errors were 0.

`imports: 87/160` on families that never needed one. It over-generalised
"always import" and broke working one-liners.

Saturating in a single round was the warning. Once there is nothing left to
learn, further rounds only narrow.

## Run 3 — widened harvest (all eight families)

The fix for a narrowing loop is replay, and here replay is free: harvest the
saturated families too. 392 prompts, K=8, 2 rounds.

One thing had to change with it. In the widened pool the five saturated
families contribute ~1500 harvested pairs while the two targets contribute
~5 — **0.3% of the corpus**. At the previous 30 training steps (120 examples
seen) those five pairs are essentially never sampled, and the run would have
reported "widening does not work" for reasons that have nothing to do with
replay. Training is cheap here (800 steps ≈ 2 minutes), so steps were scaled
to cover the corpus ~2×.

| round | harvest | repaired | pass@1 |
|---|---|---|---|
| 0 | 1956/3136 | 13 | 0.635 → 0.823 |
| 1 | 2708/3136 | 1 | 0.823 → **0.990** |

### All three axes, one ruler

| | before any loop | narrow (3+5) | **wide (all 8)** |
|---|---|---|---|
| target family 3 | 0.000 | 1.000 | **1.000** |
| target family 5 | 0.000 | 1.000 | **1.000** |
| retention, five families | ~0.988 | 0.806 | **1.000** |
| transfer — emits a call | 12/12 | 4/12 | **11/12** |
| transfer — correct | 4/12 | 1/12 | **4/12** |
| imports where not needed | — | 87/160 | **0/160** |
| answer before the call | ~0 | 80/98 | **0/98** |

Replay recovers everything the narrow run cost, and the targets are still
fully learned. Retention comes out at 1.000 — marginally *above* the
pre-loop 0.988, because family 7 (0.938) was pulled up too.

The over-generalisation is gone and the selectivity is exact: **0/160 imports
on families that do not need one, 49/98 on the targets — precisely the 49
family-5 prompts.** The model learned *when* to import, not "always import".

Transfer returns to the pre-loop level: 11/12 emitted, 4/12 correct. Note
what this is and is not — the regression is repaired, but there is **no
transfer gain**. Whatever the loop taught did not make the model better at
unfamiliar families, only no worse.

## What to reuse

- **A free verifier does not make a harvest safe.** It checks the answer, not
  the method, and it is blind to anything outside what the tool receives. A
  format artifact grew from 0 to 82% with every other metric clean.
- **Harvested trajectories must be trimmed to what you intend to train.**
  Pairing a completion generated in one context with a prompt from another is
  how the contamination entered, and self-harvest amplifies it.
- **A metric must be measured before the fix that would hide it.** Moving the
  contamination detector onto the truncated string would have reported 0 by
  construction.
- **Changing a function on both the training and eval paths requires a ruler
  check** — re-score an old checkpoint with the new binary before believing a
  gain.
- **Saturating in one round is a warning, not a success.** Narrowing follows.
- **When widening a harvest, scale training steps to the corpus.** A rare
  signal at 0.3% is invisible at a step count tuned for a small pool.
- **Replay is free wherever verification is free.** Harvesting saturated
  tasks costs only generation, and it bought back 0.19 of retention and
  7/12 of transfer emission here.

## Reproducing

```bash
# baseline, per family (this is the measurement that finds the headroom)
phase23_tooluse_self_improve --init-dir <fmt-sft-dir> --baseline \
    --n-lo 12 --n-hi 43

# does sampling reach the behaviour at all?
phase23_tooluse_self_improve --init-dir <dir> --baseline --baseline-k 16 \
    --families 3,5 --repair

# the loop
phase23_tooluse_self_improve --init-dir <dir> --trainer-gpu 1 \
    --n-lo 12 --n-hi 60 --rounds 2 --samples-per-prompt 8 \
    --train-steps 800 --harvest-repair

# transfer, never in any harvest pool
phase23_python_tool_7b --checkpoint <merged.safetensors> --no-shots --novel
```

`--init-dir` is a directory holding the starting weights as
`model.safetensors` beside the snapshot's `config.json` and `tokenizer.json`
(symlinks are fine). It must be the format-SFT'd checkpoint, not the base:
a base model prompted 0-shot emits nothing dispatchable, so the harvest is
empty and the loop cannot bootstrap.
