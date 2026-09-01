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
| **transfer** — emits a dispatchable call | 12/12 | 12/12 |
| **transfer** — computes the right answer | 4/12 | **2/12** |
| **transfer** — tool dispatch errors | 5 | **8** |
| **retention** — five unharvested families | ~0.988 | **0.806** |

Not a null, a regression — but a smaller one than first reported, and the
first report's mechanism was wrong.

> **Correction.** This table originally read 4/12 emitted and 1/12 correct,
> explained as "the model stops at `(python import math\n` without ever
> closing the call". The model does not stop there. The measurement used
> `--stop "\n"`, and the narrow checkpoint writes **multi-line** snippets:
>
> ```text
> (python import math
> print(sum(1 for i in range(1,46) if math.gcd(i,45)==1)))
> ```
>
> A newline stop cuts that after `import math`, so the call never closes and
> is counted as "emitted no call". The eval path has no stop sequence, which
> is why the same checkpoint measured family 5 at 1.000 there while appearing
> to emit nothing here. Re-measured with the stop at the call boundary
> (`")\n"`), emission is 12/12 and the real cost is in correctness (4/12 →
> 2/12) and dispatch errors (5 → 8). The wide-harvest checkpoints are
> unaffected — they write single-line snippets on unfamiliar prompts, so the
> newline stop never bit them, and their numbers are identical under both
> rulers. `phase23_python_tool_7b` and `phase23_ask` now default to `")\n"`.

The over-generalisation itself is real and is what the correction exposes:
`imports: 87/160` on families that never needed one, and the multi-line
import idiom appearing on prompts that call for a one-liner. It broke working
one-liners; it did not stop the model from emitting calls.

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
| transfer — emits a call | 12/12 | 12/12 | **11/12** |
| transfer — correct | 4/12 | 2/12 | **4/12** |
| imports where not needed | — | 87/160 | **0/160** |
| answer before the call | ~0 | 80/98 | **0/98** |

Replay recovers what the narrow run cost, and the targets are still fully
learned. Retention comes out at 1.000 — marginally *above* the
pre-loop 0.988, because family 7 (0.938) was pulled up too.

The over-generalisation is gone and the selectivity is exact: **0/160 imports
on families that do not need one, 49/98 on the targets — precisely the 49
family-5 prompts.** The model learned *when* to import, not "always import".

Transfer returns to the pre-loop level: 11/12 emitted, 4/12 correct against
the narrow run's 2/12. Note what this is and is not — the regression is
repaired, but there is **no transfer gain**. Whatever the loop taught did not make the model better at
unfamiliar families, only no worse. The next section takes that apart.

## Why there is no transfer gain

Transfer emission was repaired (4/12 → 11/12) but correctness stayed at the
pre-loop 4/12. Reading all twelve trajectories, the eight failures split into
two groups, and only one of them is anything the loop trained on.

**Divisors, 4/4 both before and after.** Already worked; untouched.

**Fibonacci, 0/4 → 0/4 — wrong mathematics, not a tool-contract failure.**

```
(python print(sum([1<<i for i in range(20) if (i*(3*i-1))%20==0])))
  → 36993, want 6765
```

No `NameError`, no syntax error — the snippet runs clean and computes the
wrong thing. The model invents a closed form that does not exist. The loop
trained on contract corrections, so it had no reason to touch this. The only
visible change is that `print(fibonacci(40))` (an undefined helper) stopped
appearing.

**Collatz, 0/4 → 0/4 — `itertools` used without importing.**

```
(python print(sum(1 for i in itertools.takewhile(...))))
  → NameError: name 'itertools' is not defined
```

This is *exactly* the failure class the loop fixed for `math`. It did not
transfer.

### The loop learned the instance, not the principle

What it acquired is "when you need gcd, `import math`" — not "this tool has
an empty namespace, so import whatever you reference." Three measurements
agree:

- retention families: **0/160** imports, where none is needed
- target families: **49/98** imports — exactly the 49 family-5 prompts
- transfer set: **0 imports in 12 problems**, including the 4 that need
  `itertools`

Which follows from the harvest. Of the eight families, exactly one needs an
import, and it is always `math`. No example requiring any other module ever
entered training, so there was nothing to generalise from. An earlier draft
of this document said the model "learned WHEN to import"; the sharper reading
is that the module choice is bound to the situation it was learned in.

### Run 4 — testing whether the rule is learnable (it was not)

Two families were added whose natural solution reaches for a module other
than `math`: a digit product (`functools.reduce` — the model writes
`reduce(...)` unimported and scores 0/32) and twice the median of the
divisors. **`itertools` was deliberately left out**, because the transfer
probe's Collatz problems reach for it; training on it would have measured
"does the same module carry over" instead of "was the rule learned".

490 prompts over ten families, K=8, 2 rounds: pass@1 0.542 → 0.708 → 0.927.

| | 8 families | **10 families** |
|---|---|---|
| target 3 / 5 / 8 | 1.000 / 1.000 / — | 1.000 / 1.000 / **1.000** |
| target 9 (median) | — | 0.367 |
| retention (0,1,2,4,6,7) | — | 0.917 |
| transfer — emits a call | 11/12 | 10/12 |
| transfer — correct | 4/12 | **4/12** |
| transfer — dispatch errors | 3 | 2 |
| imports where not needed | 0/160 | **0/192** |

**The hypothesis is refuted.** With two distinct modules in training, the
model still writes

```
(python print(sum(1 for i in itertools.takewhile(...))))
  → NameError: name 'itertools' is not defined
```

on all four Collatz problems. Zero imports across the twelve transfer
problems. Family 8 reached 1.000, so a *new* module is perfectly learnable —
the ability simply does not extend to a module the harvest never contained.
What is learned is "for gcd, import math; for reduce, import functools", not
"this tool starts with an empty namespace".

Two side findings from the same run:

- **Family 6 (largest prime factor) sits at 0.500**, and the entire drop in
  retention (1.000 → 0.917) is that one family. The earlier 8-family
  retention figure omitted family 6, so it read 1.000; including it is the
  honest number. The five others are all still 1.000.
- **Family 9 only reached 0.367.** A median needs a sort and an even/odd
  branch — an algorithm problem, not a contract violation, so it belongs with
  Fibonacci among the failures this loop does not address.

So the transfer decomposition holds exactly as stated: four Fibonacci
failures the loop can never fix, and four Collatz failures it could fix in
principle but does not, because module knowledge does not generalise across
modules. Widening the harvest is not the lever here — going from one module
to two changed nothing about the third.

### Correction — the unit is the phrase template, not the module

The claim above ("a new module is learnable; the namespace rule is not") is
still directionally right and still incomplete. Asking the Run-4 checkpoint
outside the transfer set sharpened the boundary:

| prompt | behaviour |
|---|---|
| "product of the nonzero digits of N cubed" (trained phrasing) | `import numpy as np` then `np.prod(...)` ✓ — `numpy` was never in the harvest |
| "product of the digits of 234" (same idea, different phrasing) | no `reduce` import ✗ |
| "distinct permutations of the digits" | no `math.factorial` import ✗ — even though `math` was harvested in family 5 |

So what generalises is not "modules" and not "the empty-namespace rule". It
is the **learned phrase-level snippet template**. Inside a matched template
the module slot can be swapped (`functools` → `numpy`); outside that
template, nothing transfers — including modules the harvest did contain.
The Collatz/`itertools` miss is the same fact from the other side: no
trained phrase carried an `itertools` slot, so none appeared.

### One regression worth watching

```
Collatz n=27:  exec_ok=false  said=true   A: 111   (correct)
```

The tool errored and the model stated the right answer anyway. That is the
pre-SFT invent-an-answer behaviour resurfacing on an out-of-domain prompt,
i.e. the grounding that `--sabotage` established may hold only in-domain.
Worth measuring directly: run `--sabotage` on the transfer set.

## A stop sequence is part of the ruler

The correction above is worth generalising. `--stop "\n"` was chosen when the
model only ever wrote one-line calls; the self-improve loop then taught it a
multi-line idiom, and the stop silently began truncating valid output. The
measurement kept running and kept producing a plausible number.

Two properties made it hard to notice. It only bit the *narrow* checkpoint —
the one whose behaviour had changed — so it looked like a finding about that
checkpoint rather than about the harness. And it degraded the metric in the
direction the hypothesis predicted, which is the worst case: a broken ruler
that agrees with you.

The rule: when a model's output format changes, re-check every piece of the
harness that assumes the old format. A stop sequence, a truncation rule, and
a parser are all part of the measurement, not neutral plumbing.

## Using it

`phase23_ask` is the interactive entry point — a question in, the call the
model writes, the value the interpreter returns, and the answer it gives.
Seventeen assorted questions were tried against the ten-family checkpoint;
fourteen came out right, and the spread is wider than the harvest suggests.

```text
how many trailing zeros does 1000! have?             249
how many numbers from 1 to 200 are coprime to 200?    80   n far outside the trained 12..60
what is the sum of all primes below 1000?          76127
how many perfect squares are below 500?               22   family never harvested
how many vowels are in the word encyclopedia?          5   string, not number theory
how many bits are set in 12345?                        6
train travels 60 km in 45 min, km in one hour?      80.0   word problem
what is the 15th prime number?                        47   after correcting its own first attempt
```

The three failures are all one thing — a module it was never trained to
import:

```text
how many days between 2024-01-01 and 2024-12-25?   datetime  NameError
how many anagrams does the word banana have?       itertools NameError
what is the product of the digits of 234?          reduce    NameError
```

`math` and `numpy` come out fluently; nothing else does. This is the
phrase-template boundary from the section above, seen from the other side.

Two things worth recording:

- **The scope claim in this repo was too pessimistic.** An earlier version of
  `phase23_ask`'s header called the model good for "counting and
  number-theory shapes it was harvested on and close neighbours". A string
  count and a unit-conversion word problem both work. The measurements above
  replaced that claim.
- **It self-corrects, unprompted.** Asked for the 15th prime it first computed
  the *count* of primes below 1000, saw 168, re-read the question, indexed
  `[14]` and answered 47. No training trajectory looked like that; the
  agentic loop only ever handed back a tool result. That observation is what
  `--harvest-repair-context` was built to exploit.

One reporting bug this shook out, since it is the kind that flatters or
maligns a model silently: the answer display took the *first* `A:` line, so
the discarded 168 was shown instead of the final 47 — a successful
self-correction read as a wrong answer. It now takes the last. Answers can
also arrive as floats (`80.0`), which matters if something downstream
compares strings.

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
- **A stop sequence is part of the ruler.** `--stop "\n"` was correct until
  the loop taught the model multi-line calls, then it truncated them and
  reported "emitted no call". It bit only the checkpoint whose behaviour had
  changed, and it moved the number the way the hypothesis predicted.
- **Replay is free wherever verification is free.** Harvesting saturated
  tasks costs only generation, and it bought back 0.19 of retention and
  7/12 of transfer emission here.
- **A loop generalises only as far as its harvest varies.** Exactly one of
  eight families needed an import and it was always `math`, so the model
  learned that instance and failed the same way on `itertools`. If you want a
  rule learned, the harvest has to contain more than one instance of it.
- **The transferable unit is the phrase-level snippet template, not the
  module.** A never-harvested module (`numpy`) appears when the prompt matches
  a trained phrasing; a harvested module (`math`) disappears when the
  phrasing changes. Vary modules *and* phrasings if you want anything broader
  than a template.
- **Check what class each failure belongs to before blaming the loop.** Half
  the transfer failures here were wrong mathematics on code that ran cleanly
  — nothing a tool-contract loop was ever going to fix, and averaging them
  into one "transfer" number hides that.

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
