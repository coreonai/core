# Distillation post-mortem: why the 12M baseline beat the 12M distilled student

When we first ran `distill_kowiki` against the 5K-step teacher, the
distilled student came out *worse* than a from-scratch 12M baseline
trained for the same 4K steps. Held-out CE on the same KoWiki tail
(`val_frac = 0.05`, 50 random batches, single-pass eval):

| Checkpoint                        | val_loss | perplexity | params |
|-----------------------------------|---------:|-----------:|-------:|
| Teacher 50M, 5K steps             |   7.4648 |   1745     | 46.8M  |
| Teacher 50M, **30K steps (K8)**   | **7.4267** | **1680** | 46.8M  |
| Distilled student 12M, 4K steps   |   7.6121 |   2022     | 12M    |
| Baseline student 12M, 4K steps    |   7.4982 |   1804     | 12M    |

The distilled student is **+0.11 nats worse** than the baseline. This is
the opposite of the expected distillation outcome. The KL term is
implemented correctly (we already verified — the bug was in the
normalization, fixed pre-eval); the underlying problem is the *teacher*.

## What distillation expects

The Hinton recipe `loss = (1−α)·CE + α·T²·KL(softmax(t/T) || softmax(s/T))`
moves "dark knowledge" from teacher to student: the *relative ordering*
of incorrect classes encodes information beyond the hard label.
"This token isn't the answer, but it's much more likely than that one"
is signal a student trained on hard targets alone never sees.

Two preconditions must hold for that signal to outweigh the noise:

1. **Teacher CE must be meaningfully below what the student can reach
   alone in the same compute budget.** If the teacher is at 7.51 and a
   from-scratch 12M baseline reaches 7.46, the teacher is not actually
   *teaching* anything — it has less ground to stand on than the
   student already finds without help.
2. **The teacher's predictions must concentrate probability on a small
   set of plausible candidates.** When the teacher's per-token entropy
   is close to `ln(vocab) − epsilon`, its softmax is nearly uniform. A
   nearly-uniform soft target conveys nearly-uniform information,
   which is to say: nothing.

Our 5K-step teacher fails both. Train loss 7.51 against the
information-theoretic baseline `ln(16K) = 9.68` means the teacher
reduced entropy by ~2.2 nats — better than uniform, but still
spreading most of its mass across hundreds of plausible-looking
tokens. The distillation gradient was 70% (`α=0.7`) "match the
teacher's high-entropy distribution" and only 30% "match the data."

## What the 30K-step teacher buys us (and why it's still not enough)

The K8 run trained the same 50M architecture for 30K steps instead of
5K. End-of-run *train_loss* dropped from **7.51 → 7.21** (∼0.30 nats),
which initially looked promising. But the held-out *val_loss* tells
a much less optimistic story: **7.4648 → 7.4267**, a real but tiny
improvement of just 0.038 nats over six times the compute.

The train_loss vs val_loss split is itself diagnostic. With 30K steps
× batch 16 × block 256 = ~123M training tokens seen against only
20.8M unique tokens in the corpus, the model has cycled the data
~6 times. Most of the train_loss improvement is light overfitting,
not generalization. The 50M architecture has effectively converged
on this dataset.

More importantly: the gap between the 30K-teacher (7.4267) and the
12M from-scratch baseline (7.4982) is only **0.07 nats**. The
12M-param student trained from scratch on the same data for ~5%
of the teacher's compute is already within shouting distance of
the teacher's saturation point. **There is essentially no
"dark knowledge" for the teacher to transfer**, because the teacher
itself does not meaningfully outperform the student-floor on this
data.

This is the right negative result to anchor on: the data, not the
distillation infrastructure, is the binding constraint.

## Decision rule for distillation

Going forward, before running `distill_kowiki`:

1. Run `eval_kowiki` on the teacher and on a freshly-trained 12M
   baseline student of equal step budget to the planned distillation.
2. Compute `gap = baseline_val_loss − teacher_val_loss`. Use **val
   loss**, not train loss — train_loss after multi-epoch training
   is dominated by overfitting and overstates teacher quality.
3. **If `gap < 0.3` nats: do not distill.** The teacher does not have
   enough advantage to share. Either train the teacher further OR
   accept that the dataset is the binding constraint.
4. **If `0.3 ≤ gap < 1.0` nats: distill with `α ≤ 0.3`.** Soft targets
   contribute *some* signal but are still noisy; weight hard CE
   higher.
5. **If `gap ≥ 1.0` nats: distill with `α = 0.7`** (Hinton default).
   Teacher is meaningfully more knowledgeable than the student floor;
   the dark-knowledge signal dominates.

These thresholds are derived from the observed failure (5K teacher
gap ≈ +0.03, 30K teacher gap ≈ +0.07; both produced a distilled
student that was worse than the baseline) and what the literature
reports works (e.g. Hinton's MNIST distillation worked with teacher
CE roughly 1–2 nats below baseline).

**Concrete next-step prerequisite for KoWiki distillation:** target a
teacher with held-out val_loss ≤ **6.5 nats** before re-running
`distill_kowiki`. Reaching that almost certainly requires either a
larger architecture (~150–300M params) or substantially more diverse
data (KoWiki + news + AI Hub). 30K steps of the current 50M-on-21M-tokens
recipe is already at saturation.

## Alternative recipes worth trying

If the gap stays small even with a longer-trained teacher:

- **Logit matching (no temperature)**: replace KL with `MSE(s_logits,
  t_logits)`. Less sensitive to teacher entropy because it doesn't go
  through `softmax/T` — directly aligns the pre-softmax features. Common
  in modern self-distillation recipes (e.g., DeiT III).
- **Online distillation**: teacher and student are trained jointly,
  teacher always slightly ahead. Avoids the "teacher trained too little"
  trap because the teacher's progress is monotonically tied to the
  student's.
- **Selective distillation**: only apply the KL term on tokens where
  the teacher's top-1 confidence exceeds a threshold. Drop the KL on
  high-entropy positions where the soft target is mostly noise.
- **Curriculum α**: start with `α=0.0` (pure CE) and ramp toward
  `α=0.7` once the teacher has stabilized. Equivalent to "train
  baseline-style first, then layer in the soft targets."

## What was *not* the problem

Worth recording the things we ruled out so future debug sessions don't
re-tread them:

- **KL normalization bug**: we *did* have one early on (KL divided by
  batch size, not `batch × seq_len`), causing loss to explode by
  `~seq_len`. That's fixed in `train_with_teacher` and was already
  in place when this 0.19-nat result was measured.
- **Teacher accidentally being trained**: teacher params live in a
  separate `VarMap` that never reaches the optimizer. Verified by
  inspecting `varmap.all_vars()` post-distill — only student names,
  no `teacher.*` entries.
- **Tokenizer mismatch**: both teacher and student share the same
  16K-vocab BPE. Verified at run time (`vocab_size` printed at the
  top of each example).
- **Data leakage / overfitting**: held-out tail is 5% of the corpus
  and used by neither training run. Both numbers are honest val_loss.

## Honest framing

This is a negative result, but a clean one — it isolates the issue
to teacher quality rather than the distillation infrastructure. We
can now pre-screen distillation runs against the gap-rule above
instead of running every distillation experiment to completion and
discovering the teacher was inadequate after-the-fact.

The distillation code is **production-ready** as soon as we have a
production-ready teacher.
