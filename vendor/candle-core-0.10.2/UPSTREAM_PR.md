# Upstream PR draft — huggingface/candle

Target: `huggingface/candle`, file `candle-core/src/backprop.rs`
(current `main` has the identical unguarded code, lines 457–467).

## Title

    Skip gradient computation for frozen operands in matmul backward

## Body

### Problem

`Op::Matmul` backward computes and stores a gradient for **both** operands
unconditionally:

```rust
let lhs_grad = grad.matmul(&rhs.t()?)?;
let lhs_sum_grad = grads.or_insert(lhs)?;
*lhs_sum_grad = lhs_sum_grad.add(&lhs_grad)?;

let rhs_grad = lhs.t()?.matmul(&grad)?;
let rhs_sum_grad = grads.or_insert(rhs)?;   // zeros_like(rhs) — full weight-sized
*rhs_sum_grad = rhs_sum_grad.add(&rhs_grad)?;
```

When one operand is a **frozen** (non-variable) weight — the norm for
LoRA / partial fine-tuning, where the base model is frozen and only small
adapters are trainable — candle still allocates a full weight-sized
gradient (`zeros_like`) in the `GradStore` for it and keeps all of them
alive for the whole backward pass. Those gradients are never read (the
weight isn't a variable), but they dominate peak memory.

### Fix

Guard each side with `track_op()`
(`track_op() == is_variable || op.is_some()` — precisely "this operand can
receive a gradient"). An operand that is neither a variable nor a computed
node cannot need a gradient, so skipping it is always safe.

```rust
if lhs.track_op() {
    let lhs_grad = grad.matmul(&rhs.t()?)?;
    let lhs_sum_grad = grads.or_insert(lhs)?;
    *lhs_sum_grad = lhs_sum_grad.add(&lhs_grad)?;
}
if rhs.track_op() {
    let rhs_grad = lhs.t()?.matmul(&grad)?;
    let rhs_sum_grad = grads.or_insert(rhs)?;
    *rhs_sum_grad = rhs_sum_grad.add(&rhs_grad)?;
}
```

### Impact

Measured LoRA-fine-tuning peak GPU memory (Qwen2.5-Coder, bf16):

| model | frozen base | peak before | peak after |
|-------|-------------|-------------|------------|
| 0.5B  | ~1 GB  | 4.36 GB | 1.54 GB |
| 1.5B  | ~3 GB  | 12.4 GB | 3.56 GB |
| 7B    | 15 GB  | OOM (>60 GB, 40 GB card) | 15.2 GB (fits) |

Training results are unchanged — loss trajectories are bit-identical
before/after, since only gradients that were computed-and-discarded are
skipped. Full-finetune paths (weights are variables → `track_op()` true)
are unaffected.

### Notes

The same guard could be applied to other binary ops that unconditionally
grad both operands; this PR keeps the scope to matmul, the dominant cost.
