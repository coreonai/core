# Vendored candle-core 0.10.2 — LoRA peak-memory patch

This is an **unmodified copy of `candle-core` 0.10.2 from crates.io** except
for a single hunk in `src/backprop.rs`. It is wired into the workspace via
`[patch.crates-io]` in the root `Cargo.toml`, so every crate in the graph
(candle-nn, candle-transformers, nanogpt-rs, llm-actors) builds against it.

## The change

`Op::Matmul` backward previously computed and stored a gradient for **both**
operands unconditionally:

```rust
let lhs_grad = grad.matmul(&rhs.t()?)?;
let lhs_sum_grad = grads.or_insert(lhs)?;      // zeros_like(lhs), kept in GradStore
*lhs_sum_grad = lhs_sum_grad.add(&lhs_grad)?;

let rhs_grad = lhs.t()?.matmul(&grad)?;
let rhs_sum_grad = grads.or_insert(rhs)?;      // zeros_like(rhs) — full weight-sized!
*rhs_sum_grad = rhs_sum_grad.add(&rhs_grad)?;
```

In a LoRA setup ~every weight is a **frozen** (non-variable) leaf, so candle
allocated a full weight-sized gradient in the `GradStore` for every frozen
matmul and kept them all alive for the whole backward pass. That is the
dominant peak-memory cost of LoRA fine-tuning.

The patch guards each side with `track_op()`
(`track_op() == is_variable || op.is_some()` — exactly "this operand needs a
gradient"):

```rust
if lhs.track_op() { /* ... lhs grad ... */ }
if rhs.track_op() { /* ... rhs grad ... */ }
```

A frozen weight leaf is neither a variable nor a computed node, so its grad is
skipped. Variables (LoRA A/B) and intermediate activations are unaffected, so
gradients — and therefore training dynamics — are identical.

## Why it matters

Measured LoRA-training peak GPU memory (Qwen2.5-Coder, bf16, seq-len ~6):

| model | base | peak (upstream) | peak (patched) |
|-------|------|-----------------|----------------|
| 0.5B  | ~1 GB  | 4.36 GB (~4.4×) | **1.54 GB (~1.5×)** |
| 1.5B  | ~3 GB  | 12.4 GB (~4.1×) | **3.56 GB (~1.2×)** |
| 7B    | 15 GB  | **OOM (>60 GB)** | **15.2 GB — fits a 40 GB A100** |

Loss trajectories are bit-identical before/after on 0.5B and 1.5B, and the
workspace's full test suite (179 tests, incl. full-finetune / EWC / DPO
backprop where weights *are* variables) passes unchanged.

## Upstreaming

This is a general optimization, not workLLM-specific. The same single-hunk
diff is submitted upstream: huggingface/candle#3773. Once released upstream, drop
the `[patch.crates-io]` entry and this vendor directory and depend on the
published version.

Diff vs upstream: see `git log` for the vendoring commit, or regenerate with
`diff -u <crates.io candle-core-0.10.2>/src/backprop.rs src/backprop.rs`.
