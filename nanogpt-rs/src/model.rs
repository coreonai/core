//! GPT model — decoder-only transformer, Candle backend.
//!
//! Layout mirrors Karpathy's nanoGPT (`model.py`): token + position embeddings,
//! N × {LayerNorm → CausalSelfAttention → residual → LayerNorm → MLP → residual},
//! final LayerNorm, weight-tied head.

use candle_core::{DType, Device, IndexOp, Result as CResult, Tensor};
use candle_nn::{
    layer_norm, ops, rms_norm, Embedding, Init, LayerNorm, Linear, Module, RmsNorm, VarBuilder,
};

use crate::config::{ActivationKind, GPTConfig, NormKind, NormPosition};

const INIT_STD: f64 = 0.02;

fn small_normal(stdev: f64) -> Init {
    Init::Randn { mean: 0.0, stdev }
}

/// GPT-style linear: weight ~ N(0, 0.02), no bias by default.
fn lin(in_dim: usize, out_dim: usize, bias: bool, vb: VarBuilder) -> CResult<Linear> {
    let w = vb.get_with_hints((out_dim, in_dim), "weight", small_normal(INIT_STD))?;
    let b = if bias {
        Some(vb.get_with_hints(out_dim, "bias", Init::Const(0.0))?)
    } else {
        None
    };
    Ok(Linear::new(w, b))
}

/// Linear for residual projections: scale stdev by 1/sqrt(2 * n_layer) per GPT-2.
fn lin_resid(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    n_layer: usize,
    vb: VarBuilder,
) -> CResult<Linear> {
    let std = INIT_STD / ((2.0 * n_layer as f64).sqrt());
    let w = vb.get_with_hints((out_dim, in_dim), "weight", small_normal(std))?;
    let b = if bias {
        Some(vb.get_with_hints(out_dim, "bias", Init::Const(0.0))?)
    } else {
        None
    };
    Ok(Linear::new(w, b))
}

fn embed(num: usize, dim: usize, vb: VarBuilder) -> CResult<Embedding> {
    let w = vb.get_with_hints((num, dim), "weight", small_normal(INIT_STD))?;
    Ok(Embedding::new(w, dim))
}

/// Linear initialized to all-zeros (used for LoRA's `lora_b` so the adapter
/// contributes zero at init — model is identical to no-LoRA case).
fn lin_zero(in_dim: usize, out_dim: usize, vb: VarBuilder) -> CResult<Linear> {
    let w = vb.get_with_hints((out_dim, in_dim), "weight", Init::Const(0.0))?;
    Ok(Linear::new(w, None))
}

/// Reusable low-rank adapter `(in → rank → out)` with lora_a normal-init,
/// lora_b zero-init. `forward` returns just the delta (`scale * b(a(x))`) —
/// callers add it to the wrapped Linear's output.
pub struct LoraAdapter {
    a: Linear,
    b: Linear,
    scale: f64,
}

impl LoraAdapter {
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        rank: usize,
        alpha: f32,
        vb: VarBuilder,
    ) -> CResult<Self> {
        let a = lin(in_dim, rank, false, vb.pp("lora_a"))?;
        let b = lin_zero(rank, out_dim, vb.pp("lora_b"))?;
        let scale = (alpha / rank as f32) as f64;
        Ok(Self { a, b, scale })
    }

    pub fn delta(&self, x: &Tensor) -> CResult<Tensor> {
        let h = self.a.forward(x)?;
        let y = self.b.forward(&h)?;
        y * self.scale
    }
}

/// `Some(adapter)` when `lora_rank > 0`, else `None` for back-compat.
fn maybe_lora(
    in_dim: usize,
    out_dim: usize,
    cfg: &GPTConfig,
    vb: VarBuilder,
) -> CResult<Option<LoraAdapter>> {
    if cfg.lora_rank > 0 {
        Ok(Some(LoraAdapter::new(
            in_dim,
            out_dim,
            cfg.lora_rank,
            cfg.lora_alpha,
            vb,
        )?))
    } else {
        Ok(None)
    }
}

/// Add a LoRA adapter's delta to `y` if present, else passthrough.
fn lora_add(y: Tensor, adapter: &Option<LoraAdapter>, x: &Tensor) -> CResult<Tensor> {
    match adapter {
        Some(a) => y + a.delta(x)?,
        None => Ok(y),
    }
}

pub struct CausalSelfAttention {
    c_attn: Linear,
    /// Optional LoRA adapters around `c_attn` and `c_proj`. When `Some`,
    /// the corresponding output is augmented with `delta(x)`. `lora_b`
    /// init=0 ensures the adapter's initial effect is exactly zero.
    c_attn_lora: Option<LoraAdapter>,
    c_proj: Linear,
    c_proj_lora: Option<LoraAdapter>,
    n_q_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    mask: Tensor,
    /// (block_size, head_dim) — duplicated half so HF-style rotate_half
    /// applies in one broadcast_mul.
    rope_cos: Option<Tensor>,
    rope_sin: Option<Tensor>,
}

impl CausalSelfAttention {
    pub fn new(cfg: &GPTConfig, vb: VarBuilder) -> CResult<Self> {
        let n_q_head = cfg.n_head;
        let n_kv_head = if cfg.n_kv_head == 0 {
            n_q_head
        } else {
            cfg.n_kv_head
        };
        if !n_q_head.is_multiple_of(n_kv_head) {
            candle_core::bail!(
                "n_head {} not divisible by n_kv_head {}",
                n_q_head,
                n_kv_head
            );
        }
        let head_dim = cfg.n_embd / n_q_head;
        let qkv_out = (n_q_head + 2 * n_kv_head) * head_dim;
        let c_attn = lin(cfg.n_embd, qkv_out, cfg.bias, vb.pp("c_attn"))?;
        let c_attn_lora = maybe_lora(cfg.n_embd, qkv_out, cfg, vb.pp("c_attn"))?;
        let c_proj = lin_resid(
            cfg.n_embd,
            cfg.n_embd,
            cfg.bias,
            cfg.n_layer,
            vb.pp("c_proj"),
        )?;
        let c_proj_lora = maybe_lora(cfg.n_embd, cfg.n_embd, cfg, vb.pp("c_proj"))?;
        let mask = build_causal_mask(cfg.block_size, vb.device())?;
        let (rope_cos, rope_sin) = if cfg.use_rope {
            if !head_dim.is_multiple_of(2) {
                candle_core::bail!("RoPE requires head_dim even, got {head_dim}");
            }
            let (c, s) =
                build_rope_tables(cfg.block_size, head_dim, cfg.rope_base as f64, vb.device())?;
            (Some(c), Some(s))
        } else {
            (None, None)
        };
        Ok(Self {
            c_attn,
            c_attn_lora,
            c_proj,
            c_proj_lora,
            n_q_head,
            n_kv_head,
            head_dim,
            mask,
            rope_cos,
            rope_sin,
        })
    }

    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let (b, t, c) = x.dims3()?;
        let qkv = self.c_attn.forward(x)?;
        let qkv = lora_add(qkv, &self.c_attn_lora, x)?;
        let q_dim = self.n_q_head * self.head_dim;
        let kv_dim = self.n_kv_head * self.head_dim;

        // Split into q / k / v then move to (B, n_head, T, head_dim).
        let q = qkv
            .narrow(2, 0, q_dim)?
            .reshape((b, t, self.n_q_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = qkv
            .narrow(2, q_dim, kv_dim)?
            .reshape((b, t, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = qkv
            .narrow(2, q_dim + kv_dim, kv_dim)?
            .reshape((b, t, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let (q, k) = if self.rope_cos.is_some() {
            let cos = self.rope_cos.as_ref().unwrap().i(..t)?;
            let sin = self.rope_sin.as_ref().unwrap().i(..t)?;
            (apply_rope(&q, &cos, &sin)?, apply_rope(&k, &cos, &sin)?)
        } else {
            (q, k)
        };

        // GQA: repeat K/V heads so each Q head has a partner.
        let group = self.n_q_head / self.n_kv_head;
        let k = repeat_kv(&k, group)?;
        let v = repeat_kv(&v, group)?;

        let scale = (self.head_dim as f64).sqrt();
        let att = (q.matmul(&k.transpose(2, 3)?)? / scale)?;
        let mask = self.mask.i((..t, ..t))?.broadcast_as(att.shape())?;
        let att = masked_fill(&att, &mask, f32::NEG_INFINITY)?;
        let att = ops::softmax_last_dim(&att)?;
        let y = att.matmul(&v)?;
        let y = y.transpose(1, 2)?.contiguous()?.reshape((b, t, c))?;
        let proj = self.c_proj.forward(&y)?;
        lora_add(proj, &self.c_proj_lora, &y)
    }
}

/// Repeat each of the K/V heads `n_rep` times along the head dimension.
/// `(B, n_kv, T, hd)` → `(B, n_kv * n_rep, T, hd)`.
fn repeat_kv(x: &Tensor, n_rep: usize) -> CResult<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, n_kv, t, hd) = x.dims4()?;
    x.unsqueeze(2)?
        .broadcast_as((b, n_kv, n_rep, t, hd))?
        .reshape((b, n_kv * n_rep, t, hd))
}

/// Build rotary cos/sin tables of shape `(max_seq, head_dim)`. The last dim
/// is `[c0, c1, ..., c_{hd/2-1}, c0, c1, ...]` (duplicated halves) so
/// `apply_rope` can use one broadcast_mul + rotate_half.
fn build_rope_tables(
    max_seq: usize,
    head_dim: usize,
    base: f64,
    device: &Device,
) -> CResult<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| (1.0 / base.powf(2.0 * i as f64 / head_dim as f64)) as f32)
        .collect();
    let mut cos = Vec::with_capacity(max_seq * head_dim);
    let mut sin = Vec::with_capacity(max_seq * head_dim);
    for t in 0..max_seq {
        for h in 0..2 {
            let _ = h; // duplicated halves
            for &f in &inv_freq {
                let theta = t as f32 * f;
                cos.push(theta.cos());
                sin.push(theta.sin());
            }
        }
    }
    let cos = Tensor::from_vec(cos, (max_seq, head_dim), device)?;
    let sin = Tensor::from_vec(sin, (max_seq, head_dim), device)?;
    Ok((cos, sin))
}

/// HF-style rotate_half: `[a; b]` -> `[-b; a]` along the last dim.
fn rotate_half(x: &Tensor) -> CResult<Tensor> {
    let last = x.dims().last().copied().unwrap_or(0);
    let half = last / 2;
    let a = x.narrow(candle_core::D::Minus1, 0, half)?;
    let b = x.narrow(candle_core::D::Minus1, half, half)?;
    let neg_b = b.neg()?;
    Tensor::cat(&[&neg_b, &a], candle_core::D::Minus1)
}

/// Apply RoPE: `q_rot = q * cos + rotate_half(q) * sin`.
/// `x: (B, n_head, T, hd)`, `cos/sin: (T, hd)`.
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> CResult<Tensor> {
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?; // (1, 1, T, hd)
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    let x_half = rotate_half(x)?;
    let a = x.broadcast_mul(&cos)?;
    let b = x_half.broadcast_mul(&sin)?;
    a + b
}

fn build_causal_mask(block_size: usize, device: &Device) -> CResult<Tensor> {
    let rows: Vec<u8> = (0..block_size)
        .flat_map(|i| (0..block_size).map(move |j| if j <= i { 0u8 } else { 1u8 }))
        .collect();
    Tensor::from_vec(rows, (block_size, block_size), device)
}

fn masked_fill(x: &Tensor, mask: &Tensor, fill: f32) -> CResult<Tensor> {
    // mask: 1 -> fill, 0 -> keep x
    let on_true = Tensor::new(fill, x.device())?
        .to_dtype(x.dtype())?
        .broadcast_as(x.shape())?;
    mask.where_cond(&on_true, x)
}

/// MLP block. Three forms supported:
///   - **Dense GELU** (`ActivationKind::Gelu`): `x → c_fc → gelu → c_proj`
///   - **GeGLU**: `x → (gelu(c_fc) * c_gate) → c_proj`
///   - **SwiGLU**: `x → (silu(c_fc) * c_gate) → c_proj`
///
/// Each Linear may carry an optional LoRA adapter (when `cfg.lora_rank > 0`).
pub struct MLP {
    c_fc: Linear,
    c_fc_lora: Option<LoraAdapter>,
    c_gate: Option<Linear>,
    c_proj: Linear,
    c_proj_lora: Option<LoraAdapter>,
    activation: ActivationKind,
}

impl MLP {
    pub fn new(cfg: &GPTConfig, vb: VarBuilder) -> CResult<Self> {
        let inner = cfg.ffn_mult * cfg.n_embd;
        let c_fc = lin(cfg.n_embd, inner, cfg.bias, vb.pp("c_fc"))?;
        let c_fc_lora = maybe_lora(cfg.n_embd, inner, cfg, vb.pp("c_fc"))?;
        let c_gate = if cfg.activation.is_gated() {
            Some(lin(cfg.n_embd, inner, cfg.bias, vb.pp("c_gate"))?)
        } else {
            None
        };
        let c_proj = lin_resid(inner, cfg.n_embd, cfg.bias, cfg.n_layer, vb.pp("c_proj"))?;
        let c_proj_lora = maybe_lora(inner, cfg.n_embd, cfg, vb.pp("c_proj"))?;
        Ok(Self {
            c_fc,
            c_fc_lora,
            c_gate,
            c_proj,
            c_proj_lora,
            activation: cfg.activation,
        })
    }

    fn fc(&self, x: &Tensor) -> CResult<Tensor> {
        let y = self.c_fc.forward(x)?;
        lora_add(y, &self.c_fc_lora, x)
    }

    fn proj(&self, h: &Tensor) -> CResult<Tensor> {
        let y = self.c_proj.forward(h)?;
        lora_add(y, &self.c_proj_lora, h)
    }
}

impl Module for MLP {
    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        match (&self.activation, &self.c_gate) {
            (ActivationKind::Gelu, _) => {
                let h = self.fc(x)?.gelu()?;
                self.proj(&h)
            }
            (ActivationKind::GeGlu, Some(gate)) => {
                let main = self.fc(x)?.gelu()?;
                let g = gate.forward(x)?;
                self.proj(&(main * g)?)
            }
            (ActivationKind::SwiGlu, Some(gate)) => {
                let main = self.fc(x)?.silu()?;
                let g = gate.forward(x)?;
                self.proj(&(main * g)?)
            }
            _ => unreachable!("gated activation requires c_gate"),
        }
    }
}

/// FFN block. `Dense` is the standard single-MLP path (back-compat). `MoE`
/// is Mixture-of-Experts with optional top-k masking (compute cost is the
/// same — every expert still runs; only routing weights are sparsified) and
/// a Switch-Transformer-style load-balance auxiliary loss returned alongside
/// the output.
pub enum FeedForward {
    Dense(MLP),
    MoE {
        router: Linear,
        experts: Vec<MLP>,
        top_k: usize,
    },
}

impl FeedForward {
    pub fn new(cfg: &GPTConfig, vb: VarBuilder) -> CResult<Self> {
        if cfg.n_experts <= 1 {
            Ok(FeedForward::Dense(MLP::new(cfg, vb)?))
        } else {
            let router = lin(cfg.n_embd, cfg.n_experts, false, vb.pp("router"))?;
            let experts: Vec<MLP> = (0..cfg.n_experts)
                .map(|i| MLP::new(cfg, vb.pp(format!("expert.{i}"))))
                .collect::<CResult<_>>()?;
            let top_k = if cfg.moe_top_k == 0 || cfg.moe_top_k >= cfg.n_experts {
                0
            } else {
                cfg.moe_top_k
            };
            Ok(FeedForward::MoE {
                router,
                experts,
                top_k,
            })
        }
    }

    /// Forward returning `(output, optional aux loss)`. Dense returns `None`;
    /// MoE returns a scalar aux loss tensor that callers should weight + add
    /// to the main objective.
    pub fn forward_with_aux(&self, x: &Tensor) -> CResult<(Tensor, Option<Tensor>)> {
        match self {
            FeedForward::Dense(mlp) => Ok((mlp.forward(x)?, None)),
            FeedForward::MoE {
                router,
                experts,
                top_k,
            } => {
                let n_experts = experts.len();
                let logits = router.forward(x)?; // (B, T, E)
                let full_weights = ops::softmax_last_dim(&logits)?;

                // Top-k mask: keep entries >= k-th largest, zero else, renorm.
                let weights = if *top_k > 0 && *top_k < n_experts {
                    topk_mask_renorm(&full_weights, *top_k)?
                } else {
                    full_weights.clone()
                };

                // Combine experts (full weights ARE soft if top_k=0).
                let mut out: Option<Tensor> = None;
                for (i, expert) in experts.iter().enumerate() {
                    let y = expert.forward(x)?; // (B, T, C)
                    let w = weights.narrow(candle_core::D::Minus1, i, 1)?; // (B, T, 1)
                    let weighted = y.broadcast_mul(&w)?;
                    out = Some(match out {
                        None => weighted,
                        Some(prev) => (prev + weighted)?,
                    });
                }
                let out = out.expect("MoE has at least one expert");

                // Load-balance loss on the **full** softmax (not top-k masked):
                //   P_i = mean_(B,T) full_weights[..., i]
                //   L_aux = E · sum_i P_i²
                // Minimum at uniform (= 1) ; scales > 1 as routing concentrates.
                let p_mean = full_weights
                    .mean_keepdim(0)? // → (1, T, E)
                    .mean_keepdim(1)? // → (1, 1, E)
                    .flatten_all()?; // → (E,)
                let aux = (p_mean.sqr()?.sum_all()? * (n_experts as f64))?;
                Ok((out, Some(aux)))
            }
        }
    }
}

/// Top-k mask + renormalize along the last dim. `weights: (..., E)`.
fn topk_mask_renorm(weights: &Tensor, k: usize) -> CResult<Tensor> {
    // Sort descending and read the k-th value as a per-row threshold.
    let (sorted, _idx) = weights.sort_last_dim(false)?;
    let kth = sorted.narrow(candle_core::D::Minus1, k - 1, 1)?; // (..., 1)
    let mask = weights.broadcast_ge(&kth)?; // u8 (..., E)
    let mask = mask.to_dtype(weights.dtype())?;
    let masked = weights.broadcast_mul(&mask)?;
    let sum = masked.sum_keepdim(candle_core::D::Minus1)?; // (..., 1)
    masked.broadcast_div(&sum)
}

impl Module for FeedForward {
    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        Ok(self.forward_with_aux(x)?.0)
    }
}

/// Either a LayerNorm or an RmsNorm. Forward delegates to the wrapped impl.
pub enum Norm {
    Ln(LayerNorm),
    Rms(RmsNorm),
}

impl Norm {
    pub fn new(kind: NormKind, dim: usize, vb: VarBuilder) -> CResult<Self> {
        match kind {
            NormKind::LayerNorm => Ok(Norm::Ln(layer_norm(dim, 1e-5, vb)?)),
            NormKind::RmsNorm => Ok(Norm::Rms(rms_norm(dim, 1e-5, vb)?)),
        }
    }
}

impl Module for Norm {
    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        match self {
            Norm::Ln(n) => n.forward(x),
            Norm::Rms(n) => n.forward(x),
        }
    }
}

pub struct Block {
    n1: Norm,
    attn: CausalSelfAttention,
    n2: Norm,
    mlp: FeedForward,
    norm_position: NormPosition,
}

impl Block {
    pub fn new(cfg: &GPTConfig, vb: VarBuilder) -> CResult<Self> {
        // Param paths kept as `ln_1` / `ln_2` for back-compat with old tied
        // GPT-2 checkpoints — the kind is implicit in whether bias is loaded.
        let n1 = Norm::new(cfg.norm_kind, cfg.n_embd, vb.pp("ln_1"))?;
        let attn = CausalSelfAttention::new(cfg, vb.pp("attn"))?;
        let n2 = Norm::new(cfg.norm_kind, cfg.n_embd, vb.pp("ln_2"))?;
        let mlp = FeedForward::new(cfg, vb.pp("mlp"))?;
        Ok(Self {
            n1,
            attn,
            n2,
            mlp,
            norm_position: cfg.norm_position,
        })
    }

    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        Ok(self.forward_with_aux(x)?.0)
    }

    fn forward_with_aux(&self, x: &Tensor) -> CResult<(Tensor, Option<Tensor>)> {
        match self.norm_position {
            NormPosition::Pre => {
                // pre-norm: x + f(LN(x))
                let h = (x + self.attn.forward(&self.n1.forward(x)?)?)?;
                let (mlp_out, aux) = self.mlp.forward_with_aux(&self.n2.forward(&h)?)?;
                let out = (&h + mlp_out)?;
                Ok((out, aux))
            }
            NormPosition::Post => {
                // post-norm: LN(x + f(x))
                let h = self.n1.forward(&(x + self.attn.forward(x)?)?)?;
                let (mlp_out, aux) = self.mlp.forward_with_aux(&h)?;
                let out = self.n2.forward(&(&h + mlp_out)?)?;
                Ok((out, aux))
            }
        }
    }
}

pub struct GPT {
    pub cfg: GPTConfig,
    wte: Embedding,
    /// `None` when `cfg.use_rope` is true (positional info enters via RoPE
    /// inside attention).
    wpe: Option<Embedding>,
    blocks: Vec<Block>,
    ln_f: Norm,
    /// `None` when `cfg.weight_tying` is true — head reuses `wte`. When
    /// `Some`, the Linear maps `n_embd → vocab_size` directly.
    lm_head: Option<Linear>,
    /// LoRA adapter for `lm_head` when both `weight_tying=false` and
    /// `lora_rank > 0`.
    lm_head_lora: Option<LoraAdapter>,
    pub device: Device,
}

impl GPT {
    pub fn new(cfg: GPTConfig, vb: VarBuilder) -> CResult<Self> {
        let cfg = cfg.normalized();
        let device = vb.device().clone();
        let wte = embed(cfg.vocab_size, cfg.n_embd, vb.pp("wte"))?;
        let wpe = if cfg.use_rope {
            None
        } else {
            Some(embed(cfg.block_size, cfg.n_embd, vb.pp("wpe"))?)
        };
        let blocks: Vec<Block> = (0..cfg.n_layer)
            .map(|i| Block::new(&cfg, vb.pp(format!("h.{i}"))))
            .collect::<CResult<_>>()?;
        let ln_f = Norm::new(cfg.norm_kind, cfg.n_embd, vb.pp("ln_f"))?;
        let lm_head = if cfg.weight_tying {
            None
        } else {
            Some(lin(cfg.n_embd, cfg.vocab_size, false, vb.pp("lm_head"))?)
        };
        let lm_head_lora = if !cfg.weight_tying {
            maybe_lora(cfg.n_embd, cfg.vocab_size, &cfg, vb.pp("lm_head"))?
        } else {
            None
        };
        Ok(Self {
            cfg,
            wte,
            wpe,
            blocks,
            ln_f,
            lm_head,
            lm_head_lora,
            device,
        })
    }

    /// Forward: (B, T) i64 token ids → (B, T, vocab) logits.
    pub fn forward(&self, idx: &Tensor) -> CResult<Tensor> {
        let (_b, t) = idx.dims2()?;
        if t > self.cfg.block_size {
            candle_core::bail!(
                "sequence length {t} exceeds block_size {}",
                self.cfg.block_size
            );
        }
        let tok_emb = self.wte.forward(idx)?; // (B, T, C)
        let mut x = if let Some(wpe) = &self.wpe {
            let pos = Tensor::arange(0u32, t as u32, &self.device)?;
            let pos_emb = wpe.forward(&pos)?.unsqueeze(0)?;
            tok_emb.broadcast_add(&pos_emb)?
        } else {
            tok_emb
        };
        for blk in &self.blocks {
            x = blk.forward(&x)?;
        }
        let x = self.ln_f.forward(&x)?;
        let logits = self.head_logits(&x)?;
        Ok(logits)
    }

    fn head_logits(&self, x: &Tensor) -> CResult<Tensor> {
        match &self.lm_head {
            Some(head) => {
                let logits = head.forward(x)?;
                lora_add(logits, &self.lm_head_lora, x)
            }
            None => {
                // tied head: logits = x @ wte.weight^T
                let w = self.wte.embeddings();
                x.broadcast_matmul(&w.t()?)
            }
        }
    }

    /// Like `forward`, but also collects per-MoE-layer auxiliary losses.
    pub fn forward_with_aux(&self, idx: &Tensor) -> CResult<(Tensor, Vec<Tensor>)> {
        let (_b, t) = idx.dims2()?;
        if t > self.cfg.block_size {
            candle_core::bail!(
                "sequence length {t} exceeds block_size {}",
                self.cfg.block_size
            );
        }
        let tok_emb = self.wte.forward(idx)?;
        let mut x = if let Some(wpe) = &self.wpe {
            let pos = Tensor::arange(0u32, t as u32, &self.device)?;
            let pos_emb = wpe.forward(&pos)?.unsqueeze(0)?;
            tok_emb.broadcast_add(&pos_emb)?
        } else {
            tok_emb
        };
        let mut aux: Vec<Tensor> = Vec::new();
        for blk in &self.blocks {
            let (next, blk_aux) = blk.forward_with_aux(&x)?;
            x = next;
            if let Some(a) = blk_aux {
                aux.push(a);
            }
        }
        let x = self.ln_f.forward(&x)?;
        let logits = self.head_logits(&x)?;
        Ok((logits, aux))
    }

    /// Cross-entropy loss over (B, T) targets, including any MoE aux losses.
    pub fn loss(&self, idx: &Tensor, targets: &Tensor) -> CResult<Tensor> {
        let (logits, aux) = self.forward_with_aux(idx)?;
        let (b, t, v) = logits.dims3()?;
        let logits = logits.reshape((b * t, v))?;
        let targets = targets.reshape(b * t)?.to_dtype(DType::U32)?;
        let mut total = candle_nn::loss::cross_entropy(&logits, &targets)?;
        if !aux.is_empty() && self.cfg.moe_aux_weight > 0.0 {
            let alpha = self.cfg.moe_aux_weight as f64;
            let mut sum_aux: Option<Tensor> = None;
            for a in aux {
                sum_aux = Some(match sum_aux {
                    None => a,
                    Some(prev) => (prev + a)?,
                });
            }
            if let Some(s) = sum_aux {
                total = (total + (s * alpha)?)?;
            }
        }
        Ok(total)
    }

    /// Compute the model's per-token log-probability of `completion`
    /// conditioned on `prompt`. Returns the **sum** of
    /// `log P(completion[t] | prompt + completion[:t])` over all
    /// completion positions. Higher (less-negative) = model is more
    /// confident in this completion.
    ///
    /// Used by Phase 6 Shape C's `LogitCritic` as a free pre-filter:
    /// if a candidate's per-token log-prob correlates with cargo's
    /// verdict, we can rank candidates without running cargo.
    ///
    /// Caller must ensure `prompt_ids.len() + completion_ids.len() <=
    /// block_size`. `prompt_ids` must be non-empty (we need at least
    /// one prompt token to predict the first completion token).
    pub fn sequence_log_prob(
        &self,
        prompt_ids: &[u32],
        completion_ids: &[u32],
        device: &Device,
    ) -> CResult<f32> {
        if prompt_ids.is_empty() {
            candle_core::bail!("sequence_log_prob: prompt_ids is empty");
        }
        if completion_ids.is_empty() {
            return Ok(0.0);
        }
        let n = prompt_ids.len() + completion_ids.len();
        if n > self.cfg.block_size {
            candle_core::bail!(
                "sequence_log_prob: prompt+completion {} > block_size {}",
                n,
                self.cfg.block_size
            );
        }
        // Input is full[0..n-1]; we ask the model to predict positions 1..n-1.
        // Predictions at indices prompt_len-1 .. n-2 correspond to positions
        // prompt_len .. n-1 in the full sequence — i.e. the completion span.
        let full: Vec<u32> = prompt_ids
            .iter()
            .chain(completion_ids.iter())
            .copied()
            .collect();
        let x = Tensor::from_slice(&full[..n - 1], (1, n - 1), device)?;
        let logits = self.forward(&x)?; // (1, n-1, vocab)
        let (_, t, v) = logits.dims3()?;
        let log_probs = candle_nn::ops::log_softmax(&logits.reshape((t, v))?, 1)?;
        let start = prompt_ids.len() - 1;
        let lps_comp = log_probs.narrow(0, start, completion_ids.len())?;
        let targets = Tensor::from_slice(completion_ids, completion_ids.len(), device)?
            .to_dtype(DType::U32)?
            .unsqueeze(1)?;
        let gathered = lps_comp.gather(&targets, 1)?;
        gathered.sum_all()?.to_scalar::<f32>()
    }

    pub fn block_size(&self) -> usize {
        self.cfg.block_size
    }

    pub fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }

    /// Forward returning logits at the last position only (for generation).
    pub fn forward_last(&self, idx: &Tensor) -> CResult<Tensor> {
        let logits = self.forward(idx)?;
        let (_b, t, _v) = logits.dims3()?;
        logits.i((.., t - 1, ..))
    }
}

#[allow(dead_code)]
fn _check_module_bound() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<GPT>();
}

#[cfg(test)]
mod moe_tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::{VarBuilder, VarMap};

    fn cpu_cfg(n_experts: usize, top_k: usize) -> GPTConfig {
        GPTConfig {
            vocab_size: 8,
            block_size: 4,
            n_layer: 2,
            n_head: 2,
            n_embd: 16,
            dropout: 0.0,
            bias: false,
            ffn_mult: 2,
            use_rope: false,
            rope_base: 10_000.0,
            n_kv_head: 2,
            n_experts,
            moe_top_k: top_k,
            moe_aux_weight: 0.01,
            activation: ActivationKind::Gelu,
            weight_tying: true,
            norm_kind: NormKind::LayerNorm,
            norm_position: NormPosition::Pre,
            lora_rank: 0,
            lora_alpha: 16.0,
        }
    }

    #[test]
    fn dense_returns_no_aux() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let cfg = cpu_cfg(1, 0);
        let m = GPT::new(cfg, vb).unwrap();
        let idx = Tensor::from_vec(vec![0u32, 1, 2, 3], (1, 4), &device).unwrap();
        let (_logits, aux) = m.forward_with_aux(&idx).unwrap();
        assert!(aux.is_empty(), "dense FFN should not produce aux losses");
    }

    #[test]
    fn moe_returns_aux_per_layer() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let cfg = cpu_cfg(4, 2);
        let m = GPT::new(cfg, vb).unwrap();
        let idx = Tensor::from_vec(vec![0u32, 1, 2, 3], (1, 4), &device).unwrap();
        let (_logits, aux) = m.forward_with_aux(&idx).unwrap();
        assert_eq!(aux.len(), 2, "expected one aux loss per MoE block");
        for a in aux {
            let v = a.to_scalar::<f32>().unwrap();
            assert!(
                v.is_finite() && v > 0.0,
                "aux loss must be a positive finite scalar, got {v}"
            );
        }
    }

    #[test]
    fn topk_mask_keeps_top_k_no_ties() {
        let device = Device::Cpu;
        // Distinct values so top-2 has no ties: [0.4, 0.3, 0.2, 0.1].
        let v: Vec<f32> = vec![0.4, 0.3, 0.2, 0.1];
        let weights = Tensor::from_vec(v, (1, 1, 4), &device).unwrap();
        let masked = topk_mask_renorm(&weights, 2).unwrap();
        let out: Vec<f32> = masked.flatten_all().unwrap().to_vec1().unwrap();
        let nz: Vec<f32> = out.iter().copied().filter(|x| *x > 0.0).collect();
        assert_eq!(nz.len(), 2, "exactly top-2 entries survive");
        let sum: f32 = nz.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "renormalized weights must sum to 1, got {sum}"
        );
        // Specifically, the two largest (0.4, 0.3) survive and renormalize to 4/7, 3/7.
        assert!((out[0] - 4.0 / 7.0).abs() < 1e-5);
        assert!((out[1] - 3.0 / 7.0).abs() < 1e-5);
        assert!(out[2] == 0.0 && out[3] == 0.0);
    }

    fn cfg_with_activation(act: ActivationKind) -> GPTConfig {
        GPTConfig {
            activation: act,
            ..cpu_cfg(1, 0)
        }
    }

    #[test]
    fn gated_mlp_forward_shapes_match_dense() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let cfg = cfg_with_activation(ActivationKind::SwiGlu);
        let m = GPT::new(cfg.clone(), vb).unwrap();
        let idx = Tensor::from_vec(vec![0u32, 1, 2, 3], (1, 4), &device).unwrap();
        let logits = m.forward(&idx).unwrap();
        assert_eq!(logits.dims3().unwrap(), (1, 4, cfg.vocab_size));
    }

    #[test]
    fn geglu_mlp_runs() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let cfg = cfg_with_activation(ActivationKind::GeGlu);
        let m = GPT::new(cfg, vb).unwrap();
        let idx = Tensor::from_vec(vec![0u32, 1, 2, 3], (1, 4), &device).unwrap();
        let _ = m.forward(&idx).unwrap();
    }

    #[test]
    fn rms_norm_block_runs() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let cfg = GPTConfig {
            norm_kind: NormKind::RmsNorm,
            ..cpu_cfg(1, 0)
        };
        let m = GPT::new(cfg, vb).unwrap();
        let idx = Tensor::from_vec(vec![0u32, 1, 2, 3], (1, 4), &device).unwrap();
        let _ = m.forward(&idx).unwrap();
    }

    #[test]
    fn post_norm_block_runs() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let cfg = GPTConfig {
            norm_position: NormPosition::Post,
            ..cpu_cfg(1, 0)
        };
        let m = GPT::new(cfg, vb).unwrap();
        let idx = Tensor::from_vec(vec![0u32, 1, 2, 3], (1, 4), &device).unwrap();
        let _ = m.forward(&idx).unwrap();
    }

    #[test]
    fn rms_norm_has_fewer_params_than_layer_norm() {
        let ln = GPTConfig {
            norm_kind: NormKind::LayerNorm,
            ..cpu_cfg(1, 0)
        };
        let rms = GPTConfig {
            norm_kind: NormKind::RmsNorm,
            ..cpu_cfg(1, 0)
        };
        assert!(
            rms.num_params_estimate() < ln.num_params_estimate(),
            "RmsNorm has no bias so per-block param count must be smaller"
        );
    }

    #[test]
    fn untied_head_runs_and_has_more_params() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let cfg = GPTConfig {
            weight_tying: false,
            ..cpu_cfg(1, 0)
        };
        let m = GPT::new(cfg.clone(), vb).unwrap();
        let idx = Tensor::from_vec(vec![0u32, 1, 2, 3], (1, 4), &device).unwrap();
        let logits = m.forward(&idx).unwrap();
        assert_eq!(logits.dims3().unwrap(), (1, 4, cfg.vocab_size));

        let tied = GPTConfig {
            weight_tying: true,
            ..cpu_cfg(1, 0)
        };
        let untied = GPTConfig {
            weight_tying: false,
            ..cpu_cfg(1, 0)
        };
        assert!(
            untied.num_params_estimate() > tied.num_params_estimate(),
            "untied head must report more params"
        );
    }

    #[test]
    fn gated_mlp_has_more_params_than_dense() {
        let dense = cfg_with_activation(ActivationKind::Gelu);
        let gated = cfg_with_activation(ActivationKind::SwiGlu);
        assert!(
            gated.num_params_estimate() > dense.num_params_estimate(),
            "gated MLP must have ≥1× the params of dense (estimate)"
        );
    }

    #[test]
    fn topk_mask_handles_ties_inclusively() {
        let device = Device::Cpu;
        // Two-way tie at the threshold: top-2 of [0.1, 0.5, 0.2, 0.2] keeps
        // {0.5} plus both 0.2s — `>= kth` is inclusive on ties. This is the
        // intended behavior (deterministic, no random tie-break).
        let v: Vec<f32> = vec![0.1, 0.5, 0.2, 0.2];
        let weights = Tensor::from_vec(v, (1, 1, 4), &device).unwrap();
        let masked = topk_mask_renorm(&weights, 2).unwrap();
        let out: Vec<f32> = masked.flatten_all().unwrap().to_vec1().unwrap();
        let nz: Vec<f32> = out.iter().copied().filter(|x| *x > 0.0).collect();
        assert_eq!(nz.len(), 3, "ties at the k-th value are kept");
        let sum: f32 = nz.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
