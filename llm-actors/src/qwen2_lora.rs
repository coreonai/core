//! Phase 21 Stage F — Qwen2 with LoRA adapters on `q_proj` + `v_proj`,
//! forked from `candle_transformers::models::qwen2` (v0.10.2).
//!
//! Two changes vs the upstream module:
//!
//! 1. **LoRA injection** on `q_proj` and `v_proj` — matches the
//!    `target_modules = ["q_proj", "v_proj"]` recipe Phase 14-20 used
//!    via HuggingFace PEFT. The frozen base linear weights are loaded
//!    from the upstream safetensors; the LoRA `(a, b)` adapter Vars
//!    live in a separate `VarMap` so the optimizer only touches them.
//!
//! 2. **`forward_train`** on `ModelForCausalLM` returns all-position
//!    logits `(B, T, V)` instead of just the last position. Needed for
//!    next-token-prediction cross-entropy loss over the full sequence.
//!    The inference-time `forward` (last position, with KV cache) is
//!    preserved for the existing `QwenModelActor` integration.
//!
//! NOT in this commit:
//! - Adapter save/load to a separate `.safetensors` file (in-memory
//!   only). Caller can serialize the VarMap directly when ready.
//! - LoRA on `k_proj`, `o_proj`, or the MLP gate/up/down. Phase 14-20
//!   used q+v only — that's what this mirrors.
//! - Quantization. Upstream qwen2 supports it; we don't here because
//!   gradients through quantized weights are out of scope.

use std::sync::Arc;

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{linear, linear_no_bias, Activation, Embedding, Linear, RmsNorm, VarBuilder};
use nanogpt_rs::model::LoraAdapter;

pub use candle_transformers::models::qwen2::Config;

/// Apply an RmsNorm on either the training path (slow but bwd-supporting)
/// or inference path (fast no_bwd op via the Module trait).
fn rms_norm_apply(rms: &RmsNorm, xs: &Tensor, train: bool) -> Result<Tensor> {
    if train {
        let eps = rms.eps() as f32;
        candle_nn::ops::rms_norm_slow(&xs.contiguous()?, rms.weight(), eps)
    } else {
        rms.forward(xs)
    }
}

/// LoRA hyperparameters for the fork.
#[derive(Debug, Clone, Copy)]
pub struct LoraConfig {
    pub rank: usize,
    pub alpha: f32,
}

impl Default for LoraConfig {
    fn default() -> Self {
        // Matches Phase 14-20 Python recipe: r=16, α=32.
        Self {
            rank: 16,
            alpha: 32.0,
        }
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.hidden_size / cfg.num_attention_heads;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        // candle_nn::rotary_emb::rope uses apply_op3_no_bwd (forward only)
        // — fine for inference, but BREAKS the gradient chain during
        // training. Use rope_slow (pure tensor ops) when training so the
        // LoRA Vars upstream actually receive gradients.
        let (q_c, k_c) = (q.contiguous()?, k.contiguous()?);
        let (q_embed, k_embed) = if train {
            (
                candle_nn::rotary_emb::rope_slow(&q_c, &cos, &sin)?,
                candle_nn::rotary_emb::rope_slow(&k_c, &cos, &sin)?,
            )
        } else {
            (
                candle_nn::rotary_emb::rope(&q_c, &cos, &sin)?,
                candle_nn::rotary_emb::rope(&k_c, &cos, &sin)?,
            )
        };
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        Ok(Self {
            gate_proj: linear_no_bias(hidden_sz, intermediate_sz, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(hidden_sz, intermediate_sz, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(intermediate_sz, hidden_sz, vb.pp("down_proj"))?,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

/// Attention layer with LoRA hooks on `q_proj` and `v_proj`. `k_proj`
/// and `o_proj` stay plain frozen Linears, matching Phase 14-20's PEFT
/// `target_modules = ["q_proj", "v_proj"]`.
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_lora: Option<LoraAdapter>,
    v_lora: Option<LoraAdapter>,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        base_vb: VarBuilder,
        lora_vb: Option<VarBuilder>,
        lora_cfg: LoraConfig,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = hidden_sz / num_heads;
        let q_proj = linear(hidden_sz, num_heads * head_dim, base_vb.pp("q_proj"))?;
        let k_proj = linear(hidden_sz, num_kv_heads * head_dim, base_vb.pp("k_proj"))?;
        let v_proj = linear(hidden_sz, num_kv_heads * head_dim, base_vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_sz, base_vb.pp("o_proj"))?;
        let (q_lora, v_lora) = if let Some(lvb) = lora_vb {
            let q_a = LoraAdapter::new(
                hidden_sz,
                num_heads * head_dim,
                lora_cfg.rank,
                lora_cfg.alpha,
                lvb.pp("q_proj"),
            )?;
            let v_a = LoraAdapter::new(
                hidden_sz,
                num_kv_heads * head_dim,
                lora_cfg.rank,
                lora_cfg.alpha,
                lvb.pp("v_proj"),
            )?;
            (Some(q_a), Some(v_a))
        } else {
            (None, None)
        };
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_lora,
            v_lora,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size: hidden_sz,
            rotary_emb,
            kv_cache: None,
        })
    }

    fn project_q(&self, xs: &Tensor) -> Result<Tensor> {
        let base = self.q_proj.forward(xs)?;
        if let Some(adapter) = &self.q_lora {
            base + adapter.delta(xs)?
        } else {
            Ok(base)
        }
    }

    fn project_v(&self, xs: &Tensor) -> Result<Tensor> {
        let base = self.v_proj.forward(xs)?;
        if let Some(adapter) = &self.v_lora {
            base + adapter.delta(xs)?
        } else {
            Ok(base)
        }
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        use_kv_cache: bool,
    ) -> Result<Tensor> {
        let train = !use_kv_cache;
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self.project_q(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.project_v(xs)?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) = self.rotary_emb.apply_rotary_emb_qkv(
            &query_states,
            &key_states,
            seqlen_offset,
            !use_kv_cache,
        )?;

        let (key_states, value_states) = if use_kv_cache {
            let (kk, vv) = match &self.kv_cache {
                None => (key_states.clone(), value_states.clone()),
                Some((prev_k, prev_v)) => {
                    let k = Tensor::cat(&[prev_k, &key_states], 2)?;
                    let v = Tensor::cat(&[prev_v, &value_states], 2)?;
                    (k, v)
                }
            };
            self.kv_cache = Some((kk.clone(), vv.clone()));
            (kk, vv)
        } else {
            // Training path: no cache, no carry-over.
            self.kv_cache = None;
            (key_states, value_states)
        };

        let key_states =
            candle_transformers::utils::repeat_kv(key_states, self.num_kv_groups)?.contiguous()?;
        let value_states = candle_transformers::utils::repeat_kv(value_states, self.num_kv_groups)?
            .contiguous()?;

        let attn_output = {
            let scale = 1f64 / f64::sqrt(self.head_dim as f64);
            let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;
            let attn_weights = match attention_mask {
                None => attn_weights,
                Some(mask) => attn_weights.broadcast_add(mask)?,
            };
            // softmax_last_dim is apply_op1_no_bwd — forward only.
            // Use the bwd-supporting ops::softmax(..., D::Minus1) when training.
            let attn_weights = if train {
                candle_nn::ops::softmax(&attn_weights, D::Minus1)?
            } else {
                candle_nn::ops::softmax_last_dim(&attn_weights)?
            };
            attn_weights.matmul(&value_states)?
        };
        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.hidden_size))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        base_vb: VarBuilder,
        lora_vb: Option<VarBuilder>,
        lora_cfg: LoraConfig,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            rotary_emb,
            cfg,
            base_vb.pp("self_attn"),
            lora_vb.as_ref().map(|v| v.pp("self_attn")),
            lora_cfg,
        )?;
        let mlp = Mlp::new(cfg, base_vb.pp("mlp"))?;
        let input_layernorm = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            base_vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            base_vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        use_kv_cache: bool,
    ) -> Result<Tensor> {
        let train = !use_kv_cache;
        let residual = xs;
        let xs = rms_norm_apply(&self.input_layernorm, xs, train)?;
        let xs = self
            .self_attn
            .forward(&xs, attention_mask, seqlen_offset, use_kv_cache)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = rms_norm_apply(&self.post_attention_layernorm, &xs, train)?.apply(&self.mlp)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

pub struct Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    sliding_window: usize,
    device: Device,
    dtype: DType,
}

impl Model {
    pub fn new(
        cfg: &Config,
        base_vb: VarBuilder,
        lora_vb: Option<VarBuilder>,
        lora_cfg: LoraConfig,
    ) -> Result<Self> {
        let vb_m = base_vb.pp("model");
        let lvb_m = lora_vb.as_ref().map(|v| v.pp("model"));
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb_m.dtype(), cfg, vb_m.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        let lvb_l = lvb_m.as_ref().map(|v| v.pp("layers"));
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(
                rotary_emb.clone(),
                cfg,
                vb_l.pp(layer_idx),
                lvb_l.as_ref().map(|v| v.pp(layer_idx)),
                lora_cfg,
            )?;
            layers.push(layer);
        }
        let norm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            sliding_window: cfg.sliding_window,
            device: base_vb.device().clone(),
            dtype: base_vb.dtype(),
        })
    }

    fn prepare_causal_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + self.sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), self.dtype, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(self.dtype)
    }

    fn forward_inner(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        use_kv_cache: bool,
    ) -> Result<Tensor> {
        let train = !use_kv_cache;
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask: Option<Tensor> = if seq_len <= 1 {
            None
        } else {
            Some(self.prepare_causal_attention_mask(b_size, seq_len, seqlen_offset)?)
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset, use_kv_cache)?;
        }
        rms_norm_apply(&self.norm, &xs, train)
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        self.forward_inner(input_ids, seqlen_offset, true)
    }

    pub fn forward_train(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        self.clear_kv_cache();
        self.forward_inner(input_ids, 0, false)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }

    pub fn embeddings(&self) -> Tensor {
        self.embed_tokens.embeddings().clone()
    }
}

pub struct ModelForCausalLM {
    base_model: Model,
    lm_head: Linear,
    tied: bool,
}

impl ModelForCausalLM {
    pub fn new(
        cfg: &Config,
        base_vb: VarBuilder,
        lora_vb: Option<VarBuilder>,
        lora_cfg: LoraConfig,
    ) -> Result<Self> {
        let base_model = Model::new(cfg, base_vb.clone(), lora_vb.clone(), lora_cfg)?;
        let (lm_head, tied) = if base_vb.contains_tensor("lm_head.weight") {
            (
                linear_no_bias(cfg.hidden_size, cfg.vocab_size, base_vb.pp("lm_head"))?,
                false,
            )
        } else {
            (Linear::new(base_model.embeddings(), None), true)
        };
        Ok(Self {
            base_model,
            lm_head,
            tied,
        })
    }

    /// Inference-time forward returning logits at the last position
    /// only `(B, 1, V)` and updating the internal KV cache. Mirrors
    /// upstream `qwen2::ModelForCausalLM::forward`.
    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;
        self.base_model
            .forward(input_ids, seqlen_offset)?
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.lm_head)
    }

    /// Training-time forward returning all-position logits `(B, T, V)`.
    /// Clears the KV cache and runs a fresh causal-masked attention
    /// over the full sequence. Used by the cross-entropy loss in
    /// `train_qwen_lora_step`.
    pub fn forward_train(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let hidden = self.base_model.forward_train(input_ids)?;
        hidden.apply(&self.lm_head)
    }

    pub fn clear_kv_cache(&mut self) {
        self.base_model.clear_kv_cache();
    }

    pub fn weight_tied(&self) -> bool {
        self.tied
    }
}

/// Phase 21 Stage G — one REINFORCE-style policy-gradient step.
///
/// Each entry of `samples` is `(prompt_ids, completion_ids, reward)`.
/// For each sample:
///   1. Concat `[prompt | completion]` into a 1-d tensor, unsqueezed to `(1, P+C)`.
///   2. Forward all positions through `forward_train` → logits `(1, P+C, V)`.
///   3. Slice the logits at positions `P-1 .. P-1+C` — those are the
///      logits the model used to predict each completion token.
///   4. `mean_ce = cross_entropy(pred_logits, completion_ids)` ≈ −mean log P.
///   5. Sample contributes `reward * mean_ce` to the loss.
///
/// The aggregate loss is the mean over samples of `reward_i * mean_ce_i`.
/// Minimizing this is equivalent to ascending `reward_i * mean_log_p_i`
/// — the REINFORCE objective. Use baseline-subtracted rewards (e.g.,
/// `verdict_i − mean(verdict_for_prompt)`) at the call site for variance
/// reduction; that's the standard RLOO trick.
///
/// Returns the pre-step loss value (the conventional REINFORCE logging
/// choice).
pub fn train_qwen_lora_pg_step(
    model: &mut ModelForCausalLM,
    optimizer: &mut candle_nn::AdamW,
    device: &Device,
    samples: &[(Vec<u32>, Vec<u32>, f32)],
) -> Result<f32> {
    use candle_nn::Optimizer;
    if samples.is_empty() {
        candle_core::bail!("train_qwen_lora_pg_step: samples is empty");
    }
    let mut loss: Option<Tensor> = None;
    for (prompt, comp, reward) in samples {
        if comp.is_empty() {
            continue;
        }
        let mut full = prompt.clone();
        full.extend_from_slice(comp);
        let full_len = full.len();
        let full_t = Tensor::from_slice(&full, (1, full_len), device)?;
        let logits = model.forward_train(&full_t)?; // (1, P+C, V)
        let p_len = prompt.len();
        let c_len = comp.len();
        // logits[0..P-1] predict prompt tokens; logits[P-1..P-1+C] predict completion tokens.
        let pred = logits.narrow(1, p_len.saturating_sub(1), c_len)?;
        let (_, c, v) = pred.dims3()?;
        let pred_flat = pred.reshape((c, v))?.to_dtype(DType::F32)?;
        let comp_t = Tensor::from_slice(comp, c_len, device)?;
        let mean_ce = candle_nn::loss::cross_entropy(&pred_flat, &comp_t)?;
        let contrib = (&mean_ce * (*reward as f64))?;
        loss = Some(match loss {
            Some(prev) => (prev + contrib)?,
            None => contrib,
        });
    }
    let loss = loss.ok_or_else(|| candle_core::Error::Msg("no usable samples".into()))?;
    let n = samples.len() as f64;
    let loss = (loss / n)?;
    let loss_value = loss.to_scalar::<f32>()?;
    optimizer.backward_step(&loss)?;
    Ok(loss_value)
}

/// Phase 21 Stage E.next.next — emit a "merged" safetensors file.
///
/// Reads the original frozen-base safetensors and, for every layer's
/// `q_proj` and `v_proj`, adds the LoRA delta `B @ A * (α / r)` into
/// the base `weight` tensor in place. All other tensors (k_proj /
/// o_proj / norms / embed / lm_head) pass through unchanged. The
/// resulting file is **drop-in compatible with the upstream
/// `candle_transformers::models::qwen2` loader** — `QwenModelActor`
/// can pick it up via the existing `ReloadCheckpoint` message,
/// reflecting the trained adapter at inference time without any
/// LoRA-awareness on the inference side.
///
/// Trade-off vs runtime LoRA loading: merging is one-time, but the
/// merged file is the size of the base (~1.5 GB for 0.5B) instead of
/// the ~4 MB LoRA-only adapter. For deployment / inference scaling
/// this is the right shape.
pub fn save_merged_lora(
    base_safetensors_path: &std::path::Path,
    lora_map: &candle_nn::VarMap,
    cfg: &Config,
    lora_cfg: LoraConfig,
    device: &Device,
    out_path: &std::path::Path,
) -> Result<()> {
    let mut tensors = candle_core::safetensors::load(base_safetensors_path, device)?;
    let scale = (lora_cfg.alpha / lora_cfg.rank as f32) as f64;

    let lora_vars_guard = lora_map.data().lock().unwrap();
    let lora_vars: std::collections::HashMap<String, candle_core::Var> = lora_vars_guard
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    drop(lora_vars_guard);

    for layer_idx in 0..cfg.num_hidden_layers {
        for proj in ["q_proj", "v_proj"] {
            let base_key = format!("model.layers.{layer_idx}.self_attn.{proj}.weight");
            let a_key = format!("model.layers.{layer_idx}.self_attn.{proj}.lora_a.weight");
            let b_key = format!("model.layers.{layer_idx}.self_attn.{proj}.lora_b.weight");
            let base = tensors
                .get(&base_key)
                .ok_or_else(|| candle_core::Error::Msg(format!("missing base key {base_key}")))?
                .clone();
            let a = lora_vars
                .get(&a_key)
                .ok_or_else(|| candle_core::Error::Msg(format!("missing LoRA Var {a_key}")))?;
            let b = lora_vars
                .get(&b_key)
                .ok_or_else(|| candle_core::Error::Msg(format!("missing LoRA Var {b_key}")))?;
            // a: (rank, in_dim), b: (out_dim, rank) → delta = b @ a (out, in)
            let a_t = a.as_tensor().to_dtype(base.dtype())?;
            let b_t = b.as_tensor().to_dtype(base.dtype())?;
            let delta = b_t.matmul(&a_t)?;
            let delta = (delta * scale)?;
            let merged = (&base + &delta)?;
            tensors.insert(base_key, merged);
        }
    }
    candle_core::safetensors::save(&tensors, out_path)?;
    Ok(())
}

/// Single training step over a fixed (input, target) batch.
///
/// `input_ids` and `target_ids` must both be `(B, T)` and represent the
/// shifted next-token-prediction pair — i.e., target_ids[i, t] should
/// be the token the model is asked to predict from input_ids[i, t].
/// Cross-entropy is computed at every position.
///
/// Returns the scalar loss BEFORE the optimizer step (the conventional
/// reporting choice — easier to log loss-vs-step curves).
///
/// The optimizer should already be bound to the LoRA `VarMap` only —
/// base model Vars are frozen mmap tensors and aren't touched.
pub fn train_qwen_lora_step(
    model: &mut ModelForCausalLM,
    optimizer: &mut candle_nn::AdamW,
    input_ids: &Tensor,
    target_ids: &Tensor,
) -> Result<f32> {
    let logits = model.forward_train(input_ids)?;
    let (b, t, v) = logits.dims3()?;
    // CE expects (N, V) logits + (N,) class targets — flatten over (B*T).
    let logits_flat = logits.reshape((b * t, v))?.to_dtype(DType::F32)?;
    let targets_flat = target_ids.reshape(b * t)?;
    let loss = candle_nn::loss::cross_entropy(&logits_flat, &targets_flat)?;
    let loss_value = loss.to_scalar::<f32>()?;
    use candle_nn::Optimizer;
    optimizer.backward_step(&loss)?;
    Ok(loss_value)
}

/// Phase 22 Stage D fix — **completion-only SFT**, the Phase 17 Python
/// recipe (`labels[:prompt_ids.shape[0]] = -100`).
///
/// `input_ids`, `target_ids`: `(1, P+C−1)` shifted next-token pair for a
///   single example with prompt length `P` and completion length `C`.
/// `prompt_len = P`: number of tokens in the prompt before the
///   completion starts. Loss is computed ONLY on the last `C` positions
///   of `(input, target)` (positions `P−1..P+C−1`), which correspond to
///   predicting completion tokens `c_0..c_{C-1}` from the prompt's last
///   token and the preceding completion tokens.
///
/// Without this masking, the `(P+C−1)`-position CE loss is dominated
/// by the `P−1` prompt-internal predictions (since P >> C typical for
/// HumanEval / MBPP prompts). The model collapses onto prompt
/// reproduction → catastrophic over-training, exactly the r=2 < base
/// regression observed in Phase 22 Stage D's A-batch and G1 batches.
///
/// `B > 1` not supported here (per-example prompt boundary varies);
/// callers pre-batch by tokenizing one (prompt, completion) at a time.
pub fn train_qwen_lora_step_masked(
    model: &mut ModelForCausalLM,
    optimizer: &mut candle_nn::AdamW,
    input_ids: &Tensor,
    target_ids: &Tensor,
    prompt_len: usize,
) -> Result<f32> {
    let logits = model.forward_train(input_ids)?;
    let (b, t, v) = logits.dims3()?;
    if b != 1 {
        candle_core::bail!("train_qwen_lora_step_masked: only B=1 supported (got B={b})");
    }
    if prompt_len == 0 {
        // Nothing to mask → identical to the unmasked path.
        let logits_flat = logits.reshape((b * t, v))?.to_dtype(DType::F32)?;
        let targets_flat = target_ids.reshape(b * t)?;
        let loss = candle_nn::loss::cross_entropy(&logits_flat, &targets_flat)?;
        let loss_value = loss.to_scalar::<f32>()?;
        use candle_nn::Optimizer;
        optimizer.backward_step(&loss)?;
        return Ok(loss_value);
    }
    // After the shift, (input, target) has length P+C−1.
    //   target[0..P−1] = prompt-internal predictions (mask)
    //   target[P−1..P+C−1] = completion predictions (keep, C positions)
    // We slice the logits/targets to the last (t − (P − 1)) = t − P + 1 = C
    // positions and compute CE over those.
    if prompt_len.saturating_sub(1) >= t {
        candle_core::bail!(
            "train_qwen_lora_step_masked: prompt_len={prompt_len} >= seq_len+1={}; \
             no completion tokens to score",
            t + 1
        );
    }
    let start = prompt_len - 1;
    let len = t - start;
    // logits is (1, t, v) → narrow on dim 1.
    let logits_comp = logits.narrow(1, start, len)?;
    let logits_flat = logits_comp.reshape((len, v))?.to_dtype(DType::F32)?;
    // target_ids is (1, t) → narrow on dim 1, then squeeze batch.
    let targets_comp = target_ids.narrow(1, start, len)?.reshape(len)?;
    let loss = candle_nn::loss::cross_entropy(&logits_flat, &targets_comp)?;
    let loss_value = loss.to_scalar::<f32>()?;
    use candle_nn::Optimizer;
    optimizer.backward_step(&loss)?;
    Ok(loss_value)
}

/// Diagnostic — compute and report the gradient L2 norm of all Vars
/// passed in `vars` after a single backward pass through the loss. Used
/// by the LoRA smoke to verify gradients are actually flowing into the
/// LoRA adapters before declaring training broken.
pub fn lora_grad_norms(
    model: &mut ModelForCausalLM,
    input_ids: &Tensor,
    target_ids: &Tensor,
    vars: &[candle_core::Var],
) -> Result<Vec<(String, f32)>> {
    let logits = model.forward_train(input_ids)?;
    let (b, t, v) = logits.dims3()?;
    let logits_flat = logits.reshape((b * t, v))?.to_dtype(DType::F32)?;
    let targets_flat = target_ids.reshape(b * t)?;
    let loss = candle_nn::loss::cross_entropy(&logits_flat, &targets_flat)?;
    let grads = loss.backward()?;
    let mut out = Vec::with_capacity(vars.len());
    for (i, var) in vars.iter().enumerate() {
        let g = grads.get(var);
        let norm = match g {
            Some(t) => t
                .to_dtype(DType::F32)?
                .sqr()?
                .sum_all()?
                .sqrt()?
                .to_scalar::<f32>()?,
            None => f32::NAN,
        };
        out.push((format!("var{i}({:?})", var.dims()), norm));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    #[test]
    fn lora_config_default_matches_phase14_20_recipe() {
        let cfg = LoraConfig::default();
        assert_eq!(cfg.rank, 16);
        assert!((cfg.alpha - 32.0).abs() < 1e-6);
    }

    #[test]
    fn isolated_lora_adapter_gets_gradient_through_backprop() -> Result<()> {
        // Smallest possible test: build a LoraAdapter directly, do a
        // forward + backward, verify the Var(s) registered in the VarMap
        // appear in the GradStore. If this fails, the bug is in nanogpt-rs's
        // LoraAdapter; if it passes, the bug is in our Qwen2 integration.
        let device = Device::Cpu;
        let vmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&vmap, DType::F32, &device);
        let adapter = LoraAdapter::new(8, 4, 2, 4.0, vb)?;
        let x = Tensor::randn(0f32, 1.0, (1, 8), &device)?;
        let y = adapter.delta(&x)?;
        let loss = y.sqr()?.sum_all()?;
        let grads = loss.backward()?;
        let vars = vmap.all_vars();
        assert!(!vars.is_empty(), "no LoRA Vars registered");
        let any_grad = vars.iter().any(|v| grads.get(v).is_some());
        assert!(
            any_grad,
            "no LoRA Var has a gradient — autograd broken at LoraAdapter scale"
        );
        Ok(())
    }
}
