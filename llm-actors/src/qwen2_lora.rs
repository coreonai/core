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
/// Compute REINFORCE policy-gradient loss for one (prompt, completion, reward) sample.
/// Returns `None` if the completion is empty (no gradient contribution).
fn pg_sample_loss(
    model: &mut ModelForCausalLM,
    device: &Device,
    prompt: &[u32],
    comp: &[u32],
    reward: f32,
) -> Result<Option<Tensor>> {
    if comp.is_empty() {
        return Ok(None);
    }
    let mut full = prompt.to_vec();
    full.extend_from_slice(comp);
    let full_len = full.len();
    let full_t = Tensor::from_slice(&full, (1, full_len), device)?;
    let logits = model.forward_train(&full_t)?; // (1, P+C, V)
    let p_len = prompt.len();
    let c_len = comp.len();
    let pred = logits.narrow(1, p_len.saturating_sub(1), c_len)?;
    let (_, c, v) = pred.dims3()?;
    let pred_flat = pred.reshape((c, v))?.to_dtype(DType::F32)?;
    let comp_t = Tensor::from_slice(comp, c_len, device)?;
    let mean_ce = candle_nn::loss::cross_entropy(&pred_flat, &comp_t)?;
    Ok(Some((&mean_ce * (reward as f64))?))
}

/// How a prompt-group's per-sample verifier rewards become REINFORCE
/// advantages (Phase 22 RL variance-reduction study). For binary verifier
/// rewards `v ∈ {0,1}` over `k` samples that share a prompt:
///
/// `MeanCenter`: `a_i = v_i − mean(v)`. The historical default; GRPO without
/// the std normalization. A prompt passing 3/4 and one passing 1/4 push with
/// different-magnitude gradients.
///
/// `Rloo`: leave-one-out baseline `a_i = (k·v_i − Σv)/(k−1)`. Unbiased; for
/// binary rewards it equals `MeanCenter × k/(k−1)`, i.e. only a rescale.
///
/// `Grpo`: group-relative normalization `a_i = (v_i − mean(v)) / (std(v) +
/// ε)`. Equalizes each prompt's advantage magnitude so easy and hard prompts
/// push with comparable scale — the variance-reduction lever under test. For
/// binary rewards std stays well away from 0 whenever the group has signal
/// (≥ 0.43 for k=4 mixed), so the divisor is stable.
///
/// A group with no signal (all verdicts equal) maps to all-zero advantages
/// under every mode; the PG step then skips it via `skip_zero_advantage`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdvantageMode {
    MeanCenter,
    Rloo,
    Grpo,
}

impl AdvantageMode {
    /// Parse a CLI spelling. Returns `None` for unknown input so the caller
    /// can error with a usage message.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "mean" | "meancenter" | "mean-center" | "mean_center" => Some(Self::MeanCenter),
            "rloo" | "loo" | "leave-one-out" => Some(Self::Rloo),
            "grpo" | "norm" | "group-norm" => Some(Self::Grpo),
            _ => None,
        }
    }
}

/// Turn one prompt-group's verdicts into advantages under `mode`, with an
/// optional symmetric clip to `[−clip, clip]` applied last. `eps` guards the
/// GRPO divisor. All modes return all-zero for a no-signal group.
pub fn group_advantages(verdicts: &[f32], mode: AdvantageMode, clip: Option<f32>) -> Vec<f32> {
    let k = verdicts.len();
    if k == 0 {
        return Vec::new();
    }
    let sum: f32 = verdicts.iter().copied().sum();
    let mean = sum / k as f32;
    let mut adv: Vec<f32> = match mode {
        AdvantageMode::MeanCenter => verdicts.iter().map(|v| v - mean).collect(),
        AdvantageMode::Rloo => {
            if k < 2 {
                // No leave-one-out baseline exists with a single sample.
                vec![0.0; k]
            } else {
                let denom = (k - 1) as f32;
                verdicts
                    .iter()
                    .map(|v| (k as f32 * v - sum) / denom)
                    .collect()
            }
        }
        AdvantageMode::Grpo => {
            let var = verdicts.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / k as f32;
            let std = var.sqrt();
            const EPS: f32 = 1e-6;
            verdicts.iter().map(|v| (v - mean) / (std + EPS)).collect()
        }
    };
    if let Some(c) = clip {
        let c = c.abs();
        for a in adv.iter_mut() {
            *a = a.clamp(-c, c);
        }
    }
    adv
}

/// Phase 22 follow-up C3 — knobs for one REINFORCE policy-gradient update.
///
/// The Stage E defaults (`micro_batch_size` chunking, one AdamW update per
/// chunk, every sample forwarded) collapsed the 7B hard-tail policy at the
/// first adapter sync. Two things were wrong, both fixed by this config:
///
/// - **`accumulate_grads`**: micro-batching was introduced purely to bound
///   peak GPU memory, but it also turned one PG step into `n_samples /
///   micro_batch` *optimizer* steps. At the 7B hard-tail setting (64 prompts
///   × k=4, `--pg-micro-batch-size 1`) that is **256 AdamW updates per RL
///   step** — vs 30 for a whole SFT round. AdamW normalises by gradient
///   magnitude, so a numerically tiny loss is *not* a tiny step; 4 RL steps
///   applied ~1024 full-size updates before the first sync ever revealed
///   them. Accumulating the per-chunk gradients and issuing a single
///   `Optimizer::step` restores "one PG step = one update" while keeping the
///   memory bound.
/// - **`skip_zero_advantage`**: under RLOO, every prompt whose k samples all
///   share a verdict gets advantage exactly 0. On the hard tail that is ~94%
///   of samples. They contribute no gradient, but they still cost a
///   forward+backward — and, without accumulation, each one still *moved the
///   weights*, because `AdamW::step` applies `m_hat/(sqrt(v_hat)+eps)` from
///   the momentum tail even when the incoming gradient is all zeros.
#[derive(Debug, Clone, Copy)]
pub struct PgStepConfig {
    /// Samples per forward/backward chunk. `0` = one chunk for everything
    /// (may OOM with long completions).
    pub micro_batch_size: usize,
    /// Accumulate chunk gradients and apply ONE optimizer update per call.
    /// `false` restores the Stage E per-chunk-update behaviour.
    pub accumulate_grads: bool,
    /// Drop samples whose `|reward|` is below `f32::EPSILON` before the
    /// forward pass.
    pub skip_zero_advantage: bool,
    /// Phase 22 follow-up C4 — keep only strictly-positive-advantage
    /// samples, i.e. train on the completions that *passed* the verifier.
    ///
    /// This removes the unbounded term from the objective. `pg_sample_loss`
    /// computes `mean_ce * reward`, so a negative-advantage sample is
    /// gradient *ascent* on cross-entropy — which has no upper bound, and
    /// under RLOO with k=4 is ~75% of the surviving samples. That is what
    /// makes the policy run away (C3 measured 2/2 seeds collapsing to
    /// 0/256 even with an adapter sync every step). With only positive
    /// rewards the loss is `reward * CE >= 0`: bounded below, and
    /// equivalent to reward-weighted SFT on verified-correct completions
    /// — i.e. rejection-sampling fine-tuning / RAFT, the same family as
    /// the Phase 22 SFT recipe that is worth +0.254 on this hard tail.
    ///
    /// Implies `skip_zero_advantage` (zero is not positive).
    pub positive_advantage_only: bool,
}

impl Default for PgStepConfig {
    fn default() -> Self {
        Self {
            micro_batch_size: 0,
            accumulate_grads: true,
            skip_zero_advantage: true,
            positive_advantage_only: false,
        }
    }
}

/// What one `train_qwen_lora_pg_step_cfg` call actually did — the
/// diagnostics that would have caught the Stage E collapse from the logs.
#[derive(Debug, Clone, Copy, Default)]
pub struct PgStepStats {
    /// Mean per-chunk loss (the conventional REINFORCE logging choice).
    pub loss: f32,
    /// Samples that reached a forward pass.
    pub n_used: usize,
    /// Samples dropped by `skip_zero_advantage` (or for being empty).
    pub n_skipped: usize,
    /// Optimizer updates issued. `1` when accumulating.
    pub n_updates: usize,
}

/// Phase 21 Stage G / Phase 22 Stage E — REINFORCE policy-gradient update.
///
/// Processes `samples` in chunks of `cfg.micro_batch_size` to bound peak GPU
/// memory. With `cfg.accumulate_grads` (the default) the chunk gradients are
/// summed and a single optimizer update is applied; otherwise each chunk gets
/// its own update (the Stage E behaviour — see [`PgStepConfig`]).
///
/// `trainable` is the LoRA `Var` set the optimizer owns. Accumulation keeps
/// **only** those gradients between chunks: a candle `GradStore` also holds a
/// gradient for every intermediate activation, so retaining whole stores
/// across chunks OOMs a 7B backward at max_new=192. The last chunk's store is
/// reused as the carrier handed to `Optimizer::step`, which keeps peak memory
/// at exactly one chunk's backward — the same as the per-chunk-update path.
pub fn train_qwen_lora_pg_step_cfg(
    model: &mut ModelForCausalLM,
    optimizer: &mut candle_nn::AdamW,
    device: &Device,
    samples: &[(Vec<u32>, Vec<u32>, f32)],
    trainable: &[candle_core::Var],
    cfg: PgStepConfig,
) -> Result<PgStepStats> {
    use candle_core::backprop::GradStore;
    use candle_nn::Optimizer;
    if samples.is_empty() {
        candle_core::bail!("train_qwen_lora_pg_step: samples is empty");
    }
    let mut n_skipped = 0usize;
    let kept: Vec<&(Vec<u32>, Vec<u32>, f32)> = samples
        .iter()
        .filter(|(_, comp, reward)| {
            let no_advantage = cfg.skip_zero_advantage && reward.abs() <= f32::EPSILON;
            let not_positive = cfg.positive_advantage_only && *reward <= f32::EPSILON;
            let keep = !comp.is_empty() && !no_advantage && !not_positive;
            if !keep {
                n_skipped += 1;
            }
            keep
        })
        .collect();
    if kept.is_empty() {
        // Every sample was filtered out. Under `skip_zero_advantage` that is
        // an ordinary outcome on a sparse-reward step (no prompt had a mixed
        // verdict) — report a no-op rather than failing the RL run. Without
        // the filter it means every completion was empty, which is the
        // Stage E error case.
        if cfg.skip_zero_advantage || cfg.positive_advantage_only {
            return Ok(PgStepStats {
                loss: 0.0,
                n_used: 0,
                n_skipped,
                n_updates: 0,
            });
        }
        candle_core::bail!("train_qwen_lora_pg_step: no usable samples");
    }
    let mb = if cfg.micro_batch_size == 0 {
        kept.len()
    } else {
        cfg.micro_batch_size
    };

    let mut total_loss = 0.0f32;
    let mut n_chunks = 0usize;
    let mut n_used = 0usize;
    let mut n_updates = 0usize;
    // Summed gradients for the trainable Vars only — a whole `GradStore` per
    // chunk would pin every intermediate activation gradient and OOM 7B.
    let mut acc: std::collections::HashMap<candle_core::TensorId, Tensor> =
        std::collections::HashMap::new();
    // The most recent chunk's store, reused as the carrier for the final
    // `Optimizer::step`. Only kept past the loop body on the last chunk.
    let mut carrier: Option<GradStore> = None;
    let n_total_chunks = kept.chunks(mb).len();

    for (chunk_idx, chunk) in kept.chunks(mb).enumerate() {
        let mut loss: Option<Tensor> = None;
        let mut n_in_chunk = 0usize;
        for (prompt, comp, reward) in chunk {
            if let Some(contrib) = pg_sample_loss(model, device, prompt, comp, *reward)? {
                loss = Some(match loss {
                    Some(prev) => (prev + contrib)?,
                    None => contrib,
                });
                n_in_chunk += 1;
            }
        }
        let Some(loss) = loss else { continue };
        let loss = (loss / n_in_chunk as f64)?;
        total_loss += loss.to_scalar::<f32>()?;
        n_used += n_in_chunk;
        n_chunks += 1;
        if cfg.accumulate_grads {
            let grads = loss.backward()?;
            for var in trainable {
                let Some(g) = grads.get(var.as_tensor()) else {
                    continue;
                };
                let id = var.as_tensor().id();
                let merged = match acc.remove(&id) {
                    Some(prev) => (prev + g)?,
                    None => g.clone(),
                };
                acc.insert(id, merged);
            }
            // Drop every other chunk's store right here; keep the last one
            // to hand to the optimizer.
            if chunk_idx + 1 == n_total_chunks {
                carrier = Some(grads);
            }
        } else {
            optimizer.step(&loss.backward()?)?;
            n_updates += 1;
        }
    }
    if n_chunks == 0 {
        candle_core::bail!("train_qwen_lora_pg_step: no usable samples");
    }
    if let Some(mut carrier) = carrier {
        // Mean over chunks: each chunk loss is already a per-sample mean, so
        // dividing the summed gradients by the chunk count makes the update
        // independent of how the batch happened to be split.
        let scale = 1.0 / n_chunks as f64;
        for var in trainable {
            if let Some(g) = acc.remove(&var.as_tensor().id()) {
                carrier.insert(var.as_tensor(), (g * scale)?);
            }
        }
        optimizer.step(&carrier)?;
        n_updates += 1;
    } else if !acc.is_empty() {
        // Unreachable while every kept sample has a non-empty completion
        // (the filter above guarantees it), but never silently swallow an
        // accumulated gradient.
        candle_core::bail!("train_qwen_lora_pg_step: accumulated gradients with no carrier store");
    }
    Ok(PgStepStats {
        loss: total_loss / n_chunks as f32,
        n_used,
        n_skipped,
        n_updates,
    })
}

/// Back-compat wrapper preserving the exact Phase 22 Stage E semantics
/// (one optimizer update per micro-batch, zero-advantage samples kept).
/// New call sites should use [`train_qwen_lora_pg_step_cfg`].
pub fn train_qwen_lora_pg_step(
    model: &mut ModelForCausalLM,
    optimizer: &mut candle_nn::AdamW,
    device: &Device,
    samples: &[(Vec<u32>, Vec<u32>, f32)],
    micro_batch_size: usize,
) -> Result<f32> {
    train_qwen_lora_pg_step_cfg(
        model,
        optimizer,
        device,
        samples,
        // The legacy path steps per chunk and never accumulates, so it needs
        // no Var list.
        &[],
        PgStepConfig {
            micro_batch_size,
            accumulate_grads: false,
            skip_zero_advantage: false,
            positive_advantage_only: false,
        },
    )
    .map(|s| s.loss)
}

/// Resolve the safetensors shard file(s) that make up a model.
///
/// `path` may be either:
/// - a single `.safetensors` file (e.g. a merged checkpoint) → returned
///   as-is;
/// - an HF snapshot **directory** → resolved to its shard set. Large
///   models (e.g. Qwen2.5-Coder-7B) shard weights into
///   `model-0000N-of-0000M.safetensors` listed by a
///   `model.safetensors.index.json` `weight_map`; single-file models
///   (0.5B / 1.5B) expose one `model.safetensors`.
///
/// The shard list is deduplicated and sorted (BTreeSet) so mmap order is
/// deterministic. This is the single choke point that makes every loader
/// work for both the single-file and sharded layouts.
pub fn resolve_safetensors(path: &std::path::Path) -> Result<Vec<std::path::PathBuf>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    if !path.is_dir() {
        return Err(candle_core::Error::Msg(format!(
            "resolve_safetensors: {path:?} is neither a file nor a directory"
        )));
    }
    let index = path.join("model.safetensors.index.json");
    if index.is_file() {
        let text = std::fs::read_to_string(&index)
            .map_err(|e| candle_core::Error::Msg(format!("read {index:?}: {e}")))?;
        let json: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| candle_core::Error::Msg(format!("parse {index:?}: {e}")))?;
        let weight_map = json
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| candle_core::Error::Msg(format!("no weight_map object in {index:?}")))?;
        let shards: std::collections::BTreeSet<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        if shards.is_empty() {
            return Err(candle_core::Error::Msg(format!(
                "empty weight_map in {index:?}"
            )));
        }
        Ok(shards.into_iter().map(|s| path.join(s)).collect())
    } else {
        let single = path.join("model.safetensors");
        if !single.is_file() {
            return Err(candle_core::Error::Msg(format!(
                "no model.safetensors or model.safetensors.index.json in {path:?}"
            )));
        }
        Ok(vec![single])
    }
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
    _device: &Device,
    out_path: &std::path::Path,
) -> Result<()> {
    // Merge on CPU, NOT on `_device`. This is a one-time, I/O-bound op, and
    // loading the full base onto the GPU here would be a THIRD ~15GB copy on
    // top of the resident inference model + trainer base (~30GB) — that OOMs
    // a 40GB A100 for 7B (the pipeline peaks at ~39GB before the merge). Host
    // RAM is ample, and the per-layer delta matmuls are tiny, so CPU is free.
    let merge_dev = Device::Cpu;
    // Sharded-aware: `base_safetensors_path` may be a single merged file
    // (0.5B, or a previously merged checkpoint) or an HF snapshot dir whose
    // weights span multiple shards (7B). Load every shard into one map.
    let mut tensors = std::collections::HashMap::new();
    for shard in resolve_safetensors(base_safetensors_path)? {
        tensors.extend(candle_core::safetensors::load(&shard, &merge_dev)?);
    }
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
            // a: (rank, in_dim), b: (out_dim, rank) → delta = b @ a (out, in).
            // Merge in F32 on the CPU device: candle's CPU backend has no bf16
            // matmul, so cast base + LoRA vars up to F32, compute, then cast
            // the merged weight back to the base dtype for saving.
            let out_dtype = base.dtype();
            let base_f32 = base.to_device(&merge_dev)?.to_dtype(DType::F32)?;
            let a_t = a.as_tensor().to_device(&merge_dev)?.to_dtype(DType::F32)?;
            let b_t = b.as_tensor().to_device(&merge_dev)?.to_dtype(DType::F32)?;
            let delta = (b_t.matmul(&a_t)? * scale)?;
            let merged = (&base_f32 + &delta)?.to_dtype(out_dtype)?;
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
    let loss = cross_entropy_with_prompt_mask(&logits, target_ids, prompt_len)?;
    let loss_value = loss.to_scalar::<f32>()?;
    use candle_nn::Optimizer;
    optimizer.backward_step(&loss)?;
    Ok(loss_value)
}

/// Phase 22 Stage D follow-up — cosine LR schedule with linear warmup,
/// matching Phase 17's Python recipe (`get_cosine_schedule_with_warmup`
/// from HuggingFace Transformers).
///
/// For `step < warmup_steps`: linear ramp from 0 → base_lr.
/// For `step >= warmup_steps`: cosine decay from base_lr → 0 over the
/// remaining `total_steps − warmup_steps` steps.
///
/// Phase 17 used `warmup_steps = max(1, total_steps / 10)` (10%
/// warmup). The training step is in [0, total_steps).
pub fn cosine_warmup_lr(step: usize, warmup_steps: usize, total_steps: usize, base_lr: f64) -> f64 {
    if total_steps == 0 {
        return base_lr;
    }
    if step < warmup_steps {
        // Linear warmup: step=0 → 0, step=warmup_steps-1 → ~base_lr.
        return base_lr * (step + 1) as f64 / warmup_steps.max(1) as f64;
    }
    // Cosine decay from base_lr at step=warmup_steps to 0 at step=total_steps.
    let progress = (step - warmup_steps) as f64 / (total_steps - warmup_steps).max(1) as f64;
    base_lr * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
}

/// Phase 22 Stage D fix — pure helper that computes Phase 17 Python's
/// completion-only cross-entropy from `(1, T, V)` logits and `(1, T)`
/// target token IDs.
///
/// - `prompt_len == 0`: identical to standard `cross_entropy` over all
///   `T` positions.
/// - `prompt_len > 0`: skip the first `prompt_len − 1` shifted-target
///   positions (which are prompt-internal predictions) and CE-loss
///   only the last `T − (prompt_len − 1)` positions (completion-token
///   predictions).
///
/// Extracted from `train_qwen_lora_step_masked` so the masking logic
/// can be unit-tested without instantiating a full Qwen model. Returns
/// the loss `Tensor` (caller decides to `.backward()` or just inspect).
pub fn cross_entropy_with_prompt_mask(
    logits: &Tensor,
    target_ids: &Tensor,
    prompt_len: usize,
) -> Result<Tensor> {
    let (b, t, v) = logits.dims3()?;
    if b != 1 {
        candle_core::bail!("cross_entropy_with_prompt_mask: only B=1 supported (got B={b})");
    }
    if prompt_len == 0 {
        let logits_flat = logits.reshape((b * t, v))?.to_dtype(DType::F32)?;
        let targets_flat = target_ids.reshape(b * t)?;
        return candle_nn::loss::cross_entropy(&logits_flat, &targets_flat);
    }
    // After the shift, (input, target) has length P+C−1.
    //   target[0..P−1] = prompt-internal predictions (mask)
    //   target[P−1..P+C−1] = completion predictions (keep, C positions)
    if prompt_len.saturating_sub(1) >= t {
        candle_core::bail!(
            "cross_entropy_with_prompt_mask: prompt_len={prompt_len} >= seq_len+1={}; \
             no completion tokens to score",
            t + 1
        );
    }
    let start = prompt_len - 1;
    let len = t - start;
    let logits_comp = logits.narrow(1, start, len)?;
    let logits_flat = logits_comp.reshape((len, v))?.to_dtype(DType::F32)?;
    let targets_comp = target_ids.narrow(1, start, len)?.reshape(len)?;
    candle_nn::loss::cross_entropy(&logits_flat, &targets_comp)
}

/// Phase 22 Stage D G7 — batched completion-only cross-entropy.
///
/// Generalizes [`cross_entropy_with_prompt_mask`] to `B > 1` so the
/// trainer can pad several `(prompt, completion)` pairs into one
/// forward/backward (Phase 17 used `batch_size = 4`). Because each
/// example has its own prompt boundary AND its own real length (after
/// right-padding a batch to a common width), a single `narrow` can't
/// express the per-example completion span — so this takes an explicit
/// per-position `loss_mask`.
///
/// Shapes (all in the SHIFTED frame, i.e. position `j` scores the
/// prediction of `targets[.., j]`):
/// - `logits`: `(B, T, V)`
/// - `targets`: `(B, T)` — `u32` token ids
/// - `loss_mask`: `(B, T)` — `1.0` on completion positions, `0.0` on
///   prompt-internal AND right-padding positions.
///
/// Returns the mean negative-log-likelihood over all masked-in
/// positions across the whole batch (token-level mean, matching
/// HuggingFace's default `CrossEntropyLoss(reduction="mean")` over the
/// un-ignored labels — Phase 17's recipe).
///
/// Right-padding is safe without an explicit attention mask: the causal
/// mask means a real token at position `t` only attends to `[0, t]`, so
/// padding (always appended AFTER the real tokens) never leaks into a
/// real token's representation, and padded outputs are dropped here via
/// `loss_mask = 0`.
pub fn cross_entropy_with_completion_mask_batched(
    logits: &Tensor,
    targets: &Tensor,
    loss_mask: &Tensor,
) -> Result<Tensor> {
    let (b, t, v) = logits.dims3()?;
    let logits_flat = logits.reshape((b * t, v))?.to_dtype(DType::F32)?;
    let log_probs = candle_nn::ops::log_softmax(&logits_flat, 1)?; // (B*T, V)
    let targets_flat = targets.reshape((b * t, 1))?;
    // gather log P(target) at each position → (B*T, 1) → (B*T,)
    let tgt_lp = log_probs.gather(&targets_flat, 1)?.squeeze(1)?;
    let nll = tgt_lp.neg()?; // (B*T,)
    let mask_flat = loss_mask.reshape((b * t,))?.to_dtype(DType::F32)?;
    let masked = nll.mul(&mask_flat)?;
    let total = masked.sum_all()?;
    let count = mask_flat.sum_all()?;
    total.broadcast_div(&count)
}

/// Phase 22 Stage D G7 — one batched, padded, completion-masked AdamW
/// step. `input_ids`/`target_ids`/`loss_mask` are all `(B, T)` in the
/// shifted frame (caller right-pads + builds the mask). Returns the
/// pre-step loss value.
pub fn train_qwen_lora_step_masked_batched(
    model: &mut ModelForCausalLM,
    optimizer: &mut candle_nn::AdamW,
    input_ids: &Tensor,
    target_ids: &Tensor,
    loss_mask: &Tensor,
) -> Result<f32> {
    let logits = model.forward_train(input_ids)?;
    let loss = cross_entropy_with_completion_mask_batched(&logits, target_ids, loss_mask)?;
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

    fn approx_eq(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len(), "len {a:?} vs {b:?}");
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-4, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn advantage_mode_parses_spellings() {
        assert_eq!(
            AdvantageMode::parse("mean"),
            Some(AdvantageMode::MeanCenter)
        );
        assert_eq!(
            AdvantageMode::parse("Mean-Center"),
            Some(AdvantageMode::MeanCenter)
        );
        assert_eq!(AdvantageMode::parse("rloo"), Some(AdvantageMode::Rloo));
        assert_eq!(AdvantageMode::parse("GRPO"), Some(AdvantageMode::Grpo));
        assert_eq!(AdvantageMode::parse("nope"), None);
    }

    #[test]
    fn group_advantages_mean_center_is_current_default() {
        // v - mean; matches the historical inline baseline in the RL example.
        let a = group_advantages(&[1.0, 0.0, 0.0, 0.0], AdvantageMode::MeanCenter, None);
        approx_eq(&a, &[0.75, -0.25, -0.25, -0.25]);
        assert!(
            (a.iter().sum::<f32>()).abs() < 1e-5,
            "centered advantages sum to 0"
        );
    }

    #[test]
    fn group_advantages_rloo_is_mean_center_times_k_over_km1() {
        // Leave-one-out on binary rewards = mean-center * k/(k-1).
        let a = group_advantages(&[1.0, 0.0, 0.0, 0.0], AdvantageMode::Rloo, None);
        approx_eq(&a, &[1.0, -1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0]);
        // k=1: no baseline possible -> 0.
        approx_eq(&group_advantages(&[1.0], AdvantageMode::Rloo, None), &[0.0]);
    }

    #[test]
    fn group_advantages_grpo_normalizes_by_group_std() {
        // mean=0.25, std=sqrt(0.1875)=0.43301; passing=0.75/std, fail=-0.25/std.
        let a = group_advantages(&[1.0, 0.0, 0.0, 0.0], AdvantageMode::Grpo, None);
        approx_eq(&a, &[1.73205, -0.57735, -0.57735, -0.57735]);
        // Two prompts of different difficulty get the SAME passing magnitude
        // under GRPO — the equalization that MeanCenter lacks.
        let easy = group_advantages(&[1.0, 1.0, 1.0, 0.0], AdvantageMode::Grpo, None); // 3/4
        let hard = group_advantages(&[1.0, 0.0, 0.0, 0.0], AdvantageMode::Grpo, None); // 1/4
        assert!(
            (easy[3].abs() - hard[0].abs()).abs() < 1e-4,
            "|adv| equalized: {easy:?} {hard:?}"
        );
    }

    #[test]
    fn group_advantages_no_signal_is_all_zero() {
        for v in [&[1.0f32, 1.0, 1.0, 1.0][..], &[0.0, 0.0, 0.0][..]] {
            for m in [
                AdvantageMode::MeanCenter,
                AdvantageMode::Rloo,
                AdvantageMode::Grpo,
            ] {
                let a = group_advantages(v, m, None);
                assert!(a.iter().all(|x| x.abs() < 1e-5), "{m:?} on {v:?} -> {a:?}");
            }
        }
        assert!(group_advantages(&[], AdvantageMode::Grpo, None).is_empty());
    }

    #[test]
    fn group_advantages_clip_is_symmetric_and_last() {
        let a = group_advantages(&[1.0, 0.0, 0.0, 0.0], AdvantageMode::Grpo, Some(1.0));
        // 1.732 clipped to 1.0; -0.577 untouched.
        approx_eq(&a, &[1.0, -0.57735, -0.57735, -0.57735]);
    }

    /// Phase 22 Stage D fix — masking helper tests. Verify that
    /// `cross_entropy_with_prompt_mask` slices to the correct
    /// position range and produces the same CE value as a
    /// hand-computed reference on those positions.
    #[test]
    fn prompt_mask_zero_matches_full_cross_entropy() -> Result<()> {
        let dev = Device::Cpu;
        let t = 6usize;
        let v = 5usize;
        // Deterministic logits + targets.
        let logits_data: Vec<f32> = (0..(t * v)).map(|i| (i as f32) * 0.1 - 0.5).collect();
        let logits = Tensor::from_vec(logits_data, (1, t, v), &dev)?;
        let targets: Vec<u32> = (0..t).map(|i| (i as u32) % v as u32).collect();
        let target_ids = Tensor::from_vec(targets, (1, t), &dev)?;
        // prompt_len = 0 → no masking, all positions count.
        let loss_zero =
            cross_entropy_with_prompt_mask(&logits, &target_ids, 0)?.to_scalar::<f32>()?;
        let logits_flat = logits.reshape((t, v))?.to_dtype(DType::F32)?;
        let targets_flat = target_ids.reshape(t)?;
        let loss_ref =
            candle_nn::loss::cross_entropy(&logits_flat, &targets_flat)?.to_scalar::<f32>()?;
        assert!(
            (loss_zero - loss_ref).abs() < 1e-5,
            "prompt_len=0 should match plain CE: {loss_zero} vs {loss_ref}"
        );
        Ok(())
    }

    #[test]
    fn prompt_mask_skips_prompt_positions() -> Result<()> {
        let dev = Device::Cpu;
        let t = 6usize;
        let v = 5usize;
        // Make positions [0..P-1] have LOW probability (high loss)
        // and positions [P-1..T] have HIGH probability (low loss),
        // so the masked CE (which keeps only P-1..T) should be
        // much LESS than the unmasked CE.
        // Position pos targets token y[pos]. For y[pos] to have low
        // loss the logit at column y[pos] should dominate.
        let prompt_len = 3usize;
        // shift indices [0..t]: positions 0..prompt_len-1 = prompt-internal (target = wrong)
        // positions prompt_len-1..t = completion (target = "correct")
        let mut logits_data = vec![0.0f32; t * v];
        let mut targets: Vec<u32> = Vec::with_capacity(t);
        for pos in 0..t {
            if pos < prompt_len - 1 {
                // prompt-internal: target token is 0 but logits favor 4
                logits_data[pos * v + 4] = 10.0;
                targets.push(0);
            } else {
                // completion: target token is 4 and logits favor 4
                logits_data[pos * v + 4] = 10.0;
                targets.push(4);
            }
        }
        let logits = Tensor::from_vec(logits_data, (1, t, v), &dev)?;
        let target_ids = Tensor::from_vec(targets, (1, t), &dev)?;
        let loss_full =
            cross_entropy_with_prompt_mask(&logits, &target_ids, 0)?.to_scalar::<f32>()?;
        let loss_masked =
            cross_entropy_with_prompt_mask(&logits, &target_ids, prompt_len)?.to_scalar::<f32>()?;
        // Masked loss should be much lower (only completion positions,
        // all "correct" → log P ≈ 0). Unmasked loss is high because
        // prompt-internal positions have very wrong targets.
        assert!(
            loss_masked < loss_full,
            "masked CE should be < unmasked when prompt positions have wrong targets: \
             masked={loss_masked}, full={loss_full}"
        );
        // Masked should be ~0 because all completion positions favor
        // the correct token by ~10 logits margin.
        assert!(
            loss_masked < 0.01,
            "masked CE should be near zero when completion targets are easy: {loss_masked}"
        );
        // Unmasked should be substantially positive because prompt
        // positions cost log(softmax_diff) ≈ 10 each (out of t-prompt+1
        // good positions vs prompt-1 bad ones).
        assert!(
            loss_full > 1.0,
            "unmasked CE should be substantial when prompt positions are wrong: {loss_full}"
        );
        Ok(())
    }

    #[test]
    fn prompt_mask_rejects_prompt_len_too_large() -> Result<()> {
        let dev = Device::Cpu;
        let t = 4usize;
        let v = 3usize;
        let logits = Tensor::zeros((1, t, v), DType::F32, &dev)?;
        let target_ids = Tensor::zeros((1, t), DType::U32, &dev)?;
        // prompt_len = T + 1 → start = T, len = 0 → bail
        let result = cross_entropy_with_prompt_mask(&logits, &target_ids, t + 1);
        assert!(
            result.is_err(),
            "prompt_len > T should error, got Ok({:?})",
            result.ok().map(|t| t.to_scalar::<f32>().unwrap_or(0.0))
        );
        Ok(())
    }

    #[test]
    fn prompt_mask_rejects_batch_greater_than_one() -> Result<()> {
        let dev = Device::Cpu;
        let logits = Tensor::zeros((2, 4, 3), DType::F32, &dev)?;
        let target_ids = Tensor::zeros((2, 4), DType::U32, &dev)?;
        let result = cross_entropy_with_prompt_mask(&logits, &target_ids, 2);
        assert!(result.is_err(), "B=2 should error");
        Ok(())
    }

    /// Phase 22 G7 — the batched completion-mask loss must agree with
    /// the (tested) narrow-based B=1 loss when fed a single example with
    /// a mask that is 1.0 exactly on the completion span `[P-1, T)`.
    #[test]
    fn batched_mask_loss_matches_narrow_b1() -> Result<()> {
        let dev = Device::Cpu;
        let (t, v) = (6usize, 5usize);
        let prompt_len = 3usize;
        let logits = Tensor::randn(0f32, 1.0, (1, t, v), &dev)?;
        let targets = Tensor::from_slice(&[1u32, 4, 2, 0, 3, 1], (1, t), &dev)?;
        let narrow =
            cross_entropy_with_prompt_mask(&logits, &targets, prompt_len)?.to_scalar::<f32>()?;
        // mask: 1.0 on [prompt_len-1, T), 0.0 before.
        let mask_vals: Vec<f32> = (0..t)
            .map(|j| if j >= prompt_len - 1 { 1.0 } else { 0.0 })
            .collect();
        let loss_mask = Tensor::from_slice(&mask_vals, (1, t), &dev)?;
        let batched = cross_entropy_with_completion_mask_batched(&logits, &targets, &loss_mask)?
            .to_scalar::<f32>()?;
        assert!(
            (narrow - batched).abs() < 1e-5,
            "narrow={narrow} batched={batched}"
        );
        Ok(())
    }

    /// Phase 22 G7 — a B=2 batch with different per-example completion
    /// spans (one right-padded) must equal the token-level mean NLL
    /// computed by hand over only the masked-in positions.
    #[test]
    fn batched_mask_loss_b2_token_mean() -> Result<()> {
        let dev = Device::Cpu;
        let (b, t, v) = (2usize, 4usize, 3usize);
        // Deterministic logits so we can hand-check.
        let logits = Tensor::randn(0f32, 1.0, (b, t, v), &dev)?;
        let targets = Tensor::from_slice(&[0u32, 1, 2, 0, 2, 1, 0, 0], (b, t), &dev)?;
        // Example 0: completion at positions {1,2,3}; example 1
        // (shorter, right-padded): completion at {1} only.
        let mask_vals: Vec<f32> = vec![0., 1., 1., 1., 0., 1., 0., 0.];
        let loss_mask = Tensor::from_slice(&mask_vals, (b, t), &dev)?;
        let got = cross_entropy_with_completion_mask_batched(&logits, &targets, &loss_mask)?
            .to_scalar::<f32>()?;
        // Hand-compute: mean NLL over the 4 masked positions.
        let logits_flat = logits.reshape((b * t, v))?;
        let lp = candle_nn::ops::log_softmax(&logits_flat, 1)?;
        let tgt = targets.reshape((b * t, 1))?;
        let nll: Vec<f32> = lp.gather(&tgt, 1)?.squeeze(1)?.neg()?.to_vec1()?;
        let idx = [1usize, 2, 3, 5];
        let want: f32 = idx.iter().map(|&i| nll[i]).sum::<f32>() / idx.len() as f32;
        assert!((got - want).abs() < 1e-5, "got={got} want={want}");
        Ok(())
    }

    /// Phase 22 Stage D follow-up — cosine warmup schedule sanity:
    /// linear ramp during warmup, cosine decay after, base_lr peak
    /// at the warmup→decay boundary, 0 at the end.
    #[test]
    fn cosine_warmup_schedule_basic_shape() {
        let total = 100usize;
        let warmup = 10usize;
        let base = 2e-4_f64;
        // Step 0 in warmup → small lr (1/warmup × base)
        let lr_0 = cosine_warmup_lr(0, warmup, total, base);
        assert!(lr_0 > 0.0 && lr_0 < base);
        // Step = warmup-1 → near base_lr (final warmup step)
        let lr_warm_end = cosine_warmup_lr(warmup - 1, warmup, total, base);
        assert!((lr_warm_end - base).abs() < 1e-9);
        // Step = warmup (first cosine step, progress = 0) → cos(0) = 1 → base_lr
        let lr_cos_start = cosine_warmup_lr(warmup, warmup, total, base);
        assert!((lr_cos_start - base).abs() < 1e-9);
        // Step = total-1 → near 0 (final cosine step, progress ≈ 1)
        let lr_end = cosine_warmup_lr(total - 1, warmup, total, base);
        assert!(lr_end < base * 0.01);
        // Monotonic decay during cosine phase
        let lr_mid = cosine_warmup_lr((warmup + total) / 2, warmup, total, base);
        assert!(lr_end < lr_mid && lr_mid < lr_cos_start);
    }

    #[test]
    fn cosine_warmup_total_steps_zero_returns_base_lr() {
        // Edge case: total=0 → just return base_lr (no-op).
        let lr = cosine_warmup_lr(0, 0, 0, 1e-3);
        assert!((lr - 1e-3).abs() < 1e-12);
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

    /// Unique scratch dir under the system temp, keyed by test name +
    /// process id so parallel `cargo test` runs never collide.
    fn scratch(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("workllm-resolve-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn resolve_safetensors_single_file_returns_itself() {
        let dir = scratch("single-file");
        let f = dir.join("model.safetensors");
        std::fs::write(&f, b"stub").unwrap();
        // A direct file path resolves to exactly that file.
        let got = resolve_safetensors(&f).unwrap();
        assert_eq!(got, vec![f.clone()]);
        // A directory containing one model.safetensors resolves to it.
        let got_dir = resolve_safetensors(&dir).unwrap();
        assert_eq!(got_dir, vec![f]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_safetensors_sharded_dir_reads_index_dedup_sorted() {
        let dir = scratch("sharded");
        for shard in [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ] {
            std::fs::write(dir.join(shard), b"stub").unwrap();
        }
        // weight_map lists many tensors across 2 shards (out of order, with
        // duplicate shard references) — resolve must dedup + sort.
        let index = r#"{
            "metadata": {"total_size": 42},
            "weight_map": {
                "b.weight": "model-00002-of-00002.safetensors",
                "a.weight": "model-00001-of-00002.safetensors",
                "c.weight": "model-00002-of-00002.safetensors"
            }
        }"#;
        std::fs::write(dir.join("model.safetensors.index.json"), index).unwrap();
        let got = resolve_safetensors(&dir).unwrap();
        assert_eq!(
            got,
            vec![
                dir.join("model-00001-of-00002.safetensors"),
                dir.join("model-00002-of-00002.safetensors"),
            ]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_safetensors_empty_dir_errors() {
        let dir = scratch("empty");
        assert!(resolve_safetensors(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- Phase 22 follow-up C3 — policy-gradient step semantics ----

    /// A randomly-initialised toy Qwen2 (2 layers, hidden 16) small enough
    /// to run a real forward+backward on CPU in a unit test. Both the base
    /// weights and the LoRA adapters are created by the VarBuilder's
    /// default init, so gradients are non-trivial.
    fn toy_model(dev: &Device) -> Result<(ModelForCausalLM, VarMap, VarMap)> {
        let cfg = Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            max_position_embeddings: 64,
            sliding_window: 64,
            max_window_layers: 2,
            tie_word_embeddings: false,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            use_sliding_window: false,
            hidden_act: candle_nn::Activation::Silu,
        };
        let base_map = VarMap::new();
        let base_vb = VarBuilder::from_varmap(&base_map, DType::F32, dev);
        let lora_map = VarMap::new();
        let lora_vb = VarBuilder::from_varmap(&lora_map, DType::F32, dev);
        let model = ModelForCausalLM::new(
            &cfg,
            base_vb,
            Some(lora_vb),
            LoraConfig {
                rank: 4,
                alpha: 8.0,
            },
        )?;
        Ok((model, base_map, lora_map))
    }

    fn toy_samples() -> Vec<(Vec<u32>, Vec<u32>, f32)> {
        vec![
            (vec![1, 2, 3], vec![4, 5], 0.75),
            (vec![1, 2, 3], vec![6, 7], -0.25),
            (vec![2, 3, 4], vec![8, 9], -0.25),
            (vec![2, 3, 4], vec![10, 11], -0.25),
        ]
    }

    fn lora_snapshot(map: &VarMap) -> Vec<f32> {
        let data = map.data().lock().unwrap();
        let mut names: Vec<&String> = data.keys().collect();
        names.sort();
        names
            .iter()
            .flat_map(|n| {
                data[*n]
                    .as_tensor()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            })
            .collect()
    }

    fn adamw(map: &VarMap) -> Result<candle_nn::AdamW> {
        use candle_nn::Optimizer;
        candle_nn::AdamW::new(
            map.all_vars(),
            candle_nn::ParamsAdamW {
                lr: 1e-3,
                ..Default::default()
            },
        )
    }

    /// The core Stage E defect: micro-batching (a *memory* knob) silently
    /// multiplied the number of AdamW updates per RL step. With
    /// accumulation, splitting the same batch into chunks of 1 must land on
    /// the same weights as a single un-chunked pass — one update either way.
    #[test]
    fn pg_accumulation_makes_micro_batching_update_equivalent() -> Result<()> {
        let dev = Device::Cpu;
        let samples = toy_samples();

        // One model for all three variants: `VarBuilder` init draws from a
        // fresh RNG per model, so re-creating it would change the starting
        // point. Instead we restore the LoRA vars between variants — the
        // base weights are frozen anyway.
        let (mut model, _base, lora) = toy_model(&dev)?;
        let initial: Vec<(String, Tensor)> = {
            let data = lora.data().lock().unwrap();
            data.iter()
                .map(|(name, var)| Ok((name.clone(), var.as_tensor().copy()?)))
                .collect::<Result<Vec<_>>>()?
        };
        let restore = || -> Result<()> {
            let data = lora.data().lock().unwrap();
            for (name, t) in &initial {
                data[name].set(t)?;
            }
            Ok(())
        };

        let mut updated = Vec::new();
        for mb in [0usize, 1, 2] {
            restore()?;
            let before = lora_snapshot(&lora);
            let mut opt = adamw(&lora)?;
            let stats = train_qwen_lora_pg_step_cfg(
                &mut model,
                &mut opt,
                &dev,
                &samples,
                &lora.all_vars(),
                PgStepConfig {
                    micro_batch_size: mb,
                    accumulate_grads: true,
                    skip_zero_advantage: true,
                    positive_advantage_only: false,
                },
            )?;
            assert_eq!(stats.n_updates, 1, "mb={mb}: one PG step = one update");
            assert_eq!(stats.n_used, 4, "mb={mb}: all 4 samples have advantage");
            let after = lora_snapshot(&lora);
            assert!(
                before.iter().zip(&after).any(|(b, a)| (b - a).abs() > 1e-9),
                "mb={mb}: the update must actually move the LoRA weights"
            );
            updated.push(after);
        }
        for (i, w) in updated.iter().enumerate().skip(1) {
            let max_diff = updated[0]
                .iter()
                .zip(w)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < 1e-6,
                "micro_batch variant {i} diverged from the un-chunked update: max_diff={max_diff}"
            );
        }
        Ok(())
    }

    /// Without accumulation (the Stage E path) the same batch issues one
    /// AdamW update *per chunk* — 4 updates for 4 samples at mb=1. This is
    /// the behaviour that applied ~1024 updates before the first adapter
    /// sync on the 7B hard tail.
    #[test]
    fn pg_legacy_path_issues_one_update_per_micro_batch() -> Result<()> {
        let dev = Device::Cpu;
        let (mut model, _base, lora) = toy_model(&dev)?;
        let mut opt = adamw(&lora)?;
        let stats = train_qwen_lora_pg_step_cfg(
            &mut model,
            &mut opt,
            &dev,
            &toy_samples(),
            &lora.all_vars(),
            PgStepConfig {
                micro_batch_size: 1,
                accumulate_grads: false,
                skip_zero_advantage: false,
                positive_advantage_only: false,
            },
        )?;
        assert_eq!(stats.n_updates, 4);
        assert_eq!(stats.n_used, 4);
        Ok(())
    }

    /// RLOO gives advantage exactly 0 to every prompt whose k completions
    /// share a verdict — ~94% of the 7B hard-tail batch. They must be
    /// dropped before they cost a forward pass.
    #[test]
    fn pg_skips_zero_advantage_samples() -> Result<()> {
        let dev = Device::Cpu;
        let mut samples = toy_samples();
        samples.push((vec![3, 4, 5], vec![12, 13], 0.0));
        samples.push((vec![3, 4, 5], vec![14, 15], 0.0));

        let (mut model, _base, lora) = toy_model(&dev)?;
        let mut opt = adamw(&lora)?;
        let stats = train_qwen_lora_pg_step_cfg(
            &mut model,
            &mut opt,
            &dev,
            &samples,
            &lora.all_vars(),
            PgStepConfig::default(),
        )?;
        assert_eq!(stats.n_used, 4, "the 4 non-zero-advantage samples");
        assert_eq!(stats.n_skipped, 2, "the 2 zero-advantage samples");
        Ok(())
    }

    /// Phase 22 C4 — positive-advantage-only keeps just the verifier-passing
    /// completions, so the loss is `reward * CE >= 0` (bounded below) rather
    /// than unbounded ascent on the ~75% of RLOO samples with negative
    /// advantage. Zero-advantage samples go too: zero is not positive.
    #[test]
    fn pg_positive_only_drops_negative_and_zero_advantage() -> Result<()> {
        let dev = Device::Cpu;
        let mut samples = toy_samples(); // one +0.75, three -0.25
        samples.push((vec![3, 4, 5], vec![12, 13], 0.0));

        let (mut model, _base, lora) = toy_model(&dev)?;
        let mut opt = adamw(&lora)?;
        let stats = train_qwen_lora_pg_step_cfg(
            &mut model,
            &mut opt,
            &dev,
            &samples,
            &lora.all_vars(),
            PgStepConfig {
                positive_advantage_only: true,
                ..PgStepConfig::default()
            },
        )?;
        assert_eq!(stats.n_used, 1, "only the +0.75 sample survives");
        assert_eq!(stats.n_skipped, 4, "3 negative + 1 zero");
        assert_eq!(stats.n_updates, 1);
        Ok(())
    }

    /// A step where nothing passed the verifier yields no positive-advantage
    /// samples at all — a no-op, not an error that kills the RL run.
    #[test]
    fn pg_positive_only_with_no_passes_is_a_noop() -> Result<()> {
        let dev = Device::Cpu;
        let samples = vec![
            (vec![1, 2, 3], vec![4, 5], -0.25),
            (vec![1, 2, 3], vec![6, 7], -0.25),
        ];
        let (mut model, _base, lora) = toy_model(&dev)?;
        let before = lora_snapshot(&lora);
        let mut opt = adamw(&lora)?;
        let stats = train_qwen_lora_pg_step_cfg(
            &mut model,
            &mut opt,
            &dev,
            &samples,
            &lora.all_vars(),
            PgStepConfig {
                positive_advantage_only: true,
                ..PgStepConfig::default()
            },
        )?;
        assert_eq!(stats.n_updates, 0);
        assert_eq!(stats.n_skipped, 2);
        assert_eq!(before, lora_snapshot(&lora));
        Ok(())
    }

    /// A sparse-reward RL step where no prompt had a mixed verdict is an
    /// ordinary outcome, not an error: report a no-op instead of killing
    /// the run.
    #[test]
    fn pg_all_zero_advantage_is_a_noop_not_an_error() -> Result<()> {
        let dev = Device::Cpu;
        let samples = vec![
            (vec![1, 2, 3], vec![4, 5], 0.0),
            (vec![1, 2, 3], vec![6, 7], 0.0),
        ];
        let (mut model, _base, lora) = toy_model(&dev)?;
        let before = lora_snapshot(&lora);
        let mut opt = adamw(&lora)?;
        let stats = train_qwen_lora_pg_step_cfg(
            &mut model,
            &mut opt,
            &dev,
            &samples,
            &lora.all_vars(),
            PgStepConfig::default(),
        )?;
        assert_eq!(stats.n_updates, 0);
        assert_eq!(stats.n_used, 0);
        assert_eq!(stats.n_skipped, 2);
        assert_eq!(
            before,
            lora_snapshot(&lora),
            "a no-signal step must leave the weights untouched"
        );
        Ok(())
    }
}
