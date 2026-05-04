use serde::{Deserialize, Serialize};

/// MLP activation. `Gelu` is the classic dense GPT-2 path. `SwiGlu` and
/// `GeGlu` are gated variants used in Llama, Mistral, PaLM — they add a
/// third Linear (the gate) so MLP has ~1.5× the params of dense at the
/// same `ffn_mult`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationKind {
    Gelu,
    SwiGlu,
    GeGlu,
}

impl ActivationKind {
    pub fn is_gated(&self) -> bool {
        !matches!(self, ActivationKind::Gelu)
    }
}

impl Default for ActivationKind {
    fn default() -> Self {
        ActivationKind::Gelu
    }
}

/// Normalization layer kind. `LayerNorm` is the GPT-2 default; `RmsNorm`
/// (used by Llama / Mistral) skips the mean centering — typically a small
/// quality + speed win.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormKind {
    LayerNorm,
    RmsNorm,
}

impl Default for NormKind {
    fn default() -> Self {
        NormKind::LayerNorm
    }
}

/// Where the per-block norm is applied. Pre-norm (the GPT-2 / nanoGPT and
/// Llama default) puts norm before each sublayer and adds the unnormalized
/// residual. Post-norm puts norm after the residual sum (original
/// Transformer paper) — usually less stable but occasionally wins at
/// quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormPosition {
    Pre,
    Post,
}

impl Default for NormPosition {
    fn default() -> Self {
        NormPosition::Pre
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPTConfig {
    pub vocab_size: usize,
    pub block_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    pub dropout: f32,
    pub bias: bool,
    /// MLP intermediate width = `ffn_mult * n_embd`. Standard GPT uses 4;
    /// search space typically explores {2, 4, 6, 8}.
    #[serde(default = "default_ffn_mult")]
    pub ffn_mult: usize,
    /// If true, replace learned positional embedding with rotary applied to
    /// Q/K. `rope_base` controls the frequency base (default 10000).
    #[serde(default = "default_use_rope")]
    pub use_rope: bool,
    #[serde(default = "default_rope_base")]
    pub rope_base: f32,
    /// Number of K/V heads. `n_kv_head == n_head` is standard MHA;
    /// `n_kv_head < n_head` is GQA (must divide `n_head`); `1` is MQA.
    /// Defaults to `n_head` via the `Default` impl below.
    #[serde(default)]
    pub n_kv_head: usize,
    /// Number of MLP experts per block. `1` (default) is the standard dense
    /// FFN; `>1` enables Mixture-of-Experts.
    #[serde(default = "default_n_experts")]
    pub n_experts: usize,
    /// Top-k experts kept per token in MoE routing. `0` = soft (all experts
    /// weighted by softmax); `1..=n_experts` = top-k masked + renormalized.
    /// Note: compute cost is unchanged — every expert still runs; only the
    /// routing weights are sparsified.
    #[serde(default = "default_moe_top_k")]
    pub moe_top_k: usize,
    /// Weight on the load-balance auxiliary loss. Switch-Transformer style:
    /// `α · E · sum_i P_i²` where `P_i` is the mean router probability for
    /// expert i over the batch. Pushes routing toward uniform.
    #[serde(default = "default_moe_aux_weight")]
    pub moe_aux_weight: f32,
    /// MLP activation kind. `Gelu` (default) uses the standard dense MLP;
    /// `SwiGlu`/`GeGlu` add a parallel gate Linear (params ×1.5).
    #[serde(default)]
    pub activation: ActivationKind,
    /// If true (default), the LM head reuses `wte`'s weights (`logits = x @
    /// wte.T`). If false, a separate `lm_head: Linear(n_embd, vocab_size)`
    /// is allocated — more params, sometimes better quality.
    #[serde(default = "default_weight_tying")]
    pub weight_tying: bool,
    /// Which normalization layer to use inside each Block.
    #[serde(default)]
    pub norm_kind: NormKind,
    /// Pre-norm vs post-norm placement.
    #[serde(default)]
    pub norm_position: NormPosition,
    /// LoRA rank for the attention `c_attn` adapter. `0` = no LoRA. Adds
    /// `lora_a: (n_embd, rank)` + `lora_b: (rank, qkv_out)` per block.
    /// `lora_b` is initialized to zero so the model is identical to the
    /// no-LoRA case at the start of training.
    #[serde(default = "default_lora_rank")]
    pub lora_rank: usize,
    /// LoRA scaling: `effective = lora_alpha / lora_rank`. Standard
    /// recommendation is `lora_alpha = 2 × lora_rank`.
    #[serde(default = "default_lora_alpha")]
    pub lora_alpha: f32,
}

fn default_ffn_mult() -> usize { 4 }
fn default_use_rope() -> bool { false }
fn default_rope_base() -> f32 { 10_000.0 }
fn default_n_experts() -> usize { 1 }
fn default_moe_top_k() -> usize { 0 }
fn default_moe_aux_weight() -> f32 { 0.01 }
fn default_weight_tying() -> bool { true }
fn default_lora_rank() -> usize { 0 }
fn default_lora_alpha() -> f32 { 16.0 }

impl GPTConfig {
    /// ~50M-param config tuned to the Phase 3 NAS-discovered Llama recipe
    /// (RoPE + 4× GQA + SwiGLU + RmsNorm-Pre + untied head). Override
    /// `vocab_size` to match your tokenizer.
    pub fn nano_50m() -> Self {
        Self {
            vocab_size: 32_000,
            block_size: 512,
            n_layer: 8,
            n_head: 8,
            n_embd: 512,
            dropout: 0.0,
            bias: false,
            ffn_mult: 4,
            use_rope: true,
            rope_base: 10_000.0,
            n_experts: 1,
            moe_top_k: 0,
            moe_aux_weight: 0.0,
            activation: ActivationKind::SwiGlu,
            weight_tying: false,
            norm_kind: NormKind::RmsNorm,
            norm_position: NormPosition::Pre,
            lora_rank: 0,
            lora_alpha: 16.0,
            n_kv_head: 2, // 4× GQA compression
        }
    }

    pub fn nano_125m() -> Self {
        Self {
            vocab_size: 50_257,
            block_size: 1024,
            n_layer: 12,
            n_head: 12,
            n_embd: 768,
            dropout: 0.0,
            bias: false,
            ffn_mult: 4,
            use_rope: false,
            rope_base: 10_000.0,
            n_experts: 1,
            moe_top_k: 0,
            moe_aux_weight: 0.01,
            activation: ActivationKind::Gelu,
            weight_tying: true,
            norm_kind: NormKind::LayerNorm,
            norm_position: NormPosition::Pre,
            lora_rank: 0,
            lora_alpha: 16.0,
            n_kv_head: 12,
        }
    }

    pub fn nano_300m() -> Self {
        Self {
            vocab_size: 50_257,
            block_size: 1024,
            n_layer: 24,
            n_head: 16,
            n_embd: 1024,
            dropout: 0.0,
            bias: false,
            ffn_mult: 4,
            use_rope: false,
            rope_base: 10_000.0,
            n_experts: 1,
            moe_top_k: 0,
            moe_aux_weight: 0.01,
            activation: ActivationKind::Gelu,
            weight_tying: true,
            norm_kind: NormKind::LayerNorm,
            norm_position: NormPosition::Pre,
            lora_rank: 0,
            lora_alpha: 16.0,
            n_kv_head: 16,
        }
    }

    pub fn shakespeare_char(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            block_size: 256,
            n_layer: 6,
            n_head: 6,
            n_embd: 384,
            dropout: 0.2,
            bias: false,
            ffn_mult: 4,
            use_rope: false,
            rope_base: 10_000.0,
            n_experts: 1,
            moe_top_k: 0,
            moe_aux_weight: 0.01,
            activation: ActivationKind::Gelu,
            weight_tying: true,
            norm_kind: NormKind::LayerNorm,
            norm_position: NormPosition::Pre,
            lora_rank: 0,
            lora_alpha: 16.0,
            n_kv_head: 6,
        }
    }

    /// `n_kv_head` defaults to `n_head` after deserialization if zero.
    pub fn normalized(mut self) -> Self {
        if self.n_kv_head == 0 {
            self.n_kv_head = self.n_head;
        }
        self
    }

    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }

    pub fn num_params_estimate(&self) -> usize {
        let e = self.n_embd;
        let v = self.vocab_size;
        let l = self.n_layer;
        let block_size = self.block_size;
        let f = self.ffn_mult;
        let nq = self.n_head as f64;
        let nkv = if self.n_kv_head == 0 { nq } else { self.n_kv_head as f64 };
        let n_exp = self.n_experts.max(1);
        // embeddings: token (always) + learned position (only if not RoPE)
        let emb = v * e + if self.use_rope { 0 } else { block_size * e };
        // per block:
        //   c_attn: e * (1 + 2*kv_ratio) * e
        //   c_proj: e * e
        //   mlp per expert: dense = 2 * f * e * e; gated = 3 * f * e * e
        //   moe router (only if n_exp > 1): e * n_exp
        //   layer-norms ≈ 4*e
        let kv_ratio = nkv / nq;
        let attn = (e as f64 * e as f64 * (1.0 + 2.0 * kv_ratio)) as usize;
        let mlp_per_expert = if self.activation.is_gated() {
            3 * f * e * e
        } else {
            2 * f * e * e
        };
        let mlp = n_exp * mlp_per_expert;
        let router = if n_exp > 1 { e * n_exp } else { 0 };
        // 2 norms per block. LayerNorm: weight + bias (2e each). RmsNorm:
        // weight only (e each).
        let norm_per_layer = match self.norm_kind {
            NormKind::LayerNorm => 4 * e,
            NormKind::RmsNorm => 2 * e,
        };
        // LoRA adapters (when active) on every Linear we apply them to:
        //   c_attn:  e × r  +  qkv_out × r
        //   c_proj:  e × r  +  e × r
        //   c_fc:    e × r  +  (f*e) × r
        //   c_proj (mlp): (f*e) × r  +  e × r
        let lora_per_block = if self.lora_rank > 0 {
            let r = self.lora_rank;
            let qkv_out = (e as f64 * (1.0 + 2.0 * kv_ratio)) as usize;
            // sum of all (in × r + r × out) pairs for the 4 LoRA-augmented sites
            let attn_lora = r * e + r * qkv_out;
            let attn_proj_lora = r * e + r * e;
            let mlp_fc_lora = r * e + r * f * e;
            let mlp_proj_lora = r * f * e + r * e;
            attn_lora + attn_proj_lora + mlp_fc_lora + mlp_proj_lora
        } else {
            0
        };
        let per_block = attn + e * e + mlp + router + norm_per_layer + lora_per_block;
        // tied head shares wte; untied adds a Linear(n_embd, vocab).
        let head = if self.weight_tying { e } else { e * v + e };
        let head_lora = if self.lora_rank > 0 && !self.weight_tying {
            self.lora_rank * e + self.lora_rank * v
        } else {
            0
        };
        emb + l * per_block + head + head_lora
    }
}
