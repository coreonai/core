//! Phase 10 S1 — JEPA-style auxiliary objective for next-token training.
//!
//! Standard LLM pretraining minimizes token-level cross-entropy. Recent
//! "LLM-JEPA" lines (Joint-Embedding Predictive Architecture, à la
//! LeCun) add a *latent-space* prediction loss alongside CE: from the
//! hidden state at position `i`, predict the hidden state at position
//! `i + k` (with stop-gradient on the target). This is meant to push
//! the model to develop semantic representations that aren't purely
//! tied to token-level frequencies.
//!
//! In our project, this is interesting because Phase 9 S2 found that
//! K8 Korean training over-fits onto the most-frequent token (`\n`)
//! when scaled to 100K steps, *worsening* sum-AUC for Shape-C. JEPA's
//! latent objective is one candidate fix: a model that has to predict
//! future hidden states cannot collapse onto a single high-frequency
//! token.
//!
//! ## Setup (single-encoder, stop-gradient style)
//!
//! Given a batch of hidden states `H ∈ [B, T, d]` (post-`ln_f` output
//! of `GPT::forward_with_hidden`), and an offset `k ≥ 1`:
//!
//! 1. Context: `H[:, :T-k, :]`
//! 2. Target:  `H[:, k:, :].detach()` (stop-gradient — gradients
//!    flow through the context branch only)
//! 3. Predict: `predictor(context)` — a small MLP, `d → d`
//! 4. Loss:    mean squared error between predictor output and target.
//!
//! Conceptually similar to BYOL/SimSiam but for sequences: the encoder
//! is shared, and stop-gradient on the target is what prevents
//! representation collapse.
//!
//! ## Why a 2-layer MLP, not just a Linear
//!
//! A linear predictor would let the model trivially solve JEPA by
//! learning an identity-ish transform; nonlinearity forces a real
//! representation transformation. Following I-JEPA / V-JEPA, we use
//! `Linear → GELU → Linear` with `predictor_hidden = 2 * n_embd`.

use candle_core::{Module, Result as CResult, Tensor};
use candle_nn::{linear, ops, Linear, VarBuilder};

/// Two-layer MLP predictor `d → 2d → d` applied position-wise to
/// hidden states `[B, T, d]`.
pub struct JepaPredictor {
    fc1: Linear,
    fc2: Linear,
}

impl JepaPredictor {
    pub fn new(n_embd: usize, vb: VarBuilder) -> CResult<Self> {
        let hidden = 2 * n_embd;
        let fc1 = linear(n_embd, hidden, vb.pp("fc1"))?;
        let fc2 = linear(hidden, n_embd, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    /// `[B, T, d] → [B, T, d]`. GELU between the two Linear layers.
    pub fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let h = self.fc1.forward(x)?;
        let h = h.gelu()?;
        self.fc2.forward(&h)
    }
}

/// JEPA loss given full-sequence hidden states `[B, T, d]` and offset
/// `k`. Returns the per-element mean squared error between
/// `predictor(H[:, :T-k, :])` and `H[:, k:, :].detach()`.
///
/// Errors if `T <= k` (no valid positions to compare).
pub fn jepa_loss(predictor: &JepaPredictor, hidden: &Tensor, offset: usize) -> CResult<Tensor> {
    let (_b, t, _d) = hidden.dims3()?;
    if t <= offset {
        candle_core::bail!("JEPA offset {offset} must be < sequence length {t}");
    }
    let context = hidden.narrow(1, 0, t - offset)?;
    let target = hidden.narrow(1, offset, t - offset)?.detach();
    let predicted = predictor.forward(&context)?;
    let diff = (predicted - target)?;
    let sq = diff.sqr()?;
    sq.mean_all()
}

/// Mode-collapse metric: returns the mass that the **single most
/// confident** token claims, averaged over `(B, T)` positions.
///
/// Computed over the full vocab softmax of `logits ∈ [B, T, V]`. A
/// healthy model has top-1 mass roughly inversely proportional to
/// vocab; a mode-collapsed model has top-1 mass close to 1.0
/// (e.g. KoWiki K8 100K loved emitting `\n` — top-1 mass ~0.5+).
///
/// This is a *measurement*, not a loss; intended for periodic eval.
pub fn top1_mass(logits: &Tensor) -> CResult<f32> {
    let probs = ops::softmax_last_dim(logits)?;
    let max_per_pos = probs.max(candle_core::D::Minus1)?;
    max_per_pos.mean_all()?.to_scalar::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn predictor_round_trips_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let pred = JepaPredictor::new(8, vb).expect("build");
        let x = Tensor::randn(0f32, 1.0, (2, 5, 8), &device).expect("rand");
        let y = pred.forward(&x).expect("fwd");
        assert_eq!(y.dims(), &[2, 5, 8]);
    }

    #[test]
    fn jepa_loss_runs_and_scales_with_offset_validity() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let pred = JepaPredictor::new(4, vb).expect("build");
        let hidden = Tensor::randn(0f32, 1.0, (2, 6, 4), &device).expect("rand");
        let l = jepa_loss(&pred, &hidden, 2).expect("loss");
        let v = l.to_scalar::<f32>().expect("scalar");
        assert!(v.is_finite() && v >= 0.0);
        // offset >= seq_len must error
        assert!(jepa_loss(&pred, &hidden, 6).is_err());
    }

    #[test]
    fn top1_mass_uniform_logits_match_inverse_vocab() {
        let device = Device::Cpu;
        // Uniform logits = uniform softmax = 1/V mass on every token.
        let logits = Tensor::zeros((1, 3, 5), DType::F32, &device).expect("zeros");
        let m = top1_mass(&logits).expect("top1");
        assert!((m - 0.2).abs() < 1e-5, "expected ~0.2, got {m}");
    }

    #[test]
    fn top1_mass_concentrated_distribution_close_to_one() {
        let device = Device::Cpu;
        // Make one logit huge → softmax pushes top-1 mass → 1.
        let mut data = vec![0.0f32; 5];
        data[2] = 50.0;
        let logits = Tensor::from_slice(&data, (1, 1, 5), &device).expect("from");
        let m = top1_mass(&logits).expect("top1");
        assert!(m > 0.99, "expected ~1.0, got {m}");
    }
}
