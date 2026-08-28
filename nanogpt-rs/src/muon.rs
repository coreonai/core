//! Phase 12 S1 — Muon optimizer (gradient orthogonalization).
//!
//! The Muon optimizer (Jordan et al. 2024, used in DeepSeek V4 2026)
//! is a variant of SGD with momentum where the *momentum tensor* is
//! orthogonalized via Newton-Schulz iterations before the parameter
//! update. The orthogonalization removes correlations between
//! gradient directions, giving faster convergence and better
//! stability than AdamW at large scale (DeepSeek used Muon for the
//! majority of modules in V4-Pro at 1.6T params).
//!
//! ## Per-parameter dispatch
//!
//! Muon's orthogonalization step only makes sense for 2-D weight
//! matrices. 1-D parameters (LayerNorm scales, biases, embedding
//! row vectors) fall back to standard SGD-with-momentum:
//!
//!   2-D `W ∈ ℝ^{m×n}`: g → momentum → NS-orthogonalize → step
//!   1-D `b ∈ ℝ^n`:     g → momentum → step (no NS)
//!
//! Following DeepSeek's hybrid pattern: Muon is the *primary*
//! optimizer; AdamW handles the small minority of 1-D params. We
//! simplify here by using SGD-momentum for those, which suffices
//! for our 50M-class models.
//!
//! ## Newton-Schulz iteration
//!
//! For an input matrix `X ∈ ℝ^{m×n}`, NS computes an approximation of
//! the orthogonal polar factor `Q` of `X` (i.e. `X = Q · S` with
//! `Q^T Q = I`). The iteration:
//!
//!   X_{k+1} = a · X_k + b · (X_k X_k^T) X_k + c · (X_k X_k^T)^2 X_k
//!
//! With DeepSeek V4's coefficient schedule:
//!
//!   stage 1 (4 iters): a, b, c = 3.4445, -4.7750, 2.0315 (fast convergence)
//!   stage 2 (1 iter):  a, b, c = 2.0,    -1.5,    0.5     (stabilize σ ≈ 1)
//!
//! Total 5 iterations. Pre-normalize `X` by spectral norm before NS
//! so all singular values start in `[0, 1]`, where the iteration is
//! well-behaved.
//!
//! ## Use from training
//!
//! ```rust,ignore
//! let opt = Muon::new(varmap.all_vars(), MuonConfig::default())?;
//! opt.backward_step(&loss)?;
//! ```
//!
//! Drop-in replacement for `AdamW::new(...)`. Same `Optimizer`
//! trait, same `set_learning_rate` / `backward_step` API.

use candle_core::{Result as CResult, Tensor, Var};
use candle_nn::Optimizer;

#[derive(Clone, Debug)]
pub struct MuonConfig {
    pub lr: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    /// Number of Newton-Schulz iterations. DeepSeek V4 uses 5 (4
    /// fast-stage + 1 stabilize-stage).
    pub ns_steps: usize,
    /// Adam-like fallback hyperparameters for 1-D parameters where NS
    /// doesn't apply. We use the same `lr` and `weight_decay` as the
    /// 2-D path; these are momentum/eps for the SGD-mom fallback.
    pub fallback_momentum: f64,
}

impl Default for MuonConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            momentum: 0.95,
            weight_decay: 0.01,
            ns_steps: 5,
            fallback_momentum: 0.9,
        }
    }
}

#[derive(Debug)]
struct VarMuon {
    var: Var,
    momentum_buf: Var,
    is_matrix: bool,
}

#[derive(Debug)]
pub struct Muon {
    vars: Vec<VarMuon>,
    cfg: MuonConfig,
    step_t: usize,
}

impl Optimizer for Muon {
    type Config = MuonConfig;

    fn new(vars: Vec<Var>, cfg: MuonConfig) -> CResult<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| {
                let shape = var.shape();
                let dtype = var.dtype();
                let device = var.device();
                let momentum_buf = Var::zeros(shape, dtype, device)?;
                let is_matrix = var.dims().len() == 2;
                Ok(VarMuon {
                    var,
                    momentum_buf,
                    is_matrix,
                })
            })
            .collect::<CResult<Vec<_>>>()?;
        Ok(Self {
            vars,
            cfg,
            step_t: 0,
        })
    }

    fn learning_rate(&self) -> f64 {
        self.cfg.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.cfg.lr = lr;
    }

    fn step(&mut self, grads: &candle_core::backprop::GradStore) -> CResult<()> {
        self.step_t += 1;
        let lr = self.cfg.lr;
        let wd = self.cfg.weight_decay;
        for v in &self.vars {
            let Some(g) = grads.get(&v.var) else {
                continue;
            };
            // momentum update: m ← β·m + g
            let beta = if v.is_matrix {
                self.cfg.momentum
            } else {
                self.cfg.fallback_momentum
            };
            let new_m = ((v.momentum_buf.as_tensor() * beta)? + g)?;
            v.momentum_buf.set(&new_m)?;

            // For 2-D matrices: orthogonalize the momentum (this is the
            // Muon step). For 1-D: skip NS, use raw momentum (SGD-mom).
            let direction = if v.is_matrix {
                newton_schulz(&new_m, self.cfg.ns_steps)?
            } else {
                new_m.clone()
            };

            // weight decay (decoupled, AdamW-style):
            //   θ ← θ · (1 − lr·wd) − lr · direction
            let theta = v.var.as_tensor();
            let decayed = (theta * (1.0 - lr * wd))?;
            let new_theta = (decayed - (direction * lr)?)?;
            v.var.set(&new_theta)?;
        }
        Ok(())
    }
}

/// Newton-Schulz polar-factor iteration. Approximates the orthogonal
/// matrix `Q` such that `X = Q·S` (polar decomposition). Returns the
/// orthogonalized version of `x` (same shape).
///
/// Pre-normalizes by spectral norm so every singular value starts in
/// `[0, 1]`. With DeepSeek V4's coefficient schedule (4 fast steps +
/// 1 stabilize), 5 total iterations bring singular values to ≈ 1.
///
/// Numerical safety: clamps the spectral-norm pre-divisor to ≥ 1e-7.
pub fn newton_schulz(x: &Tensor, n_steps: usize) -> CResult<Tensor> {
    let dims = x.dims();
    if dims.len() != 2 {
        candle_core::bail!(
            "newton_schulz expects a 2-D matrix, got rank {}",
            dims.len()
        );
    }
    // Pre-normalize: scale so largest singular value ≤ 1. Frobenius
    // norm is an upper bound on σ_max, so dividing by it is safe.
    let frob = x.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()? as f64;
    let denom = frob.max(1e-7);
    let mut x = (x / denom)?;

    // (m, n) → operate in the smaller side. If m < n we'd prefer to
    // work with X^T so the inner dimension is small. For our 50M
    // model the matrices are square-ish; keep simple form.
    let stage1 = (3.4445_f64, -4.7750, 2.0315);
    let stage2 = (2.0_f64, -1.5, 0.5);
    let n_stage1 = n_steps.saturating_sub(1);
    for i in 0..n_steps {
        let (a, b, c) = if i < n_stage1 { stage1 } else { stage2 };
        // X_{k+1} = a·X + b·X X^T X + c·(X X^T)² X
        let xxt = x.matmul(&x.t()?)?;
        let xxt_x = xxt.matmul(&x)?;
        let xxt2_x = xxt.matmul(&xxt_x)?;
        x = ((x * a)? + (xxt_x * b)? + (xxt2_x * c)?)?;
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{Module, VarBuilder, VarMap};

    #[test]
    fn newton_schulz_identity_returns_near_identity() {
        // NS on the identity matrix should keep it the identity (its
        // own polar factor). Small numerical drift is acceptable.
        let device = Device::Cpu;
        let id = Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 1.0], (2, 2), &device).unwrap();
        let out = newton_schulz(&id, 5).unwrap();
        let out_v = out.to_vec2::<f32>().unwrap();
        // Off-diagonals near 0, diagonals near 1.
        assert!((out_v[0][0] - 1.0).abs() < 0.05);
        assert!(out_v[0][1].abs() < 0.05);
        assert!(out_v[1][0].abs() < 0.05);
        assert!((out_v[1][1] - 1.0).abs() < 0.05);
    }

    #[test]
    fn newton_schulz_gives_orthogonal_columns() {
        // After NS, X^T X should be near identity (columns orthonormal).
        //
        // Fixed matrix, not `randn`. Unseeded this test failed about 1 run in
        // 6: a random 4x4 Gaussian is occasionally near-singular, and 5
        // Newton-Schulz iterations do not bring such a matrix inside the 0.1
        // tolerance. That made it assert "NS orthogonalizes ANY random matrix
        // in 5 steps", which is false and not what it is here to check — and
        // CI runs `cargo test --workspace` as a strict gate, so it was a ~17%
        // chance of a red build for no reason. `Device::Cpu` rejects
        // `set_seed` ("cannot seed the CPU rng"), so the fix is a literal
        // well-conditioned matrix rather than a seeded draw.
        let device = Device::Cpu;
        let x = Tensor::from_vec(
            vec![
                1.2_f32, -0.4, 0.3, 0.1, //
                0.5, 1.4, -0.2, 0.6, //
                -0.3, 0.2, 1.1, -0.5, //
                0.4, -0.6, 0.7, 1.3,
            ],
            (4, 4),
            &device,
        )
        .unwrap();
        let q = newton_schulz(&x, 5).unwrap();
        // Q^T Q ≈ I_4
        let qtq = q.t().unwrap().matmul(&q).unwrap();
        let qtq_v = qtq.to_vec2::<f32>().unwrap();
        for (i, row) in qtq_v.iter().enumerate().take(4) {
            for (j, val) in row.iter().enumerate().take(4) {
                let target = if i == j { 1.0_f32 } else { 0.0 };
                assert!(
                    (val - target).abs() < 0.1,
                    "Q^T Q[{i}][{j}]={val} expected {target}"
                );
            }
        }
    }

    #[test]
    fn newton_schulz_rejects_non_2d() {
        let device = Device::Cpu;
        let v = Tensor::randn(0_f32, 1.0, 5, &device).unwrap();
        assert!(newton_schulz(&v, 5).is_err());
        let cube = Tensor::randn(0_f32, 1.0, (2, 3, 4), &device).unwrap();
        assert!(newton_schulz(&cube, 5).is_err());
    }

    #[test]
    fn muon_step_decreases_quadratic_loss() {
        // Train W ∈ ℝ^{4×4} to minimize ||W − target||² via Muon.
        // After a few steps, the loss should decrease.
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let lin = candle_nn::linear_no_bias(4, 4, vb).unwrap();

        let target = Tensor::from_vec(
            (0..16).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            (4, 4),
            &device,
        )
        .unwrap();

        let mut opt = Muon::new(
            varmap.all_vars(),
            MuonConfig {
                lr: 1e-1,
                ..MuonConfig::default()
            },
        )
        .unwrap();

        // Use the linear layer to drive a loss against the target.
        let x = Tensor::randn(0_f32, 1.0, (8, 4), &device).unwrap();
        let target_y = x.matmul(&target).unwrap();
        let loss0 = (lin.forward(&x).unwrap() - &target_y)
            .unwrap()
            .sqr()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        for _ in 0..30 {
            let pred = lin.forward(&x).unwrap();
            let loss = (pred - &target_y)
                .unwrap()
                .sqr()
                .unwrap()
                .mean_all()
                .unwrap();
            opt.backward_step(&loss).unwrap();
        }
        let loss1 = (lin.forward(&x).unwrap() - &target_y)
            .unwrap()
            .sqr()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            loss1 < loss0 * 0.5,
            "Muon should at least halve loss; loss0={loss0} loss1={loss1}"
        );
    }

    #[test]
    fn muon_set_learning_rate_round_trips() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let _vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut opt = Muon::new(varmap.all_vars(), MuonConfig::default()).unwrap();
        opt.set_learning_rate(5e-4);
        assert!((opt.learning_rate() - 5e-4).abs() < 1e-12);
    }

    #[test]
    fn muon_handles_1d_parameters_via_sgd_momentum_fallback() {
        // 1-D bias vector: NS should NOT be invoked (bail otherwise).
        // Loss = mean((b − target)²). Muon should still drive b → target.
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let target = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], 4, &device).unwrap();
        let b = candle_core::Var::zeros(4, DType::F32, &device).unwrap();
        // Manually register in varmap so Muon picks it up.
        // Easiest: build a varmap with a custom var and pass via from_slice.
        let mut opt = Muon::from_slice(
            &[&b],
            MuonConfig {
                lr: 5e-2,
                ..MuonConfig::default()
            },
        )
        .unwrap();
        let _ = varmap;
        for _ in 0..40 {
            let loss = (b.as_tensor() - &target)
                .unwrap()
                .sqr()
                .unwrap()
                .mean_all()
                .unwrap();
            opt.backward_step(&loss).unwrap();
        }
        let final_v = b.as_tensor().to_vec1::<f32>().unwrap();
        for (got, want) in final_v.iter().zip([1.0_f32, 2.0, 3.0, 4.0].iter()) {
            assert!(
                (got - want).abs() < 0.5,
                "1-D bias should converge toward target; got {final_v:?}"
            );
        }
    }
}
