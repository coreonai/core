//! Catastrophic-forgetting prevention via weight anchoring.
//!
//! Implements a simplified Elastic Weight Consolidation: rather than
//! computing the per-parameter Fisher diagonal, we use **uniform Fisher = 1**
//! and penalize squared deviation from a snapshot of the pretrained
//! weights. This is the L2-toward-θ* regularizer:
//!
//!   `L = L_task + (λ/2) Σ_i (θ_i − θ_i*)²`
//!
//! It captures the spirit of EWC at far lower cost; in practice, on small
//! continual-learning tasks, it tends to recover a sizable fraction of the
//! benefit. A future upgrade can add per-Var Fisher weights computed from
//! the pretrain dataset.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{Device, Result as CResult, Tensor};
use candle_nn::VarMap;

use crate::config::GPTConfig;
use crate::data::TokenDataset;
use crate::model::GPT;

/// A snapshot of a `VarMap`'s tensors plus an EWC strength `lambda`.
/// Cheaply cloneable (the inner `Tensor`s are Arc-shared but materialized
/// from the source so optimizer updates after `snapshot()` don't mutate
/// the captured values).
///
/// When `fisher` is `Some`, the penalty uses the proper EWC form
/// `(λ/2) Σ_i F_i (θ_i − θ_i*)²`. When `None`, uniform Fisher = 1 is
/// used (the simpler L2-toward-θ* regularizer).
#[derive(Clone)]
pub struct WeightAnchor {
    /// Snapshot of `var.as_tensor()` per Var name at snapshot time.
    pub reference: Arc<HashMap<String, Tensor>>,
    /// Optional per-Var diagonal Fisher estimate. Same shapes as the
    /// corresponding reference tensors.
    pub fisher: Option<Arc<HashMap<String, Tensor>>>,
    /// Quadratic penalty strength.
    pub lambda: f64,
}

impl WeightAnchor {
    /// Take a deep snapshot of every Var in the map. We use `Tensor::copy`
    /// to force fresh storage — `Tensor::clone` would only share the Arc
    /// and would be invalidated by subsequent `AdamW` updates.
    pub fn snapshot(varmap: &VarMap, lambda: f64) -> CResult<Self> {
        let data = varmap.data().lock().expect("varmap mutex poisoned");
        let mut reference = HashMap::with_capacity(data.len());
        for (name, var) in data.iter() {
            let snap = var.as_tensor().copy()?;
            reference.insert(name.clone(), snap);
        }
        Ok(Self {
            reference: Arc::new(reference),
            fisher: None,
            lambda,
        })
    }

    /// Same as `snapshot`, but additionally estimates the per-Var diagonal
    /// Fisher information by running `n_batches` of forward+backward over
    /// `pretrain_ds` and averaging squared gradients. Memory cost: 2× the
    /// VarMap (reference + fisher tensors).
    ///
    /// `n_batches=0` is treated as "uniform Fisher = 1" — equivalent to
    /// the simpler `snapshot()` constructor.
    pub fn snapshot_with_fisher(
        gpt_cfg: &GPTConfig,
        varmap: &VarMap,
        pretrain_ds: &TokenDataset,
        n_batches: usize,
        batch_size: usize,
        device: &Device,
        lambda: f64,
    ) -> crate::error::Result<Self> {
        // 1. Snapshot reference + collect Vars by name (drop guard early
        //    so the model's own forward pass can re-acquire the lock if
        //    Candle ever needs it during backward).
        let vars: Vec<(String, candle_core::Var)> = {
            let data = varmap.data().lock().expect("varmap mutex poisoned");
            data.iter().map(|(n, v)| (n.clone(), v.clone())).collect()
        };
        let mut reference = HashMap::with_capacity(vars.len());
        for (name, var) in &vars {
            reference.insert(name.clone(), var.as_tensor().copy()?);
        }
        let reference = Arc::new(reference);

        if n_batches == 0 {
            return Ok(Self {
                reference,
                fisher: None,
                lambda,
            });
        }

        // 2. Build a model bound to this varmap so backward fills the
        //    correct GradStore.
        let vb = candle_nn::VarBuilder::from_varmap(varmap, candle_core::DType::F32, device);
        let model = GPT::new(gpt_cfg.clone(), vb)?;

        // 3. Init Fisher accumulators (same shape as each Var).
        let mut fisher: HashMap<String, Tensor> = HashMap::with_capacity(vars.len());
        for (name, var) in &vars {
            let z = Tensor::zeros_like(var.as_tensor())?;
            fisher.insert(name.clone(), z);
        }

        // 4. Accumulate Σ (∂L/∂θ_i)² over batches.
        for _ in 0..n_batches {
            let (x, y) = pretrain_ds.random_batch(batch_size, device)?;
            let loss = model.loss(&x, &y)?;
            let grads = loss.backward()?;
            for (name, var) in &vars {
                let Some(g) = grads.get(var.as_tensor()) else {
                    continue;
                };
                let g_sq = g.sqr()?;
                let acc = fisher.remove(name).expect("fisher entry");
                fisher.insert(name.clone(), (acc + g_sq)?);
            }
        }
        // 5. Mean over batches.
        let n = n_batches as f64;
        let mut fisher_mean: HashMap<String, Tensor> = HashMap::with_capacity(fisher.len());
        for (k, v) in fisher {
            fisher_mean.insert(k, (v / n)?);
        }

        Ok(Self {
            reference,
            fisher: Some(Arc::new(fisher_mean)),
            lambda,
        })
    }

    /// Compute `(λ/2) Σ_i (θ_i − θ_i*)²` as a scalar tensor that
    /// participates in autograd through the live Vars in `varmap`.
    pub fn penalty(&self, varmap: &VarMap) -> CResult<Tensor> {
        let data = varmap.data().lock().expect("varmap mutex poisoned");
        let mut accum: Option<Tensor> = None;
        let mut device = Device::Cpu;
        for (name, var) in data.iter() {
            device = var.as_tensor().device().clone();
            let Some(ref_t) = self.reference.get(name) else {
                continue;
            };
            let diff = (var.as_tensor() - ref_t)?;
            let sq = diff.sqr()?;
            // Weight by Fisher diagonal if available (proper EWC), else
            // uniform=1 (L2 toward θ*).
            let weighted = match &self.fisher {
                Some(f) => match f.get(name) {
                    Some(fi) => (fi * sq)?,
                    None => sq,
                },
                None => sq,
            };
            let term = weighted.sum_all()?;
            accum = Some(match accum {
                None => term,
                Some(prev) => (prev + term)?,
            });
        }
        match accum {
            Some(a) => a * (self.lambda / 2.0),
            None => Tensor::new(0f32, &device),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::{Init, VarBuilder};

    #[test]
    fn zero_penalty_when_weights_unchanged() {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &Device::Cpu);
        let _w = vb
            .get_with_hints(
                (4, 3),
                "w",
                Init::Randn {
                    mean: 0.0,
                    stdev: 0.02,
                },
            )
            .unwrap();
        let anchor = WeightAnchor::snapshot(&varmap, 1.0).unwrap();
        let p = anchor.penalty(&varmap).unwrap();
        let v = p.to_scalar::<f32>().unwrap();
        assert!(v.abs() < 1e-6, "expected ~0 penalty, got {v}");
    }

    #[test]
    fn fisher_weights_modulate_penalty() {
        // Build a tiny varmap with one weight and a hand-rolled fisher.
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &Device::Cpu);
        let _w = vb.get_with_hints((2, 2), "w", Init::Const(0.0)).unwrap();

        // Snapshot first (uniform Fisher path: penalty would be sum(diff²)).
        let mut anchor = WeightAnchor::snapshot(&varmap, 2.0).unwrap();

        // Inject a non-uniform Fisher: 1.0 in the (0,0) slot, zero elsewhere.
        let custom_fisher =
            Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 0.0], (2, 2), &Device::Cpu).unwrap();
        let mut fisher_map = HashMap::new();
        fisher_map.insert("w".to_string(), custom_fisher);
        anchor.fisher = Some(Arc::new(fisher_map));

        // Mutate var to all-ones — sum(diff²) = 4.
        // With Fisher [[1,0],[0,0]] only the (0,0) entry contributes, so
        // sum(F·diff²) = 1, and penalty = (λ/2) * 1 = 1.
        {
            let data = varmap.data().lock().unwrap();
            let var = data.get("w").unwrap();
            let new_t = Tensor::ones((2, 2), candle_core::DType::F32, &Device::Cpu).unwrap();
            var.set(&new_t).unwrap();
        }
        let p = anchor.penalty(&varmap).unwrap().to_scalar::<f32>().unwrap();
        assert!((p - 1.0).abs() < 1e-3, "expected penalty=1, got {p}");
    }

    #[test]
    fn penalty_grows_when_weights_drift() {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &Device::Cpu);
        let _w = vb.get_with_hints((4, 3), "w", Init::Const(0.0)).unwrap();
        let anchor = WeightAnchor::snapshot(&varmap, 2.0).unwrap();
        // Mutate the var to all-ones — total drift = 4*3 = 12 squared sum,
        // penalty = (λ/2) * 12 = 12.
        {
            let data = varmap.data().lock().unwrap();
            let var = data.get("w").unwrap();
            let new_t = Tensor::ones((4, 3), candle_core::DType::F32, &Device::Cpu).unwrap();
            var.set(&new_t).unwrap();
        }
        let p = anchor.penalty(&varmap).unwrap();
        let v = p.to_scalar::<f32>().unwrap();
        assert!((v - 12.0).abs() < 1e-3, "expected penalty=12, got {v}");
    }
}
