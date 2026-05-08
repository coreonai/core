//! Phase 11 prep — Direct Preference Optimization (DPO) loss.
//!
//! DPO replaces the SFT step in self-improve loops with a contrastive
//! objective over (chosen, rejected) pairs:
//!
//!   L_DPO(π, π_ref) = -E[ log σ( β · ( (logπ(y_w|x) - logπ_ref(y_w|x))
//!                                    - (logπ(y_l|x) - logπ_ref(y_l|x)) ) ) ]
//!
//! Reference: Rafailov et al., "Direct Preference Optimization: Your
//! Language Model is Secretly a Reward Model" (NeurIPS 2023).
//!
//! ## Why DPO fits the project
//!
//! Phase 9 S5's self-improve loop already produces (winner, loser)
//! pairs every round: verifier-passed completions are winners,
//! verifier-rejected ones are losers. Today we feed only winners into
//! SFT. With DPO we use the rejected completions too — negative
//! gradient signal that pushes the policy away from them, on top of
//! the positive signal from chosen.
//!
//! This module implements the loss only. Integration into
//! `train_from_full` (a `dpo_beta` config field, a reference checkpoint
//! load, a (chosen, rejected) batch shape) is Phase 11 work. Right now
//! we have the function + tests so Phase 11 can wire it up in one
//! session.
//!
//! ## Numerical stability
//!
//! `-log σ(z) = log(1 + exp(-z)) = softplus(-z)`. We compute
//! `softplus` as `log(1 + exp(x))` (naive form). For our use
//! (`β ∈ [0.05, 0.5]`, `|logp - logp_ref|` typically a few-tens),
//! `β · d` lies safely in `[-50, 50]` where the naive form is exact in
//! f32. If callers go far outside that range, add a stable form
//! `max(x, 0) + log(1 + exp(-|x|))`.

use candle_core::{Result as CResult, Tensor};

/// Numerically reasonable `log(1 + exp(x))`. Used by [`dpo_loss`].
fn softplus(x: &Tensor) -> CResult<Tensor> {
    let exp = x.exp()?;
    let one_plus = (exp + 1.0)?;
    one_plus.log()
}

/// DPO loss given per-pair sequence log-probabilities under both the
/// **policy** (`π`, the model being trained) and a frozen **reference**
/// (`π_ref`, typically the SFT initialization).
///
/// All four inputs are 1-D tensors of shape `[B]` (one element per
/// (chosen, rejected) pair). `beta` is the DPO temperature; common
/// values are `0.1 – 0.5`.
///
/// Returns the mean DPO loss across the batch.
pub fn dpo_loss(
    policy_chosen_logp: &Tensor,
    policy_rejected_logp: &Tensor,
    ref_chosen_logp: &Tensor,
    ref_rejected_logp: &Tensor,
    beta: f64,
) -> CResult<Tensor> {
    // π log-ratio: logπ(y_w|x) - logπ(y_l|x)
    let pi_logratios = (policy_chosen_logp - policy_rejected_logp)?;
    // reference log-ratio: logπ_ref(y_w|x) - logπ_ref(y_l|x)
    let ref_logratios = (ref_chosen_logp - ref_rejected_logp)?;
    // implicit reward delta z = β · (pi_logratios - ref_logratios)
    let delta = (pi_logratios - ref_logratios)?;
    let z = (delta * beta)?;
    // L = -log σ(z) = softplus(-z)
    let neg_z = z.neg()?;
    let per_pair = softplus(&neg_z)?;
    per_pair.mean_all()
}

/// Implicit reward of a (prompt, completion) pair under DPO:
///
///   r_θ(x, y) = β · (logπ(y|x) - logπ_ref(y|x))
///
/// Useful for *measurement*, not training — e.g. comparing post-DPO
/// reward gaps across rounds, or sanity-checking that
/// `r(chosen) - r(rejected) > 0` after fine-tune.
pub fn dpo_implicit_reward(policy_logp: &Tensor, ref_logp: &Tensor, beta: f64) -> CResult<Tensor> {
    let delta = (policy_logp - ref_logp)?;
    delta * beta
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn t1(v: &[f32], dev: &Device) -> Tensor {
        Tensor::from_slice(v, v.len(), dev).expect("from")
    }

    #[test]
    fn loss_zero_when_pi_equals_ref_and_chosen_equals_rejected() {
        // If π == π_ref AND chosen logp == rejected logp, then z = 0,
        // and softplus(0) = log 2 ≈ 0.6931. Establishes the reference
        // value for "no signal."
        let dev = Device::Cpu;
        let chosen_p = t1(&[1.0_f32, -2.0, 0.5], &dev);
        let rejected_p = chosen_p.clone();
        let chosen_r = chosen_p.clone();
        let rejected_r = chosen_p.clone();
        let l = dpo_loss(&chosen_p, &rejected_p, &chosen_r, &rejected_r, 0.1).unwrap();
        let v = l.to_scalar::<f32>().unwrap();
        assert!(
            (v - (2.0_f32.ln())).abs() < 1e-5,
            "expected log 2 ≈ 0.693, got {v}"
        );
    }

    #[test]
    fn loss_decreases_when_chosen_outperforms_rejected_under_policy() {
        // If policy boosts chosen more than rejected (relative to ref),
        // z > 0, so softplus(-z) < log 2.
        let dev = Device::Cpu;
        let chosen_p = t1(&[2.0_f32, 2.0], &dev); // policy thinks chosen is high
        let rejected_p = t1(&[0.0_f32, 0.0], &dev); // policy thinks rejected is low
        let chosen_r = t1(&[0.0_f32, 0.0], &dev); // ref is neutral
        let rejected_r = t1(&[0.0_f32, 0.0], &dev);
        let l = dpo_loss(&chosen_p, &rejected_p, &chosen_r, &rejected_r, 0.5).unwrap();
        let v = l.to_scalar::<f32>().unwrap();
        // delta = (2-0) - (0-0) = 2, z = 0.5*2 = 1.0, softplus(-1) ≈ 0.3133
        assert!((v - 0.3133_f32).abs() < 1e-3, "expected ~0.313, got {v}");
        assert!(
            v < 2.0_f32.ln(),
            "loss must drop below log 2 when chosen wins"
        );
    }

    #[test]
    fn loss_increases_when_rejected_outperforms_chosen_under_policy() {
        // Symmetric: policy boosting rejected over chosen pushes loss
        // above log 2.
        let dev = Device::Cpu;
        let chosen_p = t1(&[0.0_f32, 0.0], &dev);
        let rejected_p = t1(&[2.0_f32, 2.0], &dev);
        let chosen_r = t1(&[0.0_f32, 0.0], &dev);
        let rejected_r = t1(&[0.0_f32, 0.0], &dev);
        let l = dpo_loss(&chosen_p, &rejected_p, &chosen_r, &rejected_r, 0.5).unwrap();
        let v = l.to_scalar::<f32>().unwrap();
        // delta = -2, z = -1, softplus(1) ≈ 1.3133
        assert!((v - 1.3133_f32).abs() < 1e-3, "expected ~1.313, got {v}");
        assert!(v > 2.0_f32.ln());
    }

    #[test]
    fn beta_zero_reduces_loss_to_log_two_uniformly() {
        // β = 0 zeros out everything; softplus(0) = log 2 always.
        let dev = Device::Cpu;
        let chosen_p = t1(&[5.0_f32, -3.0, 1.0], &dev);
        let rejected_p = t1(&[-5.0_f32, 7.0, -1.0], &dev);
        let chosen_r = t1(&[1.0_f32, 1.0, 1.0], &dev);
        let rejected_r = t1(&[-1.0_f32, -1.0, -1.0], &dev);
        let l = dpo_loss(&chosen_p, &rejected_p, &chosen_r, &rejected_r, 0.0).unwrap();
        let v = l.to_scalar::<f32>().unwrap();
        assert!((v - 2.0_f32.ln()).abs() < 1e-5, "expected log 2, got {v}");
    }

    #[test]
    fn implicit_reward_is_beta_times_logp_delta() {
        let dev = Device::Cpu;
        let policy_p = t1(&[1.0_f32, 2.0, 3.0], &dev);
        let ref_p = t1(&[0.5_f32, 1.5, 2.5], &dev);
        let r = dpo_implicit_reward(&policy_p, &ref_p, 0.2)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // each delta = 0.5; reward = 0.5 * 0.2 = 0.1
        for v in r {
            assert!((v - 0.1_f32).abs() < 1e-5, "expected 0.1, got {v}");
        }
    }

    #[test]
    fn softplus_handles_extreme_values() {
        // Sanity: positive large → softplus ≈ x; negative large → ≈ 0.
        let dev = Device::Cpu;
        let big_pos = t1(&[20.0_f32], &dev);
        let big_neg = t1(&[-20.0_f32], &dev);
        let sp_pos = softplus(&big_pos).unwrap().to_vec1::<f32>().unwrap()[0];
        let sp_neg = softplus(&big_neg).unwrap().to_vec1::<f32>().unwrap()[0];
        assert!((sp_pos - 20.0_f32).abs() < 1e-3);
        assert!(sp_neg.abs() < 1e-7);
    }

    #[test]
    fn dpo_loss_is_finite_under_typical_logp_magnitudes() {
        // Self-improve loops produce sequence log-probs in roughly
        // [-200, 0]. Make sure DPO loss doesn't blow up there.
        let dev = Device::Cpu;
        let chosen_p = t1(&[-50.0_f32, -120.0, -30.0], &dev);
        let rejected_p = t1(&[-80.0_f32, -90.0, -45.0], &dev);
        let chosen_r = t1(&[-55.0_f32, -110.0, -32.0], &dev);
        let rejected_r = t1(&[-78.0_f32, -100.0, -40.0], &dev);
        let l = dpo_loss(&chosen_p, &rejected_p, &chosen_r, &rejected_r, 0.1).unwrap();
        let v = l.to_scalar::<f32>().unwrap();
        assert!(
            v.is_finite() && v > 0.0,
            "DPO loss should be finite > 0, got {v}"
        );
    }

    // Discard tensor types we don't need at compile time.
    #[allow(dead_code)]
    fn _force_dtype_use() {
        let _ = DType::F32;
    }
}
