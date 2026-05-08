//! Phase 12 S2 — On-Policy Distillation (OPD).
//!
//! Reference: DeepSeek V4 technical report (2026-04-24). DeepSeek
//! abandoned the mixed-RL post-training stage that V3 used and
//! replaced it entirely with multi-teacher OPD: a unified student
//! is trained on its *own* rollouts against a weighted sum of
//! frozen specialist teacher distributions.
//!
//! Why this matters here: Phase 11 found that pure DPO multi-round
//! collapses on K9 (S3/S4) and that hybrid SFT+DPO (S5) doesn't
//! beat plain SFT either. DeepSeek hit the same wall and decided
//! the answer is *not* better RL but offline distillation from
//! pre-converged specialists. Phase 12 S2 builds the loss + step
//! function so S3 can measure whether OPD outperforms our
//! Phase 11 5-session matrix at toy scale.
//!
//! ## OPD loss
//!
//! Per generated token at position `t` in a student rollout:
//!
//!   p_student(·) = softmax(student_logits[t] / T)
//!   p_teacher_i(·) = softmax(teacher_i_logits[t] / T)
//!
//!   L_OPD(t) = Σ_i w_i · KL( p_student(·) || p_teacher_i(·) )
//!
//! Aggregated as the mean over kept positions in the batch.
//!
//! Reverse-KL (student-as-reference) was the original DeepSeek
//! choice; this implementation also exposes forward-KL via a flag
//! since the choice is empirical at our scale. Default: forward-KL
//! (KL(teacher || student)) which empirically tracks better when
//! the student is initialized from a poor checkpoint.
//!
//! Important: full-vocabulary KL, not token-level estimate. With
//! n teachers and vocab V, each step cost is `O(B·T·V·n)` for the
//! KL part — same order as standard CE.
//!
//! ## Why this is a *loss function*, not a full trainer
//!
//! S2 ships only the loss. The trainer (which orchestrates rollout
//! generation, multi-teacher forward, batching) lives in
//! `train::train_opd` (added in S3 once we know the loss shape is
//! right). Keeping the loss self-contained lets us unit-test it on
//! synthetic logits in CPU before any GPU runs.

use candle_core::{Result as CResult, Tensor, D};
use candle_nn::ops;

/// Direction of the KL divergence used by OPD. Forward = `KL(teacher
/// || student)` (teacher anchors the loss; standard distillation
/// direction). Reverse = `KL(student || teacher)` (DeepSeek V4
/// default — student-as-reference).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KlDirection {
    #[default]
    Forward,
    Reverse,
}

/// OPD loss given full-vocabulary logits from a student and a list
/// of teachers.
///
/// Inputs:
/// - `student_logits`: `[B, T, V]` from the policy on its own rollout.
/// - `teacher_logits`: `[(weight, [B, T, V])]` for each teacher. Weights
///   should sum to ~1 but are **not** auto-normalized — we keep the
///   caller in charge so a 0-weight teacher can be cheaply silenced.
/// - `temperature`: softmax temperature `T` (default 1.0). DeepSeek's
///   distillation literature commonly uses 2.0 to soften both
///   distributions; lower `T` makes the loss closer to argmax-KL.
/// - `direction`: forward vs reverse KL.
///
/// Output: scalar loss tensor (mean over `[B, T]`).
pub fn opd_loss(
    student_logits: &Tensor,
    teacher_logits: &[(f64, Tensor)],
    temperature: f64,
    direction: KlDirection,
) -> CResult<Tensor> {
    if teacher_logits.is_empty() {
        candle_core::bail!("opd_loss: at least one teacher required");
    }
    if temperature <= 0.0 {
        candle_core::bail!("opd_loss: temperature must be > 0, got {temperature}");
    }
    let s_dims = student_logits.dims3()?;
    for (_, t) in teacher_logits {
        let td = t.dims3()?;
        if td != s_dims {
            candle_core::bail!(
                "opd_loss: teacher logits {:?} mismatch student {:?}",
                td,
                s_dims
            );
        }
    }

    // Pre-compute scaled log-softmax once per call.
    let s_scaled = (student_logits / temperature)?;
    let s_log_p = ops::log_softmax(&s_scaled, D::Minus1)?;

    // Accumulate Σ_i w_i · KL_i.
    let mut total: Option<Tensor> = None;
    for (w, t_logits) in teacher_logits {
        let t_scaled = (t_logits / temperature)?;
        let t_log_p = ops::log_softmax(&t_scaled, D::Minus1)?;
        // KL(p || q) = Σ p · (log p − log q)
        let kl_per_pos = match direction {
            KlDirection::Forward => {
                // teacher || student: Σ p_t · (log p_t − log p_s)
                let p_t = t_log_p.exp()?;
                let diff = (&t_log_p - &s_log_p)?;
                (p_t * diff)?.sum(D::Minus1)?
            }
            KlDirection::Reverse => {
                // student || teacher: Σ p_s · (log p_s − log p_t)
                let p_s = s_log_p.exp()?;
                let diff = (&s_log_p - &t_log_p)?;
                (p_s * diff)?.sum(D::Minus1)?
            }
        };
        let kl_mean = kl_per_pos.mean_all()?;
        let weighted = (kl_mean * *w)?;
        total = Some(match total {
            None => weighted,
            Some(prev) => (prev + weighted)?,
        });
    }
    Ok(total.expect("non-empty teachers checked above"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    fn logits(values: &[f32], shape: (usize, usize, usize), dev: &Device) -> Tensor {
        Tensor::from_slice(values, shape, dev).expect("from")
    }

    #[test]
    fn opd_loss_zero_when_student_equals_single_teacher() {
        // KL(p || p) = 0 for any p. Single teacher, identical logits → loss 0.
        let dev = Device::Cpu;
        let s = Tensor::randn(0_f32, 1.0, (2, 3, 5), &dev).unwrap();
        let t = s.clone();
        let l = opd_loss(&s, &[(1.0, t)], 1.0, KlDirection::Forward)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(l.abs() < 1e-5, "expected 0, got {l}");
    }

    #[test]
    fn opd_loss_zero_under_reverse_kl_too() {
        let dev = Device::Cpu;
        let s = Tensor::randn(0_f32, 1.0, (2, 3, 5), &dev).unwrap();
        let t = s.clone();
        let l = opd_loss(&s, &[(1.0, t)], 1.0, KlDirection::Reverse)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(l.abs() < 1e-5);
    }

    #[test]
    fn opd_loss_positive_when_student_disagrees() {
        // Make student and teacher disagree on the argmax.
        let dev = Device::Cpu;
        let s = logits(
            &[
                10.0, 0.0, 0.0, 0.0, 0.0, // (0, 0): argmax=0
                10.0, 0.0, 0.0, 0.0, 0.0, // (0, 1): argmax=0
            ],
            (1, 2, 5),
            &dev,
        );
        let t = logits(
            &[
                0.0, 0.0, 10.0, 0.0, 0.0, // (0, 0): argmax=2
                0.0, 0.0, 10.0, 0.0, 0.0, // (0, 1): argmax=2
            ],
            (1, 2, 5),
            &dev,
        );
        let l = opd_loss(&s, &[(1.0, t)], 1.0, KlDirection::Forward)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(l > 5.0, "expected large positive KL, got {l}");
    }

    #[test]
    fn opd_loss_weighted_sum_matches_individual() {
        // Loss with two equal-weight teachers should equal the average
        // of two single-teacher losses (weights summing to 1).
        let dev = Device::Cpu;
        let s = Tensor::randn(0_f32, 1.0, (1, 4, 6), &dev).unwrap();
        let t1 = Tensor::randn(0_f32, 1.0, (1, 4, 6), &dev).unwrap();
        let t2 = Tensor::randn(0_f32, 1.0, (1, 4, 6), &dev).unwrap();
        let combined = opd_loss(
            &s,
            &[(0.5, t1.clone()), (0.5, t2.clone())],
            1.0,
            KlDirection::Forward,
        )
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
        let l1 = opd_loss(&s, &[(1.0, t1)], 1.0, KlDirection::Forward)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let l2 = opd_loss(&s, &[(1.0, t2)], 1.0, KlDirection::Forward)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let expected = 0.5 * l1 + 0.5 * l2;
        assert!(
            (combined - expected).abs() < 1e-4,
            "weighted sum {combined} vs expected {expected}"
        );
    }

    #[test]
    fn opd_loss_temperature_softens_disagreement() {
        // Higher T should reduce the magnitude of KL between two
        // disagreeing distributions (peaks become softer).
        let dev = Device::Cpu;
        let s = logits(&[10.0, 0.0, 0.0], (1, 1, 3), &dev);
        let t = logits(&[0.0, 10.0, 0.0], (1, 1, 3), &dev);
        let l_t1 = opd_loss(&s, &[(1.0, t.clone())], 1.0, KlDirection::Forward)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let l_t5 = opd_loss(&s, &[(1.0, t)], 5.0, KlDirection::Forward)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            l_t5 < l_t1,
            "T=5 should soften the loss; T=1 -> {l_t1}, T=5 -> {l_t5}"
        );
    }

    #[test]
    fn opd_loss_rejects_empty_teachers() {
        let dev = Device::Cpu;
        let s = Tensor::zeros((1, 1, 3), DType::F32, &dev).unwrap();
        assert!(opd_loss(&s, &[], 1.0, KlDirection::Forward).is_err());
    }

    #[test]
    fn opd_loss_rejects_shape_mismatch() {
        let dev = Device::Cpu;
        let s = Tensor::zeros((1, 2, 3), DType::F32, &dev).unwrap();
        let t = Tensor::zeros((1, 2, 4), DType::F32, &dev).unwrap();
        assert!(opd_loss(&s, &[(1.0, t)], 1.0, KlDirection::Forward).is_err());
    }

    #[test]
    fn opd_loss_rejects_non_positive_temperature() {
        let dev = Device::Cpu;
        let s = Tensor::zeros((1, 1, 3), DType::F32, &dev).unwrap();
        let t = Tensor::zeros((1, 1, 3), DType::F32, &dev).unwrap();
        assert!(opd_loss(&s, &[(1.0, t)], 0.0, KlDirection::Forward).is_err());
        let t2 = Tensor::zeros((1, 1, 3), DType::F32, &Device::Cpu).unwrap();
        assert!(opd_loss(&s, &[(1.0, t2)], -0.1, KlDirection::Forward).is_err());
    }
}
