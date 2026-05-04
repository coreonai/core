//! Training loop.
//!
//! Single-GPU AdamW + cosine LR schedule. No gradient accumulation yet
//! (Phase 1 keeps the loop minimal so the actor wiring stays the focus).

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::{ops, AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use indicatif::{ProgressBar, ProgressStyle};
use tracing::info;

use crate::config::GPTConfig;
use crate::data::TokenDataset;
use crate::error::Result;
use crate::model::GPT;

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub max_steps: usize,
    pub batch_size: usize,
    pub eval_interval: usize,
    pub eval_iters: usize,
    pub lr: f64,
    pub min_lr: f64,
    pub warmup_steps: usize,
    pub weight_decay: f64,
    pub grad_clip: f64,
    pub dtype: DType,
}

impl TrainConfig {
    pub fn smoke() -> Self {
        Self {
            max_steps: 200,
            batch_size: 32,
            eval_interval: 50,
            eval_iters: 20,
            lr: 3e-4,
            min_lr: 3e-5,
            warmup_steps: 20,
            weight_decay: 0.1,
            grad_clip: 1.0,
            dtype: DType::F32,
        }
    }
}

fn cosine_lr(step: usize, cfg: &TrainConfig) -> f64 {
    if step < cfg.warmup_steps {
        return cfg.lr * (step + 1) as f64 / cfg.warmup_steps.max(1) as f64;
    }
    let progress = (step - cfg.warmup_steps) as f64
        / (cfg.max_steps.saturating_sub(cfg.warmup_steps).max(1)) as f64;
    let progress = progress.clamp(0.0, 1.0);
    let coeff = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
    cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)
}

pub struct TrainOutcome {
    pub final_step: usize,
    pub last_train_loss: f32,
    pub last_val_loss: Option<f32>,
}

pub fn train(
    gpt_cfg: &GPTConfig,
    train_ds: &TokenDataset,
    val_ds: Option<&TokenDataset>,
    cfg: &TrainConfig,
    device: &Device,
    save_path: Option<&Path>,
) -> Result<TrainOutcome> {
    train_from(gpt_cfg, train_ds, val_ds, cfg, device, save_path, None)
}

/// Train, optionally starting from a checkpoint on disk (continual training).
///
/// When `init_from` is `Some(path)`, weights are loaded into the freshly-built
/// VarMap before the optimizer is created. The optimizer state always starts
/// fresh — that's intentional for short continual rounds.
pub fn train_from(
    gpt_cfg: &GPTConfig,
    train_ds: &TokenDataset,
    val_ds: Option<&TokenDataset>,
    cfg: &TrainConfig,
    device: &Device,
    save_path: Option<&Path>,
    init_from: Option<&Path>,
) -> Result<TrainOutcome> {
    train_from_with_anchor(
        gpt_cfg, train_ds, val_ds, cfg, device, save_path, init_from, None,
    )
}

/// Like [`train_from`] but optionally adds an EWC weight anchor's penalty
/// to every step's loss. Use this for continual fine-tuning rounds where
/// you want to keep parameters close to the pretrained checkpoint.
pub fn train_from_with_anchor(
    gpt_cfg: &GPTConfig,
    train_ds: &TokenDataset,
    val_ds: Option<&TokenDataset>,
    cfg: &TrainConfig,
    device: &Device,
    save_path: Option<&Path>,
    init_from: Option<&Path>,
    anchor: Option<&crate::ewc::WeightAnchor>,
) -> Result<TrainOutcome> {
    train_from_full(gpt_cfg, train_ds, val_ds, cfg, device, save_path, init_from, anchor, false)
}

/// Most general training entrypoint. When `freeze_base` is `true`, only
/// Vars whose name contains `"lora"` are trainable — the rest are loaded
/// from `init_from` and held fixed. Standard LoRA fine-tune.
#[allow(clippy::too_many_arguments)]
pub fn train_from_full(
    gpt_cfg: &GPTConfig,
    train_ds: &TokenDataset,
    val_ds: Option<&TokenDataset>,
    cfg: &TrainConfig,
    device: &Device,
    save_path: Option<&Path>,
    init_from: Option<&Path>,
    anchor: Option<&crate::ewc::WeightAnchor>,
    freeze_base: bool,
) -> Result<TrainOutcome> {
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, cfg.dtype, device);
    let model = GPT::new(gpt_cfg.clone(), vb)?;
    if let Some(path) = init_from {
        varmap.load(path)?;
        tracing::info!(?path, "loaded init weights for continual training");
    }

    let params = ParamsAdamW {
        lr: cfg.lr,
        weight_decay: cfg.weight_decay,
        ..Default::default()
    };
    // For freeze_base=true LoRA fine-tune: only the adapter Vars
    // (whose names contain `"lora"`) get gradient updates. The base
    // weights are loaded from `init_from` and never enter the optimizer.
    let trainable_vars: Vec<candle_core::Var> = if freeze_base {
        let data = varmap.data().lock().expect("varmap mutex");
        let only_lora: Vec<_> = data
            .iter()
            .filter(|(name, _)| name.contains("lora"))
            .map(|(_, v)| v.clone())
            .collect();
        if only_lora.is_empty() {
            return Err(crate::error::Error::Config(
                "freeze_base=true but no `lora_*` Vars found — set lora_rank > 0 in GPTConfig"
                    .into(),
            ));
        }
        tracing::info!(n_trainable = only_lora.len(), "LoRA-only fine-tune (base frozen)");
        only_lora
    } else {
        varmap.all_vars()
    };
    let mut opt = AdamW::new(trainable_vars, params)?;

    let pb = ProgressBar::new(cfg.max_steps as u64);
    pb.set_style(
        ProgressStyle::with_template("{bar:40.cyan/blue} {pos:>5}/{len:5} {msg}")
            .expect("progress style"),
    );

    let mut last_train_loss = f32::NAN;
    let mut last_val_loss: Option<f32> = None;

    for step in 0..cfg.max_steps {
        let lr = cosine_lr(step, cfg);
        opt.set_learning_rate(lr);

        let (x, y) = train_ds.random_batch(cfg.batch_size, device)?;
        let task_loss = model.loss(&x, &y)?;
        let total = if let Some(a) = anchor {
            (&task_loss + a.penalty(&varmap)?)?
        } else {
            task_loss.clone()
        };
        opt.backward_step(&total)?;
        last_train_loss = total.to_scalar::<f32>()?;

        if (step + 1) % cfg.eval_interval == 0 {
            if let Some(val) = val_ds {
                let mut acc = 0.0f32;
                for _ in 0..cfg.eval_iters {
                    let (xv, yv) = val.random_batch(cfg.batch_size, device)?;
                    let l = model.loss(&xv, &yv)?.to_scalar::<f32>()?;
                    acc += l;
                }
                let mean = acc / cfg.eval_iters as f32;
                last_val_loss = Some(mean);
                info!(step = step + 1, train = last_train_loss, val = mean, lr, "eval");
                pb.set_message(format!(
                    "train={:.4} val={:.4} lr={:.2e}",
                    last_train_loss, mean, lr
                ));
            }
        } else {
            pb.set_message(format!("loss={:.4} lr={:.2e}", last_train_loss, lr));
        }

        pb.inc(1);
    }
    pb.finish();

    if let Some(path) = save_path {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        varmap.save(path)?;
        let cfg_path = path.with_extension("cfg.json");
        let cfg_json = serde_json::to_string_pretty(gpt_cfg)?;
        std::fs::write(cfg_path, cfg_json)?;
    }

    Ok(TrainOutcome {
        final_step: cfg.max_steps,
        last_train_loss,
        last_val_loss,
    })
}

#[derive(Debug, Clone)]
pub struct DistillConfig {
    /// Softmax temperature for both student and teacher logits.
    pub temperature: f32,
    /// Mix between hard CE and KL distillation: `loss = (1-α)·CE + α·KL`.
    pub kl_weight: f32,
}

impl Default for DistillConfig {
    fn default() -> Self {
        Self { temperature: 2.0, kl_weight: 0.7 }
    }
}

/// Knowledge-distillation training. The student is the architecture being
/// trained; the teacher is loaded from `teacher_path` and held fixed (no
/// gradients flow through it). Loss combines:
///   - hard cross-entropy on the dataset targets
///   - KL divergence between softmax(student / T) and softmax(teacher / T)
/// scaled by `kl_weight`. Standard recipe: T=2, weight=0.7.
///
/// Teacher and student must share `vocab_size`; other architecture is free.
pub fn train_with_teacher(
    student_cfg: &GPTConfig,
    teacher_cfg: &GPTConfig,
    teacher_path: &Path,
    train_ds: &TokenDataset,
    val_ds: Option<&TokenDataset>,
    cfg: &TrainConfig,
    distill: &DistillConfig,
    device: &Device,
    save_path: Option<&Path>,
    student_init_from: Option<&Path>,
) -> Result<TrainOutcome> {
    if student_cfg.vocab_size != teacher_cfg.vocab_size {
        return Err(crate::error::Error::Config(format!(
            "student vocab {} ≠ teacher vocab {}",
            student_cfg.vocab_size, teacher_cfg.vocab_size
        )));
    }

    // ---- Teacher: frozen.
    let mut teacher_varmap = VarMap::new();
    let teacher_vb = VarBuilder::from_varmap(&teacher_varmap, cfg.dtype, device);
    let teacher = GPT::new(teacher_cfg.clone(), teacher_vb)?;
    teacher_varmap.load(teacher_path)?;
    info!(?teacher_path, "loaded teacher for distillation");

    // ---- Student: trainable.
    let mut student_varmap = VarMap::new();
    let student_vb = VarBuilder::from_varmap(&student_varmap, cfg.dtype, device);
    let student = GPT::new(student_cfg.clone(), student_vb)?;
    if let Some(p) = student_init_from {
        student_varmap.load(p)?;
    }

    let params = ParamsAdamW {
        lr: cfg.lr,
        weight_decay: cfg.weight_decay,
        ..Default::default()
    };
    let mut opt = AdamW::new(student_varmap.all_vars(), params)?;

    let pb = ProgressBar::new(cfg.max_steps as u64);
    pb.set_style(
        ProgressStyle::with_template("{bar:40.cyan/blue} {pos:>5}/{len:5} {msg}")
            .expect("progress style"),
    );

    let mut last_train_loss = f32::NAN;
    let mut last_val_loss: Option<f32> = None;
    let temp = distill.temperature.max(1e-4) as f64;
    let kl_w = distill.kl_weight as f64;

    for step in 0..cfg.max_steps {
        let lr = cosine_lr(step, cfg);
        opt.set_learning_rate(lr);

        let (x, y) = train_ds.random_batch(cfg.batch_size, device)?;

        // Teacher: forward only, stop gradients (Candle has no requires_grad
        // toggle but teacher's vars aren't in the optimizer, so this is fine).
        let t_logits = teacher.forward(&x)?;
        let t_logits = (&t_logits / temp)?;
        let t_probs = ops::softmax_last_dim(&t_logits)?;

        // Student.
        let s_logits = student.forward(&x)?;
        let s_logits_scaled = (&s_logits / temp)?;
        let s_log_probs = ops::log_softmax(&s_logits_scaled, candle_core::D::Minus1)?;

        // KL(t || s) = sum_v t * (log t - log s). Implemented as
        // -sum_v t * log_s + const(t).
        let neg_kl_part = (&t_probs * &s_log_probs)?;
        let kl = neg_kl_part.sum_all()?.neg()?;
        let kl = (kl * (temp * temp))?; // scale per Hinton
        let kl = (kl / (cfg.batch_size as f64))?;

        // Hard CE on student logits (no temperature).
        let (b, t, v) = s_logits.dims3()?;
        let s_flat = s_logits.reshape((b * t, v))?;
        let y_flat = y.reshape(b * t)?.to_dtype(DType::U32)?;
        let ce = candle_nn::loss::cross_entropy(&s_flat, &y_flat)?;

        let loss = ((&ce * (1.0 - kl_w))? + (&kl * kl_w)?)?;
        opt.backward_step(&loss)?;
        last_train_loss = loss.to_scalar::<f32>()?;

        if (step + 1) % cfg.eval_interval == 0 {
            if let Some(val) = val_ds {
                let mut acc = 0.0f32;
                for _ in 0..cfg.eval_iters {
                    let (xv, yv) = val.random_batch(cfg.batch_size, device)?;
                    let l = student.loss(&xv, &yv)?.to_scalar::<f32>()?;
                    acc += l;
                }
                let mean = acc / cfg.eval_iters as f32;
                last_val_loss = Some(mean);
                info!(step = step + 1, train = last_train_loss, val = mean, lr, "distill eval");
            }
        }
        pb.set_message(format!("distill loss={:.4} lr={:.2e}", last_train_loss, lr));
        pb.inc(1);
    }
    pb.finish();

    if let Some(path) = save_path {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        student_varmap.save(path)?;
    }

    // Suppress unused-tensor warnings.
    let _ = Tensor::new(0u32, device);
    Ok(TrainOutcome {
        final_step: cfg.max_steps,
        last_train_loss,
        last_val_loss,
    })
}
