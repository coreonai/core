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
use crate::jepa::{jepa_loss, JepaPredictor};
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
    /// Phase 10 S1: weight on the JEPA auxiliary loss. `0.0` disables
    /// it entirely (predictor is not built; existing call sites are
    /// unaffected). Typical values: 0.05 – 0.5.
    pub jepa_lambda: f32,
    /// Future-position offset `k` for JEPA target. Hidden state at
    /// position `i` predicts hidden state at `i + jepa_offset`.
    /// Ignored when `jepa_lambda == 0.0`. Must be `< block_size`.
    pub jepa_offset: usize,
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
            jepa_lambda: 0.0,
            jepa_offset: 4,
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
#[allow(clippy::too_many_arguments)]
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
    train_from_full(
        gpt_cfg, train_ds, val_ds, cfg, device, save_path, init_from, anchor, false,
    )
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
    let model = GPT::new(gpt_cfg.clone(), vb.clone())?;
    // Phase 10 S1: optional JEPA predictor. Only built when
    // `jepa_lambda > 0`, so existing call sites incur zero overhead.
    let jepa_predictor = if cfg.jepa_lambda > 0.0 {
        if freeze_base {
            return Err(crate::error::Error::Config(
                "jepa_lambda > 0 with freeze_base=true is not supported \
                 (JEPA is a pretraining objective; freeze_base is for LoRA fine-tune)"
                    .into(),
            ));
        }
        if cfg.jepa_offset == 0 || cfg.jepa_offset >= gpt_cfg.block_size {
            return Err(crate::error::Error::Config(format!(
                "jepa_offset={} must satisfy 0 < offset < block_size={}",
                cfg.jepa_offset, gpt_cfg.block_size
            )));
        }
        if gpt_cfg.n_experts > 1 {
            return Err(crate::error::Error::Config(
                "jepa_lambda > 0 is not yet wired to combine with MoE aux loss \
                 (would need to use forward_with_aux instead of forward_with_hidden)"
                    .into(),
            ));
        }
        Some(JepaPredictor::new(gpt_cfg.n_embd, vb.pp("jepa_predictor"))?)
    } else {
        None
    };
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
        tracing::info!(
            n_trainable = only_lora.len(),
            "LoRA-only fine-tune (base frozen)"
        );
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
    let mut last_jepa_loss: Option<f32> = None;

    for step in 0..cfg.max_steps {
        let lr = cosine_lr(step, cfg);
        opt.set_learning_rate(lr);

        let (x, y) = train_ds.random_batch(cfg.batch_size, device)?;
        let task_loss = if let Some(predictor) = &jepa_predictor {
            // JEPA branch: one forward pass yields both logits and
            // hidden states; CE + λ·JEPA shares the backward.
            let (logits, hidden) = model.forward_with_hidden(&x)?;
            let (b, t, v) = logits.dims3()?;
            let logits_flat = logits.reshape((b * t, v))?;
            let targets_flat = y.reshape(b * t)?.to_dtype(DType::U32)?;
            let ce = candle_nn::loss::cross_entropy(&logits_flat, &targets_flat)?;
            let jl = jepa_loss(predictor, &hidden, cfg.jepa_offset)?;
            last_jepa_loss = Some(jl.to_scalar::<f32>()?);
            (ce + (jl * cfg.jepa_lambda as f64)?)?
        } else {
            model.loss(&x, &y)?
        };
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
                info!(
                    step = step + 1,
                    train = last_train_loss,
                    val = mean,
                    jepa = last_jepa_loss.unwrap_or(f32::NAN),
                    lr,
                    "eval"
                );
                pb.set_message(format!(
                    "train={:.4} val={:.4} lr={:.2e}",
                    last_train_loss, mean, lr
                ));
            }
        } else if let Some(jl) = last_jepa_loss {
            pb.set_message(format!(
                "loss={:.4} jepa={:.4} lr={:.2e}",
                last_train_loss, jl, lr
            ));
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
        Self {
            temperature: 2.0,
            kl_weight: 0.7,
        }
    }
}

/// Knowledge-distillation training. The student is the architecture being
/// trained; the teacher is loaded from `teacher_path` and held fixed (no
/// gradients flow through it). Loss combines:
///   - hard cross-entropy on the dataset targets
///   - KL divergence between softmax(student / T) and softmax(teacher / T)
///     scaled by `kl_weight`. Standard recipe: T=2, weight=0.7.
///
/// Teacher and student must share `vocab_size`; other architecture is free.
#[allow(clippy::too_many_arguments)]
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

        // KL(t || s) = sum_v t * (log t - log s). The `-sum_v t * log_s`
        // half is what carries the student's gradient; the entropy of t
        // is constant w.r.t. the student so we drop it. Average over
        // (B × T) so the loss magnitude stays comparable to the hard CE
        // term — without the T-dim normalization the KL ends up
        // `seq_len` times too large and dominates training.
        let (b, t, v) = s_logits.dims3()?;
        let neg_kl_part = (&t_probs * &s_log_probs)?;
        let kl = neg_kl_part.sum_all()?.neg()?;
        let kl = (kl * (temp * temp))?; // scale per Hinton
        let kl = (kl / ((b * t) as f64))?;

        // Hard CE on student logits (no temperature).
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
                info!(
                    step = step + 1,
                    train = last_train_loss,
                    val = mean,
                    lr,
                    "distill eval"
                );
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
        // Save the student config alongside so `eval_kowiki` (and any
        // other consumer) can rebuild the architecture without external
        // hints. Mirrors what `train_from_full` does.
        let cfg_path = path.with_extension("cfg.json");
        let cfg_json = serde_json::to_string_pretty(student_cfg)?;
        std::fs::write(cfg_path, cfg_json)?;
    }

    // Suppress unused-tensor warnings.
    let _ = Tensor::new(0u32, device);
    Ok(TrainOutcome {
        final_step: cfg.max_steps,
        last_train_loss,
        last_val_loss,
    })
}

#[cfg(test)]
mod cfg_persistence_tests {
    //! Round-trip tests for the `<ckpt>.cfg.json` sibling that
    //! `train_from_full` and `train_with_teacher` write next to the
    //! `.safetensors`. `eval_kowiki`, `self_improve_korean`, and
    //! `distill_kowiki` all rely on this file to load checkpoints
    //! whose architecture differs from any of the named presets, so
    //! a regression here silently breaks those examples.
    use super::*;
    use crate::config::{ActivationKind, GPTConfig, NormKind, NormPosition};

    /// 8-vocab / 16-dim toy config — small enough to train a few steps on CPU.
    fn toy_cfg() -> GPTConfig {
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
            n_experts: 1,
            moe_top_k: 0,
            moe_aux_weight: 0.0,
            activation: ActivationKind::Gelu,
            weight_tying: true,
            norm_kind: NormKind::LayerNorm,
            norm_position: NormPosition::Pre,
            lora_rank: 0,
            lora_alpha: 16.0,
        }
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("workllm-train-cfg-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        dir
    }

    fn toy_dataset(cfg: &GPTConfig) -> TokenDataset {
        // 64 token-ids cycling through the vocab — long enough to allow a few
        // sliding-window samples at block_size=4.
        let ids: Vec<u32> = (0..64).map(|i| i % cfg.vocab_size as u32).collect();
        TokenDataset::new(ids, cfg.block_size)
    }

    fn tiny_train_cfg() -> TrainConfig {
        TrainConfig {
            max_steps: 2,
            batch_size: 2,
            eval_interval: 100,
            eval_iters: 1,
            lr: 1e-3,
            min_lr: 1e-4,
            warmup_steps: 1,
            weight_decay: 0.0,
            grad_clip: 1.0,
            dtype: DType::F32,
            jepa_lambda: 0.0,
            jepa_offset: 4,
        }
    }

    #[test]
    fn train_from_writes_cfg_json_sibling_that_round_trips() {
        let dir = temp_dir("train-from");
        let ckpt = dir.join("toy.safetensors");
        let cfg = toy_cfg();
        let ds = toy_dataset(&cfg);
        let device = Device::Cpu;

        train_from(
            &cfg,
            &ds,
            None,
            &tiny_train_cfg(),
            &device,
            Some(&ckpt),
            None,
        )
        .expect("train_from");

        // Both files must exist.
        assert!(ckpt.exists(), "missing .safetensors at {ckpt:?}");
        let cfg_path = ckpt.with_extension("cfg.json");
        assert!(cfg_path.exists(), "missing .cfg.json at {cfg_path:?}");

        // .cfg.json must deserialize back to the exact GPTConfig we passed.
        let s = std::fs::read_to_string(&cfg_path).expect("read cfg.json");
        let round: GPTConfig = serde_json::from_str(&s).expect("parse cfg.json");
        assert_eq!(round, cfg, "cfg.json round-trip mismatch");
    }

    #[test]
    fn train_with_teacher_writes_student_cfg_json() {
        let dir = temp_dir("train-with-teacher");
        let teacher_ckpt = dir.join("teacher.safetensors");
        let student_ckpt = dir.join("student.safetensors");
        let teacher_cfg = toy_cfg();
        let student_cfg = toy_cfg();
        let device = Device::Cpu;

        // First train (then save) a teacher so train_with_teacher can load it.
        let ds = toy_dataset(&teacher_cfg);
        train_from(
            &teacher_cfg,
            &ds,
            None,
            &tiny_train_cfg(),
            &device,
            Some(&teacher_ckpt),
            None,
        )
        .expect("train teacher");

        // Now distill.
        let distill = DistillConfig::default();
        train_with_teacher(
            &student_cfg,
            &teacher_cfg,
            &teacher_ckpt,
            &ds,
            None,
            &tiny_train_cfg(),
            &distill,
            &device,
            Some(&student_ckpt),
            None,
        )
        .expect("train_with_teacher");

        // Both student artifacts must exist.
        assert!(student_ckpt.exists(), "missing student.safetensors");
        let student_cfg_path = student_ckpt.with_extension("cfg.json");
        assert!(
            student_cfg_path.exists(),
            "missing student.cfg.json at {student_cfg_path:?} — the K8 eval flow \
             relies on this file to know the student's architecture"
        );

        // The persisted cfg must be the STUDENT's, not the teacher's.
        let s = std::fs::read_to_string(&student_cfg_path).expect("read");
        let round: GPTConfig = serde_json::from_str(&s).expect("parse");
        assert_eq!(round, student_cfg);
    }

    #[test]
    fn save_path_with_no_extension_still_writes_cfg_json() {
        // `path.with_extension("cfg.json")` does the right thing even if the
        // input has no extension — `eval_kowiki` callers occasionally pass
        // "checkpoints/foo" without `.safetensors`. Lock in this contract.
        let dir = temp_dir("no-ext");
        let ckpt = dir.join("toy"); // no .safetensors
        let cfg = toy_cfg();
        let ds = toy_dataset(&cfg);

        train_from(
            &cfg,
            &ds,
            None,
            &tiny_train_cfg(),
            &Device::Cpu,
            Some(&ckpt),
            None,
        )
        .expect("train_from");

        let cfg_path = ckpt.with_extension("cfg.json");
        assert!(
            cfg_path.exists(),
            "expected cfg.json next to extensionless save path"
        );
    }

    #[test]
    fn jepa_lambda_positive_runs_end_to_end_and_writes_predictor_vars() {
        // Smoke: train_from_full with jepa_lambda > 0 should run, the
        // checkpoint should round-trip, and the saved safetensors must
        // include the predictor's vars (under `jepa_predictor.*`).
        let dir = temp_dir("jepa-smoke");
        let ckpt = dir.join("jepa.safetensors");
        let cfg = toy_cfg();
        let ds = toy_dataset(&cfg);
        let mut tcfg = tiny_train_cfg();
        tcfg.jepa_lambda = 0.1;
        tcfg.jepa_offset = 1; // toy_cfg has block_size=4, so offset must be < 4
        train_from_full(
            &cfg,
            &ds,
            None,
            &tcfg,
            &Device::Cpu,
            Some(&ckpt),
            None,
            None,
            false,
        )
        .expect("train_from_full with JEPA");

        // Inspect the saved tensors to confirm jepa_predictor.* is present.
        let mut vm = VarMap::new();
        let _vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        // Re-create model + predictor to register their Vars in the same
        // namespace, then load the checkpoint.
        let _gpt = GPT::new(
            cfg.clone(),
            VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu),
        )
        .expect("build gpt");
        let _pred = JepaPredictor::new(
            cfg.n_embd,
            VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu).pp("jepa_predictor"),
        )
        .expect("build predictor");
        vm.load(&ckpt).expect("load");
        let names: Vec<String> = {
            let data = vm.data().lock().expect("varmap mutex");
            data.keys().cloned().collect()
        };
        assert!(
            names.iter().any(|n| n.starts_with("jepa_predictor")),
            "expected jepa_predictor.* vars in checkpoint, got: {names:?}"
        );
    }

    #[test]
    fn jepa_lambda_with_freeze_base_errors() {
        let cfg = toy_cfg();
        let ds = toy_dataset(&cfg);
        let mut tcfg = tiny_train_cfg();
        tcfg.jepa_lambda = 0.1;
        tcfg.jepa_offset = 1;
        let res = train_from_full(
            &cfg,
            &ds,
            None,
            &tcfg,
            &Device::Cpu,
            None,
            None,
            None,
            true, // freeze_base + JEPA = config error
        );
        assert!(res.is_err(), "expected config error for jepa+freeze_base");
    }
}
