//! Knowledge distillation: KoWiki 50M teacher → smaller student.
//!
//! Uses [`train_with_teacher`] (KL divergence on softmaxed logits + hard
//! cross-entropy, mixed by `kl_weight`). Teacher weights are loaded from
//! the safetensors checkpoint produced by `train_kowiki`. Student is
//! freshly initialized at a smaller size and trained on the same corpus.
//!
//! For a fair comparison, this example optionally also trains the same
//! student architecture from scratch (no teacher) on the same data, so
//! you can read off the distillation gain directly.
//!
//! Run:
//!   cargo run -p nanogpt-rs --example distill_kowiki --features cuda --release -- \
//!       --teacher checkpoints/kowiki_50m_clean.safetensors \
//!       --tokenizer data/kowiki/kowiki_bpe.json \
//!       --data data/kowiki/kowiki_clean.txt \
//!       --steps 4000 \
//!       --student-save checkpoints/kowiki_student.safetensors \
//!       --baseline-save checkpoints/kowiki_baseline.safetensors

use std::path::PathBuf;

use candle_core::{DType, Device};
use clap::Parser;
use nanogpt_rs::{
    config::{ActivationKind, GPTConfig, NormKind, NormPosition},
    data::TokenDataset,
    generate::{generate, GenerateConfig},
    model::GPT,
    tokenizer::Tokenizer,
    train::{train_from, train_with_teacher, DistillConfig, TrainConfig},
};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    teacher: PathBuf,
    #[arg(long, default_value = "data/kowiki/kowiki_bpe.json")]
    tokenizer: PathBuf,
    #[arg(long, default_value = "data/kowiki/kowiki_clean.txt")]
    data: PathBuf,
    #[arg(long, default_value_t = 4000)]
    steps: usize,
    #[arg(long, default_value_t = 32)]
    batch_size: usize,
    #[arg(long, default_value_t = 256)]
    block_size: usize,
    #[arg(long, default_value_t = 1e-3)]
    lr: f64,
    #[arg(long, default_value_t = 2.0)]
    temperature: f32,
    #[arg(long, default_value_t = 0.7)]
    kl_weight: f32,
    /// If set, also train the same student architecture from scratch
    /// on the same data (no teacher). Doubles the wallclock but gives
    /// a clean A/B comparison.
    #[arg(long)]
    train_baseline: bool,
    #[arg(long, default_value = "checkpoints/kowiki_student.safetensors")]
    student_save: PathBuf,
    #[arg(long, default_value = "checkpoints/kowiki_baseline.safetensors")]
    baseline_save: PathBuf,
    #[arg(long, default_value = "대한민국의 수도는 ")]
    sample_prompt: String,
    #[arg(long, default_value_t = 60)]
    sample_tokens: usize,
}

fn pick_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        if let Ok(d) = Device::new_cuda(0) {
            return d;
        }
    }
    Device::Cpu
}

/// Smaller student than `nano_50m`: 4 layers / 256 dim / GQA-2 / SwiGLU /
/// untied / RmsNorm-Pre. Roughly ~12-15M params at typical vocab sizes.
fn student_config(vocab: usize, block_size: usize) -> GPTConfig {
    GPTConfig {
        vocab_size: vocab,
        block_size,
        n_layer: 4,
        n_head: 4,
        n_embd: 256,
        dropout: 0.0,
        bias: false,
        ffn_mult: 4,
        use_rope: true,
        rope_base: 10_000.0,
        n_kv_head: 2,
        n_experts: 1,
        moe_top_k: 0,
        moe_aux_weight: 0.0,
        activation: ActivationKind::SwiGlu,
        weight_tying: false,
        norm_kind: NormKind::RmsNorm,
        norm_position: NormPosition::Pre,
        lora_rank: 0,
        lora_alpha: 16.0,
    }
}

fn sample_from(
    checkpoint: &PathBuf,
    gpt_cfg: &GPTConfig,
    tk: &Tokenizer,
    prompt: &str,
    max_new: usize,
    device: &Device,
) -> anyhow::Result<String> {
    let mut varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = GPT::new(gpt_cfg.clone(), vb)?;
    varmap.load(checkpoint)?;
    let prompt_ids = tk.encode(prompt)?;
    let cfg = GenerateConfig {
        max_new_tokens: max_new,
        temperature: 0.8,
        top_k: Some(40),
        top_p: Some(0.9),
        seed: Some(42),
    };
    let out = generate(&model, &prompt_ids, &cfg, device)?;
    Ok(tk.decode(&out)?)
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    let args = Args::parse();
    let device = pick_device();
    tracing::info!(?device, "device");

    // ---- Tokenizer + corpus.
    let tk = Tokenizer::from_hf_file(&args.tokenizer)?;
    let vocab = tk.vocab_size();
    tracing::info!(vocab, "tokenizer ready");

    let text = std::fs::read_to_string(&args.data)?;
    let ids = tk.encode(&text)?;
    tracing::info!(chars = text.len(), tokens = ids.len(), "corpus loaded");
    let ds = TokenDataset::new(ids, args.block_size);

    // ---- Teacher = nano_50m matching the train_kowiki training.
    let mut teacher_cfg = GPTConfig::nano_50m();
    teacher_cfg.vocab_size = vocab;
    teacher_cfg.block_size = args.block_size;
    tracing::info!(
        params = teacher_cfg.num_params_estimate(),
        "teacher config (frozen)"
    );

    // ---- Student.
    let student_cfg = student_config(vocab, args.block_size);
    tracing::info!(params = student_cfg.num_params_estimate(), "student config");

    let mut tcfg = TrainConfig::smoke();
    tcfg.max_steps = args.steps;
    tcfg.batch_size = args.batch_size;
    tcfg.eval_interval = args.steps; // skip mid-training eval
    tcfg.lr = args.lr;
    tcfg.min_lr = args.lr * 0.1;
    tcfg.warmup_steps = (args.steps / 30).max(50);
    tcfg.weight_decay = 0.1;

    // ---- Distillation run.
    let distill_cfg = DistillConfig {
        temperature: args.temperature,
        kl_weight: args.kl_weight,
    };
    tracing::info!(
        temperature = args.temperature,
        kl_weight = args.kl_weight,
        "running distillation"
    );
    let distill_outcome = train_with_teacher(
        &student_cfg,
        &teacher_cfg,
        &args.teacher,
        &ds,
        None,
        &tcfg,
        &distill_cfg,
        &device,
        Some(&args.student_save),
        None,
    )?;
    tracing::info!(
        train_loss = distill_outcome.last_train_loss,
        "distillation done"
    );

    // ---- Optional from-scratch baseline.
    let baseline_outcome = if args.train_baseline {
        tracing::info!("running from-scratch baseline (same student, no teacher)");
        let r = train_from(
            &student_cfg,
            &ds,
            None,
            &tcfg,
            &device,
            Some(&args.baseline_save),
            None,
        )?;
        tracing::info!(train_loss = r.last_train_loss, "baseline done");
        Some(r)
    } else {
        None
    };

    // ---- Sample side-by-side.
    println!("\n=== samples (prompt: {:?}) ===", args.sample_prompt);
    let s1 = sample_from(
        &args.student_save,
        &student_cfg,
        &tk,
        &args.sample_prompt,
        args.sample_tokens,
        &device,
    )?;
    println!("\n--- distilled student ---\n{s1}\n");
    if baseline_outcome.is_some() {
        let s2 = sample_from(
            &args.baseline_save,
            &student_cfg,
            &tk,
            &args.sample_prompt,
            args.sample_tokens,
            &device,
        )?;
        println!("--- from-scratch baseline ---\n{s2}\n");
    }
    let s3 = sample_from(
        &args.teacher,
        &teacher_cfg,
        &tk,
        &args.sample_prompt,
        args.sample_tokens,
        &device,
    )?;
    println!("--- teacher (50M) ---\n{s3}\n");

    println!("=== final losses ===");
    println!("distilled student: {:.4}", distill_outcome.last_train_loss);
    if let Some(b) = baseline_outcome {
        println!("from-scratch baseline: {:.4}", b.last_train_loss);
    }
    Ok(())
}
