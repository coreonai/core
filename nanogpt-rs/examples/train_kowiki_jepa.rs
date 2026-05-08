//! Phase 10 S1: KoWiki training with optional JEPA auxiliary loss.
//!
//! Same model + corpus as `train_kowiki.rs`, but exposes
//! `--jepa-lambda` and `--jepa-offset` so we can run paired
//! comparisons with and without the latent-prediction objective.
//!
//! Hypothesis (from `docs/phase7-design.md` risk #9 and Phase 9 S2):
//! K8 Korean training over-fits onto high-frequency tokens (`\n` mode
//! collapse) when scaled past ~30K steps, *worsening* sum-AUC for
//! Shape C. JEPA's stop-gradient latent-prediction term should slow
//! that collapse. Falsifiable: measure (a) val loss curve, (b) top-1
//! token mass on a held-out batch, (c) sum-AUC on
//! KoreanCompletionDomain via `critic_baseline_korean`.
//!
//! Run:
//!   cargo run -p nanogpt-rs --example train_kowiki_jepa --features cuda --release \
//!       -- --steps 30000 --jepa-lambda 0.1
//!   # baseline (jepa-lambda 0 ⇒ same code path, predictor not built):
//!   cargo run -p nanogpt-rs --example train_kowiki_jepa --features cuda --release \
//!       -- --steps 30000 --jepa-lambda 0.0 --save checkpoints/k8_30k_baseline.safetensors

use std::path::PathBuf;

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use clap::Parser;
use nanogpt_rs::{
    config::GPTConfig,
    data::TokenDataset,
    generate::{generate, GenerateConfig},
    jepa::top1_mass,
    model::GPT,
    tokenizer::Tokenizer,
    train::{train_from_full, OptimizerKind, TrainConfig},
};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "data/kowiki/kowiki_plain.txt")]
    data: PathBuf,
    #[arg(long, default_value = "data/kowiki/kowiki_bpe.json")]
    tokenizer: PathBuf,
    #[arg(long)]
    retrain_tokenizer: bool,
    #[arg(long, default_value_t = 16_000)]
    vocab_size: usize,
    #[arg(long, default_value = "checkpoints/k8_30k_jepa.safetensors")]
    save: PathBuf,
    #[arg(long, default_value_t = 30_000)]
    steps: usize,
    #[arg(long, default_value_t = 16)]
    batch_size: usize,
    #[arg(long, default_value_t = 6e-4)]
    lr: f64,
    #[arg(long, default_value_t = 200)]
    eval_interval: usize,
    #[arg(long, default_value_t = 256)]
    block_size: usize,
    /// Phase 10 S1: weight on the JEPA aux loss. `0.0` disables.
    #[arg(long, default_value_t = 0.1)]
    jepa_lambda: f32,
    /// Future-token offset for JEPA target.
    #[arg(long, default_value_t = 8)]
    jepa_offset: usize,
    /// Phase 10 S2: optional EMA decay for a separate target encoder
    /// (BYOL/I-JEPA style). `None` (default) uses single-encoder
    /// stop-gradient. Typical values: 0.99 – 0.999.
    #[arg(long)]
    jepa_ema_decay: Option<f32>,
    /// Phase 12 S1: optimizer choice. `adam` (default) or `muon`.
    /// Muon is the DeepSeek V4 style Newton-Schulz orthogonalized
    /// SGD-momentum optimizer.
    #[arg(long, default_value = "adam")]
    optimizer: String,
    #[arg(long, default_value_t = 200)]
    sample_tokens: usize,
    #[arg(long, default_value = "대한민국의 수도는 ")]
    sample_prompt: String,
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

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    let args = Args::parse();
    let device = pick_device();
    tracing::info!(?device, "device");

    let tk = if args.retrain_tokenizer || !args.tokenizer.exists() {
        if let Some(parent) = args.tokenizer.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Tokenizer::train_bpe(
            std::slice::from_ref(&args.data),
            args.vocab_size,
            args.tokenizer.clone(),
        )?
    } else {
        Tokenizer::from_hf_file(&args.tokenizer)?
    };
    let vocab = tk.vocab_size();
    tracing::info!(vocab, "tokenizer ready");

    let text = std::fs::read_to_string(&args.data)?;
    let ids = tk.encode(&text)?;
    tracing::info!(tokens = ids.len(), "corpus encoded");
    let ds = TokenDataset::new(ids, args.block_size);

    let mut gpt_cfg = GPTConfig::nano_50m();
    gpt_cfg.vocab_size = vocab;
    gpt_cfg.block_size = args.block_size;
    tracing::info!(
        params = gpt_cfg.num_params_estimate(),
        n_layer = gpt_cfg.n_layer,
        n_embd = gpt_cfg.n_embd,
        block_size = gpt_cfg.block_size,
        "model config"
    );

    let mut tcfg = TrainConfig::smoke();
    tcfg.max_steps = args.steps;
    tcfg.batch_size = args.batch_size;
    tcfg.eval_interval = args.eval_interval;
    tcfg.lr = args.lr;
    tcfg.min_lr = args.lr * 0.1;
    tcfg.warmup_steps = (args.steps / 30).max(50);
    tcfg.weight_decay = 0.1;
    tcfg.jepa_lambda = args.jepa_lambda;
    tcfg.jepa_offset = args.jepa_offset;
    tcfg.jepa_ema_decay = args.jepa_ema_decay;
    tcfg.optimizer = match args.optimizer.to_lowercase().as_str() {
        "muon" => OptimizerKind::Muon,
        "adam" | "adamw" => OptimizerKind::Adam,
        other => anyhow::bail!("unknown optimizer {other:?}; expected adam or muon"),
    };
    tracing::info!(
        steps = tcfg.max_steps,
        lr = tcfg.lr,
        jepa_lambda = tcfg.jepa_lambda,
        jepa_offset = tcfg.jepa_offset,
        jepa_ema_decay = ?tcfg.jepa_ema_decay,
        optimizer = ?tcfg.optimizer,
        "training..."
    );

    let outcome = train_from_full(
        &gpt_cfg,
        &ds,
        None,
        &tcfg,
        &device,
        Some(&args.save),
        None,
        None,
        false,
    )?;
    tracing::info!(
        train_loss = outcome.last_train_loss,
        steps = outcome.final_step,
        "training done"
    );

    // ---- Reload + measure mode-collapse signal + sample.
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = GPT::new(gpt_cfg.clone(), vb)?;
    varmap.load(&args.save)?;

    // Mode-collapse metric: top-1 softmax mass on a random batch from
    // the training set. Anti-pattern: top1 mass ≫ 1/vocab implies the
    // model has collapsed onto a small set of tokens (the K8 100K `\n`
    // pathology). Healthy: top1 mass roughly comparable to 1/vocab to
    // a few-× higher.
    let (xv, _yv) = ds.random_batch(args.batch_size, &device)?;
    let logits = model.forward(&xv)?;
    let top1 = top1_mass(&logits)?;
    let uniform = 1.0 / vocab as f32;
    println!("\n=== Phase 10 S1 — mode-collapse metric ===");
    println!(
        "top-1 softmax mass: {top1:.4}  (uniform = {uniform:.4}; ratio = {:.1}×)",
        top1 / uniform
    );

    let prompt_ids = tk.encode(&args.sample_prompt)?;
    let cfg = GenerateConfig {
        max_new_tokens: args.sample_tokens,
        temperature: 0.8,
        top_k: Some(40),
        top_p: Some(0.9),
        seed: Some(42),
    };
    let out_ids = generate(&model, &prompt_ids, &cfg, &device)?;
    let out_text = tk.decode(&out_ids)?;
    println!(
        "\n=== sample (prompt: {:?}) ===\n{out_text}\n=== end ===",
        args.sample_prompt
    );

    Ok(())
}
