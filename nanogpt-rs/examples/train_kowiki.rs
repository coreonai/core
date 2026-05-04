//! KoWiki BPE + 50M GPT training pipeline.
//!
//! Steps:
//!   1. Train a fresh ByteLevel-BPE tokenizer on the extracted plaintext
//!      (or reuse an existing one with `--tokenizer`).
//!   2. Tokenize the corpus into a single `Vec<u32>`.
//!   3. Build a Phase-3-Llama-recipe ~50M GPT (vocab from the tokenizer).
//!   4. Train via `train_from` with cosine LR schedule.
//!   5. Sample Korean continuations from given prompts.
//!
//! Pipeline (one-liner):
//!   bzcat data/kowiki/kowiki-latest-pages-articles.xml.bz2 \
//!     | cargo run -p nanogpt-rs --example extract_kowiki --release \
//!     > data/kowiki/kowiki_plain.txt
//!   cargo run -p nanogpt-rs --example train_kowiki --release --features cuda \
//!     -- --data data/kowiki/kowiki_plain.txt --steps 5000

use std::path::PathBuf;

use candle_core::{DType, Device};
use clap::Parser;
use nanogpt_rs::{
    config::GPTConfig,
    data::TokenDataset,
    generate::{generate, GenerateConfig},
    model::GPT,
    tokenizer::Tokenizer,
    train::{train_from, TrainConfig},
};

#[derive(Parser, Debug)]
struct Args {
    /// Plaintext corpus (output of `extract_kowiki`).
    #[arg(long, default_value = "data/kowiki/kowiki_plain.txt")]
    data: PathBuf,
    /// Where to write / read the BPE tokenizer (HF JSON).
    #[arg(long, default_value = "data/kowiki/kowiki_bpe.json")]
    tokenizer: PathBuf,
    /// If true, train a fresh BPE on `--data` even if `--tokenizer` exists.
    #[arg(long)]
    retrain_tokenizer: bool,
    #[arg(long, default_value_t = 16_000)]
    vocab_size: usize,
    #[arg(long, default_value = "checkpoints/kowiki_50m.safetensors")]
    save: PathBuf,
    #[arg(long, default_value_t = 3000)]
    steps: usize,
    #[arg(long, default_value_t = 16)]
    batch_size: usize,
    #[arg(long, default_value_t = 6e-4)]
    lr: f64,
    #[arg(long, default_value_t = 200)]
    eval_interval: usize,
    /// Override block_size from the model preset (set lower if VRAM is tight).
    #[arg(long, default_value_t = 256)]
    block_size: usize,
    #[arg(long, default_value_t = 200)]
    sample_tokens: usize,
    /// Default sampling prompt — overridable on CLI.
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

    // ---- Tokenizer: train or reuse.
    let tk = if args.retrain_tokenizer || !args.tokenizer.exists() {
        if let Some(parent) = args.tokenizer.parent() {
            std::fs::create_dir_all(parent)?;
        }
        tracing::info!(?args.data, vocab = args.vocab_size, "training BPE tokenizer");
        Tokenizer::train_bpe(&[args.data.clone()], args.vocab_size, args.tokenizer.clone())?
    } else {
        tracing::info!(?args.tokenizer, "reusing existing tokenizer");
        Tokenizer::from_hf_file(&args.tokenizer)?
    };
    let vocab = tk.vocab_size();
    tracing::info!(vocab, "tokenizer ready");

    // ---- Tokenize the whole corpus once.
    tracing::info!("reading corpus...");
    let text = std::fs::read_to_string(&args.data)?;
    tracing::info!(chars = text.len(), "corpus loaded");
    tracing::info!("encoding corpus to ids (this is the slow part)...");
    let ids = tk.encode(&text)?;
    tracing::info!(tokens = ids.len(), "corpus encoded");
    let ds = TokenDataset::new(ids, args.block_size);

    // ---- Phase 3 Llama-recipe at ~50M params, vocab patched in.
    let mut gpt_cfg = GPTConfig::nano_50m();
    gpt_cfg.vocab_size = vocab;
    gpt_cfg.block_size = args.block_size;
    tracing::info!(
        params = gpt_cfg.num_params_estimate(),
        n_layer = gpt_cfg.n_layer,
        n_embd = gpt_cfg.n_embd,
        n_kv_head = gpt_cfg.n_kv_head,
        block_size = gpt_cfg.block_size,
        "model config (RoPE + GQA + SwiGLU + RmsNorm-Pre + untied)"
    );

    // ---- Training config.
    let mut tcfg = TrainConfig::smoke();
    tcfg.max_steps = args.steps;
    tcfg.batch_size = args.batch_size;
    tcfg.eval_interval = args.eval_interval;
    tcfg.lr = args.lr;
    tcfg.min_lr = args.lr * 0.1;
    tcfg.warmup_steps = (args.steps / 30).max(50);
    tcfg.weight_decay = 0.1;
    tracing::info!(steps = tcfg.max_steps, lr = tcfg.lr, "training...");

    let outcome = train_from(
        &gpt_cfg,
        &ds,
        None,
        &tcfg,
        &device,
        Some(&args.save),
        None,
    )?;
    tracing::info!(
        train_loss = outcome.last_train_loss,
        steps = outcome.final_step,
        "training done"
    );

    // ---- Reload + sample to confirm the saved model rounds-trips.
    let mut varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = GPT::new(gpt_cfg.clone(), vb)?;
    varmap.load(&args.save)?;

    let prompt_ids = tk.encode(&args.sample_prompt)?;
    let cfg = GenerateConfig {
        max_new_tokens: args.sample_tokens,
        temperature: 0.8,
        top_k: Some(40),
        top_p: Some(0.9),
        seed: None,
    };
    let out_ids = generate(&model, &prompt_ids, &cfg, &device)?;
    let text = tk.decode(&out_ids)?;
    println!("\n=== sample (prompt: {:?}) ===\n{text}\n=== end ===", args.sample_prompt);
    Ok(())
}
