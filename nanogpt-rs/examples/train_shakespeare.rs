//! Char-level Shakespeare smoke run.
//!
//! Expects `data/input.txt` (Karpathy's nanoGPT shakespeare dataset). To fetch:
//!   curl -L -o data/input.txt \
//!     https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
//!
//! Run:
//!   cargo run -p nanogpt-rs --example train_shakespeare --release -- \
//!       --steps 500 --batch-size 64
//! With CUDA:
//!   cargo run -p nanogpt-rs --example train_shakespeare --features cuda --release -- ...

use std::path::PathBuf;

use candle_core::{DType, Device};
use clap::Parser;
use nanogpt_rs::{
    config::GPTConfig,
    data::TokenDataset,
    generate::{generate, GenerateConfig},
    model::GPT,
    tokenizer::Tokenizer,
    train::{train, TrainConfig},
};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "data/input.txt")]
    data: PathBuf,
    #[arg(long, default_value = "data/shakespeare.tok.json")]
    tok_path: PathBuf,
    #[arg(long, default_value = "checkpoints/shakespeare.safetensors")]
    save: PathBuf,
    #[arg(long, default_value_t = 500)]
    steps: usize,
    #[arg(long, default_value_t = 64)]
    batch_size: usize,
    #[arg(long, default_value_t = 50)]
    eval_interval: usize,
    #[arg(long, default_value_t = 3e-4)]
    lr: f64,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value = "ROMEO:")]
    sample_prompt: String,
    #[arg(long, default_value_t = 200)]
    sample_tokens: usize,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).init();
    let args = Args::parse();

    let device = pick_device();
    tracing::info!(?device, "device selected");

    let text = std::fs::read_to_string(&args.data)
        .map_err(|e| anyhow::anyhow!("read {:?}: {} (download instructions in file header)", args.data, e))?;
    tracing::info!(chars = text.len(), "loaded corpus");

    let tk = Tokenizer::char_from_text(&text);
    let vocab = tk.vocab_size();
    tracing::info!(vocab, "char tokenizer built");
    if let Tokenizer::Char(c) = &tk {
        if let Some(parent) = args.tok_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        c.save(&args.tok_path)?;
    }

    let ids = tk.encode(&text)?;
    let ds = TokenDataset::new(ids, GPTConfig::shakespeare_char(vocab).block_size);
    let (train_ds, val_ds) = ds.split_train_val(0.05);
    tracing::info!(train_tokens = train_ds.tokens.len(), val_tokens = val_ds.tokens.len(), "split");

    let gpt_cfg = GPTConfig::shakespeare_char(vocab);
    tracing::info!(
        params_estimate = gpt_cfg.num_params_estimate(),
        block_size = gpt_cfg.block_size,
        n_layer = gpt_cfg.n_layer,
        n_embd = gpt_cfg.n_embd,
        "model config"
    );

    let mut tcfg = TrainConfig::smoke();
    tcfg.max_steps = args.steps;
    tcfg.batch_size = args.batch_size;
    tcfg.eval_interval = args.eval_interval;
    tcfg.lr = args.lr;
    tcfg.min_lr = args.lr * 0.1;

    let outcome = train(&gpt_cfg, &train_ds, Some(&val_ds), &tcfg, &device, Some(&args.save))?;
    tracing::info!(
        train = outcome.last_train_loss,
        val = ?outcome.last_val_loss,
        "training done"
    );

    // Reload + sample to confirm save/load + generate path works
    let mut varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = GPT::new(gpt_cfg.clone(), vb)?;
    varmap.load(&args.save)?;

    let prompt_ids = tk.encode(&args.sample_prompt)?;
    let cfg = GenerateConfig {
        max_new_tokens: args.sample_tokens,
        temperature: 0.8,
        top_k: Some(40),
        top_p: None,
        seed: if args.seed == 0 { None } else { Some(args.seed) },
    };
    let out_ids = generate(&model, &prompt_ids, &cfg, &device)?;
    let text = tk.decode(&out_ids)?;
    println!("\n=== sample ===\n{text}\n=== end ===");
    Ok(())
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
