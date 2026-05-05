//! Held-out cross-entropy evaluation for one or more KoWiki checkpoints.
//!
//! Loads each checkpoint, slices off the tail of the corpus as a held-out
//! split, and reports mean CE + perplexity over `--eval-batches` random
//! windows. Designed for apples-to-apples comparison after distillation:
//! the distilled student's training loss carries a `T²·KL` term that
//! inflates it relative to a from-scratch baseline's pure CE; running this
//! eval on the same data + same metric removes the metric mismatch.
//!
//! Each checkpoint is paired with its `*.cfg.json` (saved by `train_from`),
//! so different architectures (50M teacher, 12M student, etc.) can be
//! compared in one pass.
//!
//! Run:
//!   cargo run -p nanogpt-rs --example eval_kowiki --features cuda --release -- \
//!       --tokenizer data/kowiki/kowiki_bpe.json \
//!       --data data/kowiki/kowiki_clean.txt \
//!       --val-frac 0.05 \
//!       --eval-batches 50 \
//!       --checkpoints checkpoints/kowiki_50m_clean.safetensors \
//!       --checkpoints checkpoints/kowiki_distill_student.safetensors \
//!       --checkpoints checkpoints/kowiki_distill_baseline.safetensors

use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use clap::Parser;
use nanogpt_rs::{config::GPTConfig, data::TokenDataset, model::GPT, tokenizer::Tokenizer};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    tokenizer: PathBuf,
    #[arg(long)]
    data: PathBuf,
    /// Fraction of the corpus reserved for evaluation (taken from the
    /// tail). Each checkpoint sees the SAME held-out chunk.
    #[arg(long, default_value_t = 0.05)]
    val_frac: f32,
    #[arg(long, default_value_t = 50)]
    eval_batches: usize,
    #[arg(long, default_value_t = 16)]
    batch_size: usize,
    /// One or more `.safetensors` checkpoints. Each must have a sibling
    /// `.cfg.json` describing the architecture (saved by `train_from`).
    #[arg(long, num_args = 1..)]
    checkpoints: Vec<PathBuf>,
    #[arg(long, default_value_t = 0xE0)]
    seed: u64,
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

fn load_cfg(checkpoint: &Path) -> anyhow::Result<GPTConfig> {
    let cfg_path = checkpoint.with_extension("cfg.json");
    let s = std::fs::read_to_string(&cfg_path)
        .map_err(|e| anyhow::anyhow!("read {:?}: {e}", cfg_path))?;
    let cfg: GPTConfig = serde_json::from_str(&s)?;
    Ok(cfg)
}

fn eval_one(
    checkpoint: &Path,
    val_ds: &TokenDataset,
    args: &Args,
    device: &Device,
) -> anyhow::Result<(f32, f32)> {
    let cfg = load_cfg(checkpoint)?;
    let mut varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = GPT::new(cfg.clone(), vb)?;
    varmap.load(checkpoint)?;

    let mut total = 0.0f32;
    let mut count = 0usize;
    for _ in 0..args.eval_batches {
        let (x, y) = val_ds.random_batch(args.batch_size, device)?;
        let loss = model.loss(&x, &y)?.to_scalar::<f32>()?;
        if loss.is_finite() {
            total += loss;
            count += 1;
        }
    }
    let mean_loss = if count > 0 {
        total / count as f32
    } else {
        f32::NAN
    };
    let ppl = mean_loss.exp();
    Ok((mean_loss, ppl))
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    let args = Args::parse();
    if args.checkpoints.is_empty() {
        anyhow::bail!("need at least one --checkpoints arg");
    }
    let device = pick_device();
    tracing::info!(?device, "device");

    // Tokenize once, share across checkpoints.
    let tk = Tokenizer::from_hf_file(&args.tokenizer)?;
    let vocab = tk.vocab_size();
    tracing::info!(vocab, "tokenizer ready");
    let text = std::fs::read_to_string(&args.data)?;
    let ids = tk.encode(&text)?;
    let n = ids.len();
    let split = ((1.0 - args.val_frac) * n as f32) as usize;
    let val_ids: Vec<u32> = ids[split..].to_vec();
    tracing::info!(
        train_tokens = split,
        val_tokens = val_ids.len(),
        val_frac = args.val_frac,
        "split"
    );

    println!(
        "\n{:<60} {:>14} {:>14} {:>14}",
        "checkpoint", "val_loss", "perplexity", "params"
    );
    println!("{}", "-".repeat(60 + 14 * 3 + 3));

    // The block size is per-checkpoint (different architectures may have
    // different block sizes); build the val dataset per-checkpoint with
    // the right block_size.
    for ckpt in &args.checkpoints {
        let cfg = match load_cfg(ckpt) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skip {:?}: {e}", ckpt);
                continue;
            }
        };
        let val_ds = TokenDataset::new(val_ids.clone(), cfg.block_size);
        match eval_one(ckpt, &val_ds, &args, &device) {
            Ok((loss, ppl)) => println!(
                "{:<60} {:>14.4} {:>14.2} {:>14}",
                ckpt.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
                loss,
                ppl,
                cfg.num_params_estimate()
            ),
            Err(e) => eprintln!("eval {:?} failed: {e}", ckpt),
        }
    }
    println!();
    Ok(())
}
