//! Phase 21 Stage D — Candle-native Qwen2.5-Coder-0.5B smoke.
//!
//! Loads the SAME `Qwen2.5-Coder-0.5B` checkpoint that Phase 14–20's
//! Python scripts used (cached at
//! `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B`),
//! tokenizes a prompt with the HF `tokenizers` crate, runs
//! greedy + temperature sampling through `candle_transformers::models::qwen2::ModelForCausalLM`,
//! and prints the generated text.
//!
//! This is **Stage D's proof-of-concept** that Candle can serve the
//! Phase 14–20 model **natively in Rust, with no Python sidecar**.
//! Full `QwenModelActor` integration (trait-ifying `ModelActor` so it
//! can hold any LM) is the follow-on. This binary proves the loading
//! + tokenization + sampling path works end-to-end.
//!
//! Run:
//!   cargo run -p llm-actors --example phase21_qwen_candle_smoke \
//!       --features cuda --release -- \
//!       --prompt 'def fibonacci(n):' --max-new-tokens 32
//!
//! Or with the path explicit:
//!   --model-dir /home/paulyu/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B/snapshots/<sha>
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::{Config as Qwen2Config, ModelForCausalLM};
use clap::Parser;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
struct Args {
    /// HF snapshot directory holding config.json + tokenizer.json + model.safetensors.
    /// Defaults to the snapshot under ~/.cache that Phase 14-20 already
    /// downloaded.
    #[arg(long)]
    model_dir: Option<PathBuf>,
    #[arg(long, default_value = "def fibonacci(n):")]
    prompt: String,
    #[arg(long, default_value_t = 32)]
    max_new_tokens: usize,
    /// 0.0 = greedy argmax. > 0 enables temperature sampling.
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Use the f16 dtype on GPU. Falls back to f32 on CPU.
    #[arg(long, default_value_t = true)]
    f16: bool,
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

fn resolve_default_snapshot() -> Result<PathBuf> {
    let home = std::env::var("HOME").context("HOME unset")?;
    let snapshots_dir = PathBuf::from(format!(
        "{home}/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B/snapshots"
    ));
    let entries = std::fs::read_dir(&snapshots_dir)
        .with_context(|| format!("read_dir {snapshots_dir:?}"))?
        .collect::<Result<Vec<_>, _>>()?;
    entries
        .into_iter()
        .map(|e| e.path())
        .find(|p| p.is_dir() && p.join("config.json").exists())
        .ok_or_else(|| anyhow!("no snapshot under {snapshots_dir:?} has a config.json"))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = pick_device();
    let on_cuda = device.is_cuda();
    println!("[Phase21D] device = {:?}, on_cuda = {on_cuda}", device);

    let dir = match args.model_dir {
        Some(p) => p,
        None => resolve_default_snapshot()?,
    };
    println!("[Phase21D] model_dir = {}", dir.display());

    // --- Tokenizer
    let tokenizer =
        Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("tokenizer: {e}"))?;
    println!(
        "[Phase21D] tokenizer loaded (vocab={})",
        tokenizer.get_vocab_size(true)
    );

    // --- Config
    let cfg_text = std::fs::read_to_string(dir.join("config.json")).context("read config.json")?;
    let cfg: Qwen2Config = serde_json::from_str(&cfg_text).context("parse Qwen2Config")?;
    println!(
        "[Phase21D] config loaded (hidden={}, layers={}, heads={}/{}, vocab={})",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.vocab_size,
    );

    // --- Model
    let dtype = if on_cuda && args.f16 {
        DType::F16
    } else {
        DType::F32
    };
    let safetensors = dir.join("model.safetensors");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&safetensors], dtype, &device)? };
    let mut model = ModelForCausalLM::new(&cfg, vb)?;
    println!("[Phase21D] model loaded (dtype={:?})", dtype);

    // --- Tokenize prompt
    let encoded = tokenizer
        .encode(args.prompt.as_str(), true)
        .map_err(|e| anyhow!("encode: {e}"))?;
    let prompt_ids: Vec<u32> = encoded.get_ids().to_vec();
    println!(
        "[Phase21D] prompt = {:?} ({} tokens)",
        args.prompt,
        prompt_ids.len()
    );

    // --- Autoregressive generate
    let mut tokens = prompt_ids.clone();
    let mut seqlen_offset: usize = 0;
    let mut rng = rand_via_xorshift(args.seed);
    let prompt_chunk_len = tokens.len();

    // Prime the model with the full prompt (offset=0, all tokens).
    let mut logits = forward_step(&mut model, &tokens, seqlen_offset, &device)?;
    seqlen_offset += prompt_chunk_len;

    let mut generated_ids: Vec<u32> = Vec::with_capacity(args.max_new_tokens);
    for _ in 0..args.max_new_tokens {
        let next = sample_next(&logits, args.temperature, &mut rng)?;
        if next as usize == cfg.vocab_size {
            break;
        }
        generated_ids.push(next);
        tokens.push(next);
        // Subsequent steps feed just the new token with the offset.
        logits = forward_step(&mut model, &[next], seqlen_offset, &device)?;
        seqlen_offset += 1;
    }

    let generated_text = tokenizer
        .decode(&generated_ids, true)
        .map_err(|e| anyhow!("decode: {e}"))?;
    println!("\n=== Generated ===\n{}\n=== End ===", generated_text);
    println!(
        "[Phase21D] generated {} tokens, total {} tokens",
        generated_ids.len(),
        tokens.len(),
    );
    println!("phase21_qwen_candle_smoke: PASS");
    Ok(())
}

fn forward_step(
    model: &mut ModelForCausalLM,
    chunk: &[u32],
    seqlen_offset: usize,
    device: &Device,
) -> Result<Tensor> {
    let input = Tensor::from_slice(chunk, (1, chunk.len()), device)?;
    let logits = model.forward(&input, seqlen_offset)?; // (1, 1, vocab)
    let logits = logits.squeeze(0)?.squeeze(0)?; // (vocab,)
    Ok(logits.to_dtype(DType::F32)?)
}

fn sample_next(logits: &Tensor, temperature: f64, rng: &mut Xorshift) -> Result<u32> {
    if temperature <= 0.0 {
        let v = logits.to_vec1::<f32>()?;
        let argmax = v
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        return Ok(argmax);
    }
    let logits = (logits / temperature)?;
    let probs = candle_nn::ops::softmax_last_dim(&logits)?;
    let probs_v: Vec<f32> = probs.to_vec1()?;
    let u = rng.next_f32();
    let mut acc = 0.0f32;
    for (i, &p) in probs_v.iter().enumerate() {
        acc += p;
        if u <= acc {
            return Ok(i as u32);
        }
    }
    Ok((probs_v.len() - 1) as u32)
}

/// Tiny seedable xorshift32 — keeps the example self-contained and
/// avoids pulling rand_chacha into the workspace just for this PoC.
struct Xorshift(u32);
impl Xorshift {
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f64 / u32::MAX as f64) as f32
    }
}
fn rand_via_xorshift(seed: u64) -> Xorshift {
    let mut s = (seed as u32) ^ 0x9E3779B9;
    if s == 0 {
        s = 1;
    }
    Xorshift(s)
}

#[allow(dead_code)]
fn _ensure_path_exists(p: &Path) -> Result<()> {
    if !p.exists() {
        anyhow::bail!("path does not exist: {}", p.display());
    }
    Ok(())
}
