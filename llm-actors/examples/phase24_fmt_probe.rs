//! Phase 24 — generation probe for `p24_fmt_sft`.
//!
//! Loads the JSONL seed pairs, generates greedy completions from a merged
//! checkpoint, and reports exact / normalized match rates on:
//!   - holdout (first `--holdout` fraction of lines — same split as SFT)
//!   - train (the rest — memorisation check)
//!
//! ```text
//! cargo run -p llm-actors --example phase24_fmt_probe --features cuda --release -- \
//!     --checkpoint scratch-7b-sft/p24_fmt_sft.safetensors
//! ```

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device};
use clap::Parser;
use llm_actors::{ModelMessage, QwenModelActor};
use nanogpt_rs::{generate::GenerateConfig, Tokenizer as NgptTokenizer};
use pekko_actor::ActorSystem;
use serde::Deserialize;
use tokio::sync::oneshot;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    model_dir: Option<PathBuf>,
    #[arg(long, default_value = "Qwen2.5-Coder-7B")]
    model_id: String,
    #[arg(long, default_value = "scripts/phase24/fmt_seed_pairs.jsonl")]
    pairs: PathBuf,
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long, default_value_t = 0.15)]
    holdout: f64,
    #[arg(long, default_value_t = 256)]
    max_new_tokens: usize,
    #[arg(long, default_value = "f16")]
    dtype: String,
    #[arg(long, default_value_t = 8)]
    show: usize,
}

#[derive(Debug, Deserialize)]
struct SeedPair {
    prompt: String,
    completion: String,
    #[serde(default)]
    family: Option<String>,
}

fn unescape(s: &str) -> String {
    s.replace("\\n", "\n").replace("\\t", "\t")
}

fn norm(s: &str) -> String {
    s.chars()
        .filter(|c| !c.is_whitespace())
        .collect::<String>()
        .to_lowercase()
}

fn load_pairs(path: &std::path::Path) -> Result<Vec<(String, String, String)>> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut out = Vec::new();
    for (i, line) in BufReader::new(f).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: SeedPair =
            serde_json::from_str(&line).with_context(|| format!("jsonl line {}", i + 1))?;
        out.push((
            row.family.unwrap_or_else(|| "unknown".into()),
            unescape(&row.prompt),
            unescape(&row.completion),
        ));
    }
    Ok(out)
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

fn resolve_snapshot(dir: Option<&std::path::Path>, id: &str) -> Result<PathBuf> {
    if let Some(d) = dir {
        return Ok(d.to_path_buf());
    }
    let home = std::env::var("HOME").context("HOME unset")?;
    let snaps = PathBuf::from(format!(
        "{home}/.cache/huggingface/hub/models--Qwen--{id}/snapshots"
    ));
    std::fs::read_dir(&snaps)
        .with_context(|| format!("read_dir {snaps:?}"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.is_dir() && p.join("config.json").exists())
        .ok_or_else(|| anyhow!("no snapshot under {snaps:?}"))
}

fn dtype_arg(s: &str) -> Result<DType> {
    match s {
        "f16" | "F16" => Ok(DType::F16),
        "bf16" | "BF16" => Ok(DType::BF16),
        "f32" | "F32" => Ok(DType::F32),
        other => anyhow::bail!("bad --dtype {other}"),
    }
}

struct Score {
    n: usize,
    exact: usize,
    soft: usize,
}

impl Score {
    fn new() -> Self {
        Self {
            n: 0,
            exact: 0,
            soft: 0,
        }
    }
    fn add(&mut self, got: &str, want: &str) {
        self.n += 1;
        if got.trim() == want.trim() {
            self.exact += 1;
        }
        if !want.trim().is_empty() && norm(got).contains(&norm(want)) {
            self.soft += 1;
        }
    }
    fn report(&self, label: &str) {
        let e = if self.n == 0 {
            0.0
        } else {
            self.exact as f64 / self.n as f64
        };
        let s = if self.n == 0 {
            0.0
        } else {
            self.soft as f64 / self.n as f64
        };
        println!(
            "[Phase24Probe] {label}: n={} exact={:.1}% ({}/{}) soft-contain={:.1}% ({}/{})",
            self.n,
            100.0 * e,
            self.exact,
            self.n,
            100.0 * s,
            self.soft,
            self.n
        );
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .init();
    let args = Args::parse();
    let device = pick_device();
    if !device.is_cuda() && std::env::var("PHASE22_ALLOW_CPU").is_err() {
        anyhow::bail!("Refusing CPU. Rebuild with --features cuda.");
    }
    let dtype = dtype_arg(&args.dtype)?;
    let snapshot = resolve_snapshot(args.model_dir.as_deref(), &args.model_id)?;
    println!(
        "[Phase24Probe] device={device:?} dtype={dtype:?} ckpt={}",
        args.checkpoint.display()
    );

    let all = load_pairs(&args.pairs)?;
    let n_hold = ((all.len() as f64) * args.holdout).round() as usize;
    let n_hold = n_hold.max(1).min(all.len().saturating_sub(1));
    let (hold, train) = all.split_at(n_hold);
    println!(
        "[Phase24Probe] {} holdout / {} train from {}",
        hold.len(),
        train.len(),
        args.pairs.display()
    );

    let tk = Arc::new(NgptTokenizer::from_hf_file(
        snapshot.join("tokenizer.json"),
    )?);
    let cfg_text = std::fs::read_to_string(snapshot.join("config.json"))?;
    let config: candle_transformers::models::qwen2::Config = serde_json::from_str(&cfg_text)?;
    let tokenizer = tokenizers::Tokenizer::from_file(snapshot.join("tokenizer.json"))
        .map_err(|e| anyhow!("tokenizer: {e}"))?;
    let model = QwenModelActor::new(
        args.checkpoint.clone(),
        Arc::new(tokenizer),
        config,
        device.clone(),
        dtype,
    )?;
    let system = ActorSystem::new("phase24-fmt-probe");
    let model_ref = system.spawn(model, "qwen-model").await?;

    let mut score_hold = Score::new();
    let mut score_train = Score::new();
    let mut shown = 0usize;

    for (side, rows, score) in [
        ("holdout", hold, &mut score_hold),
        ("train", train, &mut score_train),
    ] {
        for (fi, (fam, prompt, want)) in rows.iter().enumerate() {
            let prompt_ids = tk.encode(prompt)?;
            let cfg = GenerateConfig {
                max_new_tokens: args.max_new_tokens,
                temperature: 0.0,
                top_k: None,
                top_p: None,
                seed: Some(7u64.wrapping_add(fi as u64)),
            };
            let (tx, rx) = oneshot::channel();
            model_ref
                .tell(ModelMessage::GenerateTokens {
                    prompt_ids: prompt_ids.clone(),
                    cfg,
                    reply: tx,
                })
                .map_err(|e| anyhow!("{e:?}"))?;
            let full = rx.await??;
            let comp_ids = if full.len() > prompt_ids.len() {
                &full[prompt_ids.len()..]
            } else {
                &[][..]
            };
            let got = tk.decode(comp_ids)?;
            score.add(&got, want);
            if shown < args.show {
                println!(
                    "  [{side}/{fam}] want {:?} got {:?}",
                    want.chars().take(80).collect::<String>(),
                    got.chars().take(80).collect::<String>()
                );
                shown += 1;
            }
        }
    }

    score_hold.report("holdout");
    score_train.report("train (memorisation)");
    Ok(())
}
