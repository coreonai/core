//! Phase 24 — format-SFT for Pekko/MSA agent harvest families.
//!
//! Trains completion-only CE on `scripts/phase24/fmt_seed_pairs.jsonl`
//! (prompt → reference completion, multiple phrasings per family). Same
//! trainer path as `phase23_toolcall_sft`: LoRA on Qwen2.5-Coder-7B, then
//! merge to a safetensors checkpoint.
//!
//! Init is the **base** coder weights, not `p23_py_sft` — that checkpoint is
//! specialized to `(python …)` tool calls and fights Rust/Pekko completions.
//!
//! ```text
//! cargo run -p llm-actors --example phase24_fmt_sft --features cuda --release -- \
//!     --pairs scripts/phase24/fmt_seed_pairs.jsonl \
//!     --train-steps 200 \
//!     --out scratch-7b-sft/p24_fmt_sft.safetensors
//! ```

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device};
use clap::Parser;
use llm_actors::{qwen2_lora::LoraConfig, QwenTrainerActor, QwenTrainerMessage};
use pekko_actor::ActorSystem;
use serde::Deserialize;
use tokio::sync::oneshot;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    model_dir: Option<PathBuf>,
    #[arg(long, default_value = "Qwen2.5-Coder-7B")]
    model_id: String,
    /// JSONL with {prompt, completion, family?}. Literal `\n` in strings is
    /// unescaped to real newlines on load.
    #[arg(long, default_value = "scripts/phase24/fmt_seed_pairs.jsonl")]
    pairs: PathBuf,
    #[arg(long, default_value_t = 200)]
    train_steps: usize,
    #[arg(long, default_value_t = 2e-4)]
    lr: f64,
    #[arg(long, default_value_t = 16)]
    lora_rank: usize,
    #[arg(long, default_value_t = 32.0)]
    lora_alpha: f32,
    #[arg(long, default_value_t = 4)]
    batch_size: usize,
    #[arg(long, default_value = "scratch-7b-sft/p24_fmt_sft.safetensors")]
    out: PathBuf,
    /// Hold out this fraction of pairs (by line order) for a quick smoke.
    #[arg(long, default_value_t = 0.15)]
    holdout: f64,
}

#[derive(Debug, Deserialize)]
struct SeedPair {
    prompt: String,
    completion: String,
    #[serde(default)]
    family: Option<String>,
}

fn unescape_newlines(s: &str) -> String {
    s.replace("\\n", "\n").replace("\\t", "\t")
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
        let fam = row.family.unwrap_or_else(|| "unknown".into());
        out.push((
            fam,
            unescape_newlines(&row.prompt),
            unescape_newlines(&row.completion),
        ));
    }
    if out.is_empty() {
        anyhow::bail!("no pairs in {}", path.display());
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

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .init();
    let args = Args::parse();
    let device = pick_device();
    let on_cuda = device.is_cuda();
    println!("[Phase24SFT] device = {device:?}, on_cuda = {on_cuda}");
    if !on_cuda && std::env::var("PHASE22_ALLOW_CPU").is_err() {
        anyhow::bail!("Refusing to run on CPU. Rebuild with `--features cuda`.");
    }

    let all = load_pairs(&args.pairs)?;
    let n_hold = ((all.len() as f64) * args.holdout).round() as usize;
    let n_hold = n_hold.max(1).min(all.len().saturating_sub(1));
    let (hold, train) = all.split_at(n_hold);
    let pairs: Vec<(String, String)> = train
        .iter()
        .map(|(_, p, c)| (p.clone(), c.clone()))
        .collect();
    println!(
        "[Phase24SFT] {} train / {} holdout from {} (holdout first {} lines)",
        pairs.len(),
        hold.len(),
        args.pairs.display(),
        n_hold
    );
    for (fam, _, _) in hold {
        println!("[Phase24SFT] holdout family sample: {fam}");
    }
    println!(
        "[Phase24SFT] sample: {:?} -> {:?}",
        pairs[0].0.chars().take(60).collect::<String>(),
        pairs[0].1.chars().take(60).collect::<String>()
    );

    let snapshot = resolve_snapshot(args.model_dir.as_deref(), &args.model_id)?;
    println!("[Phase24SFT] snapshot = {}", snapshot.display());

    let trainer = QwenTrainerActor::from_snapshot_dir(
        &snapshot,
        device.clone(),
        DType::BF16,
        LoraConfig {
            rank: args.lora_rank,
            alpha: args.lora_alpha,
        },
        args.lr,
    )?
    .with_sft_batch_size(args.batch_size)
    .with_fresh_optimizer(true);

    let system = ActorSystem::new("phase24-fmt-sft");
    let trainer_ref = system.spawn(trainer, "qwen-trainer").await?;

    println!("[Phase24SFT] training {} steps...", args.train_steps);
    let t0 = std::time::Instant::now();
    let (tx, rx) = oneshot::channel();
    trainer_ref
        .tell(QwenTrainerMessage::TrainSftPairs {
            pairs,
            train_steps: args.train_steps,
            reply: tx,
        })
        .map_err(|e| anyhow!("{e:?}"))?;
    let outcome = rx.await??;
    println!(
        "[Phase24SFT] loss {:.4} -> {:.4} in {:.1}s",
        outcome.initial_loss,
        outcome.final_loss,
        t0.elapsed().as_secs_f64()
    );

    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let (tx, rx) = oneshot::channel();
    trainer_ref
        .tell(QwenTrainerMessage::SaveMergedCheckpoint {
            base_path: snapshot.clone(),
            out_path: args.out.clone(),
            reply: tx,
        })
        .map_err(|e| anyhow!("{e:?}"))?;
    rx.await??;
    println!("[Phase24SFT] merged checkpoint -> {}", args.out.display());
    Ok(())
}
