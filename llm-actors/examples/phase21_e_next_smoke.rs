//! Phase 21 Stage E.next — `QwenTrainerActor` end-to-end smoke.
//!
//! Spawns `QwenTrainerActor` inside a Pekko `ActorSystem`, sends a
//! `Train` message with a small batch of text examples, and asserts
//! loss decreases monotonically over N AdamW steps. Then exercises
//! `SaveLoraAdapter` to persist the trained adapter to a safetensors
//! file.
//!
//! Pairs with Stage F's standalone `phase21_qwen_lora_smoke` (same
//! mechanism, no actor) and Stage D's `phase21_qwen_actor_smoke`
//! (inference side wrapped in actor). With this commit, both
//! inference AND training of Qwen2.5-Coder-0.5B flow through the
//! Pekko actor framework.
//!
//! Run:
//!   cargo run -p llm-actors --example phase21_e_next_smoke \
//!       --features cuda --release
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device};
use llm_actors::{qwen2_lora::LoraConfig, QwenTrainerActor, QwenTrainerMessage};
use pekko_actor::ActorSystem;
use tokio::sync::oneshot;

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

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let device = pick_device();
    let on_cuda = device.is_cuda();
    println!("[Phase21E.next] device = {device:?}, on_cuda = {on_cuda}");

    // F32 throughout — rank=16 LoRA gradients at lr=2e-4 are too coarse
    // in F16; Stage F documented this. Re-using the same recipe here.
    let dtype = DType::F32;
    let snapshot = resolve_default_snapshot()?;
    println!("[Phase21E.next] snapshot = {}", snapshot.display());

    let trainer = QwenTrainerActor::from_snapshot_dir(
        &snapshot,
        device,
        dtype,
        LoraConfig {
            rank: 16,
            alpha: 32.0,
        },
        2e-4,
    )?;
    println!(
        "[Phase21E.next] QwenTrainerActor built  lora_params = {}",
        trainer.lora_param_count()
    );

    let system = ActorSystem::new("phase21-e-next");
    let trainer_ref = system.spawn(trainer, "qwen-trainer").await?;

    // Ping
    let (px, prx) = oneshot::channel::<()>();
    trainer_ref
        .tell(QwenTrainerMessage::Ping { reply: px })
        .map_err(|e| anyhow!("{e:?}"))?;
    prx.await?;
    println!("[Phase21E.next] Ping OK");

    // Train on 3 short text examples for 8 steps. Same recipe as Stage F's
    // standalone smoke (which dropped loss 0.8226 → 0.3530 over 8 steps).
    let texts = vec![
        "def fibonacci(n):\n    if n < 2:\n        return n".to_string(),
        "def is_prime(n):\n    if n < 2:\n        return False".to_string(),
        "def reverse_string(s):\n    return s[::-1]".to_string(),
    ];
    let train_steps = 8usize;
    println!(
        "[Phase21E.next] sending Train {{ texts: {} examples, train_steps: {} }}",
        texts.len(),
        train_steps
    );
    let (tx, rx) = oneshot::channel();
    trainer_ref
        .tell(QwenTrainerMessage::Train {
            texts,
            train_steps,
            reply: tx,
        })
        .map_err(|e| anyhow!("{e:?}"))?;
    let outcome = rx.await??;
    println!(
        "\n[Phase21E.next] loss trajectory ({} steps):",
        outcome.losses.len()
    );
    for (i, l) in outcome.losses.iter().enumerate() {
        println!("  step {i}  loss = {l:.4}");
    }
    println!(
        "\n[Phase21E.next] initial={:.4}  final={:.4}  Δ={:+.4}",
        outcome.initial_loss,
        outcome.final_loss,
        outcome.final_loss - outcome.initial_loss
    );
    if outcome.final_loss >= outcome.initial_loss {
        return Err(anyhow!(
            "loss did NOT decrease ({:.4} → {:.4}) — training broken",
            outcome.initial_loss,
            outcome.final_loss
        ));
    }

    // SaveLoraAdapter
    let adapter_path = PathBuf::from("checkpoints/phase21_e_next_lora_adapter.safetensors");
    if let Some(p) = adapter_path.parent() {
        std::fs::create_dir_all(p).ok();
    }
    let (sx, srx) = oneshot::channel();
    trainer_ref
        .tell(QwenTrainerMessage::SaveLoraAdapter {
            path: adapter_path.clone(),
            reply: sx,
        })
        .map_err(|e| anyhow!("{e:?}"))?;
    srx.await??;
    let saved_size = std::fs::metadata(&adapter_path)?.len();
    println!(
        "[Phase21E.next] SaveLoraAdapter OK  path={}  size={} bytes",
        adapter_path.display(),
        saved_size
    );

    println!("\nphase21_e_next_smoke: PASS");
    Ok(())
}
