//! Phase 21 Stage D — `QwenModelActor` actor-pipeline smoke.
//!
//! Spawns `QwenModelActor` inside a Pekko `ActorSystem`, sends three
//! `ModelMessage::Generate` requests, asserts each returns plausible
//! tokens. Acts as proof that Phase 14-20's production model
//! (Qwen2.5-Coder-0.5B) is now reachable through the Rust actor
//! framework — Phase 17 S6 pass@k can in principle be driven by
//! actor traffic against the real model.
//!
//! Run:
//!   cargo run -p llm-actors --example phase21_qwen_actor_smoke \
//!       --features cuda --release
//!
//! Defaults to the HF cache snapshot for Qwen2.5-Coder-0.5B.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device};
use llm_actors::{ModelMessage, QwenModelActor};
use nanogpt_rs::generate::GenerateConfig;
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
    println!("[Phase21D] device = {device:?}, on_cuda = {on_cuda}");
    let dtype = if on_cuda { DType::F16 } else { DType::F32 };

    let snapshot = resolve_default_snapshot()?;
    println!("[Phase21D] snapshot = {}", snapshot.display());

    // Build the actor and spawn it.
    let actor = QwenModelActor::from_snapshot_dir(&snapshot, device, dtype)?;
    let tokenizer: Arc<tokenizers::Tokenizer> = actor.tokenizer.clone();
    println!("[Phase21D] actor built");

    let system = ActorSystem::new("phase21-d");
    let model_ref = system.spawn(actor, "qwen-model").await?;

    // Ping
    let (px, prx) = oneshot::channel::<()>();
    model_ref
        .tell(ModelMessage::Ping { reply: px })
        .map_err(|e| anyhow!("ping send: {e:?}"))?;
    prx.await?;
    println!("[Phase21D] Ping OK");

    let prompts = [
        "def fibonacci(n):",
        "def is_prime(n):",
        "def reverse_string(s):",
    ];

    for (i, prompt) in prompts.iter().enumerate() {
        let cfg = GenerateConfig {
            max_new_tokens: 28,
            temperature: 0.0, // greedy for determinism
            top_k: Some(1),
            top_p: None,
            seed: Some(42 + i as u64),
        };
        let (gx, grx) = oneshot::channel();
        model_ref
            .tell(ModelMessage::Generate {
                prompt: prompt.to_string(),
                cfg,
                reply: gx,
            })
            .map_err(|e| anyhow!("send: {e:?}"))?;
        let reply = grx.await??;
        assert!(!reply.tokens.is_empty(), "empty tokens");
        // Decode the completion-only span via the same tokenizer the
        // actor used — handy for the print, not load-bearing.
        let prompt_ids: Vec<u32> = tokenizer
            .encode(*prompt, true)
            .map_err(|e| anyhow!("encode: {e}"))?
            .get_ids()
            .to_vec();
        let comp_n = reply.tokens.len().saturating_sub(prompt_ids.len());
        println!(
            "\n[prompt {i}] {prompt}\n  -> +{comp_n} tokens (total {})\n  text: {}",
            reply.tokens.len(),
            reply.text.trim_end()
        );
    }

    println!("\nphase21_qwen_actor_smoke: PASS");
    Ok(())
}
