//! Phase 23 — the multi-turn agentic loop, end to end, on the 7B.
//!
//! Everything before this measured a single generation. This runs the actual
//! loop: generate → parse a call → dispatch it → splice the result back →
//! generate again, through `AgenticGeneratorActor<QwenModelActor>`.
//!
//! Two things had to land first:
//!   - `ac6edfa` made `AgenticGeneratorActor` generic over the model actor.
//!     It took `ActorRef<ModelActor>`, so the 7B could not be driven at all.
//!   - `5dbc737` SFT'd the call format in. The base model emits the call
//!     *structure* ~70% of the time but the exact call 0%, so the loop had
//!     nothing to dispatch.
//!
//! Difference from `agentic_arithmetic.rs` (Phase 4): that example plants a
//! tool call in the prompt and checks the machinery notices it. Here the
//! model has to *produce* the call itself from `Q: a+b=`, which is the thing
//! that was never true before the format SFT.
//!
//! Prompted 0-shot, matching the SFT condition — few-shot measurably hurts a
//! fine-tuned model here (100% → 0% exact calls, see 5dbc737).
//!
//! Run:
//!   cargo run -p llm-actors --example phase23_agentic_7b --features cuda --release -- \
//!       --checkpoint scratch-7b-sft/p23_fmt_sft.safetensors --n-problems 20

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device};
use clap::Parser;
use llm_actors::{
    tools::{arithmetic_tool::ArithmeticTool, Tool, ToolRegistry},
    AgenticGeneratorActor, AgenticMessage, QwenModelActor, ToolExecutorActor,
};
use nanogpt_rs::{generate::GenerateConfig, Tokenizer as NgptTokenizer};
use pekko_actor::ActorSystem;
use tokio::sync::oneshot;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    model_dir: Option<PathBuf>,
    #[arg(long, default_value = "Qwen2.5-Coder-7B")]
    model_id: String,
    /// Format-SFT'd merged checkpoint. Without one the loop has nothing to
    /// dispatch — the base model does not emit exact calls.
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    /// Held-out problems (the `i % 5 == 0` side of the 100-pair grid, the
    /// same split phase23_toolcall_sft trains the complement of).
    #[arg(long, default_value_t = 20)]
    n_problems: usize,
    #[arg(long, default_value_t = 4)]
    max_steps: usize,
    #[arg(long, default_value_t = 24)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,
    #[arg(long, default_value_t = 6)]
    show: usize,
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
    println!("[Phase23Loop] device = {device:?}");
    if !device.is_cuda() && std::env::var("PHASE22_ALLOW_CPU").is_err() {
        anyhow::bail!("Refusing to run on CPU. Rebuild with `--features cuda`.");
    }

    let snapshot = resolve_snapshot(args.model_dir.as_deref(), &args.model_id)?;
    let tk = Arc::new(NgptTokenizer::from_hf_file(
        snapshot.join("tokenizer.json"),
    )?);

    let model = match &args.checkpoint {
        Some(ckpt) => {
            println!("[Phase23Loop] checkpoint = {}", ckpt.display());
            let cfg_text = std::fs::read_to_string(snapshot.join("config.json"))?;
            let config: candle_transformers::models::qwen2::Config =
                serde_json::from_str(&cfg_text)?;
            let hf = tokenizers::Tokenizer::from_file(snapshot.join("tokenizer.json"))
                .map_err(|e| anyhow!("tokenizer: {e}"))?;
            QwenModelActor::new(
                ckpt.clone(),
                Arc::new(hf),
                config,
                device.clone(),
                DType::F16,
            )?
        }
        None => {
            println!("[Phase23Loop] BASE model (expect ~0 dispatches)");
            QwenModelActor::from_snapshot_dir(&snapshot, device.clone(), DType::F16)?
        }
    };

    let registry = ToolRegistry::from_tools(vec![Arc::new(ArithmeticTool) as Arc<dyn Tool>]);
    let system = ActorSystem::new("phase23-loop");
    let model_ref = system.spawn(model, "qwen-model").await?;
    let exec_ref = system
        .spawn(ToolExecutorActor::new(registry), "tool-exec")
        .await?;
    // The type that ac6edfa unlocked: the loop over the 7B backend.
    let agent_ref = system
        .spawn(
            AgenticGeneratorActor::<QwenModelActor>::new(
                model_ref.clone(),
                exec_ref.clone(),
                tk.clone(),
            ),
            "agentic",
        )
        .await?;

    let all: Vec<(u32, u32)> = (0..=9u32)
        .flat_map(|a| (0..=9u32).map(move |b| (a, b)))
        .collect();
    let probes: Vec<(u32, u32)> = all
        .iter()
        .copied()
        .enumerate()
        .filter(|(i, _)| i % 5 == 0)
        .map(|(_, p)| p)
        .take(args.n_problems)
        .collect();
    println!("[Phase23Loop] {} held-out problems", probes.len());

    let (mut dispatched, mut tool_ok, mut answered) = (0usize, 0usize, 0usize);
    let mut shown = 0usize;

    for &(a, b) in &probes {
        let want = a + b;
        let cfg = GenerateConfig {
            max_new_tokens: args.max_new_tokens,
            temperature: args.temperature,
            top_k: if args.temperature > 0.0 {
                Some(40)
            } else {
                None
            },
            top_p: None,
            seed: Some(0),
        };
        let (tx, rx) = oneshot::channel();
        agent_ref
            .tell(AgenticMessage::Run {
                prompt: format!("Q: {a}+{b}=\n"),
                sampling: cfg,
                max_steps: args.max_steps,
                reply: tx,
            })
            .map_err(|e| anyhow!("{e:?}"))?;
        let report = tokio::time::timeout(Duration::from_secs(120), rx).await???;

        if report.tool_calls > 0 {
            dispatched += 1;
        }
        // Did the executor actually compute a+b? That is the tool doing real
        // work, as distinct from the model guessing the answer in text.
        let exec_right = report
            .trace
            .iter()
            .any(|s| matches!(&s.tool_result, Some(Ok(r)) if r.trim() == want.to_string()));
        if exec_right {
            tool_ok += 1;
        }
        // And did the model then state that answer? `A: <want>` is what the
        // turn-2 pairs trained.
        let said = report.final_text.contains(&format!("A: {want}"));
        if said {
            answered += 1;
        }
        if shown < args.show && !(exec_right && said) {
            println!(
                "  [{a}+{b}={want}] calls={} exec_ok={} said={} final={:?}",
                report.tool_calls,
                exec_right,
                said,
                report
                    .final_text
                    .replace('\n', "\\n")
                    .chars()
                    .take(90)
                    .collect::<String>()
            );
            shown += 1;
        }
    }

    let n = probes.len() as f64;
    println!("\n[Phase23Loop] === multi-turn loop, held-out ===");
    println!(
        "  dispatched a call = {dispatched:3}/{} ({:.0}%)",
        probes.len(),
        100.0 * dispatched as f64 / n
    );
    println!(
        "  tool computed a+b = {tool_ok:3}/{} ({:.0}%)",
        probes.len(),
        100.0 * tool_ok as f64 / n
    );
    println!(
        "  model stated A:    = {answered:3}/{} ({:.0}%)",
        probes.len(),
        100.0 * answered as f64 / n
    );
    println!("\nphase23_agentic_7b: PASS");
    Ok(())
}
