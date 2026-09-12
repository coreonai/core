//! Serve the Phase 23 tool-using 7B over HTTP.
//!
//! `serve_inference` puts the small nanoGPT `ModelActor` behind HTTP and
//! cannot do anything else: `InferenceServerActor` took an
//! `ActorRef<ModelActor>`, so the 7B built in Phase 22/23 was reachable from
//! example binaries and nowhere else, and tool use was not on the serving
//! path at all. Both are now generic, and this is the binary that uses it.
//!
//! ```bash
//! phase23_serve_7b --checkpoint scratch-7b-sft/p23_si_all10/r0_merged.r1.safetensors
//!
//! # single generation, no tools
//! curl -s localhost:8080/inference -H 'content-type: application/json' \
//!   -d '{"prompt":"Q: how many divisors does 720 have?\n","max_new_tokens":64}'
//!
//! # the agentic loop
//! curl -s localhost:8080/inference -H 'content-type: application/json' \
//!   -d '{"prompt":"Q: how many divisors does 720 have?\n","tools":true}'
//! ```
//!
//! The tool-backed response carries `tool_calls` and a `tool_trace` of what
//! actually executed. Read it. This repo has measured a model stating an
//! answer its tool never produced — `exec_ok=false said=true` on a Collatz
//! problem whose tool call raised — so "the completion contains a number"
//! says nothing about whether a tool produced it.
//!
//! A request with `"tools": true` against a server started `--no-tools`
//! fails rather than quietly answering without them. A caller that asked for
//! a grounded answer must not receive an ungrounded one and be unable to tell.
//!
//! F32 by default (CLAUDE.md gotcha #11): dense code completions corrupt at
//! F16 even on memorised inputs.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device};
use clap::Parser;
use llm_actors::{
    inference_http,
    tools::{arithmetic_tool::ArithmeticTool, python_tool::PythonTool, Tool, ToolRegistry},
    AgenticGeneratorActor, InferenceServerActor, QwenModelActor, ToolExecutorActor,
};
use nanogpt_rs::Tokenizer as NgptTokenizer;
use pekko_actor::ActorSystem;

#[derive(Parser, Debug)]
struct Args {
    /// Self-improved checkpoint. Without one the base model is served, and it
    /// does not emit dispatchable calls 0-shot.
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    #[arg(long, default_value = "Qwen2.5-Coder-7B")]
    model_id: String,
    #[arg(long)]
    model_dir: Option<PathBuf>,
    #[arg(long, default_value = "127.0.0.1:8080")]
    addr: String,
    #[arg(long, default_value = "f32")]
    dtype: String,
    /// Serve generation only. `"tools": true` requests then fail loudly.
    #[arg(long)]
    no_tools: bool,
    /// Loop budget for tool requests that do not specify one.
    #[arg(long, default_value_t = 4)]
    max_steps: usize,
    #[arg(long, default_value_t = 300)]
    timeout_secs: u64,
    /// Stop sequences for the agentic loop. The default is the CALL
    /// BOUNDARY: a bare newline cuts a multi-line snippet after its import
    /// line, so the call never closes and never dispatches.
    #[arg(long, num_args = 1.., default_values_t = vec![")\n".to_string()])]
    stop: Vec<String>,
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
        .with_max_level(tracing::Level::INFO)
        .init();
    let args = Args::parse();
    let device = pick_device();
    if !device.is_cuda() && std::env::var("PHASE22_ALLOW_CPU").is_err() {
        anyhow::bail!("Refusing to run on CPU. Rebuild with `--features cuda`.");
    }
    let dtype = match args.dtype.as_str() {
        "f32" => DType::F32,
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        other => anyhow::bail!("unknown --dtype {other:?}"),
    };
    let addr: SocketAddr = args.addr.parse().context("--addr")?;

    let snapshot = resolve_snapshot(args.model_dir.as_deref(), &args.model_id)?;
    let tk = Arc::new(NgptTokenizer::from_hf_file(
        snapshot.join("tokenizer.json"),
    )?);

    println!("[Serve7B] loading {dtype:?} ...");
    let model = match &args.checkpoint {
        Some(ckpt) => {
            let cfg_text = std::fs::read_to_string(snapshot.join("config.json"))?;
            let config: candle_transformers::models::qwen2::Config =
                serde_json::from_str(&cfg_text)?;
            let hf = tokenizers::Tokenizer::from_file(snapshot.join("tokenizer.json"))
                .map_err(|e| anyhow!("tokenizer: {e}"))?;
            QwenModelActor::new(ckpt.clone(), Arc::new(hf), config, device.clone(), dtype)?
        }
        None => {
            println!("[Serve7B] WARNING: no --checkpoint; base model emits no calls 0-shot");
            QwenModelActor::from_snapshot_dir(&snapshot, device.clone(), dtype)?
        }
    };
    // Qwen's control tokens outrank code; without suppression the model
    // derails into `(python<|fim_prefix|>`. EOS is kept so generation ends.
    let tj: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(snapshot.join("tokenizer.json"))?)?;
    let suppress: Vec<u32> = tj["added_tokens"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter(|t| t["content"].as_str().unwrap_or("") != "<|endoftext|>")
                .filter_map(|t| t["id"].as_u64().map(|x| x as u32))
                .collect()
        })
        .unwrap_or_default();
    // The same stops go to the MODEL, not only the loop. The loop cuts text
    // after generation; the model stops generating. Without this a 20-token
    // call still pays for the full max_new_tokens — ~1.1 s per step at 26
    // ms/token.
    let model = model
        .with_suppressed_tokens(suppress)
        .with_stop_sequences(args.stop.iter().filter(|s| !s.is_empty()).cloned());

    let system = ActorSystem::new("phase23-serve");
    let model_ref = system.spawn(model, "qwen-model").await?;

    let mut server = InferenceServerActor::<QwenModelActor>::new(model_ref.clone())
        .with_default_max_steps(args.max_steps);
    if !args.no_tools {
        let registry = ToolRegistry::from_tools(vec![
            Arc::new(PythonTool::new()) as Arc<dyn Tool>,
            Arc::new(ArithmeticTool) as Arc<dyn Tool>,
        ]);
        let exec_ref = system
            .spawn(ToolExecutorActor::new(registry), "tool-exec")
            .await?;
        let agent_ref = system
            .spawn(
                AgenticGeneratorActor::<QwenModelActor>::new(
                    model_ref.clone(),
                    exec_ref,
                    tk.clone(),
                )
                .with_stop_sequences(args.stop.iter().filter(|s| !s.is_empty()).cloned()),
                "agentic",
            )
            .await?;
        server = server.with_agent(agent_ref);
        println!("[Serve7B] tools: python, arith");
    } else {
        println!("[Serve7B] tools disabled; \"tools\": true will be refused");
    }
    let server_ref = system.spawn(server, "inference-server").await?;

    println!("[Serve7B] listening on http://{addr}");
    inference_http::serve(addr, server_ref, args.timeout_secs).await
}
