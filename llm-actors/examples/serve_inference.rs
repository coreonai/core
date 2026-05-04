//! HTTP-fronted inference server.
//!
//! Spawns:
//!   - a `ModelActor` loaded from a checkpoint (or fresh-init for smoke
//!     testing without a checkpoint),
//!   - an `InferenceServerActor` wrapping it,
//!   - an axum HTTP server exposing `/inference` and `/health`.
//!
//! Run (smoke, no checkpoint — the model will produce garbage but the
//! pipeline is exercised end-to-end):
//!   cargo run -p llm-actors --example serve_inference --release -- --port 8080
//!
//! Run (with a real checkpoint, e.g. the trained KoWiki model):
//!   cargo run -p llm-actors --example serve_inference --release --features cuda -- \
//!       --port 8080 \
//!       --checkpoint checkpoints/kowiki_50m_clean.safetensors \
//!       --tokenizer data/kowiki/kowiki_bpe.json \
//!       --arch llama-50m
//!
//! Then:
//!   curl http://localhost:8080/health
//!   curl -X POST http://localhost:8080/inference \
//!        -H 'content-type: application/json' \
//!        -d '{"prompt":"대한민국의 수도는 ","max_new_tokens":40,"temperature":0.8}'

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use candle_core::Device;
use clap::Parser;
use llm_actors::{
    inference_http,
    inference_server_actor::InferenceServerActor,
    ModelActor,
};
use nanogpt_rs::{
    config::{ActivationKind, GPTConfig, NormKind, NormPosition},
    tokenizer::Tokenizer,
};
use pekko_actor::ActorSystem;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// Optional safetensors checkpoint to load. Without this, the model
    /// is freshly initialized — useful for smoke testing the HTTP path
    /// without paying for training.
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    /// Optional HuggingFace tokenizer JSON. If absent, a tiny char
    /// tokenizer over a fixed seed string is built (smoke only).
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    /// Architecture preset:
    ///   - `tiny`: 32-dim 2-layer model for fast smoke
    ///   - `llama-50m`: Phase-3-evolved 50M-param Llama recipe (RoPE +
    ///     GQA-2 + SwiGLU + RmsNorm-Pre + untied)
    #[arg(long, default_value = "tiny")]
    arch: String,
    /// Override the model's vocab size (defaults to the tokenizer's).
    #[arg(long)]
    vocab_size: Option<usize>,
    #[arg(long, default_value_t = 60)]
    request_timeout_secs: u64,
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

fn arch_preset(arch: &str, vocab: usize) -> anyhow::Result<GPTConfig> {
    match arch {
        "tiny" => Ok(GPTConfig {
            vocab_size: vocab,
            block_size: 64,
            n_layer: 2,
            n_head: 2,
            n_embd: 32,
            dropout: 0.0,
            bias: false,
            ffn_mult: 2,
            use_rope: false,
            rope_base: 10_000.0,
            n_kv_head: 2,
            n_experts: 1,
            moe_top_k: 0,
            moe_aux_weight: 0.0,
            activation: ActivationKind::Gelu,
            weight_tying: true,
            norm_kind: NormKind::LayerNorm,
            norm_position: NormPosition::Pre,
            lora_rank: 0,
            lora_alpha: 16.0,
        }),
        "llama-50m" => {
            let mut cfg = GPTConfig::nano_50m();
            cfg.vocab_size = vocab;
            // Match training setup
            cfg.block_size = 256;
            Ok(cfg)
        }
        other => anyhow::bail!("unknown --arch {other:?} (tiny | llama-50m)"),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    let args = Args::parse();
    let device = pick_device();
    tracing::info!(?device, "device");

    // -------- Tokenizer (HF BPE if given, else a tiny char tokenizer).
    let tk = if let Some(p) = &args.tokenizer {
        tracing::info!(?p, "loading HF tokenizer");
        Tokenizer::from_hf_file(p)?
    } else {
        tracing::info!("no --tokenizer; building tiny char tokenizer for smoke testing");
        Tokenizer::char_from_text(
            "Hello world! ROMEO: To be or not to be.\n0123456789+-=()/ 대한민국의 수도는 서울입니다.\n",
        )
    };
    let vocab = args.vocab_size.unwrap_or_else(|| tk.vocab_size());
    tracing::info!(vocab, "tokenizer ready");

    // -------- Build (and optionally load) the model.
    let cfg = arch_preset(&args.arch, vocab)?;
    tracing::info!(arch = %args.arch, params = cfg.num_params_estimate(), "model config");
    let tk = Arc::new(tk);
    let model_actor = match &args.checkpoint {
        Some(path) => {
            tracing::info!(?path, "loading checkpoint");
            ModelActor::from_checkpoint(cfg, device, tk, path)?
        }
        None => {
            tracing::info!("no --checkpoint; using fresh-init weights (smoke mode)");
            ModelActor::new(cfg, device, tk)?
        }
    };

    // -------- Spawn actors.
    let system = ActorSystem::new("inference-http");
    let model_ref = system.spawn(model_actor, "model").await?;
    let infer = InferenceServerActor::new(model_ref);
    let infer_ref = system.spawn(infer, "inference").await?;

    // -------- Serve.
    let addr: SocketAddr = ([0, 0, 0, 0], args.port).into();
    inference_http::serve(addr, infer_ref, args.request_timeout_secs).await?;
    Ok(())
}
