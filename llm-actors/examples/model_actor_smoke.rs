//! Spawn a `ModelActor` with a tiny randomly-initialized GPT and a char tokenizer,
//! send a Generate message, and assert we get tokens back.
//!
//! This is a wiring smoke test (output is gibberish — model is untrained).
//!
//! Run:
//!   cargo run -p llm-actors --example model_actor_smoke

use std::sync::Arc;
use std::time::Duration;

use candle_core::Device;
use llm_actors::{ModelActor, ModelMessage};
use nanogpt_rs::{config::GPTConfig, generate::GenerateConfig, tokenizer::Tokenizer};
use pekko_actor::ActorSystem;
use tokio::sync::oneshot;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).init();

    let device = Device::Cpu;

    // Tiny model just for wiring. Char vocab over the prompt charset.
    let seed_text = "Hello world! ROMEO: To be or not to be.\n";
    let tk = Tokenizer::char_from_text(seed_text);
    let vocab = tk.vocab_size();

    let cfg = GPTConfig {
        vocab_size: vocab,
        block_size: 32,
        n_layer: 2,
        n_head: 2,
        n_embd: 32,
        dropout: 0.0,
        bias: false,
        ffn_mult: 4,
        use_rope: false,
        rope_base: 10_000.0,
        n_kv_head: 2,
        n_experts: 1,
        moe_top_k: 0,
        moe_aux_weight: 0.01,
        activation: nanogpt_rs::config::ActivationKind::Gelu,
        weight_tying: true,
        norm_kind: nanogpt_rs::config::NormKind::LayerNorm,
        norm_position: nanogpt_rs::config::NormPosition::Pre,
        lora_rank: 0,
        lora_alpha: 16.0,
    };

    let actor = ModelActor::new(cfg, device, Arc::new(tk))?;

    let system = ActorSystem::new("llm-test");
    let actor_ref = system.spawn(actor, "model").await?;

    // Ping
    let (px, prx) = oneshot::channel::<()>();
    actor_ref.tell(ModelMessage::Ping { reply: px }).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    tokio::time::timeout(Duration::from_secs(2), prx).await??;
    println!("Ping OK");

    // Generate
    let (gx, grx) = oneshot::channel();
    let gcfg = GenerateConfig {
        max_new_tokens: 16,
        temperature: 1.0,
        top_k: Some(8),
        top_p: None,
        seed: Some(42),
    };
    actor_ref
        .tell(ModelMessage::Generate {
            prompt: "ROMEO:".to_string(),
            cfg: gcfg,
            reply: gx,
        })
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    let reply = tokio::time::timeout(Duration::from_secs(30), grx).await??;
    let reply = reply?;
    assert!(!reply.tokens.is_empty(), "empty tokens");
    println!("Generated {} tokens: {:?}", reply.tokens.len(), reply.text);

    Ok(())
}
