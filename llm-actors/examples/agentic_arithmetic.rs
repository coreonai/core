//! Phase 4 end-to-end smoke: agentic loop with tool dispatch.
//!
//! Wires up a fresh (untrained) ModelActor + ToolExecutorActor + AgenticGen.
//! The model itself can't actually emit `(arith add 3 4)\n` since we
//! haven't trained on that grammar — instead, this demo proves the LOOP
//! mechanics end-to-end by feeding a hand-crafted prompt that already
//! contains a tool call. The agent should detect, dispatch, splice the
//! result, and continue.
//!
//! Run:
//!   cargo run -p llm-actors --example agentic_arithmetic --release

use std::sync::Arc;
use std::time::Duration;

use candle_core::Device;
use llm_actors::{
    agentic_generator_actor::{AgenticGeneratorActor, AgenticMessage},
    tool_executor_actor::ToolExecutorActor,
    tools::{arithmetic_tool::ArithmeticTool, Tool, ToolRegistry},
    ModelActor,
};
use nanogpt_rs::{
    config::{ActivationKind, GPTConfig, NormKind, NormPosition},
    generate::GenerateConfig,
    tokenizer::Tokenizer,
};
use pekko_actor::ActorSystem;
use tokio::sync::oneshot;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).init();

    // Char tokenizer over the chars used in the demo (digits, ops, parens, ws).
    let charset = "0123456789+-*/=() abcdefghijklmnopqrstuvwxyz\nA";
    let tk = Arc::new(Tokenizer::char_from_text(charset));
    let vocab = tk.vocab_size();
    tracing::info!(vocab, "tokenizer ready");

    let device = Device::Cpu;
    let cfg = GPTConfig {
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
        moe_aux_weight: 0.01,
        activation: ActivationKind::Gelu,
        weight_tying: true,
        norm_kind: NormKind::LayerNorm,
        norm_position: NormPosition::Pre,
        lora_rank: 0,
        lora_alpha: 16.0,
    };

    let model_actor = ModelActor::new(cfg, device, tk.clone())?;
    let system = ActorSystem::new("agentic-demo");
    let model_ref = system.spawn(model_actor, "model").await?;

    let registry = ToolRegistry::from_tools(vec![Arc::new(ArithmeticTool) as Arc<dyn Tool>]);
    let executor = ToolExecutorActor::new(registry);
    let executor_ref = system.spawn(executor, "tool_executor").await?;

    let agent = AgenticGeneratorActor::new(model_ref.clone(), executor_ref.clone(), tk.clone());
    let agent_ref = system.spawn(agent, "agentic_gen").await?;

    // Hand-crafted prompt with a tool call already present. The model will
    // generate a few extra (gibberish) tokens past the call but the agent
    // should detect the call inside the prompt itself when it parses the
    // first generation chunk.
    let prompt = "thinking...\n(arith add 3 4)\n".to_string();
    let cfg = GenerateConfig {
        max_new_tokens: 8,
        temperature: 0.7,
        top_k: Some(5),
        top_p: None,
        seed: Some(0),
    };

    let (tx, rx) = oneshot::channel();
    agent_ref
        .tell(AgenticMessage::Run {
            prompt: prompt.clone(),
            sampling: cfg,
            max_steps: 4,
            reply: tx,
        })
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let report = tokio::time::timeout(Duration::from_secs(60), rx).await???;

    println!("\n=== agentic report ===");
    println!("steps:        {}", report.steps);
    println!("tool_calls:   {}", report.tool_calls);
    println!("trace:");
    for s in &report.trace {
        println!(
            "  step {}: tokens={} tool={:?} result={:?}",
            s.step, s.generated_tokens, s.tool_called, s.tool_result
        );
    }
    println!("\nfinal text:\n{}", report.final_text);

    if report.tool_calls == 0 {
        return Err(anyhow::anyhow!(
            "expected at least 1 tool call dispatched (tool was inside prompt)"
        ));
    }
    Ok(())
}
