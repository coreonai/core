//! Train a model on tool-use arithmetic trajectories, then evaluate it via
//! the agentic loop.
//!
//! Pipeline:
//!   1. Synthesize corpus of `Q: A+B=\n(arith add A B=R)\nA: R\n`.
//!   2. Build char tokenizer over the corpus.
//!   3. Pretrain a small model on the corpus.
//!   4. Spawn ModelActor + ToolExecutor + AgenticGenerator.
//!   5. For N test prompts, run the agent, parse `A: N`, verify.
//!   6. Report pass-rate and a few sample trajectories.
//!
//! Run:
//!   cargo run -p llm-actors --example train_tool_use_arithmetic --release \
//!       --features cuda -- --pretrain-steps 3000 --eval-n 50

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use candle_core::Device;
use clap::Parser;
use llm_actors::{
    agentic_generator_actor::{AgenticGeneratorActor, AgenticMessage},
    domain::{tool_use::ToolUseArithmeticDomain, Domain},
    tool_executor_actor::ToolExecutorActor,
    tools::{arithmetic_tool::ArithmeticTool, Tool, ToolRegistry},
    ModelActor,
};
use nanogpt_rs::{
    config::{ActivationKind, GPTConfig, NormKind, NormPosition},
    data::TokenDataset,
    generate::GenerateConfig,
    tokenizer::Tokenizer,
    train::{train_from, TrainConfig},
};
use pekko_actor::ActorSystem;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tokio::sync::oneshot;
use tracing::info;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_t = 3000)]
    pretrain_examples: usize,
    #[arg(long, default_value_t = 5000)]
    pretrain_steps: usize,
    #[arg(long, default_value_t = 128)]
    batch_size: usize,
    #[arg(long, default_value_t = 50)]
    eval_n: usize,
    #[arg(long, default_value_t = 0xE0)]
    seed: u64,
    #[arg(long, default_value = "checkpoints/tool_use.safetensors")]
    save: PathBuf,
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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).init();
    let args = Args::parse();
    let device = pick_device();
    info!(?device, "device");

    // -------- Domain + corpus + tokenizer.
    let domain = Arc::new(ToolUseArithmeticDomain::default());
    let corpus = domain.synth_corpus(args.pretrain_examples, args.seed);
    let mut seed_chars = String::from(domain.charset());
    seed_chars.push_str(&corpus);
    let tk = Arc::new(Tokenizer::char_from_text(&seed_chars));
    let vocab = tk.vocab_size();
    info!(vocab, corpus_chars = corpus.len(), "tokenizer + corpus ready");

    // -------- Architecture: small modern transformer (the Llama-flavored
    // recipe Phase 3 evolution converged on, scaled down).
    let gpt_cfg = GPTConfig {
        vocab_size: vocab,
        block_size: 64, // a single full trajectory fits in ~25–30 chars
        n_layer: 6,
        n_head: 6,
        n_embd: 192,
        dropout: 0.0,
        bias: false,
        ffn_mult: 4,
        use_rope: true,
        rope_base: 10_000.0,
        n_kv_head: 6,
        n_experts: 1,
        moe_top_k: 0,
        moe_aux_weight: 0.0,
        activation: ActivationKind::SwiGlu,
        weight_tying: false,
        norm_kind: NormKind::RmsNorm,
        norm_position: NormPosition::Pre,
        lora_rank: 0,
        lora_alpha: 16.0,
    };
    info!(params = gpt_cfg.num_params_estimate(), "model config");

    // -------- Pretraining.
    let ids = tk.encode(&corpus)?;
    let ds = TokenDataset::new(ids, gpt_cfg.block_size);
    let mut tcfg = TrainConfig::smoke();
    tcfg.max_steps = args.pretrain_steps;
    tcfg.batch_size = args.batch_size;
    tcfg.eval_interval = args.pretrain_steps; // skip mid-train eval
    tcfg.lr = 1e-3;
    tcfg.min_lr = 1e-4;
    tcfg.warmup_steps = 100;
    info!(steps = tcfg.max_steps, "pretraining...");
    let outcome = train_from(
        &gpt_cfg,
        &ds,
        None,
        &tcfg,
        &device,
        Some(&args.save),
        None,
    )?;
    info!(train_loss = outcome.last_train_loss, "pretrain done");

    // -------- Wire up actors.
    let model_actor =
        ModelActor::from_checkpoint(gpt_cfg.clone(), device.clone(), tk.clone(), &args.save)?;
    let system = ActorSystem::new("tool-use-arith");
    let model_ref = system.spawn(model_actor, "model").await?;

    let registry = ToolRegistry::from_tools(vec![Arc::new(ArithmeticTool) as Arc<dyn Tool>]);
    let executor_ref = system.spawn(ToolExecutorActor::new(registry), "tool_executor").await?;

    let agent = AgenticGeneratorActor::new(model_ref.clone(), executor_ref.clone(), tk.clone());
    let agent_ref = system.spawn(agent, "agent").await?;

    // -------- Eval: run the agent on N fresh prompts, verify, report.
    let mut rng = StdRng::seed_from_u64(0xE7A1);
    let mut correct = 0usize;
    let mut sample_traces: Vec<String> = Vec::new();
    let sampling = GenerateConfig {
        max_new_tokens: 32,
        temperature: 0.0, // greedy — we want deterministic eval
        top_k: Some(1),
        top_p: None,
        seed: Some(0xE7A1),
    };

    for _ in 0..args.eval_n {
        let prompt = domain.sample_prompt(&mut rng);
        let (tx, rx) = oneshot::channel();
        agent_ref
            .tell(AgenticMessage::Run {
                prompt: prompt.clone(),
                sampling: sampling.clone(),
                max_steps: 4,
                reply: tx,
            })
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let report = tokio::time::timeout(Duration::from_secs(60), rx).await???;

        // The agent's final_text contains prompt + completion; verify on the
        // completion portion (everything after the prompt).
        let completion = report.final_text.strip_prefix(&prompt).unwrap_or(&report.final_text);
        let verdict = domain.verify(&prompt, completion);
        if verdict.is_correct() {
            correct += 1;
        }
        if sample_traces.len() < 5 {
            sample_traces.push(format!(
                "[{:?}] tool_calls={} steps={}\nprompt: {}completion: {}",
                verdict,
                report.tool_calls,
                report.steps,
                prompt.replace('\n', "\\n"),
                completion.replace('\n', "\\n"),
            ));
        }
    }

    let pass_rate = correct as f32 / args.eval_n.max(1) as f32;
    println!(
        "\n=== eval ===\npass_rate: {}/{} = {:.1}%\n",
        correct, args.eval_n, 100.0 * pass_rate
    );
    for t in &sample_traces {
        println!("--- sample ---\n{t}\n");
    }

    Ok(())
}
