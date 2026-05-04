//! Phase 2 end-to-end smoke: one self-improvement round on the arithmetic
//! domain.
//!
//! Flow:
//!   1. Build a deterministic pretraining corpus from `ArithmeticDomain`.
//!   2. Pretrain a tiny char-GPT briefly so the model has *some* signal.
//!   3. Spin up Model/Generator/Verifier/Curator/Trainer/Evaluator actors.
//!   4. Pre-load curator with a few correct seed examples (so round-1 has
//!      something to train on even before the model learns to self-supply).
//!   5. Run N rounds; report pass-rate before/after each round.
//!
//! Run:
//!   cargo run -p llm-actors --example self_improve_round --release --features cuda -- --rounds 2

use std::path::PathBuf;
use std::sync::Arc;

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use clap::Parser;
use llm_actors::{
    curator_actor::SampleMode,
    domain::{
        arithmetic::{ArithmeticDomain, SeedMode},
        Domain,
    },
    run_round, CuratorActor, CuratorMessage, EvaluatorActor, GeneratorActor, ModelActor,
    RoundActors, RoundConfig, TrainerActor, Trajectory, Verdict, VerifiedTrajectory,
    VerifierActor,
};
use nanogpt_rs::{
    config::GPTConfig,
    data::TokenDataset,
    generate::GenerateConfig,
    model::GPT,
    tokenizer::Tokenizer,
    train::{train_from, TrainConfig},
};
use pekko_actor::ActorSystem;
use tokio::sync::oneshot;
use tracing::info;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_t = 2)]
    rounds: usize,
    #[arg(long, default_value_t = 200)]
    pretrain_examples: usize,
    #[arg(long, default_value_t = 800)]
    pretrain_steps: usize,
    #[arg(long, default_value_t = 100)]
    gen_n: usize,
    #[arg(long, default_value_t = 100)]
    eval_n: usize,
    #[arg(long, default_value_t = 400)]
    round_train_steps: usize,
    #[arg(long, default_value = "checkpoints/arith_seed.safetensors")]
    seed_ckpt: PathBuf,
    #[arg(long, default_value = "checkpoints/arith_round.safetensors")]
    round_ckpt: PathBuf,
    /// full | nocarry | none — controls what (a,b) pairs are seeded into the
    /// curator before round 0. `nocarry` is the recommended setting for a
    /// honest self-improvement demo (model must discover the carry pairs).
    #[arg(long, default_value = "nocarry")]
    seed_mode: String,
    #[arg(long, default_value_t = 1.2)]
    gen_temperature: f64,
    #[arg(long, default_value_t = 10)]
    gen_top_k: usize,
    /// Curator sampling mode: uniform | priority.
    #[arg(long, default_value = "priority")]
    curator_mode: String,
    #[arg(long, default_value_t = 0.95)]
    recency_decay: f32,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).init();
    let args = Args::parse();

    let device = pick_device();
    info!(?device, "device");

    let domain = Arc::new(ArithmeticDomain::default());

    // -------- Tokenizer: seed CharTokenizer from a corpus that covers all
    // chars the model will ever see (charset + a sample corpus).
    let pretrain_text = domain.synth_corpus(args.pretrain_examples, 7);
    let mut seed_chars = String::from(domain.charset());
    seed_chars.push_str(&pretrain_text);
    let tk = Tokenizer::char_from_text(&seed_chars);
    let vocab = tk.vocab_size();
    info!(vocab, "char tokenizer built");

    // shakespeare-char config bumped to 6/6/384. Plenty of capacity to
    // memorize the (a,b) → a+b table; with block_size=32 each training
    // window covers ~5 examples.
    let gpt_cfg = GPTConfig {
        vocab_size: vocab,
        block_size: 32,
        n_layer: 6,
        n_head: 6,
        n_embd: 384,
        dropout: 0.0,
        bias: false,
        ffn_mult: 4,
        use_rope: false,
        rope_base: 10_000.0,
        n_kv_head: 6,
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
    info!(params = gpt_cfg.num_params_estimate(), "model config");

    // -------- Pretrain (so the model has *some* arithmetic signal).
    info!("pretraining...");
    let ids = tk.encode(&pretrain_text)?;
    let pretrain_ds = TokenDataset::new(ids, gpt_cfg.block_size);
    let mut pre_cfg = TrainConfig::smoke();
    pre_cfg.max_steps = args.pretrain_steps;
    pre_cfg.batch_size = 128;
    pre_cfg.eval_interval = 500;
    pre_cfg.lr = 3e-3;
    pre_cfg.min_lr = 3e-4;
    pre_cfg.warmup_steps = 50;
    let pre_outcome = train_from(
        &gpt_cfg,
        &pretrain_ds,
        None,
        &pre_cfg,
        &device,
        Some(&args.seed_ckpt),
        None,
    )?;
    info!(
        train_loss = pre_outcome.last_train_loss,
        steps = pre_outcome.final_step,
        "pretrain done"
    );

    // -------- Actors.
    let tk = Arc::new(tk);
    let model_actor =
        ModelActor::from_checkpoint(gpt_cfg.clone(), device.clone(), tk.clone(), &args.seed_ckpt)?;
    let system = ActorSystem::new("self-improve");
    let model_ref = system.spawn(model_actor, "model").await?;

    let curator = CuratorActor::new(2048);
    let curator_ref = system.spawn(curator, "curator").await?;
    let seed_mode = parse_seed_mode(&args.seed_mode)?;
    seed_curator(&curator_ref, &domain, seed_mode).await?;

    let verifier = VerifierActor::new(domain.clone() as Arc<dyn Domain>);
    let verifier_ref = system.spawn(verifier, "verifier").await?;

    let generator = GeneratorActor::new(
        model_ref.clone(),
        tk.clone(),
        domain.clone() as Arc<dyn Domain>,
        Some('\n'),
        "model".to_string(),
    );
    let generator_ref = system.spawn(generator, "generator").await?;

    let evaluator = EvaluatorActor::new(
        model_ref.clone(),
        tk.clone(),
        domain.clone() as Arc<dyn Domain>,
        Some('\n'),
    );
    let evaluator_ref = system.spawn(evaluator, "evaluator").await?;

    let trainer = TrainerActor::new(gpt_cfg.clone(), tk.clone(), device.clone());
    let trainer_ref = system.spawn(trainer, "trainer").await?;

    let actors = RoundActors {
        model: model_ref.clone(),
        generator: generator_ref,
        verifier: verifier_ref,
        curator: curator_ref.clone(),
        trainer: trainer_ref,
        evaluator: evaluator_ref,
    };

    // -------- Sanity: confirm seed checkpoint works after reload.
    let _ = sanity_generate(&model_ref, "5+3=", 4).await?;

    // -------- Run rounds.
    let curator_sample_mode = parse_curator_mode(&args.curator_mode, args.recency_decay)?;
    let mut current_ckpt = args.seed_ckpt.clone();
    let mut history = Vec::new();
    for round in 0..args.rounds {
        let round_save = args.round_ckpt.with_extension(format!("r{round}.safetensors"));

        let mut train_cfg = TrainConfig::smoke();
        train_cfg.max_steps = args.round_train_steps;
        train_cfg.batch_size = 128;
        train_cfg.eval_interval = args.round_train_steps; // no eval mid-round
        train_cfg.lr = 1e-3;
        train_cfg.min_lr = 1e-4;
        train_cfg.warmup_steps = 20;

        let cfg = RoundConfig {
            round,
            gen_n: args.gen_n,
            gen_seed: 100 + round as u64,
            gen_sampling: GenerateConfig {
                max_new_tokens: 4,
                temperature: args.gen_temperature,
                top_k: Some(args.gen_top_k),
                top_p: None,
                seed: Some(round as u64),
            },
            eval_n: args.eval_n,
            eval_seed: 0xE7A1, // fixed across rounds for comparable eval set
            eval_sampling: GenerateConfig {
                max_new_tokens: 4,
                temperature: 0.0,
                top_k: Some(1),
                top_p: None,
                seed: Some(0xE7A1),
            },
            train_cfg,
            init_from: Some(current_ckpt.clone()),
            save_path: round_save.clone(),
            min_corpus_chars: 8000,
            sample_mode: curator_sample_mode,
            corpus_seed: Some(round as u64 * 31 + 7),
        };

        let report = run_round(&actors, cfg).await?;
        println!(
            "[round {round}] generated_correct={}/{} ({:.1}%)  eval_before={}/{}  eval_after={}/{}  train_loss={:?}  elapsed={}ms",
            report.correct,
            report.generated,
            100.0 * report.pass_rate_generated(),
            report.eval_correct_before.unwrap_or(0),
            report.eval_total,
            report.eval_correct_after.unwrap_or(0),
            report.eval_total,
            report.last_train_loss,
            report.elapsed_ms,
        );
        history.push(report);
        current_ckpt = round_save;
    }

    println!("\n=== history ===");
    for r in &history {
        println!(
            "round {}: gen={}/{}  eval before={:?}/{} after={:?}/{}",
            r.round,
            r.correct,
            r.generated,
            r.eval_correct_before,
            r.eval_total,
            r.eval_correct_after,
            r.eval_total
        );
    }

    Ok(())
}

fn parse_seed_mode(s: &str) -> anyhow::Result<SeedMode> {
    match s {
        "full" => Ok(SeedMode::Full),
        "nocarry" => Ok(SeedMode::NoCarry),
        "none" => Ok(SeedMode::None),
        other => anyhow::bail!("invalid --seed-mode {other:?} (expected full|nocarry|none)"),
    }
}

fn parse_curator_mode(s: &str, recency_decay: f32) -> anyhow::Result<SampleMode> {
    match s {
        "uniform" => Ok(SampleMode::Uniform),
        "priority" => Ok(SampleMode::Priority { recency_decay }),
        other => anyhow::bail!("invalid --curator-mode {other:?} (expected uniform|priority)"),
    }
}

async fn seed_curator(
    curator: &pekko_actor::ActorRef<CuratorActor>,
    domain: &ArithmeticDomain,
    seed_mode: SeedMode,
) -> anyhow::Result<()> {
    let pairs = domain.enumerate_seed_pairs(seed_mode);
    let items: Vec<VerifiedTrajectory> = pairs
        .into_iter()
        .map(|(a, b)| VerifiedTrajectory {
            trajectory: Trajectory {
                prompt: format!("{a}+{b}="),
                completion: format!("{}\n", a + b),
                source: "seed".to_string(),
            },
            verdict: Verdict::Correct,
            score: 1.0,
        })
        .collect();
    info!(pairs = items.len(), ?seed_mode, "seeding curator");
    let (tx, rx) = oneshot::channel();
    curator
        .tell(CuratorMessage::Add { items, reply: tx })
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let report = rx.await?;
    info!(seeded = report.accepted, "curator seeded");
    Ok(())
}

async fn sanity_generate(
    model: &pekko_actor::ActorRef<ModelActor>,
    prompt: &str,
    max_new: usize,
) -> anyhow::Result<()> {
    use llm_actors::ModelMessage;
    let cfg = GenerateConfig {
        max_new_tokens: max_new,
        temperature: 0.0,
        top_k: Some(1),
        top_p: None,
        seed: Some(0),
    };
    let (tx, rx) = oneshot::channel();
    model
        .tell(ModelMessage::Generate { prompt: prompt.to_string(), cfg, reply: tx })
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let r = rx.await??;
    info!(prompt, completion = %r.text, "sanity gen");
    Ok(())
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

#[allow(dead_code)]
fn _unused_so_imports_dont_warn(_: VarMap, _: VarBuilder, _: DType, _: GPT) {}
