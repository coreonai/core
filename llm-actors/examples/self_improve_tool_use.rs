//! End-to-end self-evolving agent: Phase 2 self-improve loop × Phase 4
//! tool-use × Phase 3 modern architecture.
//!
//! Each round:
//!   1. Eval-before: agentic generation on a held-out prompt set, verified
//!      via `ToolUseArithmeticDomain.verify` (parses the `A: N` line).
//!   2. Generate: sample prompts, run agent, record full trajectories.
//!   3. Verify: each trajectory → Verdict via the domain.
//!   4. Curate: add correct trajectories to the replay buffer.
//!   5. Train: continual fine-tune on the (padded) buffer corpus.
//!   6. ModelActor reload from new checkpoint.
//!   7. Eval-after.
//!
//! Trajectories include the resolved tool call (`(arith add 3 4=7)\nA: 7`),
//! so the model trains on the post-dispatch form. At inference the agentic
//! generator handles either the resolved form (model emits directly) or
//! the unresolved form (model emits `(arith add 3 4)\n`, executor splices
//! the result inline).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use candle_core::Device;
use candle_nn::{VarBuilder, VarMap};
use clap::Parser;
use llm_actors::{
    agentic_generator_actor::{AgenticGeneratorActor, AgenticMessage},
    curator_actor::{CuratorActor, CuratorMessage, SampleMode},
    domain::{arithmetic::SeedMode, tool_use::ToolUseArithmeticDomain, Domain},
    model_actor::ModelMessage,
    tool_executor_actor::ToolExecutorActor,
    tools::{arithmetic_tool::ArithmeticTool, Tool, ToolRegistry},
    trainer_actor::{TrainerActor, TrainerMessage},
    types::{Trajectory, Verdict, VerifiedTrajectory},
    ModelActor,
};
use nanogpt_rs::{
    config::{ActivationKind, GPTConfig, NormKind, NormPosition},
    data::TokenDataset,
    ewc::WeightAnchor,
    generate::GenerateConfig,
    tokenizer::Tokenizer,
    train::{train_from, TrainConfig},
};
use pekko_actor::{ActorRef, ActorSystem};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tokio::sync::oneshot;
use tracing::info;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_t = 3)]
    rounds: usize,
    #[arg(long, default_value_t = 2000)]
    pretrain_examples: usize,
    #[arg(long, default_value_t = 4000)]
    pretrain_steps: usize,
    #[arg(long, default_value_t = 64)]
    gen_n: usize,
    #[arg(long, default_value_t = 50)]
    eval_n: usize,
    #[arg(long, default_value_t = 1500)]
    round_train_steps: usize,
    #[arg(long, default_value_t = 0xE0)]
    seed: u64,
    #[arg(long, default_value = "checkpoints/tool_use_seed.safetensors")]
    seed_ckpt: PathBuf,
    #[arg(long, default_value = "checkpoints/tool_use_round.safetensors")]
    round_ckpt: PathBuf,
    #[arg(long, default_value_t = 1.0)]
    gen_temperature: f64,
    #[arg(long, default_value_t = 4)]
    gen_top_k: usize,
    #[arg(long, default_value_t = 32)]
    max_new_tokens: usize,
    /// Architecture preset:
    ///   - `small`: 6L H6 E192 dense (~2.7M params), fast iteration
    ///   - `llama-18m`: Phase-3-evolved L6 H8/Kv2 E384 ffn=6 SwiGLU RmsNorm-Pre untied (~18M)
    #[arg(long, default_value = "small")]
    arch: String,
    /// Curriculum for the *pretrain* corpus. `nocarry` is the recommended
    /// setting for a real self-improve signal: model is taught only the
    /// no-carry half (a+b <= max_operand) and must discover the carry pairs
    /// via generation+verification+continual fine-tune. Eval and gen always
    /// use the full range. `full` matches the saturated-pretrain regime
    /// (no headroom for self-improve to work in).
    #[arg(long, default_value = "nocarry")]
    seed_mode: String,
    /// Fraction of the per-round training corpus that comes from the
    /// pretrain corpus (the rest from the curator's replay buffer).
    /// Standard ER (Experience Replay) prevents catastrophic forgetting —
    /// 0.3 is a reasonable default. 0.0 = pure replay buffer (forgetting
    /// risk), 1.0 = pure pretrain (no learning).
    #[arg(long, default_value_t = 0.3)]
    replay_mix_frac: f32,
    /// EWC strength λ on the L2-anchor toward post-pretrain weights.
    /// `0.0` disables the anchor (pure ER). Higher values pin weights more
    /// strongly to the pretrained state.
    #[arg(long, default_value_t = 0.0)]
    ewc_lambda: f64,
    /// Number of pretrain batches used to estimate the diagonal Fisher
    /// information for EWC. `0` = uniform Fisher (= L2 toward pretrain).
    /// Larger values = better Fisher estimate, more memory + setup time.
    /// Typical: 32–128.
    #[arg(long, default_value_t = 0)]
    fisher_batches: usize,
    /// LoRA rank for the attention `c_attn` adapter. `0` = no LoRA (full
    /// fine-tune). When `> 0`, pretrain still trains all params (LoRA
    /// `lora_b` starts at zero, identity init), but per-round fine-tune
    /// freezes base weights and updates only the adapter — the strongest
    /// catastrophic-forgetting prevention available short of full
    /// per-round model snapshots.
    #[arg(long, default_value_t = 0)]
    lora_rank: usize,
}

fn parse_seed_mode(s: &str) -> anyhow::Result<SeedMode> {
    match s {
        "full" => Ok(SeedMode::Full),
        "nocarry" => Ok(SeedMode::NoCarry),
        "none" => Ok(SeedMode::None),
        other => anyhow::bail!("invalid --seed-mode {other:?} (full | nocarry | none)"),
    }
}

fn arch_preset(arch: &str, vocab: usize) -> anyhow::Result<GPTConfig> {
    match arch {
        "small" => Ok(GPTConfig {
            vocab_size: vocab,
            block_size: 64,
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
            // Patched at runtime from --lora-rank.
            lora_rank: 0,
            lora_alpha: 16.0,
        }),
        // Phase 3 evolution-best recipe: 4× GQA + SwiGLU + RmsNorm-Pre +
        // untied head, scaled to ~18M params with block_size 64 (covers
        // a full trajectory).
        "llama-18m" => Ok(GPTConfig {
            vocab_size: vocab,
            block_size: 64,
            n_layer: 6,
            n_head: 8,
            n_embd: 384,
            dropout: 0.0,
            bias: false,
            ffn_mult: 6,
            use_rope: true,
            rope_base: 10_000.0,
            n_kv_head: 2,
            n_experts: 1,
            moe_top_k: 0,
            moe_aux_weight: 0.0,
            activation: ActivationKind::SwiGlu,
            weight_tying: false,
            norm_kind: NormKind::RmsNorm,
            norm_position: NormPosition::Pre,
            // Patched at runtime from --lora-rank.
            lora_rank: 0,
            lora_alpha: 16.0,
        }),
        other => anyhow::bail!("unknown --arch {other:?} (small | llama-18m)"),
    }
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
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    let args = Args::parse();
    let device = pick_device();
    info!(?device, "device");

    let domain = Arc::new(ToolUseArithmeticDomain::default());
    let seed_mode = parse_seed_mode(&args.seed_mode)?;
    info!(
        ?seed_mode,
        seed_pair_count = domain.enumerate_seed_pairs(seed_mode).len(),
        "curriculum"
    );

    // -------- Corpus + tokenizer.
    let pretrain_corpus =
        domain.synth_corpus_with_mode(args.pretrain_examples, args.seed, seed_mode);
    let mut seed_chars = String::from(domain.charset());
    seed_chars.push_str(&pretrain_corpus);
    let tk = Arc::new(Tokenizer::char_from_text(&seed_chars));
    let vocab = tk.vocab_size();
    info!(
        vocab,
        corpus_chars = pretrain_corpus.len(),
        "tokenizer + corpus ready"
    );

    let mut gpt_cfg = arch_preset(&args.arch, vocab)?;
    gpt_cfg.lora_rank = args.lora_rank;
    info!(
        arch = %args.arch,
        lora_rank = gpt_cfg.lora_rank,
        params = gpt_cfg.num_params_estimate(),
        "model config"
    );

    // -------- Pretrain seed checkpoint.
    let pretrain_ids = tk.encode(&pretrain_corpus)?;
    let pretrain_ds = TokenDataset::new(pretrain_ids, gpt_cfg.block_size);
    let mut pre_cfg = TrainConfig::smoke();
    pre_cfg.max_steps = args.pretrain_steps;
    pre_cfg.batch_size = 128;
    pre_cfg.eval_interval = args.pretrain_steps;
    pre_cfg.lr = 1e-3;
    pre_cfg.min_lr = 1e-4;
    pre_cfg.warmup_steps = 100;
    info!("pretraining...");
    let outcome = train_from(
        &gpt_cfg,
        &pretrain_ds,
        None,
        &pre_cfg,
        &device,
        Some(&args.seed_ckpt),
        None,
    )?;
    info!(train_loss = outcome.last_train_loss, "pretrain done");

    // -------- Spawn actors.
    let model_actor =
        ModelActor::from_checkpoint(gpt_cfg.clone(), device.clone(), tk.clone(), &args.seed_ckpt)?;
    let system = ActorSystem::new("self-improve-tool-use");
    let model_ref = system.spawn(model_actor, "model").await?;

    let registry = ToolRegistry::from_tools(vec![Arc::new(ArithmeticTool) as Arc<dyn Tool>]);
    let executor_ref = system
        .spawn(ToolExecutorActor::new(registry), "tool_executor")
        .await?;

    let agent = AgenticGeneratorActor::new(model_ref.clone(), executor_ref.clone(), tk.clone());
    let agent_ref = system.spawn(agent, "agent").await?;

    let curator = CuratorActor::new(8192);
    let curator_ref = system.spawn(curator, "curator").await?;

    let trainer = TrainerActor::new(gpt_cfg.clone(), tk.clone(), device.clone());
    let trainer_ref = system.spawn(trainer, "trainer").await?;

    // -------- Snapshot the post-pretrain weights for the EWC anchor.
    //  We rebuild a fresh VarMap and load the seed checkpoint into it
    //  so the snapshot tensors are independent of the trainer's varmap
    //  (which is created on demand inside the trainer's blocking task).
    let anchor = if args.ewc_lambda > 0.0 {
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        let _model = nanogpt_rs::model::GPT::new(gpt_cfg.clone(), vb)?;
        varmap.load(&args.seed_ckpt)?;
        let a = if args.fisher_batches > 0 {
            // Real EWC: estimate diagonal Fisher from pretrain data.
            let pre_ids = tk.encode(&pretrain_corpus)?;
            let pre_ds = TokenDataset::new(pre_ids, gpt_cfg.block_size);
            info!(
                fisher_batches = args.fisher_batches,
                "estimating Fisher diagonal from pretrain data"
            );
            WeightAnchor::snapshot_with_fisher(
                &gpt_cfg,
                &varmap,
                &pre_ds,
                args.fisher_batches,
                64, // batch size for Fisher estimation; modest is fine
                &device,
                args.ewc_lambda,
            )?
        } else {
            WeightAnchor::snapshot(&varmap, args.ewc_lambda)?
        };
        info!(
            lambda = args.ewc_lambda,
            vars = a.reference.len(),
            fisher = a.fisher.is_some(),
            "EWC anchor snapshotted"
        );
        Some(Arc::new(a))
    } else {
        None
    };

    // -------- Run rounds.
    let mut current_ckpt = args.seed_ckpt.clone();
    let mut history: Vec<RoundResult> = Vec::with_capacity(args.rounds);
    for round in 0..args.rounds {
        let round_save = args
            .round_ckpt
            .with_extension(format!("r{round}.safetensors"));
        let result = run_one_round(
            round,
            &args,
            &domain,
            &pretrain_corpus,
            anchor.clone(),
            &agent_ref,
            &curator_ref,
            &trainer_ref,
            &model_ref,
            &current_ckpt,
            &round_save,
        )
        .await?;
        println!(
            "[round {}] gen_correct={}/{}  eval_before={}/{}  eval_after={}/{}  buffer={}  train_loss={:?}  elapsed={}ms",
            result.round,
            result.gen_correct,
            result.gen_total,
            result.eval_before,
            result.eval_total,
            result.eval_after,
            result.eval_total,
            result.buffer_size,
            result.last_train_loss,
            result.elapsed_ms,
        );
        history.push(result);
        current_ckpt = round_save;
    }

    println!("\n=== history ===");
    for r in &history {
        println!(
            "round {}: gen={}/{}  eval before→after = {}/{} → {}/{}  Δ={:+}",
            r.round,
            r.gen_correct,
            r.gen_total,
            r.eval_before,
            r.eval_total,
            r.eval_after,
            r.eval_total,
            r.eval_after as i32 - r.eval_before as i32,
        );
    }

    Ok(())
}

#[derive(Debug)]
struct RoundResult {
    round: usize,
    gen_total: usize,
    gen_correct: usize,
    eval_total: usize,
    eval_before: usize,
    eval_after: usize,
    buffer_size: usize,
    last_train_loss: Option<f32>,
    elapsed_ms: u128,
}

#[allow(clippy::too_many_arguments)]
async fn run_one_round(
    round: usize,
    args: &Args,
    domain: &Arc<ToolUseArithmeticDomain>,
    pretrain_corpus: &str,
    anchor: Option<Arc<WeightAnchor>>,
    agent: &ActorRef<AgenticGeneratorActor>,
    curator: &ActorRef<CuratorActor>,
    trainer: &ActorRef<TrainerActor>,
    model: &ActorRef<ModelActor>,
    init_from: &Path,
    save_to: &Path,
) -> anyhow::Result<RoundResult> {
    let t0 = Instant::now();

    let gen_sampling = GenerateConfig {
        max_new_tokens: args.max_new_tokens,
        temperature: args.gen_temperature,
        top_k: Some(args.gen_top_k),
        top_p: None,
        seed: Some(round as u64 + 1000),
    };
    let eval_sampling = GenerateConfig {
        max_new_tokens: args.max_new_tokens,
        temperature: 0.0, // greedy
        top_k: Some(1),
        top_p: None,
        seed: Some(0xE7A1),
    };

    // 1. Eval-before.
    let eval_before = agentic_eval(agent, domain, args.eval_n, 0xE7A1, &eval_sampling).await?;
    info!(round, before = eval_before, "eval-before");

    // 2. Generate batch via agentic loop.
    let traj = agentic_generate(
        agent,
        domain,
        args.gen_n,
        round as u64 * 1009 + 17,
        &gen_sampling,
    )
    .await?;
    let gen_correct = traj.iter().filter(|v| v.is_correct()).count();
    info!(
        round,
        gen_correct,
        gen_total = traj.len(),
        "generate+verify done"
    );

    // 3. Curate.
    let (tx, rx) = oneshot::channel();
    curator
        .tell(CuratorMessage::Add {
            items: traj,
            reply: tx,
        })
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let add_report = rx.await?;
    info!(
        round,
        accepted = add_report.accepted,
        buffer = add_report.buffer_size,
        "curated"
    );

    // 4. Render corpus + replay-mix pretrain (forgetting prevention).
    let (tx, rx) = oneshot::channel();
    curator
        .tell(CuratorMessage::RenderCorpus {
            mode: SampleMode::Priority {
                recency_decay: 0.95,
            },
            seed: Some(round as u64 + 31),
            reply: tx,
        })
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let buffer_corpus = rx.await?;
    let mut rng = StdRng::seed_from_u64(round as u64 * 113 + 7);
    let corpus = build_round_corpus(
        pretrain_corpus,
        &buffer_corpus,
        args.replay_mix_frac,
        16_000,
        &mut rng,
    );
    info!(
        round,
        buffer_len = buffer_corpus.len(),
        round_corpus_len = corpus.len(),
        replay_mix_frac = args.replay_mix_frac,
        "round corpus assembled (buffer + pretrain replay mix)"
    );

    let mut train_cfg = TrainConfig::smoke();
    train_cfg.max_steps = args.round_train_steps;
    train_cfg.batch_size = 128;
    train_cfg.eval_interval = train_cfg.max_steps;
    train_cfg.lr = 5e-4;
    train_cfg.min_lr = 5e-5;
    train_cfg.warmup_steps = 50;

    let freeze_base = args.lora_rank > 0;
    let last_train_loss = if !corpus.is_empty() {
        let (tx, rx) = oneshot::channel();
        trainer
            .tell(TrainerMessage::Train {
                corpus,
                save_path: save_to.to_path_buf(),
                init_from: Some(init_from.to_path_buf()),
                train_cfg,
                anchor: anchor.clone(),
                freeze_base,
                reply: tx,
            })
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let outcome = rx.await??;
        Some(outcome.last_train_loss)
    } else {
        None
    };

    // 5. Reload checkpoint.
    if last_train_loss.is_some() {
        let (tx, rx) = oneshot::channel();
        model
            .tell(ModelMessage::ReloadCheckpoint {
                path: save_to.to_path_buf(),
                reply: tx,
            })
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        rx.await??;
    }

    // 6. Eval-after.
    let eval_after = agentic_eval(agent, domain, args.eval_n, 0xE7A1, &eval_sampling).await?;
    info!(round, after = eval_after, "eval-after");

    Ok(RoundResult {
        round,
        gen_total: args.gen_n,
        gen_correct,
        eval_total: args.eval_n,
        eval_before,
        eval_after,
        buffer_size: add_report.buffer_size,
        last_train_loss,
        elapsed_ms: t0.elapsed().as_millis(),
    })
}

/// Assemble a per-round training corpus from the curator's replay buffer
/// + a sample of the pretrain corpus. Mixes the two so continual fine-tune
///   won't catastrophically forget the pretrain distribution. ASCII-safe
///   (the tool-use grammar only uses ASCII), so byte slicing == char slicing.
fn build_round_corpus(
    pretrain: &str,
    buffer: &str,
    mix_frac: f32,
    min_chars: usize,
    rng: &mut StdRng,
) -> String {
    if buffer.is_empty() {
        return String::new();
    }
    let mix_frac = mix_frac.clamp(0.0, 0.99);

    // Pad buffer up to (1 - mix_frac) × min_chars by repetition so we always
    // have enough buffer text relative to the pretrain mix.
    let target_buffer = ((min_chars as f32) * (1.0 - mix_frac)).max(buffer.len() as f32) as usize;
    let mut buffer_padded = String::with_capacity(target_buffer);
    while buffer_padded.len() < target_buffer {
        buffer_padded.push_str(buffer);
    }

    if mix_frac <= f32::EPSILON || pretrain.is_empty() {
        return buffer_padded;
    }

    // Want pretrain_chars / total = mix_frac → pretrain_chars = buffer × m / (1-m).
    let target_pretrain =
        ((buffer_padded.len() as f32) * mix_frac / (1.0 - mix_frac)).max(0.0) as usize;
    let pretrain_excerpt = sample_excerpt(pretrain, target_pretrain, rng);

    let mut combined = String::with_capacity(pretrain_excerpt.len() + buffer_padded.len());
    combined.push_str(&pretrain_excerpt);
    combined.push_str(&buffer_padded);
    combined
}

fn sample_excerpt(corpus: &str, target_chars: usize, rng: &mut StdRng) -> String {
    if target_chars == 0 || corpus.is_empty() {
        return String::new();
    }
    if corpus.len() <= target_chars {
        return corpus.to_string();
    }
    let start = rng.gen_range(0..(corpus.len() - target_chars));
    corpus[start..start + target_chars].to_string()
}

async fn agentic_generate(
    agent: &ActorRef<AgenticGeneratorActor>,
    domain: &Arc<ToolUseArithmeticDomain>,
    n: usize,
    seed: u64,
    sampling: &GenerateConfig,
) -> anyhow::Result<Vec<VerifiedTrajectory>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let prompt = domain.sample_prompt(&mut rng);
        let (tx, rx) = oneshot::channel();
        agent
            .tell(AgenticMessage::Run {
                prompt: prompt.clone(),
                sampling: sampling.clone(),
                max_steps: 4,
                reply: tx,
            })
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let report = tokio::time::timeout(Duration::from_secs(60), rx).await???;
        let completion = report
            .final_text
            .strip_prefix(&prompt)
            .unwrap_or(&report.final_text)
            .to_string();
        let verdict = domain.verify(&prompt, &completion);
        let score: f32 = if matches!(verdict, Verdict::Correct) {
            1.0
        } else {
            0.0
        };
        out.push(VerifiedTrajectory {
            trajectory: Trajectory {
                prompt,
                completion,
                source: "agent".into(),
            },
            verdict,
            score,
        });
    }
    Ok(out)
}

async fn agentic_eval(
    agent: &ActorRef<AgenticGeneratorActor>,
    domain: &Arc<ToolUseArithmeticDomain>,
    n: usize,
    seed: u64,
    sampling: &GenerateConfig,
) -> anyhow::Result<usize> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut correct = 0usize;
    for _ in 0..n {
        let prompt = domain.sample_prompt(&mut rng);
        let (tx, rx) = oneshot::channel();
        agent
            .tell(AgenticMessage::Run {
                prompt: prompt.clone(),
                sampling: sampling.clone(),
                max_steps: 4,
                reply: tx,
            })
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let report = tokio::time::timeout(Duration::from_secs(60), rx).await???;
        let completion = report
            .final_text
            .strip_prefix(&prompt)
            .unwrap_or(&report.final_text);
        if domain.verify(&prompt, completion).is_correct() {
            correct += 1;
        }
    }
    Ok(correct)
}
