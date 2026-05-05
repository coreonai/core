//! Phase 5 Session 3: ensemble self-improve loop on the cargo-verified
//! Rust domain.
//!
//! Differs from `self_improve_rust` (Phase 2.5) in the curate step:
//!
//!   single-model:                 ensemble:
//!   ┌─────────┐                   ┌─────────┐ ┌─────────┐ ┌─────────┐
//!   │ ModelA  │                   │ ModelA  │ │ ModelB  │ │ ModelC  │
//!   └────┬────┘                   └────┬────┘ └────┬────┘ └────┬────┘
//!        ▼                              ▼           ▼           ▼
//!   ┌────────────┐                  ┌──────────────────────────────┐
//!   │  Verifier  │                  │ Verifier (each trajectory)   │
//!   └────┬───────┘                  └──┬───────────┬───────────┬───┘
//!        ▼                              ▼           ▼           ▼
//!   ┌────────────┐                  ┌──────────────────────────────┐
//!   │  Curator   │                  │ Curator::AddEnsemble        │
//!   │  (any 1+)  │                  │ (kept iff ≥ min_agreement   │
//!   └────────────┘                  │  models agree)               │
//!                                   └──────────────────────────────┘
//!
//! The consensus filter discards single-model fixed points — a slot
//! that only one member produced (and the others didn't independently
//! reach) is treated as overfit luck rather than signal.
//!
//! Per-member divergence comes from two sources:
//!   (a) different random VarMap init at spawn (Candle's default),
//!   (b) different pretrain RNG seeds → different per-batch sample
//!       order → different gradient sequence.
//!
//! Run:
//!   cargo run -p llm-actors --example self_improve_ensemble_rust --features cuda --release -- \
//!       --n-models 3 --rounds 3 --pretrain-steps 1500 --round-train-steps 400 \
//!       --lora-rank 32 --lora-alpha 64.0

use std::path::PathBuf;
use std::sync::Arc;

use candle_core::Device;
use clap::Parser;
use llm_actors::{
    curator_actor::SampleMode,
    domain::{rust_code::RustCodeDomain, Domain},
    ensemble::{ensemble_generate, EnsembleActors, EnsembleConfig},
    evaluator_actor::{EvalReport, EvaluatorActor, EvaluatorMessage},
    trainer_actor::{TrainerActor, TrainerMessage},
    CuratorActor, CuratorMessage, EnsembleItem, ModelMessage, Trajectory, Verdict,
    VerifiedTrajectory, VerifierActor, VerifierMessage,
};
use nanogpt_rs::{
    config::GPTConfig,
    data::TokenDataset,
    generate::GenerateConfig,
    tokenizer::Tokenizer,
    train::{train_from, TrainConfig},
};
use pekko_actor::{ActorRef, ActorSystem};
use tokio::sync::oneshot;
use tracing::info;

#[derive(Parser, Debug)]
struct Args {
    /// Number of ensemble members.
    #[arg(long, default_value_t = 3)]
    n_models: usize,
    #[arg(long, default_value_t = 3)]
    rounds: usize,
    #[arg(long, default_value_t = 1500)]
    pretrain_steps: usize,
    #[arg(long, default_value_t = 900)]
    pretrain_examples: usize,
    /// Number of generation prompts per round (each member gets 1 sample
    /// per prompt — diversity comes from the N members).
    #[arg(long, default_value_t = 24)]
    gen_n: usize,
    /// Held-out eval prompts (per member; same eval seed across rounds).
    #[arg(long, default_value_t = 21)]
    eval_n: usize,
    #[arg(long, default_value_t = 400)]
    round_train_steps: usize,
    #[arg(long, default_value_t = 0.8)]
    gen_temperature: f64,
    #[arg(long, default_value_t = 10)]
    gen_top_k: usize,
    #[arg(long, default_value_t = 16)]
    max_new_tokens: usize,
    /// LoRA rank (applied identically to all members). 0 = full FT.
    #[arg(long, default_value_t = 0)]
    lora_rank: usize,
    /// LoRA alpha (effective per-step scaling = α/r).
    #[arg(long, default_value_t = 16.0)]
    lora_alpha: f32,
    /// Minimum number of distinct models that must produce the same
    /// (prompt, completion) pair for the consensus filter to keep it.
    /// Default 0 → resolved at runtime to `majority_threshold(N)`.
    #[arg(long, default_value_t = 0)]
    min_agreement: usize,
    /// Scratch dir for the cargo verifier.
    #[arg(long, default_value = "/tmp/workllm-rust-scratch")]
    scratch_dir: PathBuf,
    /// Where per-member checkpoints land. We append `_seed_{i}.safetensors`
    /// for the post-pretrain weights and `.r{round}.{i}.safetensors` for
    /// per-round.
    #[arg(long, default_value = "checkpoints/rust_ensemble")]
    ckpt_prefix: PathBuf,
    #[arg(long, default_value_t = true)]
    run_program: bool,
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

const CHALLENGES: &[(&str, &[&str])] = &[
    (
        "fn main() { assert_eq!(",
        &[
            "2 + 3", "1 + 4", "5 + 0", "0 + 5", "10 - 5", "5 * 1", "1 * 5", "100 / 20",
        ],
    ),
    (
        "fn main() { assert_eq!(2 * (",
        &["7", "3 + 4", "4 + 3", "1 + 6", "6 + 1", "10 - 3", "14 / 2"],
    ),
    (
        "fn main() { let s: &str = ",
        &[
            r#""hello""#,
            r#""world""#,
            r#""abcde""#,
            r#""12345""#,
            r#""HELLO""#,
        ],
    ),
];

/// Pretrain corpus differs per member only in shuffle order — the
/// underlying (prompt, slot) population is identical. Each member sees
/// the same support but a different sequence of training samples,
/// which compounds with the random init to give independent solutions.
fn synth_pretrain_corpus(n: usize, seed: u64) -> String {
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = String::with_capacity(n * 64);
    for _ in 0..n {
        let (prompt, slots) = CHALLENGES.choose(&mut rng).expect("non-empty");
        let slot = slots.choose(&mut rng).expect("non-empty");
        out.push_str(prompt);
        out.push_str(slot);
        out.push('\n');
    }
    out
}

fn synth_seed_trajectories() -> Vec<VerifiedTrajectory> {
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    let mut rng = StdRng::seed_from_u64(0xC0DE);
    let mut out = Vec::new();
    for _ in 0..32 {
        let (prompt, slots) = CHALLENGES.choose(&mut rng).expect("non-empty");
        let slot = slots.choose(&mut rng).expect("non-empty");
        out.push(VerifiedTrajectory {
            trajectory: Trajectory {
                prompt: (*prompt).to_string(),
                completion: format!("{slot}\n"),
                source: "synthetic-seed".to_string(),
            },
            verdict: Verdict::Correct,
            score: 1.0,
        });
    }
    out
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    let args = Args::parse();
    let device = pick_device();
    info!(?device, n_models = args.n_models, "device");

    if args.n_models < 2 {
        anyhow::bail!("--n-models must be >= 2 (use self_improve_rust for single model)");
    }

    // ---- Domain (cargo verifier).
    let mut rcd = RustCodeDomain::new(&args.scratch_dir);
    rcd.run_program = args.run_program;
    rcd.ensure_scratch_project()?;
    let domain = Arc::new(rcd);

    // ---- Tokenizer (shared across all members).
    let charset_text: String = {
        // Cover all chars seen in pretrain + domain charset, regardless of
        // which member generates the corpus. We use member-0's corpus seed
        // for char coverage; the actual *order* differs per member.
        let pretrain_text = synth_pretrain_corpus(args.pretrain_examples, 7);
        let mut s = String::from(domain.charset());
        s.push_str(&pretrain_text);
        s
    };
    let tk = Arc::new(Tokenizer::char_from_text(&charset_text));
    let vocab = tk.vocab_size();
    info!(vocab, "tokenizer ready");

    // ---- Base GPTConfig (same for all members; per-member divergence
    // comes from random init + pretrain seed).
    let gpt_cfg = GPTConfig {
        vocab_size: vocab,
        block_size: 80,
        n_layer: 4,
        n_head: 4,
        n_embd: 128,
        dropout: 0.0,
        bias: false,
        ffn_mult: 4,
        use_rope: true,
        rope_base: 10_000.0,
        n_kv_head: 4,
        n_experts: 1,
        moe_top_k: 0,
        moe_aux_weight: 0.0,
        activation: nanogpt_rs::config::ActivationKind::SwiGlu,
        weight_tying: false,
        norm_kind: nanogpt_rs::config::NormKind::RmsNorm,
        norm_position: nanogpt_rs::config::NormPosition::Pre,
        lora_rank: args.lora_rank,
        lora_alpha: args.lora_alpha,
    };
    info!(
        params = gpt_cfg.num_params_estimate(),
        lora_rank = gpt_cfg.lora_rank,
        lora_alpha = gpt_cfg.lora_alpha,
        "model config (shared across members)"
    );

    // ---- Per-member pretrain. Different corpus seed → different
    // training-sample sequence → independent local minima.
    let mut seed_ckpts: Vec<PathBuf> = Vec::with_capacity(args.n_models);
    for i in 0..args.n_models {
        let seed = 7u64.wrapping_mul(i as u64 + 1).wrapping_add(0xE0);
        let pretrain_text = synth_pretrain_corpus(args.pretrain_examples, seed);
        let ids = tk.encode(&pretrain_text)?;
        let ds = TokenDataset::new(ids, gpt_cfg.block_size);
        let mut tcfg = TrainConfig::smoke();
        tcfg.max_steps = args.pretrain_steps;
        tcfg.batch_size = 64;
        tcfg.eval_interval = args.pretrain_steps;
        tcfg.lr = 3e-3;
        tcfg.min_lr = 3e-4;
        tcfg.warmup_steps = 50;
        let path = args
            .ckpt_prefix
            .with_extension(format!("seed_{i}.safetensors"));
        info!(member = i, ?path, ?seed, "pretraining member");
        let outcome = train_from(&gpt_cfg, &ds, None, &tcfg, &device, Some(&path), None)?;
        info!(
            member = i,
            train_loss = outcome.last_train_loss,
            "member pretrain done"
        );
        seed_ckpts.push(path);
    }

    // ---- Spawn ensemble + shared actors.
    let ens_cfg = EnsembleConfig {
        models: vec![gpt_cfg.clone(); args.n_models],
        init_paths: seed_ckpts.iter().cloned().map(Some).collect(),
        device: device.clone(),
    };
    let system = ActorSystem::new("ensemble-rust");
    let ensemble = EnsembleActors::spawn(&ens_cfg, tk.clone(), &system).await?;

    let curator = CuratorActor::new(2048);
    let curator_ref = system.spawn(curator, "curator").await?;
    seed_curator(&curator_ref, synth_seed_trajectories()).await?;

    let verifier = VerifierActor::new(domain.clone() as Arc<dyn Domain>);
    let verifier_ref = system.spawn(verifier, "verifier").await?;

    let trainer = TrainerActor::new(gpt_cfg.clone(), tk.clone(), device.clone());
    let trainer_ref = system.spawn(trainer, "trainer").await?;

    // One evaluator per member (each binds to its own model_ref).
    let mut evaluators: Vec<ActorRef<EvaluatorActor>> = Vec::with_capacity(args.n_models);
    for (i, model_ref) in ensemble.models.iter().enumerate() {
        let ev = EvaluatorActor::new(
            model_ref.clone(),
            tk.clone(),
            domain.clone() as Arc<dyn Domain>,
            Some('\n'),
        );
        evaluators.push(system.spawn(ev, &format!("evaluator-{i}")).await?);
    }

    // ---- Resolve consensus threshold.
    let min_agreement = if args.min_agreement == 0 {
        CuratorActor::majority_threshold(args.n_models)
    } else {
        args.min_agreement
    };
    info!(
        n_models = args.n_models,
        min_agreement, "consensus threshold resolved"
    );

    let eval_sampling = GenerateConfig {
        max_new_tokens: args.max_new_tokens,
        temperature: 0.0,
        top_k: Some(1),
        top_p: None,
        seed: Some(0xE5A2),
    };
    let gen_sampling = GenerateConfig {
        max_new_tokens: args.max_new_tokens,
        temperature: args.gen_temperature,
        top_k: Some(args.gen_top_k),
        top_p: None,
        seed: None, // ensemble_generate overrides per (i, k, j)
    };

    // ---- Round loop.
    let mut current_ckpts: Vec<PathBuf> = seed_ckpts.clone();
    let mut history: Vec<RoundLine> = Vec::with_capacity(args.rounds);
    for round in 0..args.rounds {
        let line = run_one_round(
            round,
            &args,
            &domain,
            &ensemble,
            &evaluators,
            &verifier_ref,
            &curator_ref,
            &trainer_ref,
            &gpt_cfg,
            &gen_sampling,
            &eval_sampling,
            &current_ckpts,
            min_agreement,
        )
        .await?;
        println!(
            "[round {}] gen={}/{} kept={} eval_before={:?} eval_after={:?} ensemble_max={}/{} loss[0]={:.4} t={}ms",
            line.round,
            line.gen_correct_total,
            args.n_models * args.gen_n,
            line.consensus_kept,
            line.per_model_eval_before,
            line.per_model_eval_after,
            line.ensemble_max_after,
            args.eval_n,
            line.last_train_loss_first,
            line.elapsed_ms,
        );
        // Update per-member checkpoint paths to the freshly-trained ones.
        current_ckpts = (0..args.n_models)
            .map(|i| {
                args.ckpt_prefix
                    .with_extension(format!("r{round}.{i}.safetensors"))
            })
            .collect();
        history.push(line);
    }

    println!("\n=== history ===");
    for h in &history {
        println!(
            "round {}: gen={}/{}  kept={}  eval_after={:?}  ensemble_max={}/{}",
            h.round,
            h.gen_correct_total,
            args.n_models * args.gen_n,
            h.consensus_kept,
            h.per_model_eval_after,
            h.ensemble_max_after,
            args.eval_n,
        );
    }
    Ok(())
}

struct RoundLine {
    round: usize,
    gen_correct_total: usize,
    consensus_kept: usize,
    per_model_eval_before: Vec<usize>,
    per_model_eval_after: Vec<usize>,
    ensemble_max_after: usize,
    last_train_loss_first: f32,
    elapsed_ms: u128,
}

#[allow(clippy::too_many_arguments)]
async fn run_one_round(
    round: usize,
    args: &Args,
    domain: &Arc<RustCodeDomain>,
    ensemble: &EnsembleActors,
    evaluators: &[ActorRef<EvaluatorActor>],
    verifier: &ActorRef<VerifierActor>,
    curator: &ActorRef<CuratorActor>,
    trainer: &ActorRef<TrainerActor>,
    gpt_cfg: &GPTConfig,
    gen_sampling: &GenerateConfig,
    eval_sampling: &GenerateConfig,
    current_ckpts: &[PathBuf],
    min_agreement: usize,
) -> anyhow::Result<RoundLine> {
    use std::time::Instant;
    let t0 = Instant::now();

    // 1. Eval before — per member.
    let mut per_model_eval_before = Vec::with_capacity(ensemble.n());
    for ev in evaluators {
        let r = ask_eval(ev, args.eval_n, 0xE5A2, eval_sampling.clone()).await?;
        per_model_eval_before.push(r.correct);
    }

    // 2. Generate — sample `gen_n` prompts, run all members on them.
    let prompts = sample_prompts(domain, args.gen_n, 0xC0DE_u64.wrapping_add(round as u64));
    let traj_per_model = ensemble_generate(
        ensemble,
        &prompts,
        1,
        gen_sampling,
        0xE0FF_u64.wrapping_add(round as u64),
    )
    .await?;

    // 3. Verify each member's batch through cargo.
    let mut items: Vec<EnsembleItem> = Vec::with_capacity(args.n_models * args.gen_n);
    let mut gen_correct_total = 0usize;
    for (i, trajs) in traj_per_model.into_iter().enumerate() {
        let (tx, rx) = oneshot::channel();
        verifier
            .tell(VerifierMessage::Verify {
                items: trajs,
                reply: tx,
            })
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let verified = rx.await?;
        for vt in verified {
            if vt.is_correct() {
                gen_correct_total += 1;
            }
            items.push(EnsembleItem {
                trajectory: vt.trajectory,
                verdict: vt.verdict,
                model_id: i,
            });
        }
    }

    // 4. Consensus curate.
    let (tx, rx) = oneshot::channel();
    curator
        .tell(CuratorMessage::AddEnsemble {
            items,
            n_models: ensemble.n(),
            min_agreement,
            reply: tx,
        })
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let report = rx.await?;
    let consensus_kept = report.accepted;

    // 5. Render shared corpus.
    let (tx, rx) = oneshot::channel();
    curator
        .tell(CuratorMessage::RenderCorpus {
            mode: SampleMode::Priority {
                recency_decay: 0.95,
            },
            seed: Some(round as u64 * 31 + 7),
            reply: tx,
        })
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let mut corpus = rx.await?;
    if !corpus.is_empty() && corpus.len() < 4_000 {
        let factor = 4_000usize.div_ceil(corpus.len());
        corpus = corpus.repeat(factor);
    }

    // 6. Train each member on the shared corpus, then reload.
    let mut last_train_loss_first = f32::NAN;
    for (i, init_path) in current_ckpts.iter().enumerate().take(ensemble.n()) {
        let save = args
            .ckpt_prefix
            .with_extension(format!("r{round}.{i}.safetensors"));
        let mut tcfg = TrainConfig::smoke();
        tcfg.max_steps = args.round_train_steps;
        tcfg.batch_size = 64;
        tcfg.eval_interval = args.round_train_steps;
        tcfg.lr = 5e-4;
        tcfg.min_lr = 5e-5;
        tcfg.warmup_steps = 20;

        let (tx, rx) = oneshot::channel();
        trainer
            .tell(TrainerMessage::Train {
                corpus: corpus.clone(),
                save_path: save.clone(),
                init_from: Some(init_path.clone()),
                train_cfg: tcfg,
                anchor: None,
                freeze_base: gpt_cfg.lora_rank > 0,
                reply: tx,
            })
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let outcome = rx.await??;
        if i == 0 {
            last_train_loss_first = outcome.last_train_loss;
        }

        let (tx, rx) = oneshot::channel();
        ensemble.models[i]
            .tell(ModelMessage::ReloadCheckpoint {
                path: save,
                reply: tx,
            })
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        rx.await??;
    }

    // 7. Eval after — per member.
    let mut per_model_eval_after = Vec::with_capacity(ensemble.n());
    for ev in evaluators {
        let r = ask_eval(ev, args.eval_n, 0xE5A2, eval_sampling.clone()).await?;
        per_model_eval_after.push(r.correct);
    }
    let ensemble_max_after = *per_model_eval_after.iter().max().unwrap_or(&0);

    Ok(RoundLine {
        round,
        gen_correct_total,
        consensus_kept,
        per_model_eval_before,
        per_model_eval_after,
        ensemble_max_after,
        last_train_loss_first,
        elapsed_ms: t0.elapsed().as_millis(),
    })
}

fn sample_prompts(domain: &Arc<RustCodeDomain>, n: usize, seed: u64) -> Vec<String> {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| domain.sample_prompt(&mut rng)).collect()
}

async fn seed_curator(
    curator: &ActorRef<CuratorActor>,
    items: Vec<VerifiedTrajectory>,
) -> anyhow::Result<()> {
    let (tx, rx) = oneshot::channel();
    curator
        .tell(CuratorMessage::Add { items, reply: tx })
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let report = rx.await?;
    info!(seeded = report.accepted, "curator seeded");
    Ok(())
}

async fn ask_eval(
    evaluator: &ActorRef<EvaluatorActor>,
    n: usize,
    seed: u64,
    sampling: GenerateConfig,
) -> anyhow::Result<EvalReport> {
    let (tx, rx) = oneshot::channel();
    evaluator
        .tell(EvaluatorMessage::Eval {
            n,
            seed,
            sampling,
            reply: tx,
        })
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    rx.await?
}
