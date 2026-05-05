//! Phase 2.5 Rust-code self-improve loop with the cargo-build verifier.
//!
//! `RustCodeDomain` writes `prompt + completion + suffix` into a scratch
//! Cargo project and runs `cargo run --offline --quiet`. Verdict is
//! `Correct` iff cargo exits 0 (compile + assert pass).
//!
//! Three challenges (see `domain::rust_code::DEFAULT_CHALLENGES`):
//!   - `equals_5`              : `fn main() { assert_eq!(<...>, 5); }`
//!   - `equals_14_via_doubling`: `fn main() { assert_eq!(2 * (<...>), 14); }`
//!   - `len_5_string`          : `fn main() { let s: &str = <...>; assert_eq!(s.len(), 5); }`
//!
//! Each challenge has a distinct prompt prefix so the verifier's
//! exact-prompt dispatch routes correctly. The model only emits the
//! `<...>` slot — typically 1–10 chars of arithmetic / integer /
//! string-literal. Char-level tokenizer is a natural fit (BPE would
//! over-fragment short ASCII patterns).
//!
//! Run:
//!   cargo run -p llm-actors --example self_improve_rust --features cuda --release -- \
//!       --rounds 2 --pretrain-steps 1500 --round-train-steps 400
//!
//! With the multi-challenge layout, the eval set draws prompts
//! uniformly across all three challenges. Greedy decode must produce
//! a slot appropriate to each prompt — i.e. cross-challenge
//! conditional generation, not a single memorized slot.
//!
//! ## Smoke result (4 rounds, 24 gen, 21 eval, 1500 pretrain, 400/round):
//!
//! With and without EWC the trajectory is identical at 4 rounds —
//! the round-1 collapse seen on a single 3-round run earlier was a
//! transient. Both runs:
//!
//!   round 0: gen 0/24 (0.0%)   eval before=0/21 after=8/21   Δ=+8
//!   round 1: gen 0/24 (0.0%)   eval before=8/21 after=7/21   Δ=-1
//!   round 2: gen 9/24 (37.5%)  eval before=7/21 after=8/21   Δ=+1
//!   round 3: gen 0/24 (0.0%)   eval before=8/21 after=8/21   Δ=+0
//!
//! Two findings:
//!
//! 1. **Round-2 stochastic gen 37.5%** is the first non-trivial
//!    sampling-side self-improve signal we've measured anywhere in
//!    the codebase (arithmetic capped near 30% on greedy + heuristic
//!    parsing; Korean stayed at 0% in greedy eval). 9/24 randomly
//!    sampled (temp=0.8, top_k=10) slots actually pass cargo's
//!    external check across three different challenges.
//!
//! 2. **EWC at λ=100 with real Fisher (64 batches) is
//!    indistinguishable from plain replay** at this task scale. The
//!    ~12-example replay buffer + the curator's priority sampling
//!    already keeps weights tight. Higher λ (1000+) or smaller
//!    `--round-train-steps` could surface a forgetting regime where
//!    EWC matters; at the smoke configuration it's no-op overhead.
//!    This matches Phase 4's "EWC vs ER net benefit unproven" result
//!    on the tool-use task — same finding generalizes here.
//!
//! ## K9 v5: LoRA-only fine-tune (`--lora-rank` + `--lora-alpha`)
//!
//! Per-round trainer freezes base weights and updates only the LoRA
//! adapters. Effective per-step scaling on the LoRA delta is
//! `alpha/rank` — separate axes from "trainable parameter count".
//!
//! Sweeping (rank, alpha) holding the rest fixed reveals that BOTH
//! axes matter independently:
//!
//! | r  | α  | scale | rounds 0–3      | peak  | stochastic gen |
//! |----|----|-------|-----------------|-------|----------------|
//! | 32 | 16 | 0.5   | 8/8/7/8/8       | 8     | 0%             |
//! |  8 |  4 | 0.5   | 7/0/0/15/0      | 15    | 0%             |
//! | 32 | 64 | 2.0   | 8/**15/15/8/8** | 15    | **9/24 (37.5%)** |
//! |  8 | 16 | 2.0   | 7/0/15/14/0     | 15    | 0%             |
//!
//! Hypotheses ruled out:
//! - "Effective scale α/r alone determines behavior": fails — r=32
//!   α=16 (stable) vs r=8 α=4 (unstable) share scale 0.5.
//! - "Rank alone determines behavior": fails — r=32 α=16 (stable, no
//!   spike) vs r=32 α=64 (immediate spike, recoverable) differ.
//!
//! Pattern that holds:
//! - **Rank controls stability**: high r → graceful learning + recovery
//!   from spikes; low r → brittle, peaks then crashes.
//! - **Scale controls learning aggressiveness**: high α/r → bigger
//!   per-round Δ in either direction; low α/r → smaller swings.
//!
//! Best configuration tested: **r=32, α=64 (scale 2.0)** — combines
//! aggressive learning (peak 15/21 immediately at round 0) with
//! enough capacity to *stay* there for round 1 *and* recover the
//! stochastic-gen 37.5% signal that previously only full-FT reached.
//! High rank gives the model enough parameters to find a generalizing
//! solution rather than a brittle fixed point.

use std::path::PathBuf;
use std::sync::Arc;

use candle_core::Device;
use candle_nn::{VarBuilder, VarMap};
use clap::Parser;
use llm_actors::{
    curator_actor::SampleMode,
    domain::{rust_code::RustCodeDomain, Domain},
    run_round, CuratorActor, CuratorMessage, EvaluatorActor, GeneratorActor, ModelActor,
    RoundActors, RoundConfig, TrainerActor, Trajectory, Verdict, VerifiedTrajectory, VerifierActor,
};
use nanogpt_rs::{
    config::GPTConfig,
    data::TokenDataset,
    ewc::WeightAnchor,
    generate::GenerateConfig,
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
    /// Pretrain steps before round 0.
    #[arg(long, default_value_t = 1500)]
    pretrain_steps: usize,
    /// Number of correct programs synthesized for pretrain.
    #[arg(long, default_value_t = 600)]
    pretrain_examples: usize,
    /// Per-round generations.
    #[arg(long, default_value_t = 24)]
    gen_n: usize,
    /// Per-round held-out eval count (same set across rounds).
    #[arg(long, default_value_t = 24)]
    eval_n: usize,
    /// Continual fine-tune steps per round.
    #[arg(long, default_value_t = 400)]
    round_train_steps: usize,
    /// Generation temperature.
    #[arg(long, default_value_t = 0.8)]
    gen_temperature: f64,
    /// Top-k for generation.
    #[arg(long, default_value_t = 10)]
    gen_top_k: usize,
    /// Max tokens to generate per completion. Long enough for ".len()"
    /// + a few literal chars.
    #[arg(long, default_value_t = 16)]
    max_new_tokens: usize,
    /// Scratch dir where the cargo verifier writes / runs.
    #[arg(long, default_value = "/tmp/workllm-rust-scratch")]
    scratch_dir: PathBuf,
    /// Seed checkpoint path.
    #[arg(long, default_value = "checkpoints/rust_seed.safetensors")]
    seed_ckpt: PathBuf,
    /// Round checkpoint stem (`.r{n}.safetensors` is appended).
    #[arg(long, default_value = "checkpoints/rust_round")]
    round_ckpt: PathBuf,
    /// `cargo check` (false) is faster but only validates compilation.
    /// `cargo run` (default true) actually executes the assert!.
    #[arg(long, default_value_t = true)]
    run_program: bool,
    /// EWC strength λ on the L2-anchor toward post-pretrain weights.
    /// 0.0 disables EWC (plain continual fine-tune). Higher pins
    /// weights closer to the pretrained state — exchanges learning
    /// capacity for forgetting resistance. Phase 4 found λ≈100 with
    /// real Fisher to be the sweet spot on the tool-use task.
    #[arg(long, default_value_t = 0.0)]
    ewc_lambda: f64,
    /// Number of pretrain batches used to estimate the diagonal Fisher
    /// for EWC. 0 = uniform Fisher (= L2 toward pretrain). 32–128 is
    /// the practical range for real Fisher.
    #[arg(long, default_value_t = 0)]
    fisher_batches: usize,
    /// LoRA rank. When `> 0`, pretrain trains everything (including
    /// the zero-init LoRA `lora_b` so it has no effect at init) and
    /// per-round continual fine-tune freezes the base weights —
    /// only `lora_*` adapters update. Phase 4 found r=32 across all
    /// linears was the best LoRA stability/capacity trade-off.
    #[arg(long, default_value_t = 0)]
    lora_rank: usize,
    /// LoRA scaling. Effective per-step delta is `alpha / rank`, so
    /// for a fixed rank, larger alpha → bigger swings. To separate
    /// "rank as capacity" from "alpha as step size", hold one fixed
    /// and sweep the other (e.g. r=8 α=4 vs r=32 α=16, both scale 0.5).
    #[arg(long, default_value_t = 16.0)]
    lora_alpha: f32,
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

/// Trajectory format the GeneratorActor produces:
///   prompt → completion(== slot) → '\n' (stop)
/// The verifier appends the challenge's `suffix` (e.g. `, 5); }\n`)
/// before running cargo, so the LM must NOT emit the suffix itself —
/// otherwise the suffix is duplicated and cargo rejects on syntax.
///
/// Each (prompt, slots) pair below pretrains the LM to emit a slot
/// that, when wrapped by the matching challenge's suffix, produces
/// a Rust program that compiles and the assertion passes. Slots are
/// hand-picked correct fillings; the model still has to learn
/// *which prompt → which slot space* via fine-tune.
const CHALLENGES: &[(&str, &[&str])] = &[
    // equals_5: `fn main() { assert_eq!(<slot>, 5); }`
    (
        "fn main() { assert_eq!(",
        &[
            "2 + 3", "1 + 4", "5 + 0", "0 + 5", "10 - 5", "5 * 1", "1 * 5", "100 / 20",
        ],
    ),
    // equals_14_via_doubling: `fn main() { assert_eq!(2 * (<slot>), 14); }`
    // Slot must equal 7.
    (
        "fn main() { assert_eq!(2 * (",
        &["7", "3 + 4", "4 + 3", "1 + 6", "6 + 1", "10 - 3", "14 / 2"],
    ),
    // len_5_string: `fn main() { let s: &str = <slot>; assert_eq!(s.len(), 5); }`
    // Slot must be a `&str` of length 5.
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

fn synth_pretrain_corpus(n: usize, seed: u64) -> String {
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = String::with_capacity(n * 64);
    for _ in 0..n {
        let (prompt, slots) = CHALLENGES.choose(&mut rng).expect("non-empty");
        let slot = slots.choose(&mut rng).expect("non-empty");
        // Train the LM to emit slot then stop (\n triggers GeneratorActor
        // termination). The verifier supplies the suffix at runtime.
        out.push_str(prompt);
        out.push_str(slot);
        out.push('\n');
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
    info!(?device, "device");

    // ---- Domain (cargo verifier).
    let mut rcd = RustCodeDomain::new(&args.scratch_dir);
    rcd.run_program = args.run_program;
    rcd.ensure_scratch_project()?;
    info!(scratch_dir = %args.scratch_dir.display(), "scratch project ready");
    let domain = Arc::new(rcd);

    // ---- Pretrain corpus.
    let pretrain_text = synth_pretrain_corpus(args.pretrain_examples, 7);
    let mut seed_chars = String::from(domain.charset());
    seed_chars.push_str(&pretrain_text);
    let tk = Arc::new(Tokenizer::char_from_text(&seed_chars));
    let vocab = tk.vocab_size();
    info!(
        vocab,
        corpus_chars = pretrain_text.len(),
        "tokenizer + corpus"
    );

    // ---- Model: small char-level. Block size large enough for one
    // full program (~50 chars) plus headroom.
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
        lora_scale = if gpt_cfg.lora_rank > 0 {
            gpt_cfg.lora_alpha / gpt_cfg.lora_rank as f32
        } else {
            0.0
        },
        "model config"
    );

    // ---- Pretrain.
    info!("pretraining...");
    let ids = tk.encode(&pretrain_text)?;
    let pretrain_ds = TokenDataset::new(ids, gpt_cfg.block_size);
    let mut pre_cfg = TrainConfig::smoke();
    pre_cfg.max_steps = args.pretrain_steps;
    pre_cfg.batch_size = 64;
    pre_cfg.eval_interval = args.pretrain_steps;
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

    // ---- Snapshot post-pretrain weights for the EWC anchor.
    // Rebuild a fresh VarMap and load the seed checkpoint so the snapshot
    // tensors are independent of the trainer's varmap (rebuilt per round
    // inside the trainer's blocking task).
    let anchor: Option<Arc<WeightAnchor>> = if args.ewc_lambda > 0.0 {
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        let _model = nanogpt_rs::model::GPT::new(gpt_cfg.clone(), vb)?;
        varmap.load(&args.seed_ckpt)?;
        let a = if args.fisher_batches > 0 {
            info!(
                fisher_batches = args.fisher_batches,
                "estimating Fisher diagonal from pretrain data"
            );
            WeightAnchor::snapshot_with_fisher(
                &gpt_cfg,
                &varmap,
                &pretrain_ds,
                args.fisher_batches,
                64,
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

    // ---- Actors.
    let model_actor =
        ModelActor::from_checkpoint(gpt_cfg.clone(), device.clone(), tk.clone(), &args.seed_ckpt)?;
    let system = ActorSystem::new("self-improve-rust");
    let model_ref = system.spawn(model_actor, "model").await?;

    let curator = CuratorActor::new(512);
    let curator_ref = system.spawn(curator, "curator").await?;
    seed_curator_from_synthetic(&curator_ref, 32).await?;

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

    // ---- Rounds.
    let mut current_ckpt = args.seed_ckpt.clone();
    let mut history = Vec::new();
    for round in 0..args.rounds {
        let round_save = args
            .round_ckpt
            .with_extension(format!("r{round}.safetensors"));

        let mut train_cfg = TrainConfig::smoke();
        train_cfg.max_steps = args.round_train_steps;
        train_cfg.batch_size = 64;
        train_cfg.eval_interval = args.round_train_steps;
        train_cfg.lr = 5e-4;
        train_cfg.min_lr = 5e-5;
        train_cfg.warmup_steps = 20;

        let cfg = RoundConfig {
            round,
            gen_n: args.gen_n,
            gen_seed: 100 + round as u64,
            gen_sampling: GenerateConfig {
                max_new_tokens: args.max_new_tokens,
                temperature: args.gen_temperature,
                top_k: Some(args.gen_top_k),
                top_p: None,
                seed: Some(round as u64),
            },
            eval_n: args.eval_n,
            eval_seed: 0xE5A2,
            eval_sampling: GenerateConfig {
                max_new_tokens: args.max_new_tokens,
                temperature: 0.0,
                top_k: Some(1),
                top_p: None,
                seed: Some(0xE5A2),
            },
            train_cfg,
            init_from: Some(current_ckpt.clone()),
            save_path: round_save.clone(),
            min_corpus_chars: 4_000,
            sample_mode: SampleMode::Priority {
                recency_decay: 0.95,
            },
            corpus_seed: Some(round as u64 * 31 + 7),
            anchor: anchor.clone(),
            freeze_base: args.lora_rank > 0,
        };

        let report = run_round(&actors, cfg).await?;
        println!(
            "[round {round}] gen_correct={}/{} ({:.1}%)  eval before={}/{}  after={}/{}  Δ={:+}  loss={:?}  t={}ms",
            report.correct,
            report.generated,
            100.0 * report.pass_rate_generated(),
            report.eval_correct_before.unwrap_or(0),
            report.eval_total,
            report.eval_correct_after.unwrap_or(0),
            report.eval_total,
            report.eval_correct_after.unwrap_or(0) as i64
                - report.eval_correct_before.unwrap_or(0) as i64,
            report.last_train_loss,
            report.elapsed_ms,
        );
        history.push(report);
        current_ckpt = round_save;
    }

    println!("\n=== history ===");
    for r in &history {
        let before = r.eval_correct_before.unwrap_or(0);
        let after = r.eval_correct_after.unwrap_or(0);
        println!(
            "round {}: gen={}/{} ({:.1}%)  eval before={}/{} after={}/{}  Δ={:+}",
            r.round,
            r.correct,
            r.generated,
            100.0 * r.pass_rate_generated(),
            before,
            r.eval_total,
            after,
            r.eval_total,
            after as i64 - before as i64,
        );
    }
    Ok(())
}

/// Pre-seed the curator with synthetic correct trajectories so round 0
/// has training material even if the model's first generations all fail
/// cargo. Same trajectory format as what GeneratorActor would emit.
async fn seed_curator_from_synthetic(
    curator: &pekko_actor::ActorRef<CuratorActor>,
    n: usize,
) -> anyhow::Result<()> {
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    let mut rng = StdRng::seed_from_u64(0xC0DE);
    let mut items: Vec<VerifiedTrajectory> = Vec::with_capacity(n);
    for _ in 0..n {
        let (prompt, slots) = CHALLENGES.choose(&mut rng).expect("non-empty");
        let slot = slots.choose(&mut rng).expect("non-empty");
        items.push(VerifiedTrajectory {
            trajectory: Trajectory {
                prompt: prompt.to_string(),
                completion: format!("{slot}\n"),
                source: "synthetic-seed".to_string(),
            },
            verdict: Verdict::Correct,
            score: 1.0,
        });
    }
    info!(count = items.len(), "seeding curator");
    let (tx, rx) = oneshot::channel();
    curator
        .tell(CuratorMessage::Add { items, reply: tx })
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let report = rx.await?;
    info!(seeded = report.accepted, "curator seeded");
    Ok(())
}
