//! Phase 4 epilogue: KoreanCompletionDomain self-improve loop on a
//! KoWiki-pretrained 50M model.
//!
//! Differs from `self_improve_round.rs` (arithmetic toy):
//!   - Domain: heuristic Korean completion verifier (Hangul + sentence
//!     ending + length window). No ground-truth label set.
//!   - Tokenizer: pretrained 16K BPE loaded from disk.
//!   - Init: K8's 30K-step KoWiki Llama-recipe checkpoint.
//!   - Pretrain phase is skipped — the model is already trained.
//!   - Curator is seeded from KoWiki itself (lines that pass the
//!     heuristic), so round 0 has *something* to train on even if the
//!     model's own generations all fail the verifier (which they
//!     mostly do at val_loss 7.43).
//!
//! The signal we're looking for: pass-rate on a fixed eval prompt set
//! climbing across rounds. With a 7.43-val_loss model, baseline pass
//! rate may be 0–10%; any round-after pass_rate increase over
//! round-before is a real self-improvement signal — and the heuristic
//! is independent of the training corpus, so the metric isn't
//! contaminated by curator content.
//!
//! ## Observed at the K8 30K-step checkpoint scale
//!
//! Smoke run with `--rounds 2 --gen-n 32 --eval-n 16
//! --round-train-steps 100`:
//!
//!   round 0: gen 0/32 (0.0%)  eval before=0/16 after=0/16  Δ=+0
//!   round 1: gen 2/32 (6.2%)  eval before=0/16 after=0/16  Δ=+0
//!
//! The generation phase (sampling at temperature 0.8) shows a real
//! signal — pass rate climbs from 0% to 6.2% after one round of
//! continual fine-tune. But the greedy eval (temperature 0) stays at
//! 0/16 because greedy decode on a 7.43-val_loss model produces
//! degenerate output ("low사관 변동..." etc) that the Hangul-and-
//! sentence-ending heuristic always rejects. The metric is too
//! strict at this teacher quality. Until the underlying KoWiki model
//! reaches fluent-Korean territory (val_loss ≤ ~5.5 nats, ~150–300M
//! params or much more diverse data — see
//! `docs/distillation-postmortem.md`), eval pass-rate Δ will likely
//! stay at 0 and only the generation Δ is informative.
//!
//! Run:
//!   cargo run -p llm-actors --example self_improve_korean --features cuda --release -- \
//!       --rounds 3 \
//!       --init checkpoints/kowiki_50m_30k.safetensors \
//!       --tokenizer data/kowiki/kowiki_bpe.json \
//!       --corpus data/kowiki/kowiki_clean.txt

use std::path::PathBuf;
use std::sync::Arc;

use candle_core::Device;
use clap::Parser;
use llm_actors::{
    curator_actor::SampleMode,
    domain::{korean_completion::KoreanCompletionDomain, Domain},
    run_round, CuratorActor, CuratorMessage, EvaluatorActor, GeneratorActor, ModelActor,
    RoundActors, RoundConfig, TrainerActor, TrainerActorHandle, Trajectory, Verdict,
    VerifiedTrajectory, VerifierActor,
};
use nanogpt_rs::{
    config::GPTConfig, generate::GenerateConfig, tokenizer::Tokenizer, train::TrainConfig,
};
use pekko_actor::ActorSystem;
use tokio::sync::oneshot;
use tracing::{info, warn};

#[derive(Parser, Debug)]
struct Args {
    /// Number of self-improve rounds to run.
    #[arg(long, default_value_t = 3)]
    rounds: usize,
    /// Pretrained KoWiki checkpoint (saved by `train_kowiki`). Must have
    /// a sibling `.cfg.json` describing the architecture.
    #[arg(long, default_value = "checkpoints/kowiki_50m_30k.safetensors")]
    init: PathBuf,
    /// Pretrained BPE tokenizer.
    #[arg(long, default_value = "data/kowiki/kowiki_bpe.json")]
    tokenizer: PathBuf,
    /// Plaintext corpus to mine seed examples from.
    #[arg(long, default_value = "data/kowiki/kowiki_clean.txt")]
    corpus: PathBuf,
    /// Where to save round-N checkpoints. The example appends `.r{n}.safetensors`.
    #[arg(long, default_value = "checkpoints/kowiki_self_improve")]
    save_prefix: PathBuf,
    /// Number of generations per round (verifier is a heuristic, so most
    /// fail at the start — keep this generous).
    #[arg(long, default_value_t = 64)]
    gen_n: usize,
    /// Number of held-out generations evaluated before/after each round.
    /// Same prompt set across rounds (fixed seed).
    #[arg(long, default_value_t = 32)]
    eval_n: usize,
    /// Continual fine-tune steps per round.
    #[arg(long, default_value_t = 200)]
    round_train_steps: usize,
    /// Per-round training batch size.
    #[arg(long, default_value_t = 8)]
    train_batch: usize,
    /// Per-round LR.
    #[arg(long, default_value_t = 5e-5)]
    lr: f64,
    /// Generation temperature for the gen step (NOT eval).
    #[arg(long, default_value_t = 0.8)]
    gen_temperature: f64,
    /// Top-k for generation.
    #[arg(long, default_value_t = 40)]
    gen_top_k: usize,
    /// Max tokens to generate per completion.
    #[arg(long, default_value_t = 80)]
    max_new_tokens: usize,
    /// Number of high-quality KoWiki lines to seed the curator with.
    #[arg(long, default_value_t = 64)]
    seed_examples: usize,
    /// Curator buffer capacity.
    #[arg(long, default_value_t = 1024)]
    curator_capacity: usize,
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

fn load_cfg(path: &std::path::Path) -> anyhow::Result<GPTConfig> {
    let cfg_path = path.with_extension("cfg.json");
    let s = std::fs::read_to_string(&cfg_path)
        .map_err(|e| anyhow::anyhow!("read {:?}: {e}", cfg_path))?;
    Ok(serde_json::from_str(&s)?)
}

/// Mine the corpus for lines that already pass `KoreanCompletionDomain.verify`.
/// We pair each with a SEED_PROMPT prefix so the trajectory format matches
/// what the generator will produce. The "completion" is the line itself
/// minus the prompt, when it happens to start with one — otherwise the
/// completion is the full line and the prompt comes from the domain's
/// fixed list (we don't try to be clever about prompt-completion alignment;
/// the verifier doesn't depend on that).
fn mine_seed_examples(
    corpus_text: &str,
    domain: &KoreanCompletionDomain,
    n: usize,
    rng_seed: u64,
) -> Vec<VerifiedTrajectory> {
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    let mut rng = StdRng::seed_from_u64(rng_seed);
    let mut prompts_rng = StdRng::seed_from_u64(rng_seed.wrapping_add(1));

    // Candidate sentences: split on `다.` `요.` `까?` so each candidate
    // ends with a sentence terminator.
    let mut candidates: Vec<String> = Vec::new();
    for raw in corpus_text.split_inclusive(['.', '?']) {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Cheap filter: keep only sentences in 20–300 chars.
        let n_chars = trimmed.chars().count();
        if !(20..=300).contains(&n_chars) {
            continue;
        }
        candidates.push(trimmed.to_string());
        if candidates.len() >= 50_000 {
            break; // we only need n, not the whole corpus
        }
    }
    candidates.shuffle(&mut rng);

    // Verify each candidate with the actual domain.verify; keep those
    // that pass.
    let mut out: Vec<VerifiedTrajectory> = Vec::new();
    for cand in candidates {
        if out.len() >= n {
            break;
        }
        if matches!(domain.verify("", &cand), Verdict::Correct) {
            // Pair with a random fixed-list prompt so the training
            // example resembles what the generator emits.
            let prompt = domain.sample_prompt(&mut prompts_rng);
            out.push(VerifiedTrajectory {
                trajectory: Trajectory {
                    prompt,
                    completion: format!("{cand}\n"),
                    source: "kowiki-mined".to_string(),
                },
                verdict: Verdict::Correct,
                score: 1.0,
            });
        }
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

    // ---- Load architecture config from sibling .cfg.json so we exactly
    // match the saved checkpoint (block_size, vocab, etc).
    let gpt_cfg = load_cfg(&args.init)?;
    info!(
        params = gpt_cfg.num_params_estimate(),
        n_layer = gpt_cfg.n_layer,
        n_embd = gpt_cfg.n_embd,
        block_size = gpt_cfg.block_size,
        vocab = gpt_cfg.vocab_size,
        "loaded model config"
    );

    // ---- Tokenizer.
    let tk = Arc::new(Tokenizer::from_hf_file(&args.tokenizer)?);
    let vocab = tk.vocab_size();
    if vocab != gpt_cfg.vocab_size {
        anyhow::bail!(
            "tokenizer vocab {vocab} ≠ checkpoint vocab {} — pass the matching --tokenizer",
            gpt_cfg.vocab_size
        );
    }

    // ---- Domain + corpus mining for seed.
    let domain = Arc::new(KoreanCompletionDomain::default());
    let corpus_text = std::fs::read_to_string(&args.corpus)
        .map_err(|e| anyhow::anyhow!("read corpus {:?}: {e}", args.corpus))?;
    info!(corpus_chars = corpus_text.len(), "corpus loaded");
    let seeds = mine_seed_examples(&corpus_text, &domain, args.seed_examples, 0xE0);
    info!(
        kept = seeds.len(),
        target = args.seed_examples,
        "mined seed examples"
    );
    if seeds.is_empty() {
        warn!("no seed examples passed the heuristic — round 0 may train on an empty corpus");
    }

    // ---- Actors.
    let model_actor =
        ModelActor::from_checkpoint(gpt_cfg.clone(), device.clone(), tk.clone(), &args.init)?;
    let system = ActorSystem::new("self-improve-korean");
    let model_ref = system.spawn(model_actor, "model").await?;

    let curator = CuratorActor::new(args.curator_capacity);
    let curator_ref = system.spawn(curator, "curator").await?;
    seed_curator(&curator_ref, seeds).await?;

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
        trainer: Arc::new(TrainerActorHandle::new(trainer_ref)),
        evaluator: evaluator_ref,
    };

    // ---- Run rounds.
    let mut current_ckpt = args.init.clone();
    let mut history = Vec::new();
    for round in 0..args.rounds {
        let round_save = args
            .save_prefix
            .with_extension(format!("r{round}.safetensors"));

        let mut train_cfg = TrainConfig::smoke();
        train_cfg.max_steps = args.round_train_steps;
        train_cfg.batch_size = args.train_batch;
        train_cfg.eval_interval = args.round_train_steps;
        train_cfg.lr = args.lr;
        train_cfg.min_lr = args.lr * 0.1;
        train_cfg.warmup_steps = (args.round_train_steps / 30).max(20);

        let cfg = RoundConfig {
            round,
            gen_n: args.gen_n,
            gen_seed: 1000 + round as u64,
            gen_sampling: GenerateConfig {
                max_new_tokens: args.max_new_tokens,
                temperature: args.gen_temperature,
                top_k: Some(args.gen_top_k),
                top_p: None,
                seed: Some(round as u64),
            },
            eval_n: args.eval_n,
            // Same eval set across rounds → before/after numbers are comparable.
            eval_seed: 0xE5A1,
            eval_sampling: GenerateConfig {
                max_new_tokens: args.max_new_tokens,
                temperature: 0.0,
                top_k: Some(1),
                top_p: None,
                seed: Some(0xE5A1),
            },
            train_cfg,
            init_from: Some(current_ckpt.clone()),
            save_path: round_save.clone(),
            min_corpus_chars: 32_000,
            sample_mode: SampleMode::Priority {
                recency_decay: 0.95,
            },
            corpus_seed: Some(round as u64 * 31 + 7),
            anchor: None,
            freeze_base: false,
            gen_oversample: 1,
            dpo_beta: None,
            dpo_reference_path: None,
            dpo_max_pairs_per_prompt: 0,
            dpo_sft_anchor_weight: 0.0,
            eval_passk: 1,
            sft_mask_prompt: true,
            samples_per_prompt: None,
        };

        let report = run_round(&actors, cfg).await?;
        println!(
            "[round {round}] generated_correct={}/{} ({:.1}%)  eval_before={}/{}  eval_after={}/{}  Δ={:+}  train_loss={:?}  elapsed={}ms",
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

async fn seed_curator(
    curator: &pekko_actor::ActorRef<CuratorActor>,
    items: Vec<VerifiedTrajectory>,
) -> anyhow::Result<()> {
    if items.is_empty() {
        return Ok(());
    }
    info!(count = items.len(), "seeding curator from mined corpus");
    let (tx, rx) = oneshot::channel();
    curator
        .tell(CuratorMessage::Add { items, reply: tx })
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let report = rx.await?;
    info!(seeded = report.accepted, "curator seeded");
    Ok(())
}
