//! Round orchestration for the self-improvement loop.
//!
//! Implemented as an async function rather than an actor because the round
//! is a single linear flow over already-async actor handles. We can promote
//! it to a real `SupervisorActor` later if we need supervision (restart
//! children on failure, halt on regression, etc).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use nanogpt_rs::{ewc::WeightAnchor, generate::GenerateConfig, train::TrainConfig};
use pekko_actor::ActorRef;
use tokio::sync::oneshot;
use tracing::info;

use crate::curator_actor::{CuratorActor, CuratorMessage, SampleMode};
use crate::evaluator_actor::{EvalReport, EvaluatorActor, EvaluatorMessage};
use crate::generator_actor::{GeneratorActor, GeneratorMessage};
use crate::model_actor::{ModelActor, ModelMessage};
use crate::trainer_actor::{TrainerActor, TrainerMessage};
use crate::types::RoundReport;
use crate::verifier_actor::{VerifierActor, VerifierMessage};

pub struct RoundActors {
    pub model: ActorRef<ModelActor>,
    pub generator: ActorRef<GeneratorActor>,
    pub verifier: ActorRef<VerifierActor>,
    pub curator: ActorRef<CuratorActor>,
    pub trainer: ActorRef<TrainerActor>,
    pub evaluator: ActorRef<EvaluatorActor>,
}

pub struct RoundConfig {
    pub round: usize,
    pub gen_n: usize,
    pub gen_seed: u64,
    pub gen_sampling: GenerateConfig,
    pub eval_n: usize,
    pub eval_seed: u64,
    pub eval_sampling: GenerateConfig,
    pub train_cfg: TrainConfig,
    pub init_from: Option<PathBuf>,
    pub save_path: PathBuf,
    /// If the curated corpus is shorter than this, repeat-tile it. Useful for
    /// small replay buffers where each round otherwise overfits a tiny window.
    pub min_corpus_chars: usize,
    /// Buffer iteration order for corpus rendering.
    pub sample_mode: SampleMode,
    /// Seed for the corpus-rendering RNG (only used when `sample_mode` is
    /// `Priority`).
    pub corpus_seed: Option<u64>,
    /// Optional EWC-style weight anchor passed through to the trainer.
    /// `None` (default) is plain continual fine-tune; `Some` adds the
    /// anchor's penalty term to every step's loss.
    pub anchor: Option<Arc<WeightAnchor>>,
    /// When `true`, the trainer freezes all non-LoRA Vars during the
    /// per-round fine-tune — only `lora_*` adapters get gradient updates.
    /// Requires the GPTConfig used to spawn the trainer to have
    /// `lora_rank > 0`. `false` is plain full-parameter fine-tune.
    pub freeze_base: bool,
    /// Phase 6 Shape C: oversample factor for the gen step. `1`
    /// (default) = current behavior. `> 1` = generate this many
    /// candidates per prompt, score each via the model's own log-prob,
    /// keep the highest. Cargo budget unchanged; per-prompt selection
    /// becomes critic-driven.
    pub gen_oversample: usize,
    /// Phase 11 S2: when `Some(beta)`, the round's training step uses
    /// DPO — `(prompt, chosen, rejected)` triples are rendered from the
    /// curator and fed to `train_dpo`. `None` (default) uses the
    /// existing SFT path through `RenderCorpus` + `train_from`.
    pub dpo_beta: Option<f64>,
    /// Phase 11 S2: reference checkpoint for DPO. Required when
    /// `dpo_beta` is `Some`. Held frozen during fine-tune; provides the
    /// `π_ref` half of the DPO objective. Typically the SFT-trained
    /// init checkpoint.
    pub dpo_reference_path: Option<PathBuf>,
    /// Phase 11 S2: cap on (chosen, rejected) cross-pairs per prompt
    /// when rendering DPO training data. Defaults to 4 if unspecified.
    /// Larger values produce more training pairs but bias toward
    /// prompts with prolific verifier hits + misses.
    pub dpo_max_pairs_per_prompt: usize,
}

pub async fn run_round(actors: &RoundActors, cfg: RoundConfig) -> anyhow::Result<RoundReport> {
    let t0 = Instant::now();
    let mut report = RoundReport {
        round: cfg.round,
        eval_total: cfg.eval_n,
        ..RoundReport::default()
    };

    // 1. Eval BEFORE
    info!(round = cfg.round, "phase: eval-before");
    let before = ask_eval(
        &actors.evaluator,
        cfg.eval_n,
        cfg.eval_seed,
        cfg.eval_sampling.clone(),
    )
    .await?;
    report.eval_correct_before = Some(before.correct);
    info!(
        round = cfg.round,
        before_correct = before.correct,
        total = before.total,
        "eval-before done"
    );
    log_samples("before", &before);

    // 2. Generate
    info!(round = cfg.round, "phase: generate");
    let (tx, rx) = oneshot::channel();
    actors
        .generator
        .tell(GeneratorMessage::GenerateBatch {
            n: cfg.gen_n,
            seed: cfg.gen_seed,
            sampling: cfg.gen_sampling,
            oversample: cfg.gen_oversample.max(1),
            reply: tx,
        })
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let trajectories = rx.await??;
    report.generated = trajectories.len();

    // 3. Verify
    info!(round = cfg.round, "phase: verify");
    let (tx, rx) = oneshot::channel();
    actors
        .verifier
        .tell(VerifierMessage::Verify {
            items: trajectories,
            reply: tx,
        })
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let verified = rx.await?;
    report.correct = verified.iter().filter(|v| v.is_correct()).count();

    // 4. Curate
    info!(round = cfg.round, "phase: curate");
    let (tx, rx) = oneshot::channel();
    actors
        .curator
        .tell(CuratorMessage::Add {
            items: verified,
            reply: tx,
        })
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let _add_report = rx.await?;

    // 5. Train. Phase 11 S2 fork: DPO if `dpo_beta` set, otherwise the
    // existing SFT path (RenderCorpus + train_from).
    let outcome = if let Some(beta) = cfg.dpo_beta {
        let reference_path = cfg
            .dpo_reference_path
            .clone()
            .ok_or_else(|| anyhow::anyhow!("dpo_beta is Some but dpo_reference_path is None"))?;
        let init_from = cfg
            .init_from
            .clone()
            .ok_or_else(|| anyhow::anyhow!("dpo_beta is Some but init_from is None"))?;
        let max_per_prompt = if cfg.dpo_max_pairs_per_prompt == 0 {
            4
        } else {
            cfg.dpo_max_pairs_per_prompt
        };
        let (tx, rx) = oneshot::channel();
        actors
            .curator
            .tell(CuratorMessage::RenderPreferencePairs {
                max_per_prompt,
                reply: tx,
            })
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let pairs = rx.await?;
        if pairs.is_empty() {
            info!(
                round = cfg.round,
                "skip DPO training: zero pairs from curator"
            );
            report.elapsed_ms = t0.elapsed().as_millis();
            return Ok(report);
        }
        info!(
            round = cfg.round,
            n_pairs = pairs.len(),
            beta,
            "phase: train (DPO)"
        );
        let (tx, rx) = oneshot::channel();
        actors
            .trainer
            .tell(TrainerMessage::TrainDpo {
                pairs,
                save_path: cfg.save_path.clone(),
                init_from,
                reference_path,
                train_cfg: cfg.train_cfg.clone(),
                beta,
                reply: tx,
            })
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        rx.await??
    } else {
        let (tx, rx) = oneshot::channel();
        actors
            .curator
            .tell(CuratorMessage::RenderCorpus {
                mode: cfg.sample_mode,
                seed: cfg.corpus_seed,
                reply: tx,
            })
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let mut corpus = rx.await?;
        if !corpus.is_empty() && corpus.len() < cfg.min_corpus_chars {
            let factor = cfg.min_corpus_chars.div_ceil(corpus.len());
            corpus = corpus.repeat(factor);
            info!(
                round = cfg.round,
                corpus_chars = corpus.len(),
                factor,
                "padded corpus"
            );
        }
        if corpus.is_empty() {
            info!(round = cfg.round, "skip training: empty corpus");
            report.elapsed_ms = t0.elapsed().as_millis();
            return Ok(report);
        }
        info!(round = cfg.round, "phase: train (SFT)");
        let (tx, rx) = oneshot::channel();
        actors
            .trainer
            .tell(TrainerMessage::Train {
                corpus,
                save_path: cfg.save_path.clone(),
                init_from: cfg.init_from.clone(),
                train_cfg: cfg.train_cfg.clone(),
                anchor: cfg.anchor.clone(),
                freeze_base: cfg.freeze_base,
                reply: tx,
            })
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        rx.await??
    };
    report.training_steps = outcome.final_step;
    report.last_train_loss = Some(outcome.last_train_loss);

    // 6. ModelActor reload
    info!(round = cfg.round, "phase: reload");
    let (tx, rx) = oneshot::channel();
    actors
        .model
        .tell(ModelMessage::ReloadCheckpoint {
            path: cfg.save_path.clone(),
            reply: tx,
        })
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    rx.await??;

    // 7. Eval AFTER
    info!(round = cfg.round, "phase: eval-after");
    let after = ask_eval(
        &actors.evaluator,
        cfg.eval_n,
        cfg.eval_seed,
        cfg.eval_sampling,
    )
    .await?;
    report.eval_correct_after = Some(after.correct);
    log_samples("after", &after);

    report.elapsed_ms = t0.elapsed().as_millis();
    Ok(report)
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

fn log_samples(tag: &str, eval: &EvalReport) {
    for (i, s) in eval.samples.iter().take(3).enumerate() {
        info!(
            tag,
            i,
            prompt = %s.prompt,
            completion = %s.completion.replace('\n', "\\n"),
            "sample"
        );
    }
}
