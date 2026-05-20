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
use pekko_actor::{Actor, ActorRef};
use tokio::sync::oneshot;
use tracing::info;

use crate::curator_actor::{CuratorActor, CuratorMessage, SampleMode};
use crate::evaluator_actor::{EvalReport, EvaluatorActor, EvaluatorMessage};
use crate::generator_actor::{GeneratorActor, GeneratorMessage};
use crate::model_actor::{ModelActor, ModelMessage};
use crate::trainer_handle::{TrainRequest, TrainerHandle};
use crate::types::RoundReport;
use crate::verifier_actor::{VerifierActor, VerifierMessage};

/// Phase 21 Stage E — generic over the backing model actor type so
/// `run_round` / `run_multi_round` drive both `ModelActor` (nanogpt_rs)
/// and `QwenModelActor` (Candle-native Qwen2) with the same flow.
/// Default `M = ModelActor` preserves every existing call site.
pub struct RoundActors<M = ModelActor>
where
    M: Actor<Message = ModelMessage>,
{
    pub model: ActorRef<M>,
    pub generator: ActorRef<GeneratorActor<M>>,
    pub verifier: ActorRef<VerifierActor>,
    pub curator: ActorRef<CuratorActor>,
    /// Phase 21 Stage H — trainer is now polymorphic via the
    /// `TrainerHandle` trait. Wrap an `ActorRef<TrainerActor>` in
    /// `TrainerActorHandle` for the historical nanogpt_rs path, or
    /// an `ActorRef<QwenTrainerActor>` in `QwenTrainerActorHandle`
    /// for the Candle-native Qwen2 LoRA path. The trait abstracts
    /// `Train` / `TrainDpo` dispatch and adapter merging.
    pub trainer: Arc<dyn TrainerHandle>,
    pub evaluator: ActorRef<EvaluatorActor<M>>,
}

#[derive(Clone)]
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
    /// Phase 11 S5: hybrid SFT+DPO weight. `0.0` (default) is pure DPO
    /// — the S2/S3/S4 behavior. `> 0.0` mixes an SFT anchor on the
    /// chosen completions: `loss = (1 - α)·DPO + α·SFT_chosen` where
    /// α is this field. Adds CE-on-chosen as a stabilizer to prevent
    /// the multi-round mode collapse seen in S3/S4. Ignored when
    /// `dpo_beta` is `None`.
    pub dpo_sft_anchor_weight: f64,
    /// Phase 21: pass@k at the eval-before / eval-after phases of the
    /// round. `1` (default) is historical pass@1. `> 1` samples k
    /// completions per prompt and counts a prompt correct if ANY of
    /// them verifies — the inference-time-scaling axis discovered in
    /// Phase 17 S6 (base Qwen pass@10 = 0.524 vs pass@1 = 0.216 on
    /// HumanEval). At the small Candle scales used here it surfaces
    /// stochastic-decode capability the greedy eval misses.
    pub eval_passk: usize,
    /// Phase 22 Stage D fix — when `true`, the supervisor pulls
    /// `(prompt, completion)` pairs from the curator alongside the
    /// concatenated corpus and passes both into `TrainRequest::Sft`.
    /// Trainers that support completion-only loss
    /// (`QwenTrainerActorHandle`) use the pairs and mask prompt
    /// positions out of CE (Phase 17 Python recipe). Trainers
    /// without masking (`TrainerActorHandle` on nanogpt_rs) ignore
    /// the pairs and fall back to the corpus string.
    ///
    /// Default `true` because Phase 22 A-batch + G1 + G2 batches
    /// all confirmed that prompt-unmasked SFT catastrophically
    /// over-trains at HumanEval/MBPP prompt scales.
    pub sft_mask_prompt: bool,
    /// Phase 22 Stage D G6 — when `Some(k)`, the generate phase uses
    /// `GeneratorMessage::GenerateSystematic` (every prompt × k
    /// completions) instead of `GenerateBatch` (`gen_n` random
    /// with-replacement draws). `gen_n` is ignored in this mode.
    /// Matches Phase 17 `self_improve.py --samples k`: 164 prompts ×
    /// 6 = 984 attempts/round → ~210 verifier-passed training pairs,
    /// vs ~10 from with-replacement sampling. Requires the domain to
    /// implement `n_prompts`/`nth_prompt`. `None` (default) preserves
    /// the with-replacement behavior.
    pub samples_per_prompt: Option<usize>,
}

pub async fn run_round<M>(actors: &RoundActors<M>, cfg: RoundConfig) -> anyhow::Result<RoundReport>
where
    M: Actor<Message = ModelMessage>,
{
    run_round_with_prev(actors, cfg, None).await
}

/// Phase 22 Stage D follow-up #1 — pipelined `run_multi_round`
/// optimization. Round N+1's eval-before is deterministically equal
/// to round N's eval-after (same model state, same `eval_seed`,
/// same `eval_sampling`, same `eval_n` since `run_multi_round`
/// doesn't change eval params per round). Passing the cached value
/// in `prev_eval_after` skips re-running eval-before, saving ~5 min
/// per round at gen-n=164 + eval-n=32 + passk=3.
///
/// `prev_eval_after` is `None` for the first round (no previous
/// model to reuse from) and `Some(prev_report)` for rounds 1+ when
/// driven by `run_multi_round`. Direct callers of `run_round` get
/// the historical full-eval behavior.
pub async fn run_round_with_prev<M>(
    actors: &RoundActors<M>,
    cfg: RoundConfig,
    prev_eval_after: Option<EvalReport>,
) -> anyhow::Result<RoundReport>
where
    M: Actor<Message = ModelMessage>,
{
    let t0 = Instant::now();
    let mut report = RoundReport {
        round: cfg.round,
        eval_total: cfg.eval_n,
        ..RoundReport::default()
    };

    // 1. Eval BEFORE — reuse previous round's eval-after when available.
    // Same model state + same eval params → deterministic, no need to
    // re-run the eval. Saves ~5 min per round at gen-n=164.
    let before = if let Some(prev) = prev_eval_after {
        info!(
            round = cfg.round,
            "phase: eval-before (reused from previous round's eval-after)"
        );
        prev
    } else {
        info!(round = cfg.round, "phase: eval-before");
        let before = ask_eval(
            &actors.evaluator,
            cfg.eval_n,
            cfg.eval_seed,
            cfg.eval_sampling.clone(),
            cfg.eval_passk,
        )
        .await?;
        info!(
            round = cfg.round,
            before_correct = before.correct,
            total = before.total,
            "eval-before done"
        );
        log_samples("before", &before);
        before
    };
    report.eval_correct_before = Some(before.correct);

    // 2. Generate — systematic (every prompt × k) when
    // `samples_per_prompt` is set (Phase 17 G6 recipe), else the
    // historical `gen_n` with-replacement batch.
    info!(round = cfg.round, "phase: generate");
    let (tx, rx) = oneshot::channel();
    let gen_msg = match cfg.samples_per_prompt {
        Some(k) => GeneratorMessage::GenerateSystematic {
            samples_per_prompt: k,
            seed: cfg.gen_seed,
            sampling: cfg.gen_sampling,
            reply: tx,
        },
        None => GeneratorMessage::GenerateBatch {
            n: cfg.gen_n,
            seed: cfg.gen_seed,
            sampling: cfg.gen_sampling,
            oversample: cfg.gen_oversample.max(1),
            reply: tx,
        },
    };
    actors
        .generator
        .tell(gen_msg)
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
    // existing SFT path (RenderCorpus + trainer). Phase 21 Stage H:
    // dispatch via `TrainerHandle` so both nanogpt_rs and Qwen2 LoRA
    // trainers can drive this loop.
    let train_req = if let Some(beta) = cfg.dpo_beta {
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
        TrainRequest::Dpo {
            pairs,
            save_path: cfg.save_path.clone(),
            init_from,
            reference_path,
            train_cfg: cfg.train_cfg.clone(),
            beta,
            sft_anchor_weight: cfg.dpo_sft_anchor_weight,
        }
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
        // Phase 22 Stage D fix — also pull `(prompt, completion)`
        // pairs so trainers that support completion-only loss
        // (`QwenTrainerActorHandle` → `TrainSftPairs`) can mask the
        // prompt tokens out of the CE loss. The legacy `corpus`
        // string is also passed for nanogpt_rs path back-compat.
        let sft_pairs = if cfg.sft_mask_prompt {
            let (tx2, rx2) = oneshot::channel();
            actors
                .curator
                .tell(CuratorMessage::RenderPairs {
                    mode: cfg.sample_mode,
                    seed: cfg.corpus_seed,
                    reply: tx2,
                })
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            Some(rx2.await?)
        } else {
            None
        };
        info!(
            round = cfg.round,
            sft_pairs_n = sft_pairs.as_ref().map(|p| p.len()).unwrap_or(0),
            "phase: train (SFT)"
        );
        TrainRequest::Sft {
            corpus,
            sft_pairs,
            save_path: cfg.save_path.clone(),
            init_from: cfg.init_from.clone(),
            train_cfg: cfg.train_cfg.clone(),
            anchor: cfg.anchor.clone(),
            freeze_base: cfg.freeze_base,
        }
    };
    let outcome = actors.trainer.train(train_req).await?;
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
        cfg.eval_passk,
    )
    .await?;
    report.eval_correct_after = Some(after.correct);
    log_samples("after", &after);

    report.elapsed_ms = t0.elapsed().as_millis();
    Ok(report)
}

async fn ask_eval<M>(
    evaluator: &ActorRef<EvaluatorActor<M>>,
    n: usize,
    seed: u64,
    sampling: GenerateConfig,
    passk: usize,
) -> anyhow::Result<EvalReport>
where
    M: Actor<Message = ModelMessage>,
{
    let (tx, rx) = oneshot::channel();
    evaluator
        .tell(EvaluatorMessage::Eval {
            n,
            seed,
            sampling,
            passk: passk.max(1),
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

/// Phase 21 Stage C — config for `run_multi_round`. Wraps a single-round
/// `base` template and applies per-round mutations so each round chains
/// `init_from ← previous round's save_path`. Seeds also bump per round
/// so the harvest set varies across rounds.
///
/// Per-round mutations applied to `base`:
/// - `round` ← current round index (0..rounds)
/// - `init_from` ← previous round's save_path (round 0 uses `base.init_from`)
/// - `save_path` ← `<base.save_path stripped of .safetensors>.r{N}.safetensors`
/// - `gen_seed` ← `base.gen_seed + round * gen_seed_stride`
/// - `gen_sampling.seed` ← `Some(base.gen_seed.seed.unwrap_or(0) + round)`
/// - `corpus_seed` ← `base.corpus_seed.map(|s| s + round)`
///
/// Everything else (eval_seed, train_cfg, anchor, dpo_*, freeze_base,
/// eval_passk, ...) is held constant across rounds.
#[derive(Clone)]
pub struct MultiRoundConfig {
    pub rounds: usize,
    pub base: RoundConfig,
    /// Per-round bump applied to `gen_seed`. Default `1` (each round
    /// is `base + round`). Larger values spread the per-round entropy
    /// further; `0` makes harvest deterministic across rounds (rarely
    /// what you want).
    pub gen_seed_stride: u64,
}

impl MultiRoundConfig {
    /// Constructor with the common defaults: `gen_seed_stride: 1`.
    pub fn new(rounds: usize, base: RoundConfig) -> Self {
        Self {
            rounds,
            base,
            gen_seed_stride: 1,
        }
    }
}

/// Phase 21 Stage C — multi-round orchestration helper.
///
/// Runs `cfg.rounds` rounds of the standard
/// `Gen → Verify → Curate → Train → Reload → Eval` cycle, chaining
/// each round's `save_path` into the next round's `init_from`. The
/// supplied callback is invoked after each round with `(round_idx,
/// report)` so callers can stream progress without owning the loop.
///
/// Bridges the Phase 17-20 multi-round SFT findings (HumanEval r=5 mean
/// 0.556, r=6 plateau 0.581) into the Rust actor stack as a first-class
/// helper instead of an ad-hoc `for round in 0..rounds` loop in every
/// example.
///
/// Returns the per-round `RoundReport` vector after all rounds complete.
pub async fn run_multi_round<M, F>(
    actors: &RoundActors<M>,
    cfg: MultiRoundConfig,
    mut on_round_done: F,
) -> anyhow::Result<Vec<RoundReport>>
where
    M: Actor<Message = ModelMessage>,
    F: FnMut(usize, &RoundReport),
{
    let mut reports = Vec::with_capacity(cfg.rounds);
    let template = save_path_template(&cfg.base.save_path);
    let mut current_init = cfg.base.init_from.clone();
    let base_gen_sampling_seed = cfg.base.gen_sampling.seed.unwrap_or(0);
    // Phase 22 Stage D follow-up #1 — track previous round's
    // eval-after to reuse as next round's eval-before. Saves ~5
    // min/round at gen-n=164 + eval-n=32 + passk=3 (the eval-before
    // and eval-after of consecutive rounds measure the SAME model
    // state with the SAME seed, so the result is deterministic and
    // re-running just wastes wallclock).
    let mut prev_eval_after: Option<EvalReport> = None;

    for r in 0..cfg.rounds {
        let save_path: PathBuf = format!("{template}.r{r}.safetensors").into();
        let mut round_cfg = cfg.base.clone();
        round_cfg.round = r;
        round_cfg.init_from = current_init.clone();
        round_cfg.save_path = save_path.clone();
        round_cfg.gen_seed = cfg
            .base
            .gen_seed
            .wrapping_add((r as u64).wrapping_mul(cfg.gen_seed_stride));
        round_cfg.gen_sampling.seed = Some(base_gen_sampling_seed.wrapping_add(r as u64));
        round_cfg.corpus_seed = cfg.base.corpus_seed.map(|s| s.wrapping_add(r as u64));

        let report = run_round_with_prev(actors, round_cfg, prev_eval_after.take()).await?;
        // Stash the eval-after if it was run (skip-training rounds
        // don't produce one).
        prev_eval_after = report.eval_correct_after.map(|correct| EvalReport {
            total: report.eval_total,
            correct,
            samples: Vec::new(),
            passk: cfg.base.eval_passk,
            total_attempts: None,
            total_passes: None,
        });
        on_round_done(r, &report);
        current_init = Some(save_path);
        reports.push(report);
    }
    Ok(reports)
}

/// Strip a trailing `.safetensors` from the supplied path and return
/// the remainder as a String — used as the template for per-round save
/// paths so `checkpoints/run.safetensors` becomes `checkpoints/run.r0.safetensors`.
fn save_path_template(p: &std::path::Path) -> String {
    p.to_string_lossy()
        .trim_end_matches(".safetensors")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_path_template_strips_safetensors_suffix() {
        let p: PathBuf = "checkpoints/run.safetensors".into();
        assert_eq!(save_path_template(&p), "checkpoints/run");
    }

    #[test]
    fn save_path_template_leaves_paths_without_suffix() {
        let p: PathBuf = "checkpoints/run".into();
        assert_eq!(save_path_template(&p), "checkpoints/run");
    }

    #[test]
    fn save_path_template_strips_only_trailing_safetensors() {
        let p: PathBuf = "ckpt.safetensors.bak".into();
        // Trailing suffix doesn't match → no change.
        assert_eq!(save_path_template(&p), "ckpt.safetensors.bak");
    }

    #[test]
    fn multi_round_config_new_defaults_stride_to_one() {
        // Build a minimal base RoundConfig — we only need to inspect
        // fields that `run_multi_round` reads, not run anything.
        let base = RoundConfig {
            round: 0,
            gen_n: 1,
            gen_seed: 42,
            gen_sampling: GenerateConfig::default(),
            eval_n: 1,
            eval_seed: 0,
            eval_sampling: GenerateConfig::default(),
            train_cfg: nanogpt_rs::train::TrainConfig::smoke(),
            init_from: None,
            save_path: PathBuf::from("checkpoints/x.safetensors"),
            min_corpus_chars: 0,
            sample_mode: crate::curator_actor::SampleMode::Uniform,
            corpus_seed: Some(7),
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
        let cfg = MultiRoundConfig::new(5, base);
        assert_eq!(cfg.rounds, 5);
        assert_eq!(cfg.gen_seed_stride, 1);
        assert_eq!(cfg.base.gen_seed, 42);
        assert_eq!(cfg.base.corpus_seed, Some(7));
    }

    #[test]
    fn round_config_is_clone() {
        // Compile-time assertion that derive(Clone) on RoundConfig holds.
        fn assert_clone<T: Clone>() {}
        assert_clone::<RoundConfig>();
        assert_clone::<MultiRoundConfig>();
    }
}
