//! Phase 21 Stage H — trainer abstraction so the supervisor pipeline
//! can drive both the nanogpt_rs `TrainerActor` and the Candle-native
//! `QwenTrainerActor` (Stage E.next).
//!
//! `RoundActors.trainer` is now `Arc<dyn TrainerHandle>` instead of
//! the concrete `ActorRef<TrainerActor>`. Callers wrap their actor
//! ref in the appropriate handle:
//!
//! ```ignore
//! let nanogpt_handle = Arc::new(TrainerActorHandle::new(trainer_ref));
//! let qwen_handle = Arc::new(QwenTrainerActorHandle::new(
//!     qwen_trainer_ref,
//!     8,                     // train_steps per round
//!     base_safetensors_path,
//! ));
//! ```
//!
//! `run_round` builds the corpus/pairs and dispatches via the trait;
//! the handle knows how to talk to its underlying actor.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use nanogpt_rs::{
    ewc::WeightAnchor,
    train::{TrainConfig, TrainOutcome},
};
use pekko_actor::ActorRef;
use tokio::sync::oneshot;

use crate::qwen_trainer_actor::{QwenTrainerActor, QwenTrainerMessage};
use crate::trainer_actor::{TrainerActor, TrainerMessage};

/// Per-round training request. Variants differ in what the curator
/// rendered (a corpus string for SFT, preference pairs for DPO) and
/// what extra fields each path needs. The `train_cfg` and recipe-
/// specific knobs come from `RoundConfig`.
pub enum TrainRequest {
    Sft {
        corpus: String,
        save_path: PathBuf,
        init_from: Option<PathBuf>,
        train_cfg: TrainConfig,
        anchor: Option<Arc<WeightAnchor>>,
        freeze_base: bool,
    },
    Dpo {
        /// `(prompt, chosen, rejected)` triples, as the curator's
        /// `RenderPreferencePairs` produces.
        pairs: Vec<(String, String, String)>,
        save_path: PathBuf,
        init_from: PathBuf,
        reference_path: PathBuf,
        train_cfg: TrainConfig,
        beta: f64,
        sft_anchor_weight: f64,
    },
}

#[async_trait]
pub trait TrainerHandle: Send + Sync + 'static {
    async fn train(&self, req: TrainRequest) -> anyhow::Result<TrainOutcome>;
}

/// Wraps an `ActorRef<TrainerActor>` so the nanogpt_rs trainer fits
/// the supervisor pipeline through the `TrainerHandle` trait. Direct
/// translation — `Sft` → `TrainerMessage::Train`, `Dpo` →
/// `TrainerMessage::TrainDpo`. Matches the pre-Stage-H behavior of
/// `run_round` exactly.
pub struct TrainerActorHandle {
    pub trainer: ActorRef<TrainerActor>,
}

impl TrainerActorHandle {
    pub fn new(trainer: ActorRef<TrainerActor>) -> Self {
        Self { trainer }
    }
}

#[async_trait]
impl TrainerHandle for TrainerActorHandle {
    async fn train(&self, req: TrainRequest) -> anyhow::Result<TrainOutcome> {
        match req {
            TrainRequest::Sft {
                corpus,
                save_path,
                init_from,
                train_cfg,
                anchor,
                freeze_base,
            } => {
                let (tx, rx) = oneshot::channel();
                self.trainer
                    .tell(TrainerMessage::Train {
                        corpus,
                        save_path,
                        init_from,
                        train_cfg,
                        anchor,
                        freeze_base,
                        reply: tx,
                    })
                    .map_err(|e| anyhow::anyhow!("{e:?}"))?;
                rx.await?
            }
            TrainRequest::Dpo {
                pairs,
                save_path,
                init_from,
                reference_path,
                train_cfg,
                beta,
                sft_anchor_weight,
            } => {
                let (tx, rx) = oneshot::channel();
                self.trainer
                    .tell(TrainerMessage::TrainDpo {
                        pairs,
                        save_path,
                        init_from,
                        reference_path,
                        train_cfg,
                        beta,
                        sft_anchor_weight,
                        reply: tx,
                    })
                    .map_err(|e| anyhow::anyhow!("{e:?}"))?;
                rx.await?
            }
        }
    }
}

/// Wraps an `ActorRef<QwenTrainerActor>` (Stage E.next) so the Candle-
/// native Qwen2 LoRA trainer fits the supervisor pipeline.
///
/// On `Sft`:
/// 1. Split corpus by newlines → texts: Vec<String> (one prompt+slot pair per line)
/// 2. Send `QwenTrainerMessage::Train { texts, train_steps, ... }`
/// 3. Send `QwenTrainerMessage::SaveMergedCheckpoint { base_path, out_path: save_path }`
/// 4. Return TrainOutcome with final loss
///
/// `Dpo` returns `Err` — Qwen LoRA training has no DPO path yet.
pub struct QwenTrainerActorHandle {
    pub trainer: ActorRef<QwenTrainerActor>,
    /// Per-round number of AdamW steps. Used to size each round's
    /// training. (The supervisor's `TrainConfig.max_steps` is ignored
    /// because Qwen training doesn't use TrainConfig; we bake this
    /// at handle construction.)
    pub train_steps: usize,
    /// Source safetensors of the frozen base Qwen — needed at
    /// `SaveMergedCheckpoint` time to construct the merged file.
    pub base_safetensors: PathBuf,
}

impl QwenTrainerActorHandle {
    pub fn new(
        trainer: ActorRef<QwenTrainerActor>,
        train_steps: usize,
        base_safetensors: PathBuf,
    ) -> Self {
        Self {
            trainer,
            train_steps,
            base_safetensors,
        }
    }
}

#[async_trait]
impl TrainerHandle for QwenTrainerActorHandle {
    async fn train(&self, req: TrainRequest) -> anyhow::Result<TrainOutcome> {
        match req {
            TrainRequest::Sft {
                corpus, save_path, ..
            } => {
                // Curator renders a single newline-delimited corpus string.
                // For Qwen we treat each non-empty line as a training text.
                let texts: Vec<String> = corpus
                    .lines()
                    .filter(|l| !l.trim().is_empty())
                    .map(|l| l.to_string())
                    .collect();
                if texts.is_empty() {
                    anyhow::bail!("QwenTrainerActorHandle::train: empty corpus");
                }
                let (tx, rx) = oneshot::channel();
                self.trainer
                    .tell(QwenTrainerMessage::Train {
                        texts,
                        train_steps: self.train_steps,
                        reply: tx,
                    })
                    .map_err(|e| anyhow::anyhow!("{e:?}"))?;
                let outcome = rx.await??;

                // Save merged checkpoint so the inference actor can
                // ReloadCheckpoint(save_path) and see the trained model.
                let (tx2, rx2) = oneshot::channel();
                self.trainer
                    .tell(QwenTrainerMessage::SaveMergedCheckpoint {
                        base_path: self.base_safetensors.clone(),
                        out_path: save_path,
                        reply: tx2,
                    })
                    .map_err(|e| anyhow::anyhow!("{e:?}"))?;
                rx2.await??;

                Ok(TrainOutcome {
                    final_step: outcome.losses.len(),
                    last_train_loss: outcome.final_loss,
                    last_val_loss: None,
                })
            }
            TrainRequest::Dpo { .. } => {
                anyhow::bail!(
                    "QwenTrainerActorHandle: DPO training is not implemented \
                     (Qwen LoRA path is SFT-only)"
                )
            }
        }
    }
}
