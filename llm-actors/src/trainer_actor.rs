//! TrainerActor: continual fine-tune.
//!
//! Receives a corpus string + (optional) base checkpoint + save path, runs
//! `nanogpt_rs::train::train_from` on a blocking task, returns the outcome.
//! Optimizer state restarts each round (intentional for short rounds).

use std::path::PathBuf;
use std::sync::Arc;

use candle_core::Device;
use nanogpt_rs::{
    config::GPTConfig,
    data::TokenDataset,
    ewc::WeightAnchor,
    train::{train_dpo, train_from_full, PreferencePair, TrainConfig, TrainOutcome},
    Tokenizer,
};
use pekko_actor::{Actor, ActorContext};
use tokio::sync::oneshot;
use tokio::task::spawn_blocking;
use tracing::{info, warn};

pub enum TrainerMessage {
    Train {
        corpus: String,
        save_path: PathBuf,
        init_from: Option<PathBuf>,
        train_cfg: TrainConfig,
        /// Optional EWC weight-anchor: penalizes drift from a pretrained
        /// snapshot. Recommended for continual fine-tune rounds. `None` =
        /// plain CE loss.
        anchor: Option<Arc<WeightAnchor>>,
        /// LoRA-only fine-tune: when `true`, only Vars named `*lora*` are
        /// updated. Requires the model to have been built with
        /// `lora_rank > 0`. Eliminates catastrophic forgetting since base
        /// weights are immutable during the round.
        freeze_base: bool,
        reply: oneshot::Sender<anyhow::Result<TrainOutcome>>,
    },
    /// Phase 11 S2: DPO fine-tune.
    ///
    /// `pairs` is `(prompt_text, chosen_completion_text, rejected_completion_text)`.
    /// The trainer encodes each side with its own `tokenizer` and calls
    /// `nanogpt_rs::train::train_dpo` on a blocking task. The reference
    /// model is loaded from `reference_path` and held frozen.
    TrainDpo {
        pairs: Vec<(String, String, String)>,
        save_path: PathBuf,
        init_from: PathBuf,
        reference_path: PathBuf,
        train_cfg: TrainConfig,
        beta: f64,
        reply: oneshot::Sender<anyhow::Result<TrainOutcome>>,
    },
}

pub struct TrainerActor {
    pub gpt_cfg: GPTConfig,
    pub tokenizer: Arc<Tokenizer>,
    pub device: Device,
}

impl TrainerActor {
    pub fn new(gpt_cfg: GPTConfig, tokenizer: Arc<Tokenizer>, device: Device) -> Self {
        Self {
            gpt_cfg,
            tokenizer,
            device,
        }
    }
}

impl Actor for TrainerActor {
    type Message = TrainerMessage;

    fn receive(
        &mut self,
        msg: Self::Message,
        _ctx: &mut ActorContext<Self>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + '_>> {
        Box::pin(async move {
            match msg {
                TrainerMessage::Train {
                    corpus,
                    save_path,
                    init_from,
                    train_cfg,
                    anchor,
                    freeze_base,
                    reply,
                } => {
                    let gpt_cfg = self.gpt_cfg.clone();
                    let tokenizer = self.tokenizer.clone();
                    let device = self.device.clone();
                    info!(
                        corpus_chars = corpus.len(),
                        ?save_path,
                        ?init_from,
                        steps = train_cfg.max_steps,
                        ewc = anchor.is_some(),
                        freeze_base,
                        "TrainerActor: launching blocking training"
                    );
                    let job = spawn_blocking(move || {
                        let ids = tokenizer.encode(&corpus).map_err(anyhow::Error::from)?;
                        if ids.len() < gpt_cfg.block_size + 2 {
                            anyhow::bail!(
                                "corpus too short to train: {} tokens < block_size {}+2",
                                ids.len(),
                                gpt_cfg.block_size
                            );
                        }
                        let ds = TokenDataset::new(ids, gpt_cfg.block_size);
                        let outcome = train_from_full(
                            &gpt_cfg,
                            &ds,
                            None,
                            &train_cfg,
                            &device,
                            Some(&save_path),
                            init_from.as_deref(),
                            anchor.as_deref(),
                            freeze_base,
                        )?;
                        Ok::<TrainOutcome, anyhow::Error>(outcome)
                    });
                    let result = match job.await {
                        Ok(inner) => inner,
                        Err(join_err) => {
                            warn!(error = %join_err, "training task panicked");
                            Err(anyhow::anyhow!("training panicked: {join_err}"))
                        }
                    };
                    let _ = reply.send(result);
                }
                TrainerMessage::TrainDpo {
                    pairs,
                    save_path,
                    init_from,
                    reference_path,
                    train_cfg,
                    beta,
                    reply,
                } => {
                    let gpt_cfg = self.gpt_cfg.clone();
                    let tokenizer = self.tokenizer.clone();
                    let device = self.device.clone();
                    info!(
                        n_pairs = pairs.len(),
                        ?save_path,
                        ?init_from,
                        ?reference_path,
                        steps = train_cfg.max_steps,
                        beta,
                        "TrainerActor: launching DPO training"
                    );
                    let job = spawn_blocking(move || {
                        // Encode (prompt, chosen, rejected) text triples
                        // into PreferencePair token-id triples. Drop pairs
                        // that don't fit in block_size.
                        let mut encoded: Vec<PreferencePair> = Vec::with_capacity(pairs.len());
                        let mut dropped = 0usize;
                        for (prompt, chosen, rejected) in pairs {
                            let prompt_ids =
                                tokenizer.encode(&prompt).map_err(anyhow::Error::from)?;
                            let chosen_ids =
                                tokenizer.encode(&chosen).map_err(anyhow::Error::from)?;
                            let rejected_ids =
                                tokenizer.encode(&rejected).map_err(anyhow::Error::from)?;
                            if prompt_ids.is_empty() {
                                dropped += 1;
                                continue;
                            }
                            let n_chosen = prompt_ids.len() + chosen_ids.len();
                            let n_rejected = prompt_ids.len() + rejected_ids.len();
                            if n_chosen > gpt_cfg.block_size || n_rejected > gpt_cfg.block_size {
                                dropped += 1;
                                continue;
                            }
                            encoded.push(PreferencePair {
                                prompt_ids,
                                chosen_ids,
                                rejected_ids,
                            });
                        }
                        if encoded.is_empty() {
                            anyhow::bail!(
                                "all {} DPO pairs dropped (block_size or empty prompt) — \
                                 nothing to train on",
                                dropped
                            );
                        }
                        if dropped > 0 {
                            tracing::warn!(dropped, kept = encoded.len(), "DPO: dropped pairs");
                        }
                        let outcome = train_dpo(
                            &gpt_cfg,
                            &encoded,
                            &train_cfg,
                            beta,
                            &init_from,
                            &reference_path,
                            &device,
                            Some(&save_path),
                        )?;
                        Ok::<TrainOutcome, anyhow::Error>(outcome)
                    });
                    let result = match job.await {
                        Ok(inner) => inner,
                        Err(join_err) => {
                            warn!(error = %join_err, "DPO training task panicked");
                            Err(anyhow::anyhow!("DPO training panicked: {join_err}"))
                        }
                    };
                    let _ = reply.send(result);
                }
            }
        })
    }
}
