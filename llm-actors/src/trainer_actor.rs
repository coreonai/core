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
    train::{train_from_full, TrainConfig, TrainOutcome},
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
}

pub struct TrainerActor {
    pub gpt_cfg: GPTConfig,
    pub tokenizer: Arc<Tokenizer>,
    pub device: Device,
}

impl TrainerActor {
    pub fn new(gpt_cfg: GPTConfig, tokenizer: Arc<Tokenizer>, device: Device) -> Self {
        Self { gpt_cfg, tokenizer, device }
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
                        let ids = tokenizer
                            .encode(&corpus)
                            .map_err(anyhow::Error::from)?;
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
            }
        })
    }
}
