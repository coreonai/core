//! ModelActor: owns a `GPT` + `VarMap` + tokenizer.
//!
//! Holds the VarMap so we can hot-reload weights in place via
//! `ReloadCheckpoint`. Linear/Embedding tensors share storage with VarMap's
//! Vars, so a successful `varmap.load()` immediately changes what the next
//! generate sees — no rebuild needed.

use std::path::PathBuf;
use std::sync::Arc;

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use nanogpt_rs::{config::GPTConfig, generate, GenerateConfig, Tokenizer, GPT};
use pekko_actor::{Actor, ActorContext};
use tokio::sync::oneshot;
use tracing::{error, info, warn};

pub enum ModelMessage {
    /// Decode-then-sample.
    Generate {
        prompt: String,
        cfg: GenerateConfig,
        reply: oneshot::Sender<anyhow::Result<GenerateReply>>,
    },
    /// Generate from raw token ids, return raw ids (no decode).
    GenerateTokens {
        prompt_ids: Vec<u32>,
        cfg: GenerateConfig,
        reply: oneshot::Sender<anyhow::Result<Vec<u32>>>,
    },
    /// Hot-swap weights from a safetensors file. Architecture must match.
    ReloadCheckpoint {
        path: PathBuf,
        reply: oneshot::Sender<anyhow::Result<()>>,
    },
    /// Compute loss on a (B, T) batch (used by EvaluatorActor for PPL).
    LossOn {
        x: candle_core::Tensor,
        y: candle_core::Tensor,
        reply: oneshot::Sender<anyhow::Result<f32>>,
    },
    /// Health check.
    Ping { reply: oneshot::Sender<()> },
}

pub struct GenerateReply {
    pub text: String,
    pub tokens: Vec<u32>,
}

pub struct ModelActor {
    pub varmap: VarMap,
    pub model: GPT,
    pub tokenizer: Arc<Tokenizer>,
    pub device: Device,
    pub config: GPTConfig,
}

impl ModelActor {
    pub fn new(
        config: GPTConfig,
        device: Device,
        tokenizer: Arc<Tokenizer>,
    ) -> anyhow::Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = GPT::new(config.clone(), vb)?;
        Ok(Self {
            varmap,
            model,
            tokenizer,
            device,
            config,
        })
    }

    pub fn from_checkpoint(
        config: GPTConfig,
        device: Device,
        tokenizer: Arc<Tokenizer>,
        path: &std::path::Path,
    ) -> anyhow::Result<Self> {
        let mut me = Self::new(config, device, tokenizer)?;
        me.varmap.load(path)?;
        Ok(me)
    }
}

impl Actor for ModelActor {
    type Message = ModelMessage;

    fn receive(
        &mut self,
        msg: Self::Message,
        _ctx: &mut ActorContext<Self>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + '_>> {
        Box::pin(async move {
            match msg {
                ModelMessage::Ping { reply } => {
                    let _ = reply.send(());
                }
                ModelMessage::Generate { prompt, cfg, reply } => {
                    let result = self.handle_generate(&prompt, &cfg);
                    log_send(reply, result, "generate");
                }
                ModelMessage::GenerateTokens {
                    prompt_ids,
                    cfg,
                    reply,
                } => {
                    let result = generate(&self.model, &prompt_ids, &cfg, &self.device)
                        .map_err(anyhow::Error::from);
                    log_send(reply, result, "generate_tokens");
                }
                ModelMessage::ReloadCheckpoint { path, reply } => {
                    let result = self.handle_reload(&path);
                    log_send(reply, result, "reload_checkpoint");
                }
                ModelMessage::LossOn { x, y, reply } => {
                    let result = self
                        .model
                        .loss(&x, &y)
                        .and_then(|l| l.to_scalar::<f32>())
                        .map_err(anyhow::Error::from);
                    log_send(reply, result, "loss_on");
                }
            }
        })
    }

    fn pre_start(
        &mut self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + '_>> {
        Box::pin(async {
            info!("ModelActor started");
        })
    }
}

fn log_send<T>(reply: oneshot::Sender<anyhow::Result<T>>, r: anyhow::Result<T>, op: &str) {
    if let Err(e) = &r {
        error!(op, error = %e, "actor op failed");
    }
    if reply.send(r).is_err() {
        warn!(op, "reply channel dropped before response");
    }
}

impl ModelActor {
    fn handle_generate(&self, prompt: &str, cfg: &GenerateConfig) -> anyhow::Result<GenerateReply> {
        let prompt_ids = self.tokenizer.encode(prompt)?;
        let tokens = generate(&self.model, &prompt_ids, cfg, &self.device)?;
        let text = self.tokenizer.decode(&tokens)?;
        Ok(GenerateReply { text, tokens })
    }

    fn handle_reload(&mut self, path: &std::path::Path) -> anyhow::Result<()> {
        self.varmap.load(path)?;
        info!(?path, "checkpoint reloaded");
        Ok(())
    }
}
