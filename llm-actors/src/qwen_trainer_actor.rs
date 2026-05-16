//! Phase 21 Stage E.next — actor wrapper around `qwen2_lora`'s training
//! step. Pairs with `QwenModelActor` (Stage D inference) to close the
//! last gap in the Pekko bridge: now both inference AND training of the
//! Phase 14-20 production Qwen2.5-Coder-0.5B model run through the
//! actor framework.
//!
//! ## What's NOT covered (deferred)
//! - **`run_multi_round` integration**. `RoundActors.trainer:
//!   ActorRef<TrainerActor>` is hardcoded to the nanogpt_rs trainer.
//!   Plumbing in `QwenTrainerActor` would either require RoundActors
//!   to be generic over the trainer too OR a redesign that splits the
//!   training step behind a trait. Both are bigger surgery; this
//!   commit ships the actor itself so the wrap-Stage-F-in-Pekko
//!   capability is in tree, and a focused E.next.next can wire it
//!   into multi-round.
//! - **Merged-base safetensors export**. After training, callers get
//!   a LoRA-only adapter file via `SaveLoraAdapter`. To hand that off
//!   to `QwenModelActor` (which uses the upstream `qwen2` module
//!   without LoRA hooks), the adapter has to be merged back into the
//!   base. That's a `W' = W + (B @ A) * scale` matmul per LoRA layer
//!   — short follow-on, but not in this commit.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use candle_transformers::models::qwen2::Config as Qwen2Config;
use pekko_actor::{Actor, ActorContext};
use tokenizers::Tokenizer as HfTokenizer;
use tokio::sync::oneshot;
use tracing::{error, info, warn};

use crate::qwen2_lora::{
    save_merged_lora, train_qwen_lora_pg_step, train_qwen_lora_step, train_qwen_lora_step_masked,
    LoraConfig, ModelForCausalLM,
};

pub enum QwenTrainerMessage {
    /// Train on a batch of free-form text examples for `train_steps`
    /// total optimizer updates. Each step picks one text round-robin
    /// and runs one `train_qwen_lora_step` (next-token CE loss on the
    /// full sequence). The model's LoRA Vars are mutated in place;
    /// the frozen base safetensors are not touched.
    ///
    /// ⚠ This is the **prompt-unmasked** path — every position
    /// (including the prompt prefix) contributes to the loss. For
    /// HumanEval / MBPP-style SFT where the prompt dominates the
    /// sequence, prefer `TrainSftPairs` below — it implements
    /// Phase 17's `labels[:prompt_ids.shape[0]] = -100` completion-
    /// only loss.
    Train {
        texts: Vec<String>,
        train_steps: usize,
        reply: oneshot::Sender<anyhow::Result<TrainOutcome>>,
    },
    /// Phase 22 Stage D fix — **completion-only SFT**. Each example is
    /// passed in as a `(prompt, completion)` pair. The trainer
    /// tokenizes prompt and completion separately, computes the
    /// prompt boundary `P = len(prompt_ids)`, encodes the full
    /// `prompt + completion`, and runs `train_qwen_lora_step_masked`
    /// to compute CE loss ONLY on the last `C` positions
    /// (completion-token predictions). Matches Phase 17 Python's
    /// `labels[:prompt_ids.shape[0]] = -100` semantics.
    ///
    /// This is the recipe that lets Phase 17's r=2 = 0.404 result
    /// reproduce; the unmasked `Train` path catastrophically
    /// over-trains on prompt reproduction (Phase 22 Stage D's
    /// A-batch + G1 + G2 batches all confirmed this).
    TrainSftPairs {
        pairs: Vec<(String, String)>,
        train_steps: usize,
        reply: oneshot::Sender<anyhow::Result<TrainOutcome>>,
    },
    /// Persist the LoRA adapter Vars (and ONLY them — the frozen base
    /// stays in the mmapped source) to a safetensors file via
    /// `VarMap::save`. Round-trips with `LoadLoraAdapter`.
    SaveLoraAdapter {
        path: PathBuf,
        reply: oneshot::Sender<anyhow::Result<()>>,
    },
    /// Phase 21 Stage E.next.next — emit a merged safetensors that
    /// folds the trained LoRA delta into the base weights for every
    /// (q_proj, v_proj) layer. The output is drop-in compatible with
    /// the upstream `candle_transformers::models::qwen2` loader, so
    /// `QwenModelActor::ReloadCheckpoint(out_path)` picks up the
    /// trained model without any LoRA-awareness on the inference
    /// side. `base_path` is the SOURCE safetensors (the original
    /// frozen Qwen checkpoint this trainer was built from).
    SaveMergedCheckpoint {
        base_path: PathBuf,
        out_path: PathBuf,
        reply: oneshot::Sender<anyhow::Result<()>>,
    },
    /// Phase 21 Stage G — REINFORCE policy-gradient training step
    /// against a batch of pre-generated `(prompt_ids, completion_ids,
    /// reward)` samples. The reward is reward-weighted log-prob loss
    /// over the completion span; baseline-subtracted rewards (e.g.,
    /// per-prompt RLOO) belong at the call site.
    ///
    /// One actor message = one optimizer step (single batch update).
    /// Run multiple via repeated sends for multi-step RL.
    TrainPolicyGradient {
        /// `(prompt_ids, completion_ids, reward)` triples. Empty
        /// completions are skipped silently.
        samples: Vec<(Vec<u32>, Vec<u32>, f32)>,
        reply: oneshot::Sender<anyhow::Result<f32>>,
    },
    /// Health check.
    Ping { reply: oneshot::Sender<()> },
}

#[derive(Debug, Clone)]
pub struct TrainOutcome {
    pub losses: Vec<f32>,
    pub initial_loss: f32,
    pub final_loss: f32,
}

pub struct QwenTrainerActor {
    pub model: ModelForCausalLM,
    pub tokenizer: Arc<HfTokenizer>,
    pub config: Qwen2Config,
    pub device: Device,
    pub dtype: DType,
    pub lora_map: VarMap,
    /// Hyperparameters used when constructing the LoRA adapters.
    /// Kept on the actor so `SaveMergedCheckpoint` can recompute the
    /// `α / r` scale needed to bake the delta back into the base.
    pub lora_cfg: LoraConfig,
    pub optimizer: AdamW,
}

impl QwenTrainerActor {
    /// Convenience loader: read `config.json` + `tokenizer.json` +
    /// `model.safetensors` from a single HF snapshot directory, attach
    /// fresh LoRA adapters, and build the AdamW optimizer over only
    /// the LoRA Vars (the frozen base stays untouched).
    pub fn from_snapshot_dir(
        snapshot_dir: &Path,
        device: Device,
        dtype: DType,
        lora_cfg: LoraConfig,
        lr: f64,
    ) -> anyhow::Result<Self> {
        let cfg_text = std::fs::read_to_string(snapshot_dir.join("config.json"))?;
        let config: Qwen2Config = serde_json::from_str(&cfg_text)?;
        let tokenizer = HfTokenizer::from_file(snapshot_dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let safetensors = snapshot_dir.join("model.safetensors");

        let base_vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[&safetensors], dtype, &device)? };
        let lora_map = VarMap::new();
        let lora_vb = VarBuilder::from_varmap(&lora_map, dtype, &device);
        let model = ModelForCausalLM::new(&config, base_vb, Some(lora_vb), lora_cfg)?;

        let optimizer = AdamW::new(
            lora_map.all_vars(),
            ParamsAdamW {
                lr,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.0,
            },
        )?;

        Ok(Self {
            model,
            tokenizer: Arc::new(tokenizer),
            config,
            device,
            dtype,
            lora_map,
            lora_cfg,
            optimizer,
        })
    }

    /// Number of trainable LoRA parameters across all registered Vars.
    /// Reported in the smoke and useful as a sanity check.
    pub fn lora_param_count(&self) -> usize {
        self.lora_map
            .all_vars()
            .iter()
            .map(|v| v.dims().iter().product::<usize>())
            .sum()
    }
}

impl Actor for QwenTrainerActor {
    type Message = QwenTrainerMessage;

    fn receive(
        &mut self,
        msg: Self::Message,
        _ctx: &mut ActorContext<Self>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + '_>> {
        Box::pin(async move {
            match msg {
                QwenTrainerMessage::Ping { reply } => {
                    let _ = reply.send(());
                }
                QwenTrainerMessage::Train {
                    texts,
                    train_steps,
                    reply,
                } => {
                    let result = self.handle_train(&texts, train_steps);
                    log_send(reply, result, "qwen_train");
                }
                QwenTrainerMessage::TrainSftPairs {
                    pairs,
                    train_steps,
                    reply,
                } => {
                    let result = self.handle_train_sft_pairs(&pairs, train_steps);
                    log_send(reply, result, "qwen_train_sft_pairs");
                }
                QwenTrainerMessage::SaveLoraAdapter { path, reply } => {
                    let result = self.lora_map.save(&path).map_err(anyhow::Error::from);
                    log_send(reply, result, "qwen_save_lora_adapter");
                }
                QwenTrainerMessage::TrainPolicyGradient { samples, reply } => {
                    let result = train_qwen_lora_pg_step(
                        &mut self.model,
                        &mut self.optimizer,
                        &self.device,
                        &samples,
                    )
                    .map_err(anyhow::Error::from);
                    log_send(reply, result, "qwen_train_pg");
                }
                QwenTrainerMessage::SaveMergedCheckpoint {
                    base_path,
                    out_path,
                    reply,
                } => {
                    let result = save_merged_lora(
                        &base_path,
                        &self.lora_map,
                        &self.config,
                        // Re-derive LoRA hyperparameters from the model
                        // by inspecting any one LoRA Var shape. Simpler:
                        // remember them on the actor. For now we keep
                        // them on the actor; bake them in below.
                        self.lora_cfg,
                        &self.device,
                        &out_path,
                    )
                    .map_err(anyhow::Error::from);
                    log_send(reply, result, "qwen_save_merged_checkpoint");
                }
            }
        })
    }

    fn pre_start(
        &mut self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + '_>> {
        Box::pin(async {
            info!("QwenTrainerActor started");
        })
    }
}

impl QwenTrainerActor {
    fn handle_train(
        &mut self,
        texts: &[String],
        train_steps: usize,
    ) -> anyhow::Result<TrainOutcome> {
        if texts.is_empty() {
            anyhow::bail!("QwenTrainerActor::Train: texts is empty");
        }
        // Pre-encode all texts once — re-tokenizing per step would
        // dominate wallclock at small step counts.
        let mut encoded: Vec<Vec<u32>> = Vec::with_capacity(texts.len());
        for t in texts {
            let enc = self
                .tokenizer
                .encode(t.as_str(), true)
                .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
            let ids = enc.get_ids().to_vec();
            if ids.len() < 2 {
                warn!(text = %t, "skipping text with < 2 tokens (no next-token loss possible)");
                continue;
            }
            encoded.push(ids);
        }
        if encoded.is_empty() {
            anyhow::bail!("QwenTrainerActor::Train: all texts shorter than 2 tokens");
        }

        let mut losses = Vec::with_capacity(train_steps);
        for step in 0..train_steps {
            let ids = &encoded[step % encoded.len()];
            let input =
                Tensor::from_slice(&ids[..ids.len() - 1], (1, ids.len() - 1), &self.device)?;
            let target = Tensor::from_slice(&ids[1..], (1, ids.len() - 1), &self.device)?;
            let loss = train_qwen_lora_step(&mut self.model, &mut self.optimizer, &input, &target)?;
            losses.push(loss);
        }
        let initial_loss = losses.first().copied().unwrap_or(f32::NAN);
        let final_loss = losses.last().copied().unwrap_or(f32::NAN);
        info!(
            steps = losses.len(),
            initial_loss, final_loss, "QwenTrainerActor train done"
        );
        Ok(TrainOutcome {
            losses,
            initial_loss,
            final_loss,
        })
    }

    /// Phase 22 Stage D fix — completion-only SFT (Phase 17 recipe).
    fn handle_train_sft_pairs(
        &mut self,
        pairs: &[(String, String)],
        train_steps: usize,
    ) -> anyhow::Result<TrainOutcome> {
        if pairs.is_empty() {
            anyhow::bail!("QwenTrainerActor::TrainSftPairs: pairs is empty");
        }
        // Pre-tokenize each (prompt, completion) pair, remembering the
        // prompt boundary so the training step can mask prompt
        // positions out of the CE loss.
        let mut encoded: Vec<(Vec<u32>, usize)> = Vec::with_capacity(pairs.len());
        for (prompt, completion) in pairs {
            let prompt_enc = self
                .tokenizer
                .encode(prompt.as_str(), true)
                .map_err(|e| anyhow::anyhow!("encode prompt: {e}"))?;
            let prompt_ids = prompt_enc.get_ids().to_vec();
            let full_enc = self
                .tokenizer
                .encode((prompt.clone() + completion).as_str(), true)
                .map_err(|e| anyhow::anyhow!("encode prompt+completion: {e}"))?;
            let full_ids = full_enc.get_ids().to_vec();
            if full_ids.len() <= prompt_ids.len() {
                warn!(
                    prompt_len = prompt_ids.len(),
                    full_len = full_ids.len(),
                    "skipping pair: full_ids no longer than prompt_ids (empty completion or tokenizer collision)"
                );
                continue;
            }
            if full_ids.len() < 2 {
                warn!("skipping pair: full_ids shorter than 2 tokens");
                continue;
            }
            encoded.push((full_ids, prompt_ids.len()));
        }
        if encoded.is_empty() {
            anyhow::bail!("QwenTrainerActor::TrainSftPairs: all pairs rejected");
        }

        let mut losses = Vec::with_capacity(train_steps);
        for step in 0..train_steps {
            let (ids, prompt_len) = &encoded[step % encoded.len()];
            let n = ids.len();
            let input = Tensor::from_slice(&ids[..n - 1], (1, n - 1), &self.device)?;
            let target = Tensor::from_slice(&ids[1..], (1, n - 1), &self.device)?;
            let loss = train_qwen_lora_step_masked(
                &mut self.model,
                &mut self.optimizer,
                &input,
                &target,
                *prompt_len,
            )?;
            losses.push(loss);
        }
        let initial_loss = losses.first().copied().unwrap_or(f32::NAN);
        let final_loss = losses.last().copied().unwrap_or(f32::NAN);
        info!(
            steps = losses.len(),
            initial_loss,
            final_loss,
            n_pairs = encoded.len(),
            "QwenTrainerActor train_sft_pairs done"
        );
        Ok(TrainOutcome {
            losses,
            initial_loss,
            final_loss,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_trainer_actor_uses_dedicated_message_enum() {
        // Compile-time assertion that QwenTrainerActor uses its own
        // QwenTrainerMessage enum (NOT TrainerMessage). This matters
        // because TrainerMessage carries nanogpt_rs-specific payloads
        // (corpus string + TrainConfig) that don't fit Qwen training.
        fn assert_uses<A>()
        where
            A: Actor<Message = QwenTrainerMessage>,
        {
        }
        assert_uses::<QwenTrainerActor>();
    }
}
