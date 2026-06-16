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
    cosine_warmup_lr, save_merged_lora, train_qwen_lora_pg_step, train_qwen_lora_step,
    train_qwen_lora_step_masked, train_qwen_lora_step_masked_batched, LoraConfig, ModelForCausalLM,
};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

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
    /// Phase 22 Stage D follow-up — peak learning rate. The optimizer's
    /// current lr is mutated by the cosine warmup schedule in
    /// `handle_train_sft_pairs`; this field stores the value the
    /// AdamW was constructed with so the schedule can ramp back up
    /// to it.
    pub base_lr: f64,
    /// Phase 22 Stage D G7 — SFT mini-batch size for `TrainSftPairs`.
    /// `1` (default) keeps the historical single-example-per-step path
    /// bit-identical. `> 1` enables padded mini-batching (right-pad to
    /// max len in batch, completion-masked loss, shuffled per epoch),
    /// matching Phase 17's `batch_size = 4`.
    pub sft_batch_size: usize,
    /// Phase 22 Stage D G8 — when `true`, rebuild the AdamW optimizer at
    /// the start of every `TrainSftPairs` call (= every MR round), so
    /// Adam's moment estimates reset between rounds. Matches Phase 17,
    /// which constructs a fresh `torch.optim.AdamW` inside each
    /// `lora_finetune` call. `false` (default) reuses the optimizer
    /// across rounds (stale moments) — the historical behavior. The
    /// LoRA weights persist either way; only the optimizer state resets.
    pub fresh_optimizer_per_round: bool,
    /// Phase 22 Stage D — AdamW weight decay. `0.0` (default) is the
    /// historical value; `0.01` matches Phase 17's
    /// `torch.optim.AdamW(trainable, lr=lr)` (PyTorch's default
    /// weight_decay). Applied via `rebuild_optimizer` — with
    /// `fresh_optimizer_per_round=true` (the G9 recipe) it takes effect
    /// from round 0; otherwise set it before the first round.
    pub weight_decay: f64,
    /// Phase 22 Stage E — micro-batch size for `TrainPolicyGradient`.
    /// `0` (default) processes all samples in a single backward pass
    /// (original behaviour). `> 0` chunks samples into groups of this
    /// size and calls `backward_step` once per chunk, bounding peak GPU
    /// memory when completions are long (max_new ≥ 64).
    pub pg_micro_batch_size: usize,
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
            base_lr: lr,
            sft_batch_size: 1,
            fresh_optimizer_per_round: false,
            weight_decay: 0.0,
            pg_micro_batch_size: 0,
        })
    }

    /// Phase 22 Stage D — set AdamW weight decay (Phase 17 uses 0.01,
    /// the PyTorch default). Builder-style. Takes effect via
    /// `rebuild_optimizer` (round 0 when `fresh_optimizer_per_round`).
    pub fn with_weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Phase 22 Stage D G7 — set the SFT mini-batch size used by
    /// `TrainSftPairs`. Builder-style so callers can write
    /// `QwenTrainerActor::from_snapshot_dir(..)?.with_sft_batch_size(4)`.
    /// `0` is clamped to `1`.
    pub fn with_sft_batch_size(mut self, batch_size: usize) -> Self {
        self.sft_batch_size = batch_size.max(1);
        self
    }

    /// Phase 22 Stage D G8 — enable fresh-AdamW-per-round (resets Adam
    /// moments at the start of each `TrainSftPairs` call). Builder-style.
    pub fn with_fresh_optimizer(mut self, enabled: bool) -> Self {
        self.fresh_optimizer_per_round = enabled;
        self
    }

    /// Phase 22 Stage E — set the policy-gradient micro-batch size.
    /// `0` (default) keeps the original single-backward-pass behaviour.
    /// `> 0` issues one `backward_step` per chunk of this many samples,
    /// bounding peak GPU memory for long completions (max_new ≥ 64).
    pub fn with_pg_micro_batch_size(mut self, size: usize) -> Self {
        self.pg_micro_batch_size = size;
        self
    }

    /// Rebuild the AdamW optimizer over the LoRA Vars at the current
    /// `base_lr`, discarding accumulated moment estimates. The LoRA
    /// weights themselves are untouched (the optimizer only holds
    /// references + moment buffers).
    fn rebuild_optimizer(&mut self) -> anyhow::Result<()> {
        self.optimizer = AdamW::new(
            self.lora_map.all_vars(),
            ParamsAdamW {
                lr: self.base_lr,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: self.weight_decay,
            },
        )?;
        Ok(())
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
                        self.pg_micro_batch_size,
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
        // Phase 22 Stage D G8 — fresh AdamW per round (reset moments).
        if self.fresh_optimizer_per_round {
            self.rebuild_optimizer()?;
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

        // Phase 22 Stage D follow-up — cosine LR schedule with linear
        // warmup (Phase 17 recipe: `num_warmup_steps = max(1, steps/10)`).
        let warmup_steps = (train_steps / 10).max(1);
        let mut losses = Vec::with_capacity(train_steps);
        if self.sft_batch_size <= 1 {
            // Historical single-example-per-step path (bit-identical).
            for step in 0..train_steps {
                let (ids, prompt_len) = &encoded[step % encoded.len()];
                let n = ids.len();
                let input = Tensor::from_slice(&ids[..n - 1], (1, n - 1), &self.device)?;
                let target = Tensor::from_slice(&ids[1..], (1, n - 1), &self.device)?;
                // Apply the schedule BEFORE the optimizer step so the step
                // uses the right lr.
                let lr = cosine_warmup_lr(step, warmup_steps, train_steps, self.base_lr);
                self.optimizer.set_learning_rate(lr);
                let loss = train_qwen_lora_step_masked(
                    &mut self.model,
                    &mut self.optimizer,
                    &input,
                    &target,
                    *prompt_len,
                )?;
                losses.push(loss);
            }
        } else {
            // Phase 22 G7 — padded mini-batch path (Phase 17 batch=4).
            // Right-pad each batch to its max full length (pad id 0;
            // value is irrelevant since padded positions are masked out
            // of the loss and, under causal attention, never leak into a
            // real token's representation). Shuffle the example order
            // each epoch (DataLoader(shuffle=True) parity).
            let batch_size = self.sft_batch_size;
            let n = encoded.len();
            let n_batches = n.div_ceil(batch_size);
            let mut order: Vec<usize> = (0..n).collect();
            let mut rng = StdRng::seed_from_u64(0x5f7_u64.wrapping_add(train_steps as u64));
            order.shuffle(&mut rng);
            for step in 0..train_steps {
                let batch_idx = step % n_batches;
                if batch_idx == 0 && step > 0 {
                    order.shuffle(&mut rng);
                }
                let start = batch_idx * batch_size;
                let end = (start + batch_size).min(n);
                let batch: Vec<&(Vec<u32>, usize)> =
                    order[start..end].iter().map(|&i| &encoded[i]).collect();
                let b = batch.len();
                let max_len = batch.iter().map(|(ids, _)| ids.len()).max().unwrap_or(2);
                let width = max_len - 1; // shifted length
                let mut input_buf = vec![0u32; b * width];
                let mut target_buf = vec![0u32; b * width];
                let mut mask_buf = vec![0f32; b * width];
                for (bi, (ids, prompt_len)) in batch.iter().enumerate() {
                    let l = ids.len();
                    for j in 0..(l - 1) {
                        input_buf[bi * width + j] = ids[j];
                        target_buf[bi * width + j] = ids[j + 1];
                        // target[j] = ids[j+1] is a completion token iff
                        // its full-sequence index j+1 >= prompt_len.
                        if j + 1 >= *prompt_len {
                            mask_buf[bi * width + j] = 1.0;
                        }
                    }
                }
                let input = Tensor::from_slice(&input_buf, (b, width), &self.device)?;
                let target = Tensor::from_slice(&target_buf, (b, width), &self.device)?;
                let loss_mask = Tensor::from_slice(&mask_buf, (b, width), &self.device)?;
                let lr = cosine_warmup_lr(step, warmup_steps, train_steps, self.base_lr);
                self.optimizer.set_learning_rate(lr);
                let loss = train_qwen_lora_step_masked_batched(
                    &mut self.model,
                    &mut self.optimizer,
                    &input,
                    &target,
                    &loss_mask,
                )?;
                losses.push(loss);
            }
        }
        // Restore peak lr after the schedule completes (so subsequent
        // rounds start fresh from base_lr, not whatever the decay
        // ended at). The optimizer is shared across rounds.
        self.optimizer.set_learning_rate(self.base_lr);
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
