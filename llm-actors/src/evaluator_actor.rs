//! EvaluatorActor: pass-rate over a held-out set of prompts.
//!
//! Generates completions via a `ModelActor`, verifies them via the active
//! `Domain`, returns count of correct out of `n`.

use std::sync::Arc;
use std::time::Duration;

use nanogpt_rs::{generate::GenerateConfig, Tokenizer};
use pekko_actor::{Actor, ActorContext, ActorRef};
use rand::rngs::StdRng;
use rand::SeedableRng;
use tokio::sync::oneshot;
use tokio::time::timeout;
use tracing::info;

use crate::domain::Domain;
use crate::model_actor::{ModelActor, ModelMessage};
use crate::types::Trajectory;

pub enum EvaluatorMessage {
    Eval {
        n: usize,
        seed: u64,
        sampling: GenerateConfig,
        /// Phase 21: number of samples per prompt; the prompt is counted
        /// correct if ANY of the `passk` completions verifies. `1` is the
        /// historical pass@1 behavior. Each k-th sample uses a deterministic
        /// per-k seed override so the eval remains reproducible.
        passk: usize,
        reply: oneshot::Sender<anyhow::Result<EvalReport>>,
    },
}

#[derive(Debug, Clone)]
pub struct EvalReport {
    pub total: usize,
    pub correct: usize,
    pub samples: Vec<Trajectory>,
    /// `passk` used for this eval (≥1). Echoed back from the message so
    /// callers can include it in logs and reports.
    pub passk: usize,
}

impl EvalReport {
    pub fn pass_rate(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.correct as f32 / self.total as f32
        }
    }
}

/// Phase 21 Stage E — generic over the backing model actor type so the
/// same evaluator pipeline can serve `ModelActor` (nanogpt_rs) and
/// `QwenModelActor` (Candle-native Qwen2) without changes. Default
/// `M = ModelActor` preserves every existing call site.
pub struct EvaluatorActor<M = ModelActor>
where
    M: Actor<Message = ModelMessage>,
{
    pub model: ActorRef<M>,
    pub tokenizer: Arc<Tokenizer>,
    pub domain: Arc<dyn Domain>,
    pub stop_char: Option<char>,
    pub per_request_timeout: Duration,
    pub keep_samples: usize,
}

impl<M> EvaluatorActor<M>
where
    M: Actor<Message = ModelMessage>,
{
    pub fn new(
        model: ActorRef<M>,
        tokenizer: Arc<Tokenizer>,
        domain: Arc<dyn Domain>,
        stop_char: Option<char>,
    ) -> Self {
        Self {
            model,
            tokenizer,
            domain,
            stop_char,
            per_request_timeout: Duration::from_secs(60),
            keep_samples: 8,
        }
    }
}

impl<M> Actor for EvaluatorActor<M>
where
    M: Actor<Message = ModelMessage>,
{
    type Message = EvaluatorMessage;

    fn receive(
        &mut self,
        msg: Self::Message,
        _ctx: &mut ActorContext<Self>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + '_>> {
        Box::pin(async move {
            match msg {
                EvaluatorMessage::Eval {
                    n,
                    seed,
                    sampling,
                    passk,
                    reply,
                } => {
                    let result = self.run(n, seed, sampling, passk).await;
                    let _ = reply.send(result);
                }
            }
        })
    }
}

impl<M> EvaluatorActor<M>
where
    M: Actor<Message = ModelMessage>,
{
    async fn run(
        &self,
        n: usize,
        seed: u64,
        sampling: GenerateConfig,
        passk: usize,
    ) -> anyhow::Result<EvalReport> {
        let passk = passk.max(1);
        let mut rng = StdRng::seed_from_u64(seed);
        let mut correct = 0usize;
        let mut total = 0usize;
        let mut samples = Vec::with_capacity(self.keep_samples);

        for prompt_idx in 0..n {
            let prompt = self.domain.sample_prompt(&mut rng);
            let prompt_ids = self.tokenizer.encode(&prompt)?;
            let mut any_pass = false;
            let mut first_completion: Option<String> = None;

            for k in 0..passk {
                // Deterministic per-(prompt, k) seed so the k samples diverge
                // and the eval remains fully reproducible across runs.
                let mut cfg_k = sampling.clone();
                let k_seed = seed
                    .wrapping_add(prompt_idx as u64)
                    .wrapping_mul(passk as u64)
                    .wrapping_add(k as u64);
                cfg_k.seed = Some(k_seed);

                let (tx, rx) = oneshot::channel();
                self.model
                    .tell(ModelMessage::GenerateTokens {
                        prompt_ids: prompt_ids.clone(),
                        cfg: cfg_k,
                        reply: tx,
                    })
                    .map_err(|e| anyhow::anyhow!("send GenerateTokens: {e:?}"))?;
                let tokens = timeout(self.per_request_timeout, rx).await???;
                let comp_ids = if tokens.len() > prompt_ids.len() {
                    &tokens[prompt_ids.len()..]
                } else {
                    &[][..]
                };
                let mut completion = self.tokenizer.decode(comp_ids)?;
                if let Some(stop) = self.stop_char {
                    if let Some(idx) = completion.find(stop) {
                        completion.truncate(idx + stop.len_utf8());
                    }
                }
                let v = self.domain.verify(&prompt, &completion);
                if v.is_correct() {
                    any_pass = true;
                }
                if first_completion.is_none() {
                    first_completion = Some(completion);
                }
                if any_pass {
                    break; // short-circuit; saves work on easy prompts
                }
            }

            total += 1;
            if any_pass {
                correct += 1;
            }
            if samples.len() < self.keep_samples {
                samples.push(Trajectory {
                    prompt,
                    completion: first_completion.unwrap_or_default(),
                    source: "eval".to_string(),
                });
            }
        }
        info!(
            total,
            correct,
            passk,
            pass_rate = correct as f32 / total.max(1) as f32,
            "EvaluatorActor done"
        );
        Ok(EvalReport {
            total,
            correct,
            samples,
            passk,
        })
    }
}
