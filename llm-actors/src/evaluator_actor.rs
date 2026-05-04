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
        reply: oneshot::Sender<anyhow::Result<EvalReport>>,
    },
}

#[derive(Debug, Clone)]
pub struct EvalReport {
    pub total: usize,
    pub correct: usize,
    pub samples: Vec<Trajectory>,
}

impl EvalReport {
    pub fn pass_rate(&self) -> f32 {
        if self.total == 0 { 0.0 } else { self.correct as f32 / self.total as f32 }
    }
}

pub struct EvaluatorActor {
    pub model: ActorRef<ModelActor>,
    pub tokenizer: Arc<Tokenizer>,
    pub domain: Arc<dyn Domain>,
    pub stop_char: Option<char>,
    pub per_request_timeout: Duration,
    pub keep_samples: usize,
}

impl EvaluatorActor {
    pub fn new(
        model: ActorRef<ModelActor>,
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

impl Actor for EvaluatorActor {
    type Message = EvaluatorMessage;

    fn receive(
        &mut self,
        msg: Self::Message,
        _ctx: &mut ActorContext<Self>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + '_>> {
        Box::pin(async move {
            match msg {
                EvaluatorMessage::Eval { n, seed, sampling, reply } => {
                    let result = self.run(n, seed, sampling).await;
                    let _ = reply.send(result);
                }
            }
        })
    }
}

impl EvaluatorActor {
    async fn run(&self, n: usize, seed: u64, sampling: GenerateConfig) -> anyhow::Result<EvalReport> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut correct = 0usize;
        let mut total = 0usize;
        let mut samples = Vec::with_capacity(self.keep_samples);

        for _ in 0..n {
            let prompt = self.domain.sample_prompt(&mut rng);
            let prompt_ids = self.tokenizer.encode(&prompt)?;
            let (tx, rx) = oneshot::channel();
            self.model
                .tell(ModelMessage::GenerateTokens {
                    prompt_ids: prompt_ids.clone(),
                    cfg: sampling.clone(),
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
            total += 1;
            if v.is_correct() {
                correct += 1;
            }
            if samples.len() < self.keep_samples {
                samples.push(Trajectory { prompt, completion, source: "eval".to_string() });
            }
        }
        info!(total, correct, pass_rate = correct as f32 / total.max(1) as f32, "EvaluatorActor done");
        Ok(EvalReport { total, correct, samples })
    }
}
