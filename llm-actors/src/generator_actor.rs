//! GeneratorActor: produces (prompt, completion) Trajectories using a ModelActor.
//!
//! Sampling is parameter-driven (temperature/top-k/top-p) and stops when
//! either `max_new_tokens` is reached or the configured `stop_token` (e.g.
//! newline) is emitted — the latter keeps each arithmetic example tight.

use std::sync::Arc;
use std::time::Duration;

use nanogpt_rs::{generate::GenerateConfig, Tokenizer};
use pekko_actor::{Actor, ActorContext, ActorRef};
use rand::rngs::StdRng;
use rand::SeedableRng;
use tokio::sync::oneshot;
use tokio::time::timeout;
use tracing::{info, warn};

use crate::domain::Domain;
use crate::model_actor::{ModelActor, ModelMessage};
use crate::types::Trajectory;

pub enum GeneratorMessage {
    /// Generate `n` trajectories from fresh random prompts.
    /// `oversample = 1` (the default) → one generation per sampled
    /// prompt. `oversample > 1` → for each prompt, generate
    /// `oversample` candidates with different seed offsets, score each
    /// via the model's own log-prob (Phase 6 Shape C `LogitCritic`),
    /// and keep the highest-scoring candidate. Cargo budget is
    /// unchanged (still `n` survivors), but per-prompt selection is
    /// now critic-driven instead of single-shot.
    GenerateBatch {
        n: usize,
        seed: u64,
        sampling: GenerateConfig,
        /// 1 = current behavior; > 1 enables critic-rerank.
        oversample: usize,
        reply: oneshot::Sender<anyhow::Result<Vec<Trajectory>>>,
    },
}

pub struct GeneratorActor {
    pub model: ActorRef<ModelActor>,
    pub tokenizer: Arc<Tokenizer>,
    pub domain: Arc<dyn Domain>,
    /// Stop the completion when this char is emitted (after decode).
    pub stop_char: Option<char>,
    /// Source label written into Trajectory.source.
    pub source: String,
    pub per_request_timeout: Duration,
}

impl GeneratorActor {
    pub fn new(
        model: ActorRef<ModelActor>,
        tokenizer: Arc<Tokenizer>,
        domain: Arc<dyn Domain>,
        stop_char: Option<char>,
        source: String,
    ) -> Self {
        Self {
            model,
            tokenizer,
            domain,
            stop_char,
            source,
            per_request_timeout: Duration::from_secs(60),
        }
    }

    async fn generate_one(
        &self,
        prompt: String,
        sampling: GenerateConfig,
    ) -> anyhow::Result<Trajectory> {
        let prompt_ids = self.tokenizer.encode(&prompt)?;
        let (tx, rx) = oneshot::channel();
        self.model
            .tell(ModelMessage::GenerateTokens {
                prompt_ids: prompt_ids.clone(),
                cfg: sampling,
                reply: tx,
            })
            .map_err(|e| anyhow::anyhow!("send GenerateTokens: {e:?}"))?;
        let tokens = timeout(self.per_request_timeout, rx).await???;
        // Decode just the completion (tokens after the prompt).
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
        Ok(Trajectory {
            prompt,
            completion,
            source: self.source.clone(),
        })
    }

    /// Phase 6 Shape C: generate `oversample` candidates for the same
    /// prompt with different seed offsets, score each via the model's
    /// own mean log-prob (`ScoreLogProb`), return the highest scorer.
    async fn generate_one_with_oversample(
        &self,
        prompt: String,
        sampling: GenerateConfig,
        oversample: usize,
        seed_offset: u64,
    ) -> anyhow::Result<Trajectory> {
        let mut best: Option<(f32, Trajectory)> = None;
        for k in 0..oversample {
            let mut s = sampling.clone();
            // Vary seed per candidate so they're independent draws.
            // If the caller gave us a specific seed, fan out from it;
            // otherwise let the underlying generator's None-seed
            // behavior take over (un-seeded → entropy).
            if let Some(base) = sampling.seed {
                s.seed = Some(base.wrapping_add(seed_offset.wrapping_mul(31) + k as u64));
            }
            let traj = self.generate_one(prompt.clone(), s).await?;
            // Score with the model's ScoreLogProb message.
            let prompt_ids = self.tokenizer.encode(&traj.prompt)?;
            let comp_ids = self.tokenizer.encode(&traj.completion)?;
            let (tx, rx) = oneshot::channel();
            self.model
                .tell(ModelMessage::ScoreLogProb {
                    prompt_ids,
                    completion_ids: comp_ids,
                    reply: tx,
                })
                .map_err(|e| anyhow::anyhow!("send ScoreLogProb: {e:?}"))?;
            let score = timeout(self.per_request_timeout, rx).await???;
            best = match best {
                None => Some((score, traj)),
                Some((s_old, t_old)) if score > s_old => Some((score, traj)),
                Some(prev) => Some(prev),
            };
        }
        Ok(best.expect("oversample > 0").1)
    }
}

impl Actor for GeneratorActor {
    type Message = GeneratorMessage;

    fn receive(
        &mut self,
        msg: Self::Message,
        _ctx: &mut ActorContext<Self>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + '_>> {
        Box::pin(async move {
            match msg {
                GeneratorMessage::GenerateBatch {
                    n,
                    seed,
                    sampling,
                    oversample,
                    reply,
                } => {
                    let mut rng = StdRng::seed_from_u64(seed);
                    let mut out = Vec::with_capacity(n);
                    let mut errs = 0usize;
                    let f = oversample.max(1);
                    for i in 0..n {
                        let prompt = self.domain.sample_prompt(&mut rng);
                        let result = if f == 1 {
                            self.generate_one(prompt, sampling.clone()).await
                        } else {
                            self.generate_one_with_oversample(prompt, sampling.clone(), f, i as u64)
                                .await
                        };
                        match result {
                            Ok(t) => out.push(t),
                            Err(e) => {
                                warn!(error = %e, "generate_one failed");
                                errs += 1;
                            }
                        }
                    }
                    info!(
                        generated = out.len(),
                        errors = errs,
                        oversample = f,
                        "GeneratorActor batch done"
                    );
                    let _ = reply.send(Ok(out));
                }
            }
        })
    }
}
