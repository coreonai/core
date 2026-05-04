//! AgenticGeneratorActor: multi-turn loop with tool dispatch.
//!
//! The classic agent loop:
//!   1. ask the model to generate up to N tokens
//!   2. scan the new text for a complete tool call
//!   3. if found: dispatch to ToolExecutor, splice result into the buffer,
//!      loop with the spliced text as the new prompt
//!   4. if not found, or budget exhausted, return the buffer
//!
//! The model has no built-in concept of tools; the parser detects
//! `(name args)\n` patterns. Phase 4 doesn't require a model that's
//! actually been trained to emit them — the wiring is what matters.

use std::sync::Arc;
use std::time::Duration;

use nanogpt_rs::generate::GenerateConfig;
use nanogpt_rs::Tokenizer;
use pekko_actor::{Actor, ActorContext, ActorRef};
use tokio::sync::oneshot;
use tokio::time::timeout;
use tracing::{info, warn};

use crate::model_actor::{ModelActor, ModelMessage};
use crate::tool_executor_actor::{ToolExecutorActor, ToolExecutorMessage};
use crate::tools::{parse_first_tool_call, splice_result};

pub enum AgenticMessage {
    /// Run the generate↔tool-execute loop starting from `prompt`.
    Run {
        prompt: String,
        sampling: GenerateConfig,
        max_steps: usize,
        reply: oneshot::Sender<anyhow::Result<AgenticReport>>,
    },
}

#[derive(Debug, Clone)]
pub struct AgenticReport {
    pub final_text: String,
    pub tool_calls: usize,
    pub steps: usize,
    /// Per-step records — useful for debugging the loop.
    pub trace: Vec<StepRecord>,
}

#[derive(Debug, Clone)]
pub struct StepRecord {
    pub step: usize,
    pub generated_tokens: usize,
    pub tool_called: Option<String>,
    pub tool_result: Option<Result<String, String>>,
}

pub struct AgenticGeneratorActor {
    pub model: ActorRef<ModelActor>,
    pub executor: ActorRef<ToolExecutorActor>,
    pub tokenizer: Arc<Tokenizer>,
    pub per_request_timeout: Duration,
}

impl AgenticGeneratorActor {
    pub fn new(
        model: ActorRef<ModelActor>,
        executor: ActorRef<ToolExecutorActor>,
        tokenizer: Arc<Tokenizer>,
    ) -> Self {
        Self {
            model,
            executor,
            tokenizer,
            per_request_timeout: Duration::from_secs(60),
        }
    }

    async fn generate_chunk(
        &self,
        text: &str,
        sampling: &GenerateConfig,
    ) -> anyhow::Result<(String, usize)> {
        let prompt_ids = self.tokenizer.encode(text)?;
        let (tx, rx) = oneshot::channel();
        self.model
            .tell(ModelMessage::GenerateTokens {
                prompt_ids: prompt_ids.clone(),
                cfg: sampling.clone(),
                reply: tx,
            })
            .map_err(|e| anyhow::anyhow!("send GenerateTokens: {e:?}"))?;
        let tokens = timeout(self.per_request_timeout, rx).await???;
        let new_token_count = tokens.len().saturating_sub(prompt_ids.len());
        let full = self.tokenizer.decode(&tokens)?;
        Ok((full, new_token_count))
    }

    async fn dispatch_tool(&self, call: &crate::tools::ToolCall) -> anyhow::Result<Result<String, String>> {
        let (tx, rx) = oneshot::channel();
        self.executor
            .tell(ToolExecutorMessage::Execute { call: call.clone(), reply: tx })
            .map_err(|e| anyhow::anyhow!("send Execute: {e:?}"))?;
        let result = timeout(self.per_request_timeout, rx).await??;
        match result {
            Ok(s) => Ok(Ok(s)),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn run_loop(
        &self,
        prompt: String,
        sampling: GenerateConfig,
        max_steps: usize,
    ) -> anyhow::Result<AgenticReport> {
        let mut buffer = prompt;
        let mut tool_calls = 0usize;
        let mut trace = Vec::with_capacity(max_steps);

        for step in 0..max_steps {
            // Generate continuation against the running buffer. The parser
            // skips already-resolved calls (those containing `=` in body),
            // so scanning the full text is safe and lets us catch tool
            // calls that arrived inside the prompt.
            let (full, new_tokens) = self.generate_chunk(&buffer, &sampling).await?;

            if let Some((range, call)) = parse_first_tool_call(&full) {
                let res = self.dispatch_tool(&call).await?;
                let inserted = match &res {
                    Ok(s) => s.clone(),
                    Err(e) => format!("ERR:{e}"),
                };
                buffer = splice_result(&full, range, &inserted);
                tool_calls += 1;
                trace.push(StepRecord {
                    step,
                    generated_tokens: new_tokens,
                    tool_called: Some(call.name.clone()),
                    tool_result: Some(res),
                });
                info!(step, tool = %call.name, inserted = %inserted, "tool call resolved");
                continue;
            }

            // No tool call this step — accept the new tokens and stop.
            buffer = full;
            trace.push(StepRecord {
                step,
                generated_tokens: new_tokens,
                tool_called: None,
                tool_result: None,
            });
            info!(step, new_tokens, "no tool call in this chunk; loop done");
            break;
        }

        Ok(AgenticReport {
            final_text: buffer,
            tool_calls,
            steps: trace.len(),
            trace,
        })
    }
}

impl Actor for AgenticGeneratorActor {
    type Message = AgenticMessage;

    fn receive(
        &mut self,
        msg: Self::Message,
        _ctx: &mut ActorContext<Self>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + '_>> {
        Box::pin(async move {
            match msg {
                AgenticMessage::Run { prompt, sampling, max_steps, reply } => {
                    let r = self.run_loop(prompt, sampling, max_steps).await;
                    if let Err(e) = &r {
                        warn!(error = %e, "agentic loop failed");
                    }
                    let _ = reply.send(r);
                }
            }
        })
    }
}
