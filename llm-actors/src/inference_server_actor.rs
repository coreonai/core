//! InferenceServerActor: external-facing request/reply for inference.
//!
//! The contract is transport-neutral, so the same actor is reachable over an
//! actor-internal channel and over HTTP ([`crate::inference_http`]).
//!
//! ## Generic over the model, and tool-aware
//!
//! Phase 4 hardcoded `ActorRef<ModelActor>` here, which meant the serving
//! path could only ever drive the small nanoGPT model — the 7B built in Phase
//! 22/23 was reachable from example binaries and nowhere else. It is now
//! generic over any actor speaking [`ModelMessage`], so `QwenModelActor`
//! plugs in unchanged. The default type parameter keeps every existing call
//! site compiling; this mirrors what `AgenticGeneratorActor` and
//! `EvaluatorActor` already needed.
//!
//! Tool use is opt-in per request. Wire an [`AgenticGeneratorActor`] with
//! [`InferenceServerActor::with_agent`] and set `tools: true` on a request to
//! route it through generate → dispatch → splice → continue instead of a
//! single generation. Without an agent wired, a request asking for tools is
//! refused rather than silently answered without them — a caller that asked
//! for a tool-backed answer must not receive an ungrounded one and be unable
//! to tell.

use std::time::{Duration, Instant};

use nanogpt_rs::generate::GenerateConfig;
use pekko_actor::{Actor, ActorContext, ActorRef};
use tokio::sync::oneshot;
use tokio::time::timeout;
use tracing::{info, warn};

use crate::agentic_generator_actor::{AgenticGeneratorActor, AgenticMessage};
use crate::model_actor::{ModelActor, ModelMessage};

#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub prompt: String,
    pub sampling: GenerateConfig,
    /// Optional client-supplied id (for logging / tracing).
    pub request_id: Option<String>,
    /// Run the agentic loop (generate → dispatch → splice → continue) rather
    /// than a single generation. Requires an agent wired via
    /// [`InferenceServerActor::with_agent`]; the request fails if none is.
    pub tools: bool,
    /// Loop budget when `tools` is set. `0` uses the server default.
    pub max_steps: usize,
}

impl InferenceRequest {
    /// A plain single-shot request — the Phase 4 behaviour.
    pub fn new(prompt: String, sampling: GenerateConfig) -> Self {
        Self {
            prompt,
            sampling,
            request_id: None,
            tools: false,
            max_steps: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InferenceResponse {
    pub request_id: Option<String>,
    pub completion: String,
    pub tokens: Vec<u32>,
    pub elapsed_ms: u128,
    /// Tools dispatched while answering. `0` on the single-shot path.
    pub tool_calls: usize,
    /// Each `(tool, args, result)` the loop ran, in order. A caller checking
    /// whether an answer is grounded needs to see what actually executed —
    /// this repo has measured models stating an answer the tool never
    /// produced, so "it used a tool" cannot be inferred from the text.
    pub tool_trace: Vec<ToolCallRecord>,
    /// Why the agentic loop stopped, when it ran.
    pub stop_reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ToolCallRecord {
    pub tool: String,
    pub args: String,
    pub result: Result<String, String>,
}

pub enum InferenceMessage {
    Serve {
        req: InferenceRequest,
        reply: oneshot::Sender<anyhow::Result<InferenceResponse>>,
    },
}

pub struct InferenceServerActor<M = ModelActor>
where
    M: Actor<Message = ModelMessage>,
{
    pub model: ActorRef<M>,
    pub per_request_timeout: Duration,
    /// Optional agentic loop, for requests that ask for tools.
    pub agent: Option<ActorRef<AgenticGeneratorActor<M>>>,
    /// Loop budget when a request does not specify one.
    pub default_max_steps: usize,
}

impl<M> InferenceServerActor<M>
where
    M: Actor<Message = ModelMessage>,
{
    pub fn new(model: ActorRef<M>) -> Self {
        Self {
            model,
            per_request_timeout: Duration::from_secs(60),
            agent: None,
            default_max_steps: 4,
        }
    }

    /// Enable tool use. Requests with `tools: true` are then routed through
    /// the agentic loop.
    pub fn with_agent(mut self, agent: ActorRef<AgenticGeneratorActor<M>>) -> Self {
        self.agent = Some(agent);
        self
    }

    pub fn with_default_max_steps(mut self, n: usize) -> Self {
        self.default_max_steps = n;
        self
    }

    async fn handle_agentic(&self, req: InferenceRequest) -> anyhow::Result<InferenceResponse> {
        let t0 = Instant::now();
        let Some(agent) = &self.agent else {
            anyhow::bail!(
                "request asked for tools but no agent is wired \
                 (build the server with InferenceServerActor::with_agent)"
            );
        };
        let max_steps = if req.max_steps == 0 {
            self.default_max_steps
        } else {
            req.max_steps
        };
        let (tx, rx) = oneshot::channel();
        agent
            .tell(AgenticMessage::Run {
                prompt: req.prompt,
                sampling: req.sampling,
                max_steps,
                reply: tx,
            })
            .map_err(|e| anyhow::anyhow!("send Run: {e:?}"))?;
        let report = timeout(self.per_request_timeout, rx).await???;
        let tool_trace = report
            .trace
            .iter()
            .filter_map(|s| {
                let tool = s.tool_called.clone()?;
                Some(ToolCallRecord {
                    tool,
                    args: s.tool_args.clone().unwrap_or_default(),
                    result: s.tool_result.clone()?,
                })
            })
            .collect();
        Ok(InferenceResponse {
            request_id: req.request_id,
            completion: report.final_text,
            // The loop re-generates against the running buffer, so there is
            // no single token sequence to hand back.
            tokens: Vec::new(),
            elapsed_ms: t0.elapsed().as_millis(),
            tool_calls: report.tool_calls,
            tool_trace,
            stop_reason: Some(format!("{:?}", report.stop_reason)),
        })
    }

    async fn handle(&self, req: InferenceRequest) -> anyhow::Result<InferenceResponse> {
        let t0 = Instant::now();
        let (tx, rx) = oneshot::channel();
        self.model
            .tell(ModelMessage::Generate {
                prompt: req.prompt,
                cfg: req.sampling,
                reply: tx,
            })
            .map_err(|e| anyhow::anyhow!("send Generate: {e:?}"))?;
        let reply = timeout(self.per_request_timeout, rx).await???;
        Ok(InferenceResponse {
            request_id: req.request_id,
            completion: reply.text,
            tokens: reply.tokens,
            elapsed_ms: t0.elapsed().as_millis(),
            tool_calls: 0,
            tool_trace: Vec::new(),
            stop_reason: None,
        })
    }
}

impl<M> Actor for InferenceServerActor<M>
where
    M: Actor<Message = ModelMessage>,
{
    type Message = InferenceMessage;

    fn receive(
        &mut self,
        msg: Self::Message,
        _ctx: &mut ActorContext<Self>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + '_>> {
        Box::pin(async move {
            match msg {
                InferenceMessage::Serve { req, reply } => {
                    info!(
                        req_id = ?req.request_id,
                        prompt_len = req.prompt.len(),
                        tools = req.tools,
                        "serving inference"
                    );
                    let r = if req.tools {
                        self.handle_agentic(req).await
                    } else {
                        self.handle(req).await
                    };
                    if let Err(e) = &r {
                        warn!(error = %e, "inference failed");
                    }
                    let _ = reply.send(r);
                }
            }
        })
    }
}
