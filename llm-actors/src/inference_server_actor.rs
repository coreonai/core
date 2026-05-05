//! InferenceServerActor: external-facing request/reply for inference.
//!
//! Phase 4 ships only the actor-internal interface (request → reply via
//! oneshot). Wrapping it in TCP/HTTP (axum) or pekko-remoting is the next
//! step — the contract here is intentionally transport-neutral so either
//! path is straightforward.

use std::time::{Duration, Instant};

use nanogpt_rs::generate::GenerateConfig;
use pekko_actor::{Actor, ActorContext, ActorRef};
use tokio::sync::oneshot;
use tokio::time::timeout;
use tracing::{info, warn};

use crate::model_actor::{ModelActor, ModelMessage};

#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub prompt: String,
    pub sampling: GenerateConfig,
    /// Optional client-supplied id (for logging / tracing).
    pub request_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct InferenceResponse {
    pub request_id: Option<String>,
    pub completion: String,
    pub tokens: Vec<u32>,
    pub elapsed_ms: u128,
}

pub enum InferenceMessage {
    Serve {
        req: InferenceRequest,
        reply: oneshot::Sender<anyhow::Result<InferenceResponse>>,
    },
}

pub struct InferenceServerActor {
    pub model: ActorRef<ModelActor>,
    pub per_request_timeout: Duration,
}

impl InferenceServerActor {
    pub fn new(model: ActorRef<ModelActor>) -> Self {
        Self {
            model,
            per_request_timeout: Duration::from_secs(60),
        }
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
        })
    }
}

impl Actor for InferenceServerActor {
    type Message = InferenceMessage;

    fn receive(
        &mut self,
        msg: Self::Message,
        _ctx: &mut ActorContext<Self>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + '_>> {
        Box::pin(async move {
            match msg {
                InferenceMessage::Serve { req, reply } => {
                    info!(req_id = ?req.request_id, prompt_len = req.prompt.len(), "serving inference");
                    let r = self.handle(req).await;
                    if let Err(e) = &r {
                        warn!(error = %e, "inference failed");
                    }
                    let _ = reply.send(r);
                }
            }
        })
    }
}
