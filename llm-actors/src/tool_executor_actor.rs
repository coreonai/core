//! ToolExecutorActor: dispatches parsed `ToolCall`s to the registry.
//!
//! Synchronous tools run inline; long-running ones can spawn_blocking
//! internally. Errors are returned as `Err(ToolError)` to the caller.

use pekko_actor::{Actor, ActorContext};
use tokio::sync::oneshot;
use tracing::{info, warn};

use crate::tools::{ToolCall, ToolError, ToolRegistry};

pub enum ToolExecutorMessage {
    Execute {
        call: ToolCall,
        reply: oneshot::Sender<Result<String, ToolError>>,
    },
    /// List registered tool names.
    ListTools { reply: oneshot::Sender<Vec<String>> },
}

pub struct ToolExecutorActor {
    pub registry: ToolRegistry,
}

impl ToolExecutorActor {
    pub fn new(registry: ToolRegistry) -> Self {
        Self { registry }
    }
}

impl Actor for ToolExecutorActor {
    type Message = ToolExecutorMessage;

    fn receive(
        &mut self,
        msg: Self::Message,
        _ctx: &mut ActorContext<Self>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + '_>> {
        Box::pin(async move {
            match msg {
                ToolExecutorMessage::Execute { call, reply } => {
                    info!(tool = %call.name, args = %call.args, "executing tool");
                    let r = self.registry.dispatch(&call);
                    if let Err(e) = &r {
                        warn!(error = %e, "tool dispatch error");
                    }
                    let _ = reply.send(r);
                }
                ToolExecutorMessage::ListTools { reply } => {
                    let names: Vec<String> = self
                        .registry
                        .names()
                        .iter()
                        .map(|s| s.to_string())
                        .collect();
                    let _ = reply.send(names);
                }
            }
        })
    }
}
