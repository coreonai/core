//! VerifierActor: applies the active Domain's verifier to each trajectory.
//!
//! Verification is currently CPU-bound and fast (parse + arithmetic), so we
//! run it inline. When we add cargo-build-based verification this will move
//! to `tokio::task::spawn_blocking` with a worker pool.

use std::sync::Arc;

use pekko_actor::{Actor, ActorContext};
use tokio::sync::oneshot;
use tracing::info;

use crate::domain::Domain;
use crate::types::{Trajectory, VerifiedTrajectory};

pub enum VerifierMessage {
    Verify {
        items: Vec<Trajectory>,
        reply: oneshot::Sender<Vec<VerifiedTrajectory>>,
    },
}

pub struct VerifierActor {
    pub domain: Arc<dyn Domain>,
}

impl VerifierActor {
    pub fn new(domain: Arc<dyn Domain>) -> Self {
        Self { domain }
    }
}

impl Actor for VerifierActor {
    type Message = VerifierMessage;

    fn receive(
        &mut self,
        msg: Self::Message,
        _ctx: &mut ActorContext<Self>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + '_>> {
        Box::pin(async move {
            match msg {
                VerifierMessage::Verify { items, reply } => {
                    let mut out = Vec::with_capacity(items.len());
                    let mut correct = 0usize;
                    for t in items {
                        let verdict = self.domain.verify(&t.prompt, &t.completion);
                        if verdict.is_correct() {
                            correct += 1;
                        }
                        let score = self.domain.score(&verdict);
                        out.push(VerifiedTrajectory {
                            trajectory: t,
                            verdict,
                            score,
                        });
                    }
                    info!(verified = out.len(), correct, "VerifierActor batch done");
                    let _ = reply.send(out);
                }
            }
        })
    }
}
