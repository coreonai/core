//! VerifierActor: applies the active Domain's verifier to each trajectory.
//!
//! Verification is heterogeneous in cost:
//! - Arithmetic / char-string domains: a few microseconds, CPU-bound inline.
//! - `HumanEvalDomain` / `MbppDomain` / `RustCodeDomain`: tens to hundreds of
//!   milliseconds because each call spawns a `python3` (or `cargo`)
//!   subprocess. These dominate batch wallclock when a Domain has hundreds
//!   of trajectories per round.
//!
//! Phase 22 Stage D follow-up: VerifierActor now runs verifies in parallel
//! via `tokio::task::spawn_blocking` bounded by a `Semaphore`. The
//! `HumanEvalDomain` and `MbppDomain` were made thread-safe in the same
//! change (per-call unique scratch filenames + `AtomicU64` counter), so
//! `Arc<dyn Domain>` clones can run truly concurrently from multiple
//! blocking threads. Net wallclock impact:
//! - Training-loop verify (164 trajectories): ~82s → ~11s at concurrency=8.
//! - Aggregate eval (1640 trajectories interleaved with gen) still
//!   bottlenecked by serial generation; the verify reduction matters when
//!   verify is the dominant phase (cargo-build domains).

use std::sync::Arc;

use pekko_actor::{Actor, ActorContext};
use tokio::sync::{oneshot, Semaphore};
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
    /// Maximum number of concurrent `domain.verify(...)` calls. Each
    /// runs in a `tokio::task::spawn_blocking` worker. Default `8` is
    /// a reasonable balance for the python3-subprocess domains
    /// (HumanEval, MBPP) on an 8+ core box.
    pub verify_concurrency: usize,
}

impl VerifierActor {
    pub fn new(domain: Arc<dyn Domain>) -> Self {
        Self {
            domain,
            verify_concurrency: 8,
        }
    }

    /// Override the default 8-worker concurrency. Pass `1` to restore
    /// the old serial behavior (useful for diagnosis or when verify
    /// has external state that's not yet thread-safe).
    pub fn with_concurrency(mut self, c: usize) -> Self {
        self.verify_concurrency = c.max(1);
        self
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
                    let domain = Arc::clone(&self.domain);
                    let sem = Arc::new(Semaphore::new(self.verify_concurrency));
                    let mut handles = Vec::with_capacity(items.len());
                    for t in items {
                        let domain = Arc::clone(&domain);
                        let sem = Arc::clone(&sem);
                        let h = tokio::spawn(async move {
                            let _permit = sem
                                .acquire()
                                .await
                                .expect("verify semaphore closed unexpectedly");
                            let (prompt, completion) = (t.prompt.clone(), t.completion.clone());
                            let (verdict, score) = tokio::task::spawn_blocking(move || {
                                let v = domain.verify(&prompt, &completion);
                                let s = domain.score(&v);
                                (v, s)
                            })
                            .await
                            .expect("verify blocking task panicked");
                            VerifiedTrajectory {
                                trajectory: t,
                                verdict,
                                score,
                            }
                        });
                        handles.push(h);
                    }
                    let mut out = Vec::with_capacity(handles.len());
                    let mut correct = 0usize;
                    for h in handles {
                        match h.await {
                            Ok(vt) => {
                                if vt.verdict.is_correct() {
                                    correct += 1;
                                }
                                out.push(vt);
                            }
                            Err(e) => {
                                tracing::error!("verify task join failed: {e}");
                            }
                        }
                    }
                    info!(
                        verified = out.len(),
                        correct,
                        concurrency = self.verify_concurrency,
                        "VerifierActor batch done"
                    );
                    let _ = reply.send(out);
                }
            }
        })
    }
}
