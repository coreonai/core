//! CuratorActor: replay buffer for verified-correct trajectories.
//!
//! - FIFO eviction at `capacity`.
//! - Dedup by `Trajectory.full_text()` (cheap: only the most recent K kept).
//! - Sampling supports two modes:
//!     - **Uniform**: classic random pick.
//!     - **Priority**: weighted by `score * recency_factor^(age)`. Recent
//!       and high-score items get sampled more — useful when the buffer
//!       is dominated by stale seed examples.
//! - `RenderCorpus` returns the buffer as a single training string. Optionally
//!   takes a `repeat` factor (caller-side; supervisor handles padding).

use std::collections::HashSet;

use pekko_actor::{Actor, ActorContext};
use rand::distributions::{Distribution, WeightedIndex};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::rngs::StdRng;
use tokio::sync::oneshot;
use tracing::info;

use crate::types::VerifiedTrajectory;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SampleMode {
    Uniform,
    /// score * recency_decay^(age_in_inserts).
    Priority { recency_decay: f32 },
}

impl Default for SampleMode {
    fn default() -> Self {
        SampleMode::Uniform
    }
}

pub enum CuratorMessage {
    /// Insert verified items (correct or not — curator decides what to keep).
    Add {
        items: Vec<VerifiedTrajectory>,
        reply: oneshot::Sender<CuratorAddReport>,
    },
    /// Sample up to `n` items.
    Sample {
        n: usize,
        seed: Option<u64>,
        mode: SampleMode,
        reply: oneshot::Sender<Vec<VerifiedTrajectory>>,
    },
    /// Concatenate items' full_text() into one training corpus.
    /// `mode` controls iteration order:
    ///   - Uniform: insertion order (FIFO).
    ///   - Priority: a single weighted shuffle of the buffer.
    RenderCorpus {
        mode: SampleMode,
        seed: Option<u64>,
        reply: oneshot::Sender<String>,
    },
    /// Current buffer size.
    Size {
        reply: oneshot::Sender<usize>,
    },
}

#[derive(Debug, Clone, Default)]
pub struct CuratorAddReport {
    pub accepted: usize,
    pub rejected_incorrect: usize,
    pub rejected_duplicate: usize,
    pub buffer_size: usize,
}

pub struct CuratorActor {
    capacity: usize,
    buf: Vec<VerifiedTrajectory>,
    seen: HashSet<String>,
    /// Insertion counter — used as recency for priority sampling.
    insert_counter: u64,
    /// Per-item insertion index (parallel to `buf`).
    insert_idx: Vec<u64>,
}

impl CuratorActor {
    pub fn new(capacity: usize) -> Self {
        let cap_hint = capacity.min(8192);
        Self {
            capacity,
            buf: Vec::with_capacity(cap_hint),
            seen: HashSet::new(),
            insert_counter: 0,
            insert_idx: Vec::with_capacity(cap_hint),
        }
    }

    fn insert(&mut self, item: VerifiedTrajectory, report: &mut CuratorAddReport) {
        if !item.is_correct() {
            report.rejected_incorrect += 1;
            return;
        }
        let key = item.trajectory.full_text();
        if self.seen.contains(&key) {
            report.rejected_duplicate += 1;
            return;
        }
        if self.buf.len() == self.capacity {
            let evicted = self.buf.remove(0);
            self.insert_idx.remove(0);
            self.seen.remove(&evicted.trajectory.full_text());
        }
        self.seen.insert(key);
        self.buf.push(item);
        self.insert_idx.push(self.insert_counter);
        self.insert_counter += 1;
        report.accepted += 1;
    }

    fn weights(&self, recency_decay: f32) -> Vec<f32> {
        let max_idx = self.insert_counter.saturating_sub(1);
        self.buf
            .iter()
            .enumerate()
            .map(|(i, it)| {
                let age = (max_idx as i64 - self.insert_idx[i] as i64).max(0) as f32;
                let recency = recency_decay.powf(age);
                (it.score.max(1e-6)) * recency.max(1e-6)
            })
            .collect()
    }

    fn weighted_indices(&self, n: usize, recency_decay: f32, rng: &mut StdRng) -> Vec<usize> {
        let weights = self.weights(recency_decay);
        if self.buf.is_empty() {
            return Vec::new();
        }
        let dist = match WeightedIndex::new(&weights) {
            Ok(d) => d,
            // All-zero weights → fall back to uniform.
            Err(_) => {
                let mut idx: Vec<usize> = (0..self.buf.len()).collect();
                idx.shuffle(rng);
                return idx.into_iter().take(n).collect();
            }
        };
        (0..n).map(|_| dist.sample(rng)).collect()
    }
}

impl Actor for CuratorActor {
    type Message = CuratorMessage;

    fn receive(
        &mut self,
        msg: Self::Message,
        _ctx: &mut ActorContext<Self>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + '_>> {
        Box::pin(async move {
            match msg {
                CuratorMessage::Add { items, reply } => {
                    let mut report = CuratorAddReport::default();
                    for it in items {
                        self.insert(it, &mut report);
                    }
                    report.buffer_size = self.buf.len();
                    info!(
                        accepted = report.accepted,
                        rejected_incorrect = report.rejected_incorrect,
                        rejected_duplicate = report.rejected_duplicate,
                        buffer = report.buffer_size,
                        "curator add"
                    );
                    let _ = reply.send(report);
                }
                CuratorMessage::Sample { n, seed, mode, reply } => {
                    let mut rng: StdRng = match seed {
                        Some(s) => StdRng::seed_from_u64(s),
                        None => StdRng::from_entropy(),
                    };
                    let out: Vec<_> = match mode {
                        SampleMode::Uniform => {
                            let take = n.min(self.buf.len());
                            let mut idx: Vec<usize> = (0..self.buf.len()).collect();
                            idx.shuffle(&mut rng);
                            idx.into_iter().take(take).map(|i| self.buf[i].clone()).collect()
                        }
                        SampleMode::Priority { recency_decay } => {
                            self.weighted_indices(n, recency_decay, &mut rng)
                                .into_iter()
                                .map(|i| self.buf[i].clone())
                                .collect()
                        }
                    };
                    let _ = reply.send(out);
                }
                CuratorMessage::RenderCorpus { mode, seed, reply } => {
                    let mut rng: StdRng = match seed {
                        Some(s) => StdRng::seed_from_u64(s),
                        None => StdRng::from_entropy(),
                    };
                    let order: Vec<usize> = match mode {
                        SampleMode::Uniform => (0..self.buf.len()).collect(),
                        SampleMode::Priority { recency_decay } => {
                            self.weighted_indices(self.buf.len(), recency_decay, &mut rng)
                        }
                    };
                    let mut s = String::new();
                    for i in order {
                        s.push_str(&self.buf[i].trajectory.full_text());
                    }
                    let _ = reply.send(s);
                }
                CuratorMessage::Size { reply } => {
                    let _ = reply.send(self.buf.len());
                }
            }
        })
    }
}
