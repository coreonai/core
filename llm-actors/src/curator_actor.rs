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

use std::collections::{HashMap, HashSet};

use pekko_actor::{Actor, ActorContext};
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use tokio::sync::oneshot;
use tracing::info;

use crate::types::{Trajectory, Verdict, VerifiedTrajectory};

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum SampleMode {
    #[default]
    Uniform,
    /// score * recency_decay^(age_in_inserts).
    Priority { recency_decay: f32 },
}

/// One generation by one ensemble member, paired with the (single,
/// shared) verifier's verdict on it.
#[derive(Debug, Clone)]
pub struct EnsembleItem {
    pub trajectory: Trajectory,
    pub verdict: Verdict,
    /// Which ensemble member produced this. 0..n_models.
    pub model_id: usize,
}

pub enum CuratorMessage {
    /// Insert verified items (correct or not — curator decides what to keep).
    Add {
        items: Vec<VerifiedTrajectory>,
        reply: oneshot::Sender<CuratorAddReport>,
    },
    /// Phase 11 S2: emit `(prompt, chosen, rejected)` triples for DPO
    /// training. For each prompt that has both correct and incorrect
    /// trajectories observed across all `Add` calls so far, cross-pair
    /// the recent ones (capped at `max_per_prompt` pairs per prompt to
    /// avoid combinatorial blowup). Source for `train_dpo`'s input batch.
    RenderPreferencePairs {
        max_per_prompt: usize,
        reply: oneshot::Sender<Vec<(String, String, String)>>,
    },
    /// Insert via N-actor ensemble consensus. Items are grouped by
    /// exact `(prompt, completion)` and the count of distinct models
    /// that emitted each pair AND verdict-correct is taken; only pairs
    /// with `>= min_agreement` agreeing models are accepted, with
    /// `score = matching_models / n_models`. The standard recipe is
    /// `min_agreement = (n_models + 1) / 2` (≥ half, rounded up) so a
    /// 2-of-3 ensemble accepts pairs ≥ 2 of 3 models agree on. Pass
    /// 1 for "any model that got it right" (no consensus filter).
    AddEnsemble {
        items: Vec<EnsembleItem>,
        n_models: usize,
        min_agreement: usize,
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
    /// Phase 22 Stage D fix — return the buffer's items as
    /// `(prompt, completion)` pairs (NOT concatenated). Lets the
    /// trainer compute the prompt boundary per pair and mask prompt
    /// positions out of the CE loss (Phase 17 Python's
    /// `labels[:prompt_ids.shape[0]] = -100` semantics). Iteration
    /// order follows the same `mode` rules as `RenderCorpus`.
    RenderPairs {
        mode: SampleMode,
        seed: Option<u64>,
        reply: oneshot::Sender<Vec<(String, String)>>,
    },
    /// Current buffer size.
    Size { reply: oneshot::Sender<usize> },
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
    /// Phase 11 S2: per-prompt rolling buffer of incorrect completions.
    /// Used by [`CuratorMessage::RenderPreferencePairs`] to assemble
    /// (chosen, rejected) DPO training pairs. Each prompt's vector is
    /// capped at `failure_cap_per_prompt` (FIFO eviction). Independent
    /// of the main `buf`, which still only stores correct items.
    failures: HashMap<String, Vec<VerifiedTrajectory>>,
    failure_cap_per_prompt: usize,
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
            failures: HashMap::new(),
            failure_cap_per_prompt: 16,
        }
    }

    fn insert(&mut self, item: VerifiedTrajectory, report: &mut CuratorAddReport) {
        if !item.is_correct() {
            // Phase 11 S2: keep incorrect items per-prompt for DPO pairing.
            // FIFO cap so a flood of failures on one prompt doesn't crowd out
            // recent ones from other prompts.
            let prompt_key = item.trajectory.prompt.clone();
            let bucket = self.failures.entry(prompt_key).or_default();
            bucket.push(item);
            if bucket.len() > self.failure_cap_per_prompt {
                bucket.remove(0);
            }
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

    /// Phase 11 S2: emit `(prompt, chosen, rejected)` triples by
    /// cross-pairing recent successes from `buf` against recent failures
    /// from `failures[prompt]`. Caps at `max_per_prompt` triples per
    /// prompt (most-recent first on both sides) so a single prolific
    /// prompt doesn't dominate.
    pub fn render_preference_pairs(&self, max_per_prompt: usize) -> Vec<(String, String, String)> {
        // Group successes by prompt (most-recent last).
        let mut by_prompt: HashMap<&str, Vec<&VerifiedTrajectory>> = HashMap::new();
        for it in &self.buf {
            by_prompt
                .entry(it.trajectory.prompt.as_str())
                .or_default()
                .push(it);
        }
        let mut out: Vec<(String, String, String)> = Vec::new();
        for (prompt, chosens) in by_prompt {
            let Some(failures) = self.failures.get(prompt) else {
                continue;
            };
            // Walk the most-recent successes and most-recent failures.
            let chosen_iter = chosens.iter().rev();
            let failure_iter = failures.iter().rev();
            for (c, r) in chosen_iter.zip(failure_iter).take(max_per_prompt) {
                out.push((
                    prompt.to_string(),
                    c.trajectory.completion.clone(),
                    r.trajectory.completion.clone(),
                ));
            }
        }
        out
    }

    /// Strict-majority threshold for an `n`-actor ensemble:
    ///   n=2 → 1, n=3 → 2, n=4 → 2, n=5 → 3.
    /// Mathematically `⌈n / 2⌉` (ceiling), with the n=2 special case
    /// rounded down to 1 since strict-majority is impossible at N=2.
    pub fn majority_threshold(n: usize) -> usize {
        n.div_ceil(2).max(1)
    }

    /// Apply the consensus filter described in `AddEnsemble`'s docstring.
    /// Returns `Vec<VerifiedTrajectory>` with `score = count / n_models`,
    /// ready for the standard `insert()` path. Pulled out as a free
    /// function so it's testable without spinning up the actor.
    pub fn consensus_filter(
        items: Vec<EnsembleItem>,
        n_models: usize,
        min_agreement: usize,
    ) -> Vec<VerifiedTrajectory> {
        // (prompt, completion) -> set of distinct model_ids that produced
        // this exact pair AND were verdict-correct on it.
        let mut groups: HashMap<(String, String), HashSet<usize>> = HashMap::new();
        for item in items {
            if !item.verdict.is_correct() {
                continue;
            }
            let key = (item.trajectory.prompt, item.trajectory.completion);
            groups.entry(key).or_default().insert(item.model_id);
        }
        let mut out = Vec::new();
        for ((prompt, completion), agree) in groups {
            let count = agree.len();
            if count >= min_agreement {
                let weight = count as f32 / n_models as f32;
                out.push(VerifiedTrajectory {
                    trajectory: Trajectory {
                        prompt,
                        completion,
                        source: format!("ensemble-{count}of{n_models}"),
                    },
                    verdict: Verdict::Correct,
                    score: weight,
                });
            }
        }
        out
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
                CuratorMessage::AddEnsemble {
                    items,
                    n_models,
                    min_agreement,
                    reply,
                } => {
                    let n_in = items.len();
                    let consensus = Self::consensus_filter(items, n_models, min_agreement);
                    let n_filtered = consensus.len();
                    let mut report = CuratorAddReport::default();
                    for it in consensus {
                        self.insert(it, &mut report);
                    }
                    report.buffer_size = self.buf.len();
                    info!(
                        n_in,
                        n_filtered,
                        n_models,
                        min_agreement,
                        accepted = report.accepted,
                        rejected_duplicate = report.rejected_duplicate,
                        buffer = report.buffer_size,
                        "curator add-ensemble (consensus filter)"
                    );
                    let _ = reply.send(report);
                }
                CuratorMessage::Sample {
                    n,
                    seed,
                    mode,
                    reply,
                } => {
                    let mut rng: StdRng = match seed {
                        Some(s) => StdRng::seed_from_u64(s),
                        None => StdRng::from_entropy(),
                    };
                    let out: Vec<_> = match mode {
                        SampleMode::Uniform => {
                            let take = n.min(self.buf.len());
                            let mut idx: Vec<usize> = (0..self.buf.len()).collect();
                            idx.shuffle(&mut rng);
                            idx.into_iter()
                                .take(take)
                                .map(|i| self.buf[i].clone())
                                .collect()
                        }
                        SampleMode::Priority { recency_decay } => self
                            .weighted_indices(n, recency_decay, &mut rng)
                            .into_iter()
                            .map(|i| self.buf[i].clone())
                            .collect(),
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
                CuratorMessage::RenderPairs { mode, seed, reply } => {
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
                    let pairs: Vec<(String, String)> = order
                        .into_iter()
                        .map(|i| {
                            let t = &self.buf[i].trajectory;
                            (t.prompt.clone(), t.completion.clone())
                        })
                        .collect();
                    let _ = reply.send(pairs);
                }
                CuratorMessage::Size { reply } => {
                    let _ = reply.send(self.buf.len());
                }
                CuratorMessage::RenderPreferencePairs {
                    max_per_prompt,
                    reply,
                } => {
                    let pairs = self.render_preference_pairs(max_per_prompt);
                    info!(
                        n_pairs = pairs.len(),
                        max_per_prompt,
                        n_successes = self.buf.len(),
                        n_prompts_with_failures = self.failures.len(),
                        "curator render-preference-pairs"
                    );
                    let _ = reply.send(pairs);
                }
            }
        })
    }
}

#[cfg(test)]
mod consensus_tests {
    use super::*;

    fn item(model_id: usize, prompt: &str, completion: &str, correct: bool) -> EnsembleItem {
        EnsembleItem {
            trajectory: Trajectory {
                prompt: prompt.into(),
                completion: completion.into(),
                source: format!("model-{model_id}"),
            },
            verdict: if correct {
                Verdict::Correct
            } else {
                Verdict::Incorrect {
                    reason: "wrong".into(),
                }
            },
            model_id,
        }
    }

    #[test]
    fn majority_threshold_table() {
        assert_eq!(CuratorActor::majority_threshold(1), 1);
        assert_eq!(CuratorActor::majority_threshold(2), 1);
        assert_eq!(CuratorActor::majority_threshold(3), 2);
        assert_eq!(CuratorActor::majority_threshold(4), 2);
        assert_eq!(CuratorActor::majority_threshold(5), 3);
        assert_eq!(CuratorActor::majority_threshold(7), 4);
    }

    #[test]
    fn consensus_2_of_3_accepts_with_weight_two_thirds() {
        // Design-doc canonical example: 3-model ensemble, 2 of 3 produce
        // `"hello"` for the string_len prompt. Strict-majority (≥ 2) → kept.
        let prompt = "fn main() { let s: &str = ";
        let items = vec![
            item(0, prompt, r#""hello"\n"#, true),
            item(1, prompt, r#""hello"\n"#, true),
            // Model 2 produced something else, irrelevant for "hello"'s count.
            item(2, prompt, r#""world"\n"#, true),
        ];
        let kept = CuratorActor::consensus_filter(items, 3, 2);
        // Two distinct (prompt, completion) groups: "hello" with 2 votes,
        // "world" with 1 vote. Threshold=2 → only "hello" kept.
        assert_eq!(kept.len(), 1);
        let v = &kept[0];
        assert_eq!(v.trajectory.completion, r#""hello"\n"#);
        assert!((v.score - 2.0 / 3.0).abs() < 1e-6, "got {}", v.score);
        assert_eq!(v.trajectory.source, "ensemble-2of3");
    }

    #[test]
    fn consensus_drops_incorrect_verdicts() {
        // Three models all emit the same completion, but cargo says it's
        // wrong. None should be kept regardless of agreement count.
        let items = vec![
            item(0, "p", "wrong", false),
            item(1, "p", "wrong", false),
            item(2, "p", "wrong", false),
        ];
        let kept = CuratorActor::consensus_filter(items, 3, 1);
        assert!(kept.is_empty());
    }

    #[test]
    fn consensus_dedups_same_model_repeated() {
        // Model 0 sampled the same completion twice. That's still ONE
        // distinct model agreeing — don't double-count.
        let items = vec![item(0, "p", "ans", true), item(0, "p", "ans", true)];
        // n_models=2, min_agreement=2 → not enough distinct models.
        let kept = CuratorActor::consensus_filter(items, 2, 2);
        assert!(kept.is_empty());
        // But min_agreement=1 → kept (1 distinct model).
        let kept = CuratorActor::consensus_filter(
            vec![item(0, "p", "ans", true), item(0, "p", "ans", true)],
            2,
            1,
        );
        assert_eq!(kept.len(), 1);
        assert!((kept[0].score - 0.5).abs() < 1e-6);
    }

    #[test]
    fn consensus_3_of_3_weight_one() {
        let items = vec![
            item(0, "p", "ans", true),
            item(1, "p", "ans", true),
            item(2, "p", "ans", true),
        ];
        let kept = CuratorActor::consensus_filter(items, 3, 2);
        assert_eq!(kept.len(), 1);
        assert!((kept[0].score - 1.0).abs() < 1e-6);
        assert_eq!(kept[0].trajectory.source, "ensemble-3of3");
    }

    #[test]
    fn consensus_drops_lone_correct_at_strict_majority() {
        // Only model 0 produced this slot — single-model fixed point.
        // The whole point of consensus is to filter these.
        let items = vec![item(0, "p", "ans", true)];
        let kept = CuratorActor::consensus_filter(items, 3, 2);
        assert!(
            kept.is_empty(),
            "lone-correct from one model must NOT pass strict-majority filter"
        );
    }

    #[test]
    fn consensus_keeps_lone_correct_when_threshold_is_one() {
        // min_agreement=1 disables the consensus filter — useful for
        // baselines that compare to "any-model-correct" semantics.
        let items = vec![item(0, "p", "ans", true)];
        let kept = CuratorActor::consensus_filter(items, 3, 1);
        assert_eq!(kept.len(), 1);
        assert!((kept[0].score - 1.0 / 3.0).abs() < 1e-6);
    }
}

#[cfg(test)]
mod preference_pair_tests {
    //! Phase 11 S2: tests for the DPO preference-pair rendering path.
    use super::*;

    fn correct(prompt: &str, completion: &str) -> VerifiedTrajectory {
        VerifiedTrajectory {
            trajectory: Trajectory {
                prompt: prompt.into(),
                completion: completion.into(),
                source: "test".into(),
            },
            verdict: Verdict::Correct,
            score: 1.0,
        }
    }
    fn wrong(prompt: &str, completion: &str) -> VerifiedTrajectory {
        VerifiedTrajectory {
            trajectory: Trajectory {
                prompt: prompt.into(),
                completion: completion.into(),
                source: "test".into(),
            },
            verdict: Verdict::Incorrect {
                reason: "test".into(),
            },
            score: 0.0,
        }
    }

    #[test]
    fn pairs_emit_chosen_from_buf_and_rejected_from_failures() {
        let mut c = CuratorActor::new(64);
        let mut report = CuratorAddReport::default();
        // Two correct + two failures on the same prompt.
        c.insert(correct("P", "ok1"), &mut report);
        c.insert(correct("P", "ok2"), &mut report);
        c.insert(wrong("P", "bad1"), &mut report);
        c.insert(wrong("P", "bad2"), &mut report);
        let pairs = c.render_preference_pairs(8);
        assert!(
            !pairs.is_empty(),
            "expected at least one (chosen, rejected) pair"
        );
        for (prompt, chosen, rejected) in &pairs {
            assert_eq!(prompt, "P");
            assert!(
                ["ok1", "ok2"].contains(&chosen.as_str()),
                "chosen must be from buf"
            );
            assert!(
                ["bad1", "bad2"].contains(&rejected.as_str()),
                "rejected must be from failures"
            );
        }
    }

    #[test]
    fn pairs_skip_prompt_with_no_failures() {
        let mut c = CuratorActor::new(64);
        let mut report = CuratorAddReport::default();
        c.insert(correct("P", "ok1"), &mut report);
        c.insert(correct("Q", "ok2"), &mut report);
        c.insert(wrong("Q", "bad1"), &mut report); // only Q has a failure
        let pairs = c.render_preference_pairs(8);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, "Q");
        assert_eq!(pairs[0].1, "ok2");
        assert_eq!(pairs[0].2, "bad1");
    }

    #[test]
    fn pairs_skip_prompt_with_no_successes() {
        let mut c = CuratorActor::new(64);
        let mut report = CuratorAddReport::default();
        // Only failures on P — no chosen, so no pairs for P.
        c.insert(wrong("P", "bad1"), &mut report);
        c.insert(wrong("P", "bad2"), &mut report);
        let pairs = c.render_preference_pairs(8);
        assert!(pairs.is_empty());
    }

    #[test]
    fn pairs_cap_per_prompt() {
        let mut c = CuratorActor::new(64);
        let mut report = CuratorAddReport::default();
        for i in 0..6 {
            c.insert(correct("P", &format!("ok{i}")), &mut report);
            c.insert(wrong("P", &format!("bad{i}")), &mut report);
        }
        let pairs = c.render_preference_pairs(3);
        assert_eq!(pairs.len(), 3, "expected cap at 3 pairs per prompt");
    }

    #[test]
    fn pairs_empty_when_buffer_empty() {
        let c = CuratorActor::new(64);
        let pairs = c.render_preference_pairs(8);
        assert!(pairs.is_empty());
    }

    #[test]
    fn failure_buffer_caps_per_prompt_at_16() {
        let mut c = CuratorActor::new(64);
        let mut report = CuratorAddReport::default();
        // Push 25 failures on the same prompt; only 16 should remain.
        for i in 0..25 {
            c.insert(wrong("P", &format!("bad{i}")), &mut report);
        }
        let bucket = c.failures.get("P").expect("P bucket");
        assert_eq!(bucket.len(), 16, "failure cap should hold at 16 per prompt");
        // Most-recent one should be `bad24`.
        assert_eq!(bucket.last().unwrap().trajectory.completion, "bad24");
    }
}
