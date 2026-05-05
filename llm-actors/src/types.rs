//! Shared types for the self-improvement loop.
//!
//! `Trajectory` is the raw (prompt, completion) pair from the model.
//! `VerifiedTrajectory` adds the verifier's verdict and a numeric score
//! that the curator can use for prioritized replay.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory {
    pub prompt: String,
    pub completion: String,
    /// Source checkpoint name/id (for tracing rounds).
    pub source: String,
}

impl Trajectory {
    /// Concatenation used for training: prompt + completion.
    pub fn full_text(&self) -> String {
        let mut s = String::with_capacity(self.prompt.len() + self.completion.len());
        s.push_str(&self.prompt);
        s.push_str(&self.completion);
        s
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Verdict {
    Correct,
    Incorrect { reason: String },
    Inconclusive { reason: String },
}

impl Verdict {
    pub fn is_correct(&self) -> bool {
        matches!(self, Verdict::Correct)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedTrajectory {
    pub trajectory: Trajectory,
    pub verdict: Verdict,
    /// 0.0..=1.0
    pub score: f32,
}

impl VerifiedTrajectory {
    pub fn is_correct(&self) -> bool {
        self.verdict.is_correct()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RoundReport {
    pub round: usize,
    pub generated: usize,
    pub correct: usize,
    pub eval_correct_before: Option<usize>,
    pub eval_correct_after: Option<usize>,
    pub eval_total: usize,
    pub training_steps: usize,
    pub last_train_loss: Option<f32>,
    pub elapsed_ms: u128,
}

impl RoundReport {
    pub fn pass_rate_generated(&self) -> f32 {
        if self.generated == 0 {
            0.0
        } else {
            self.correct as f32 / self.generated as f32
        }
    }

    pub fn pass_rate_eval_after(&self) -> Option<f32> {
        self.eval_correct_after.map(|c| {
            if self.eval_total == 0 {
                0.0
            } else {
                c as f32 / self.eval_total as f32
            }
        })
    }
}
