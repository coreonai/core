//! Tool-use arithmetic domain.
//!
//! Trains and evaluates a model that reasons via tool calls. Each
//! trajectory has three lines:
//!
//!   Q: A+B=                       ← prompt (sampled by the domain)
//!   (arith add A B=A+B)           ← tool-call line, RESOLVED form
//!   A: A+B                        ← final answer
//!
//! The training corpus uses the *resolved* tool-call form so the model
//! produces complete, verifiable trajectories. At inference time the
//! `AgenticGeneratorActor` happily handles either flavor:
//!   - if the model emits `(arith add 3 4)\n` (unresolved), the executor
//!     dispatches and splices the result;
//!   - if it emits `(arith add 3 4=7)\n` directly (resolved, learned from
//!     the corpus), the parser recognizes it and leaves it alone.
//!
//! Verification: parse the first `A: N\n` line of the completion and
//! compare `N` against `A + B`.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;

use super::arithmetic::SeedMode;
use super::Domain;
use crate::types::Verdict;

pub struct ToolUseArithmeticDomain {
    pub max_operand: u32,
}

impl Default for ToolUseArithmeticDomain {
    fn default() -> Self {
        Self { max_operand: 9 }
    }
}

impl ToolUseArithmeticDomain {
    /// `"Q: A+B=\n"` — terminator must match what the trainer sees.
    pub fn render_prompt(&self, a: u32, b: u32) -> String {
        format!("Q: {a}+{b}=\n")
    }

    /// Full trajectory used for training (resolved tool call inline).
    pub fn render_full_trajectory(&self, a: u32, b: u32) -> String {
        let r = a + b;
        format!("Q: {a}+{b}=\n(arith add {a} {b}={r})\nA: {r}\n")
    }

    /// Concatenated corpus of `n` random trajectories. Deterministic in
    /// `seed` for reproducibility across training runs.
    pub fn synth_corpus(&self, n: usize, seed: u64) -> String {
        self.synth_corpus_with_mode(n, seed, SeedMode::Full)
    }

    /// Like `synth_corpus`, but restricts the (a, b) draws to the subset
    /// implied by `mode`. `NoCarry` keeps only `a + b <= max_operand` —
    /// useful for curriculum learning where the model is trained on the
    /// "easy half" and self-improve has to discover the carry half.
    pub fn synth_corpus_with_mode(&self, n: usize, seed: u64, mode: SeedMode) -> String {
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(seed);
        let pairs = self.enumerate_seed_pairs(mode);
        if pairs.is_empty() {
            return String::new();
        }
        let mut s = String::with_capacity(n * 24);
        for _ in 0..n {
            let &(a, b) = pairs.choose(&mut rng).unwrap();
            s.push_str(&self.render_full_trajectory(a, b));
        }
        s
    }

    /// All (a, b) pairs allowed under `mode`. Mirrors `ArithmeticDomain`'s
    /// version so the same curriculum knob works for both.
    pub fn enumerate_seed_pairs(&self, mode: SeedMode) -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        match mode {
            SeedMode::None => {}
            SeedMode::Full => {
                for a in 0..=self.max_operand {
                    for b in 0..=self.max_operand {
                        out.push((a, b));
                    }
                }
            }
            SeedMode::NoCarry => {
                for a in 0..=self.max_operand {
                    for b in 0..=self.max_operand {
                        if a + b <= self.max_operand {
                            out.push((a, b));
                        }
                    }
                }
            }
        }
        out
    }

    fn parse_prompt(prompt: &str) -> Option<(u32, u32)> {
        // Accept either "Q: A+B=\n" or "Q: A+B=" (trailing newline optional).
        let s = prompt.trim();
        let s = s.strip_prefix("Q:").unwrap_or(s).trim();
        let s = s.strip_suffix('=')?;
        let (a, b) = s.split_once('+')?;
        Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
    }

    fn parse_answer(completion: &str) -> Option<u32> {
        // First "A: N" or "\nA: N" line.
        for line in completion.lines() {
            let line = line.trim_start();
            if let Some(rest) = line.strip_prefix("A:") {
                let rest = rest.trim();
                let mut digits = String::new();
                for c in rest.chars() {
                    if c.is_ascii_digit() {
                        digits.push(c);
                    } else {
                        break;
                    }
                }
                if !digits.is_empty() {
                    return digits.parse().ok();
                }
            }
        }
        None
    }
}

impl Domain for ToolUseArithmeticDomain {
    fn sample_prompt(&self, rng: &mut StdRng) -> String {
        let a = rng.gen_range(0..=self.max_operand);
        let b = rng.gen_range(0..=self.max_operand);
        self.render_prompt(a, b)
    }

    fn verify(&self, prompt: &str, completion: &str) -> Verdict {
        let (a, b) = match Self::parse_prompt(prompt) {
            Some(p) => p,
            None => {
                return Verdict::Inconclusive {
                    reason: format!("bad prompt: {prompt:?}"),
                }
            }
        };
        match Self::parse_answer(completion) {
            None => Verdict::Incorrect {
                reason: format!("no `A: N` line in completion: {completion:?}"),
            },
            Some(n) if n == a + b => Verdict::Correct,
            Some(n) => Verdict::Incorrect {
                reason: format!("expected {}, got {n}", a + b),
            },
        }
    }

    fn charset(&self) -> &str {
        // Includes everything used in render_full_trajectory.
        "0123456789+=()abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ:\n "
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_full_traj_round_trips() {
        let d = ToolUseArithmeticDomain::default();
        let traj = d.render_full_trajectory(3, 4);
        assert_eq!(traj, "Q: 3+4=\n(arith add 3 4=7)\nA: 7\n");
    }

    #[test]
    fn verify_correct_when_answer_matches() {
        let d = ToolUseArithmeticDomain::default();
        let p = d.render_prompt(3, 4);
        let comp = "(arith add 3 4=7)\nA: 7\n";
        assert!(matches!(d.verify(&p, comp), Verdict::Correct));
    }

    #[test]
    fn verify_incorrect_on_wrong_answer() {
        let d = ToolUseArithmeticDomain::default();
        let p = d.render_prompt(3, 4);
        let comp = "(arith add 3 4=7)\nA: 8\n";
        assert!(matches!(d.verify(&p, comp), Verdict::Incorrect { .. }));
    }

    #[test]
    fn verify_incorrect_when_no_answer_line() {
        let d = ToolUseArithmeticDomain::default();
        let p = d.render_prompt(3, 4);
        let comp = "no answer at all";
        assert!(matches!(d.verify(&p, comp), Verdict::Incorrect { .. }));
    }

    #[test]
    fn verify_works_when_completion_starts_with_answer_line() {
        let d = ToolUseArithmeticDomain::default();
        let p = d.render_prompt(5, 6);
        // Some agentic flows may strip the tool call; verifier still succeeds
        // as long as `A: N` is present.
        let comp = "A: 11\n";
        assert!(matches!(d.verify(&p, comp), Verdict::Correct));
    }

    #[test]
    fn corpus_is_deterministic() {
        let d = ToolUseArithmeticDomain::default();
        assert_eq!(d.synth_corpus(20, 42), d.synth_corpus(20, 42));
    }

    #[test]
    fn nocarry_corpus_omits_carry_pairs() {
        let d = ToolUseArithmeticDomain::default();
        let corpus = d.synth_corpus_with_mode(200, 7, SeedMode::NoCarry);
        // Verify every trajectory's (a, b) satisfies a + b <= 9.
        for line in corpus.lines() {
            if let Some(rest) = line.strip_prefix("Q: ") {
                if let Some(eq_pos) = rest.find('=') {
                    let lhs = &rest[..eq_pos];
                    if let Some((a_s, b_s)) = lhs.split_once('+') {
                        let a: u32 = a_s.parse().unwrap();
                        let b: u32 = b_s.parse().unwrap();
                        assert!(a + b <= 9, "NoCarry must exclude {a}+{b}");
                    }
                }
            }
        }
    }

    #[test]
    fn enumerate_seed_pairs_counts_match() {
        let d = ToolUseArithmeticDomain::default();
        assert_eq!(d.enumerate_seed_pairs(SeedMode::None).len(), 0);
        assert_eq!(d.enumerate_seed_pairs(SeedMode::Full).len(), 100);
        assert_eq!(d.enumerate_seed_pairs(SeedMode::NoCarry).len(), 55);
    }
}
