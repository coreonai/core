//! Single-digit addition domain.
//!
//! Format (one example per line): `A+B=C\n` where `A,B ∈ {0..=9}` and `C = A+B`.
//! - Prompt:     `"A+B="`
//! - Completion: `"C\n"` (1 or 2 digits, then newline)
//!
//! Verifier: extract digits after `=` up to first non-digit, compare to `A+B`.
//!
//! Charset (15): `0123456789+=\n` plus space/`Q`/`:` reserved for future
//! variants — tokenizer is built from `charset()` so vocab is deterministic.

use rand::rngs::StdRng;
use rand::Rng;

use super::Domain;
use crate::types::Verdict;

pub struct ArithmeticDomain {
    pub max_operand: u32,
}

impl Default for ArithmeticDomain {
    fn default() -> Self {
        Self { max_operand: 9 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeedMode {
    /// Seed every (a,b) pair — exhaustive, leaves no headroom for self-improvement.
    Full,
    /// Seed only pairs where `a + b <= max_operand` (no-carry half).
    /// Leaves "carry" pairs as self-improvement targets.
    NoCarry,
    /// No seeding — model must bootstrap from pretraining alone.
    None,
}

impl ArithmeticDomain {
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
}

impl ArithmeticDomain {
    pub fn render_example(&self, a: u32, b: u32) -> String {
        format!("{}+{}={}\n", a, b, a + b)
    }

    /// Build a synthetic pretraining corpus: `n` random examples concatenated.
    pub fn synth_corpus(&self, n: usize, seed: u64) -> String {
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut s = String::with_capacity(n * 8);
        for _ in 0..n {
            let a = rng.gen_range(0..=self.max_operand);
            let b = rng.gen_range(0..=self.max_operand);
            s.push_str(&self.render_example(a, b));
        }
        s
    }

    fn parse_prompt(prompt: &str) -> Option<(u32, u32)> {
        let prompt = prompt.trim_end_matches('\n');
        let prompt = prompt.strip_suffix('=')?;
        let (a_str, b_str) = prompt.split_once('+')?;
        let a: u32 = a_str.parse().ok()?;
        let b: u32 = b_str.parse().ok()?;
        Some((a, b))
    }

    fn parse_completion_answer(completion: &str) -> Option<u32> {
        let trimmed = completion.trim_start();
        let mut digits = String::new();
        for c in trimmed.chars() {
            if c.is_ascii_digit() {
                digits.push(c);
            } else {
                break;
            }
        }
        digits.parse().ok()
    }
}

impl Domain for ArithmeticDomain {
    fn sample_prompt(&self, rng: &mut StdRng) -> String {
        let a = rng.gen_range(0..=self.max_operand);
        let b = rng.gen_range(0..=self.max_operand);
        format!("{}+{}=", a, b)
    }

    fn verify(&self, prompt: &str, completion: &str) -> Verdict {
        let (a, b) = match Self::parse_prompt(prompt) {
            Some(p) => p,
            None => return Verdict::Inconclusive { reason: format!("bad prompt: {prompt:?}") },
        };
        let answer = match Self::parse_completion_answer(completion) {
            Some(n) => n,
            None => {
                return Verdict::Incorrect {
                    reason: format!("no leading digits in completion: {completion:?}"),
                }
            }
        };
        let expected = a + b;
        if answer == expected {
            Verdict::Correct
        } else {
            Verdict::Incorrect {
                reason: format!("expected {expected}, got {answer}"),
            }
        }
    }

    fn charset(&self) -> &str {
        // ONLY chars that actually appear in train/eval data — including any
        // never-seen-in-training token in the vocab leaves its embedding at
        // init, and with a weight-tied LM head that quickly becomes the
        // argmax. Keep this tight.
        "0123456789+=\n"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_correct() {
        let d = ArithmeticDomain::default();
        assert!(matches!(d.verify("3+4=", "7\n"), Verdict::Correct));
        assert!(matches!(d.verify("9+9=", "18\n"), Verdict::Correct));
    }

    #[test]
    fn verify_incorrect() {
        let d = ArithmeticDomain::default();
        assert!(matches!(d.verify("3+4=", "8\n"), Verdict::Incorrect { .. }));
        assert!(matches!(d.verify("3+4=", "abc"), Verdict::Incorrect { .. }));
    }

    #[test]
    fn verify_inconclusive_on_bad_prompt() {
        let d = ArithmeticDomain::default();
        assert!(matches!(d.verify("malformed", "5\n"), Verdict::Inconclusive { .. }));
    }

    #[test]
    fn corpus_is_deterministic_for_seed() {
        let d = ArithmeticDomain::default();
        assert_eq!(d.synth_corpus(50, 1), d.synth_corpus(50, 1));
    }

    #[test]
    fn seed_modes() {
        let d = ArithmeticDomain::default();
        assert_eq!(d.enumerate_seed_pairs(SeedMode::None).len(), 0);
        assert_eq!(d.enumerate_seed_pairs(SeedMode::Full).len(), 100);
        // NoCarry: all (a,b) with a+b <= 9 = 55 pairs.
        assert_eq!(d.enumerate_seed_pairs(SeedMode::NoCarry).len(), 55);
    }
}
