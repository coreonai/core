//! Python code completion domain.
//!
//! Mirror of [`crate::domain::rust_code::RustCodeDomain`] but for
//! Python. Each prompt is the prefix of a tiny Python program with
//! a slot; the model produces a slot completion; the verifier
//! concatenates `prompt + completion + suffix` and runs `python -c`.
//! Verdict is `Correct` iff the interpreter exits 0 (the assert
//! survived).
//!
//! Why this exists alongside RustCodeDomain: Phase 8 Session 2 uses
//! it to test whether Phase 6 Shape C generalizes to a different
//! verifier mechanism (Python interpreter vs cargo build/run). Both
//! share the "subprocess + exit code" pattern but target different
//! languages and different toolchains. If Shape C passes the AUC
//! gate on both, the mechanism generalizes; if only on one, the
//! result was language-specific.
//!
//! `python -c` is cheaper than `cargo run` (~50ms vs ~100ms on a
//! cold scratch project), so a same-prompt-budget Phase 8 measurement
//! finishes ~2× faster than the equivalent K9 measurement.

use std::process::Command;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::Rng;

use super::Domain;
use crate::types::Verdict;

#[derive(Debug, Clone)]
pub struct PythonChallenge {
    pub name: &'static str,
    pub prompt: &'static str,
    pub suffix: &'static str,
}

/// Each challenge's `prompt` MUST be unique — `verify` dispatches by
/// exact-string match (same first-match dispatch as RustCodeDomain).
/// The Phase 6 K9 v3 "all three challenges share one prompt" bug is
/// the reference failure to avoid.
pub const DEFAULT_PYTHON_CHALLENGES: &[PythonChallenge] = &[
    // Slot is an expression equal to 5. e.g. "2 + 3", "5 * 1".
    PythonChallenge {
        name: "equals_5",
        prompt: "def f(): return ",
        suffix: "\nassert f() == 5\n",
    },
    // Slot is the inner factor; full expression is 2 * <slot> = 14, so
    // <slot> == 7. Distinct prompt prefix dispatches here.
    PythonChallenge {
        name: "equals_14_via_doubling",
        prompt: "def f(): return 2 * (",
        suffix: ")\nassert f() == 14\n",
    },
    // Slot is a string literal of length 5. e.g. `"hello"`, `"world"`.
    // Distinct prompt prefix.
    PythonChallenge {
        name: "len_5_string",
        prompt: "s = ",
        suffix: "\nassert len(s) == 5\n",
    },
];

pub struct PythonCodeDomain {
    pub challenges: &'static [PythonChallenge],
    pub timeout: Duration,
    pub python_bin: String,
    /// `python -c` invocations are serialized — they don't actually
    /// share state but holding the lock makes the timing deterministic
    /// and avoids unbounded fork pressure on tiny machines.
    invoke_lock: Mutex<()>,
}

impl PythonCodeDomain {
    pub fn new() -> Self {
        Self::with_python_bin("python3")
    }

    pub fn with_python_bin(python_bin: impl Into<String>) -> Self {
        Self {
            challenges: DEFAULT_PYTHON_CHALLENGES,
            timeout: Duration::from_secs(10),
            python_bin: python_bin.into(),
            invoke_lock: Mutex::new(()),
        }
    }

    fn challenge_for_prompt(&self, prompt: &str) -> Option<&PythonChallenge> {
        self.challenges.iter().find(|c| c.prompt == prompt)
    }

    fn run_python(&self, code: &str) -> std::io::Result<RunOutcome> {
        let _guard = self.invoke_lock.lock().expect("invoke_lock poisoned");
        let start = Instant::now();
        let output = Command::new(&self.python_bin)
            .arg("-c")
            .arg(code)
            .output()?;
        Ok(RunOutcome {
            success: output.status.success(),
            elapsed: start.elapsed(),
            stderr_tail: tail_lines(&String::from_utf8_lossy(&output.stderr), 5),
        })
    }
}

impl Default for PythonCodeDomain {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
struct RunOutcome {
    success: bool,
    elapsed: Duration,
    stderr_tail: String,
}

fn tail_lines(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    let take = lines.len().saturating_sub(n);
    lines[take..].join("\n")
}

impl Domain for PythonCodeDomain {
    fn sample_prompt(&self, rng: &mut StdRng) -> String {
        let i = rng.gen_range(0..self.challenges.len());
        self.challenges[i].prompt.to_string()
    }

    fn verify(&self, prompt: &str, completion: &str) -> Verdict {
        let challenge = match self.challenge_for_prompt(prompt) {
            Some(c) => c,
            None => {
                return Verdict::Inconclusive {
                    reason: format!("unknown prompt: {prompt:?}"),
                }
            }
        };
        let code = format!("{prompt}{completion}{}", challenge.suffix);
        match self.run_python(&code) {
            Ok(out) if out.success => Verdict::Correct,
            Ok(out) => Verdict::Incorrect {
                reason: format!(
                    "python exit nonzero in {:?}: {}",
                    out.elapsed, out.stderr_tail
                ),
            },
            Err(e) => Verdict::Inconclusive {
                reason: format!("python invoke failed: {e}"),
            },
        }
    }

    fn charset(&self) -> &str {
        // Python source charset (mostly ASCII printable). Hint for
        // CharTokenizer; BPE callers don't need this.
        " \n\t!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correct_completion_passes() {
        let d = PythonCodeDomain::new();
        // Challenge 0: "def f(): return " + "2 + 3" + "\nassert f() == 5\n"
        let v = d.verify("def f(): return ", "2 + 3");
        assert!(matches!(v, Verdict::Correct), "got: {v:?}");
    }

    #[test]
    fn wrong_completion_fails() {
        let d = PythonCodeDomain::new();
        // 2 + 4 = 6 ≠ 5, so assert fails → exit nonzero.
        let v = d.verify("def f(): return ", "2 + 4");
        assert!(matches!(v, Verdict::Incorrect { .. }), "got: {v:?}");
    }

    #[test]
    fn challenge_2_doubling_dispatches_correctly() {
        let d = PythonCodeDomain::new();
        // 2 * (7) = 14 ✓
        let v = d.verify("def f(): return 2 * (", "7");
        assert!(matches!(v, Verdict::Correct), "got: {v:?}");
    }

    #[test]
    fn challenge_3_string_len_dispatches_correctly() {
        let d = PythonCodeDomain::new();
        // len("hello") == 5 ✓
        let v = d.verify("s = ", r#""hello""#);
        assert!(matches!(v, Verdict::Correct), "got: {v:?}");
    }

    #[test]
    fn unknown_prompt_inconclusive() {
        let d = PythonCodeDomain::new();
        let v = d.verify("not_a_real_prompt(", "");
        assert!(matches!(v, Verdict::Inconclusive { .. }), "got: {v:?}");
    }

    #[test]
    fn all_default_prompts_are_unique() {
        // Same regression guard as RustCodeDomain.
        let prompts: Vec<&str> = DEFAULT_PYTHON_CHALLENGES.iter().map(|c| c.prompt).collect();
        let mut seen = std::collections::HashSet::new();
        for p in &prompts {
            assert!(
                seen.insert(p),
                "duplicate prompt in DEFAULT_PYTHON_CHALLENGES: {p:?}"
            );
        }
    }
}
