//! Rust code completion domain.
//!
//! Each prompt is the prefix of a tiny Rust program with a placeholder; the
//! model produces a continuation; the verifier writes prompt+completion to a
//! scratch project's `src/main.rs` and runs `cargo run` (or `cargo check` if
//! `RustCodeDomain.run_program` is false).
//!
//! The scratch project must already exist on disk — it is NOT created by
//! this domain (we don't want the verifier to mutate parent directories at
//! runtime). Use [`RustCodeDomain::ensure_scratch_project`] once at startup.
//!
//! Phase 2.5 ships only the simplest "fill the right-hand side of an
//! `assert_eq!`" task; the surface area is intentionally small so the
//! verifier semantics are easy to test.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::Rng;

use super::Domain;
use crate::types::Verdict;

#[derive(Debug, Clone)]
pub struct RustChallenge {
    /// Free-form description (used as a comment in the generated file).
    pub name: &'static str,
    /// Prompt — code preceding the completion.
    pub prompt: &'static str,
    /// Suffix appended after the completion before compilation.
    pub suffix: &'static str,
}

/// Each challenge has a UNIQUE `prompt` so `verify`'s first-match
/// dispatch (see `challenge_for_prompt`) routes correctly. An earlier
/// version of this list had three challenges sharing one prompt prefix;
/// only the first was reachable. The rule going forward: don't add a
/// new `RustChallenge` whose `prompt` is a duplicate or a prefix of
/// another challenge's prompt.
pub const DEFAULT_CHALLENGES: &[RustChallenge] = &[
    // Slot must be an expression equal to 5. Examples: "2 + 3", "5 * 1".
    RustChallenge {
        name: "equals_5",
        prompt: "fn main() { assert_eq!(",
        suffix: ", 5); }\n",
    },
    // Slot is the inner factor; full expression is 2 * <slot> = 14, so
    // <slot> must equal 7. Distinct prompt ("2 * (" prefix) routes here.
    RustChallenge {
        name: "equals_14_via_doubling",
        prompt: "fn main() { assert_eq!(2 * (",
        suffix: "), 14); }\n",
    },
    // Slot is a `&str` whose `.len()` == 5. Examples: `"hello"`, `"world"`.
    // Distinct prompt ("let s: &str = " prefix) routes here.
    RustChallenge {
        name: "len_5_string",
        prompt: "fn main() { let s: &str = ",
        suffix: "; assert_eq!(s.len(), 5); }\n",
    },
    // Phase 13 S1 (A1): expanded challenge set — distinct prompts each
    // routing to a unique RustChallenge via exact-match dispatch.

    // Slot must be an expression equal to 10.
    RustChallenge {
        name: "equals_10",
        prompt: "fn main() { let x: i32 = ",
        suffix: "; assert_eq!(x, 10); }\n",
    },
    // Slot must be an expression equal to 0.
    RustChallenge {
        name: "equals_zero",
        prompt: "fn main() { let z: i32 = ",
        suffix: "; assert_eq!(z, 0); }\n",
    },
    // Slot must be the literal `true` or any expression evaluating to it.
    RustChallenge {
        name: "bool_true",
        prompt: "fn main() { let b: bool = ",
        suffix: "; assert_eq!(b, true); }\n",
    },
    // Slot must evaluate to `false`.
    RustChallenge {
        name: "bool_false",
        prompt: "fn main() { let f: bool = ",
        suffix: "; assert_eq!(f, false); }\n",
    },
    // Slot is a `&str` whose `.len()` == 3.
    RustChallenge {
        name: "len_3_string",
        prompt: "fn main() { let t: &str = ",
        suffix: "; assert_eq!(t.len(), 3); }\n",
    },
    // Slot is a `[i32; 3]` literal whose elements sum to 6.
    RustChallenge {
        name: "vec_sum_6",
        prompt: "fn main() { let xs: [i32; 3] = ",
        suffix: "; assert_eq!(xs.iter().sum::<i32>(), 6); }\n",
    },
    // Slot must evaluate to Some(5).
    RustChallenge {
        name: "option_some_5",
        prompt: "fn main() { let o: Option<i32> = ",
        suffix: "; assert_eq!(o, Some(5)); }\n",
    },
];

pub struct RustCodeDomain {
    pub scratch_dir: PathBuf,
    pub challenges: &'static [RustChallenge],
    /// `false`: only `cargo check` (faster, doesn't run the binary).
    /// `true`: `cargo run` + check exit status.
    pub run_program: bool,
    pub timeout: Duration,
    /// cargo invocations are serialized — they all write to the same
    /// `src/main.rs`.
    write_lock: Mutex<()>,
}

impl RustCodeDomain {
    pub fn new(scratch_dir: impl Into<PathBuf>) -> Self {
        Self {
            scratch_dir: scratch_dir.into(),
            challenges: DEFAULT_CHALLENGES,
            run_program: true,
            timeout: Duration::from_secs(30),
            write_lock: Mutex::new(()),
        }
    }

    /// Create a `Cargo.toml` + `src/main.rs` skeleton. Idempotent.
    pub fn ensure_scratch_project(&self) -> std::io::Result<()> {
        fs::create_dir_all(self.scratch_dir.join("src"))?;
        let cargo = self.scratch_dir.join("Cargo.toml");
        if !cargo.exists() {
            let mut f = fs::File::create(&cargo)?;
            writeln!(
                f,
                "[package]\nname = \"scratch\"\nversion = \"0.0.0\"\nedition = \"2021\"\n[dependencies]\n"
            )?;
        }
        let main = self.scratch_dir.join("src/main.rs");
        if !main.exists() {
            fs::write(&main, "fn main() {}\n")?;
        }
        Ok(())
    }

    fn challenge_for_prompt(&self, prompt: &str) -> Option<&RustChallenge> {
        self.challenges.iter().find(|c| c.prompt == prompt)
    }

    fn write_program(&self, prompt: &str, completion: &str, suffix: &str) -> std::io::Result<()> {
        let _guard = self.write_lock.lock().expect("write_lock poisoned");
        let path = self.scratch_dir.join("src/main.rs");
        let mut full = String::with_capacity(prompt.len() + completion.len() + suffix.len() + 32);
        full.push_str("// auto-generated by RustCodeDomain — do not edit\n");
        full.push_str(prompt);
        full.push_str(completion);
        full.push_str(suffix);
        fs::write(path, full)
    }

    fn run_cargo(&self, run: bool) -> std::io::Result<RunOutcome> {
        let start = Instant::now();
        let mut cmd = Command::new("cargo");
        cmd.arg(if run { "run" } else { "check" })
            .arg("--quiet")
            .arg("--offline")
            .current_dir(&self.scratch_dir);
        let output = cmd.output()?;
        Ok(RunOutcome {
            success: output.status.success(),
            elapsed: start.elapsed(),
            stderr_tail: tail_lines(&String::from_utf8_lossy(&output.stderr), 5),
        })
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

impl Domain for RustCodeDomain {
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
        if let Err(e) = self.write_program(prompt, completion, challenge.suffix) {
            return Verdict::Inconclusive {
                reason: format!("write failed: {e}"),
            };
        }
        match self.run_cargo(self.run_program) {
            Ok(out) if out.success => Verdict::Correct,
            Ok(out) => Verdict::Incorrect {
                reason: format!(
                    "cargo {} failed in {:?}: {}",
                    if self.run_program { "run" } else { "check" },
                    out.elapsed,
                    out.stderr_tail
                ),
            },
            Err(e) => Verdict::Inconclusive {
                reason: format!("cargo invoke failed: {e}"),
            },
        }
    }

    fn charset(&self) -> &str {
        // Rust char-level training is impractical; this charset is mostly a
        // hint. Real use of RustCodeDomain pairs it with a BPE tokenizer.
        " \n\t!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
    }

    fn truncate_completion(&self, completion: &str) -> String {
        let stops = ["\npub ", "\nfn ", "\nuse ", "\nstruct ", "\nimpl ", "\n\n", "<|fim_prefix|>"];
        let mut cut = completion.len();
        for s in stops {
            if let Some(i) = completion.find(s) {
                cut = cut.min(i);
            }
        }
        completion[..cut].trim_end().to_string()
    }

    fn repair_prompt(&self, prompt: &str, completion: &str, v: &Verdict) -> Option<String> {
        let reason = match v {
            Verdict::Incorrect { reason } => reason.as_str(),
            _ => return None,
        };
        // Hand the cargo stderr back; ask for a replacement slot only.
        Some(format!(
            "{prompt}{completion}\n// ERR:{reason}\n// Fix the expression for this prefix only:\n{prompt}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rust_code_test_{name}"));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn correct_completion_passes() {
        let dir = scratch_dir("ok");
        let d = RustCodeDomain::new(&dir);
        d.ensure_scratch_project().unwrap();
        let v = d.verify("fn main() { assert_eq!(", "2 + 3");
        assert!(matches!(v, Verdict::Correct), "got: {v:?}");
    }

    #[test]
    fn wrong_completion_fails() {
        let dir = scratch_dir("nope");
        let d = RustCodeDomain::new(&dir);
        d.ensure_scratch_project().unwrap();
        // Wrong answer (assert_eq!(2 + 4, 5) → panics at runtime).
        let v = d.verify("fn main() { assert_eq!(", "2 + 4");
        assert!(matches!(v, Verdict::Incorrect { .. }), "got: {v:?}");
    }

    #[test]
    fn unknown_prompt_inconclusive() {
        let dir = scratch_dir("unknown");
        let d = RustCodeDomain::new(&dir);
        d.ensure_scratch_project().unwrap();
        let v = d.verify("fn main() { unknown_thing(", "");
        assert!(matches!(v, Verdict::Inconclusive { .. }), "got: {v:?}");
    }

    #[test]
    fn challenge_2_doubling_dispatches_correctly() {
        let dir = scratch_dir("doubling");
        let d = RustCodeDomain::new(&dir);
        d.ensure_scratch_project().unwrap();
        // Challenge 2 prompt: "fn main() { assert_eq!(2 * (".
        // Slot 7 → assert_eq!(2 * (7), 14) passes.
        let v = d.verify("fn main() { assert_eq!(2 * (", "7");
        assert!(matches!(v, Verdict::Correct), "got: {v:?}");
    }

    #[test]
    fn challenge_3_string_len_dispatches_correctly() {
        let dir = scratch_dir("strlen");
        let d = RustCodeDomain::new(&dir);
        d.ensure_scratch_project().unwrap();
        // Challenge 3 prompt: `fn main() { let s: &str = `.
        // Slot `"hello"` → s.len() == 5 → assert passes.
        let v = d.verify("fn main() { let s: &str = ", r#""hello""#);
        assert!(matches!(v, Verdict::Correct), "got: {v:?}");
    }

    #[test]
    fn all_default_prompts_are_unique() {
        // Guards against the "all three challenges share one prompt"
        // regression that originally made challenges 2 and 3
        // unreachable: `challenge_for_prompt` does exact-string match,
        // so duplicate prompts always route to the first one.
        let prompts: Vec<&str> = DEFAULT_CHALLENGES.iter().map(|c| c.prompt).collect();
        let mut seen = std::collections::HashSet::new();
        for p in &prompts {
            assert!(
                seen.insert(p),
                "duplicate prompt in DEFAULT_CHALLENGES: {p:?}"
            );
        }
    }
}
