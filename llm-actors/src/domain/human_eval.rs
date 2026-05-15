//! Phase 22 Stage A — HumanEval-164 as a Rust `Domain`.
//!
//! Mirrors the verify pattern Phase 17–20's Python scripts used:
//! load 164 problems from `data/humaneval/HumanEval.jsonl`, concat
//! `prompt + completion + "\n\n" + test + "\ncheck(<entry_point>)\n"`,
//! write to a scratch file, run `python3` with a timeout, accept on
//! exit code 0.
//!
//! With this Domain wired into `EvaluatorActor::<QwenModelActor>`,
//! the Pekko stack now drives real HumanEval evaluation against
//! Qwen2.5-Coder-0.5B end-to-end — reproducing Phase 17 S6's baseline
//! (pass@1 ≈ 0.216, pass@10 ≈ 0.524) via the same pipeline that
//! Phase 21 Stage H ships.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

use rand::rngs::StdRng;
use rand::Rng;
use serde::Deserialize;

use crate::domain::Domain;
use crate::types::Verdict;

#[derive(Debug, Clone, Deserialize)]
pub struct HumanEvalProblem {
    pub task_id: String,
    pub prompt: String,
    pub entry_point: String,
    pub canonical_solution: String,
    pub test: String,
}

pub struct HumanEvalDomain {
    pub problems: Vec<HumanEvalProblem>,
    /// `prompt → index in problems` map for fast verify-time lookup.
    /// HumanEval prompts are unique across the 164-problem set.
    prompt_to_idx: HashMap<String, usize>,
    pub scratch_dir: PathBuf,
    pub timeout: Duration,
    /// Serializes file writes inside `scratch_dir` so verify calls
    /// from concurrent tasks don't clobber each other's `solution.py`.
    /// Mirrors `RustCodeDomain`'s write-lock pattern.
    write_lock: Mutex<()>,
}

impl HumanEvalDomain {
    /// Load all problems from a JSONL path (one problem per line).
    /// The bundled `data/humaneval/HumanEval.jsonl` is the standard
    /// OpenAI dataset.
    pub fn from_jsonl(
        jsonl_path: impl AsRef<Path>,
        scratch_dir: impl Into<PathBuf>,
    ) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(jsonl_path.as_ref())?;
        let mut problems = Vec::with_capacity(164);
        let mut prompt_to_idx = HashMap::with_capacity(164);
        for (line_idx, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let p: HumanEvalProblem = serde_json::from_str(line).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("line {line_idx}: {e}"),
                )
            })?;
            prompt_to_idx.insert(p.prompt.clone(), problems.len());
            problems.push(p);
        }
        let scratch_dir = scratch_dir.into();
        std::fs::create_dir_all(&scratch_dir)?;
        Ok(Self {
            problems,
            prompt_to_idx,
            scratch_dir,
            timeout: Duration::from_secs(8),
            write_lock: Mutex::new(()),
        })
    }

    /// Convenience: load the bundled `data/humaneval/HumanEval.jsonl`
    /// relative to the workspace root.
    pub fn from_default_data_dir() -> std::io::Result<Self> {
        let here = std::env::current_dir()?;
        let candidate = here.join("data/humaneval/HumanEval.jsonl");
        let scratch = std::env::temp_dir().join("workllm-humaneval-scratch");
        Self::from_jsonl(&candidate, &scratch)
    }

    pub fn n_problems(&self) -> usize {
        self.problems.len()
    }

    /// Build the executable Python program for a (prompt, completion)
    /// pair: appends the `test` block defining `check(candidate)`,
    /// then a top-level `check(<entry_point>)` call. The Phase 15+
    /// scripts use exactly this composition.
    fn build_program(&self, problem: &HumanEvalProblem, completion: &str) -> String {
        let mut s = String::with_capacity(problem.prompt.len() + completion.len() + 512);
        s.push_str(&problem.prompt);
        s.push_str(completion);
        s.push_str("\n\n");
        s.push_str(&problem.test);
        s.push_str("\ncheck(");
        s.push_str(&problem.entry_point);
        s.push_str(")\n");
        s
    }
}

impl Domain for HumanEvalDomain {
    fn sample_prompt(&self, rng: &mut StdRng) -> String {
        let i = rng.gen_range(0..self.problems.len());
        self.problems[i].prompt.clone()
    }

    fn verify(&self, prompt: &str, completion: &str) -> Verdict {
        let idx = match self.prompt_to_idx.get(prompt) {
            Some(i) => *i,
            None => {
                return Verdict::Inconclusive {
                    reason: format!(
                        "unknown HumanEval prompt (truncated to first 80 chars): {:?}",
                        &prompt[..prompt.len().min(80)]
                    ),
                };
            }
        };
        let problem = &self.problems[idx];
        let program = self.build_program(problem, completion);

        // Write program to scratch_dir/solution.py under the write lock
        // so concurrent verifies don't race.
        let solution_path = self.scratch_dir.join("solution.py");
        let _guard = self
            .write_lock
            .lock()
            .expect("HumanEvalDomain write lock poisoned");
        if let Err(e) = std::fs::write(&solution_path, &program) {
            return Verdict::Inconclusive {
                reason: format!("write {}: {e}", solution_path.display()),
            };
        }

        let mut cmd = Command::new("python3");
        cmd.arg(&solution_path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                return Verdict::Inconclusive {
                    reason: format!("spawn python3: {e}"),
                };
            }
        };

        // Poll-based timeout. We can't use wait_timeout directly
        // without an extra crate; this matches Phase 15/17's
        // subprocess.run(..., timeout=...) semantics closely enough.
        let start = std::time::Instant::now();
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    return if status.success() {
                        Verdict::Correct
                    } else {
                        Verdict::Incorrect {
                            reason: format!("python3 exit code {:?}", status.code()),
                        }
                    };
                }
                Ok(None) => {
                    if start.elapsed() > self.timeout {
                        let _ = child.kill();
                        return Verdict::Incorrect {
                            reason: format!("python3 timed out after {:?}", self.timeout),
                        };
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(e) => {
                    return Verdict::Inconclusive {
                        reason: format!("try_wait: {e}"),
                    };
                }
            }
        }
    }

    fn charset(&self) -> &str {
        // Unused for the BPE tokenizer path that QwenModelActor uses.
        ""
    }

    /// Stage B sequential-sweep support — exposes the fixed 164-problem
    /// set as an indexed series so `EvaluatorActor::EvalSequential` can
    /// iterate problems without replacement.
    fn n_prompts(&self) -> Option<usize> {
        Some(self.problems.len())
    }

    fn nth_prompt(&self, i: usize) -> Option<String> {
        self.problems.get(i).map(|p| p.prompt.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tiny helper to get the current function name as a literal for
    // per-test scratch-dir disambiguation. Avoids a dependency.
    macro_rules! function_name {
        () => {{
            fn f() {}
            fn type_name<T>(_: T) -> &'static str {
                std::any::type_name::<T>()
            }
            let n = type_name(f);
            let n = n.strip_suffix("::f").unwrap_or(n);
            n.rsplit("::").next().unwrap_or(n)
        }};
    }

    fn skip_if_no_jsonl(scratch_tag: &str) -> Option<HumanEvalDomain> {
        // Per-test scratch dir so parallel tests don't race on solution.py.
        let scratch = std::env::temp_dir().join(format!("workllm-humaneval-test-{scratch_tag}"));
        let p = PathBuf::from("../data/humaneval/HumanEval.jsonl");
        if !p.exists() {
            let p2 = PathBuf::from("data/humaneval/HumanEval.jsonl");
            if !p2.exists() {
                return None;
            }
            return HumanEvalDomain::from_jsonl(&p2, &scratch).ok();
        }
        HumanEvalDomain::from_jsonl(&p, &scratch).ok()
    }

    #[test]
    fn loader_returns_164_problems() {
        let Some(d) = skip_if_no_jsonl(function_name!()) else {
            eprintln!("skipping: HumanEval.jsonl not on disk");
            return;
        };
        assert_eq!(d.n_problems(), 164);
        assert_eq!(d.problems[0].task_id, "HumanEval/0");
    }

    #[test]
    fn verify_canonical_solution_passes() {
        let Some(d) = skip_if_no_jsonl(function_name!()) else {
            eprintln!("skipping: HumanEval.jsonl not on disk");
            return;
        };
        // The canonical_solution should always pass the test block.
        // Use problem 0 as a smoke check.
        let p = &d.problems[0];
        let verdict = d.verify(&p.prompt, &p.canonical_solution);
        assert!(
            verdict.is_correct(),
            "canonical solution failed verify: {verdict:?}"
        );
    }

    #[test]
    fn verify_empty_completion_fails() {
        let Some(d) = skip_if_no_jsonl(function_name!()) else {
            eprintln!("skipping: HumanEval.jsonl not on disk");
            return;
        };
        let p = &d.problems[0];
        let verdict = d.verify(&p.prompt, "    pass\n");
        assert!(
            !verdict.is_correct(),
            "`pass` body should fail check(): {verdict:?}"
        );
    }

    #[test]
    fn verify_unknown_prompt_is_inconclusive() {
        let Some(d) = skip_if_no_jsonl(function_name!()) else {
            eprintln!("skipping: HumanEval.jsonl not on disk");
            return;
        };
        let verdict = d.verify("def fake_problem():", "    pass");
        assert!(
            matches!(verdict, Verdict::Inconclusive { .. }),
            "expected Inconclusive for unknown prompt, got {verdict:?}"
        );
    }
}
