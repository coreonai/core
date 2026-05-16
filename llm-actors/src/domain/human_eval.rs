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
use std::sync::atomic::{AtomicU64, Ordering};
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
    /// Atomic counter giving each verify call a unique scratch
    /// filename (`solution_{id}.py`). Concurrent verifies are safe
    /// because each call writes to a distinct path and runs an
    /// independent `python3` subprocess. The previous `Mutex<()>`
    /// was a worst-case serializer; removing it unlocks
    /// `VerifierActor`'s parallel batch path.
    next_call_id: AtomicU64,
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
            next_call_id: AtomicU64::new(0),
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

        // Each verify call writes to its own scratch filename. No
        // Mutex needed; concurrent calls don't race because the
        // filenames are unique. `python3 solution_{id}.py` produces
        // an independent subprocess per call. The file is cleaned
        // up at the end (best-effort — failures to remove are not
        // fatal but log nothing to avoid spamming concurrent runs).
        let call_id = self.next_call_id.fetch_add(1, Ordering::Relaxed);
        let solution_path = self.scratch_dir.join(format!("solution_{call_id}.py"));
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
                let _ = std::fs::remove_file(&solution_path);
                return Verdict::Inconclusive {
                    reason: format!("spawn python3: {e}"),
                };
            }
        };

        // Poll-based timeout. We can't use wait_timeout directly
        // without an extra crate; this matches Phase 15/17's
        // subprocess.run(..., timeout=...) semantics closely enough.
        let start = std::time::Instant::now();
        let verdict = loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    break if status.success() {
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
                        break Verdict::Incorrect {
                            reason: format!("python3 timed out after {:?}", self.timeout),
                        };
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(e) => {
                    break Verdict::Inconclusive {
                        reason: format!("try_wait: {e}"),
                    };
                }
            }
        };
        let _ = std::fs::remove_file(&solution_path);
        verdict
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
    fn verify_parallel_canonical_solutions_all_pass() {
        // Phase 22 Stage D follow-up #3: verify call thread-safety.
        // 8 concurrent threads, each calls verify() on a canonical
        // solution from the first 8 HumanEval problems against the
        // same `HumanEvalDomain` instance. All must verify Correct.
        // Pre-fix this would race on `scratch_dir/solution.py`; now
        // each call writes a unique `solution_{call_id}.py`.
        let Some(d) = skip_if_no_jsonl(function_name!()) else {
            eprintln!("skipping: HumanEval.jsonl not on disk");
            return;
        };
        let d = std::sync::Arc::new(d);
        let n = 8usize.min(d.problems.len());
        let handles: Vec<_> = (0..n)
            .map(|i| {
                let d = std::sync::Arc::clone(&d);
                std::thread::spawn(move || {
                    let p = &d.problems[i];
                    d.verify(&p.prompt, &p.canonical_solution)
                })
            })
            .collect();
        let verdicts: Vec<Verdict> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        for (i, v) in verdicts.iter().enumerate() {
            assert!(
                v.is_correct(),
                "parallel canonical solution at index {i} did not verify: {v:?}"
            );
        }
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
