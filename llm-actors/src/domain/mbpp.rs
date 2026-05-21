//! Phase 22 Stage C — MBPP (Mostly Basic Python Problems) as a Rust `Domain`.
//!
//! Mirrors `HumanEvalDomain` but the source format is different:
//!   text       : natural-language task description
//!   code       : canonical Python implementation (we parse signature from this)
//!   test_list  : top-level assert statements
//!   task_id    : integer
//!
//! Phase 17 S3's `scripts/phase17_s3/problems.py` defined the prompt
//! shape we reproduce here:
//!   1. Parse first top-level `def name(args):` from `code`.
//!   2. Detect imports referenced in `code` (math, typing, re, heapq,
//!      collections, itertools, functools) → emit as prelude.
//!   3. Prompt = `<imports>\n\n<sig>:\n    """<text>"""\n`.
//!   4. Suffix = `\n<imports>\n<test_list joined>\n` (assertions run
//!      at module top-level — no `check(...)` call needed).
//!
//! By default we use task_id 11-110 (100 tasks), skipping the standard
//! MBPP few-shot prompt examples (1-10) — matches Phase 17 S3's
//! "MBPP-100 subset". Wallclock manageable at this size.

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
pub struct MbppProblem {
    pub task_id: usize,
    pub text: String,
    pub code: String,
    pub test_list: Vec<String>,
    #[serde(default)]
    pub test_setup_code: String,
}

#[derive(Debug, Clone)]
pub struct MbppChallenge {
    pub task_id: usize,
    pub prompt: String,
    pub suffix: String,
    pub entry_point: String,
}

pub struct MbppDomain {
    pub challenges: Vec<MbppChallenge>,
    prompt_to_idx: HashMap<String, usize>,
    pub scratch_dir: PathBuf,
    pub timeout: Duration,
    /// Atomic counter for unique `solution_{id}.py` filenames per
    /// concurrent verify call (mirrors `HumanEvalDomain`'s
    /// thread-safe pattern). No Mutex — calls are independent
    /// `python3` subprocesses writing to distinct files.
    next_call_id: AtomicU64,
}

/// Parse the first top-level `def name(arg1, arg2, ...):` signature
/// from a piece of Python source. Returns `(name, "def name(arg1, ...):")`.
/// Returns `None` if no such line is found (rare — would be a leading-
/// expression or class-only file).
fn parse_signature(code: &str) -> Option<(String, String)> {
    for line in code.lines() {
        let stripped = line.trim_start();
        // Only top-level defs (no leading whitespace). Indented `def`s
        // are nested helpers; the canonical solution always exposes
        // the entry point at top level.
        if line.len() == stripped.len() && stripped.starts_with("def ") {
            let after_def = &stripped[4..];
            let lparen = after_def.find('(')?;
            let rparen = after_def.find(')')?;
            if rparen <= lparen {
                continue;
            }
            let name = after_def[..lparen].trim().to_string();
            if name.is_empty() {
                continue;
            }
            let args = after_def[lparen + 1..rparen].trim();
            let sig = format!("def {name}({args}):");
            return Some((name, sig));
        }
    }
    None
}

/// Detect imports referenced in the canonical code. Matches Phase 17
/// S3's `detect_imports`. We re-emit them in the prompt prelude AND in
/// the test suffix so verify-time runtime has the right names bound.
fn detect_imports(code: &str) -> Vec<&'static str> {
    let mut out = Vec::new();
    if code.contains("import math") {
        out.push("import math");
    }
    // `from typing import *` covers List/Dict/Set/Tuple/Optional usage
    // even when only the type aliases (not the `from typing` statement)
    // appear in the code — Phase 17 S3's regex includes both branches.
    if code.contains("from typing")
        || code.contains("List[")
        || code.contains("Dict[")
        || code.contains("Set[")
        || code.contains("Tuple[")
        || code.contains("Optional[")
    {
        out.push("from typing import *");
    }
    if code.contains("import re") {
        out.push("import re");
    }
    if code.contains("import heapq") {
        out.push("import heapq");
    }
    if code.contains("import collections")
        || code.contains("Counter(")
        || code.contains("defaultdict(")
    {
        out.push("import collections");
    }
    if code.contains("from collections") {
        out.push("from collections import *");
    }
    if code.contains("import itertools") || code.contains("itertools.") {
        out.push("import itertools");
    }
    if code.contains("import functools") || code.contains("reduce(") {
        out.push("import functools");
    }
    out
}

fn build_challenge(problem: &MbppProblem) -> Option<MbppChallenge> {
    let (entry, sig) = parse_signature(&problem.code)?;
    let imports = detect_imports(&problem.code);
    let prelude = if imports.is_empty() {
        String::new()
    } else {
        format!("{}\n\n", imports.join("\n"))
    };
    let desc = problem.text.replace('\n', " ").trim().to_string();
    let prompt = format!("{prelude}{sig}\n    \"\"\"{desc}\"\"\"\n");
    let suffix_imports = if imports.is_empty() {
        String::new()
    } else {
        format!("{}\n", imports.join("\n"))
    };
    let suffix = format!("\n{}{}\n", suffix_imports, problem.test_list.join("\n"));
    Some(MbppChallenge {
        task_id: problem.task_id,
        prompt,
        suffix,
        entry_point: entry,
    })
}

impl MbppDomain {
    /// Load problems from JSONL (one problem per line), filter by
    /// task_id range, cap to `max_tasks`, drop unparseable rows.
    pub fn from_jsonl_range(
        jsonl_path: impl AsRef<Path>,
        scratch_dir: impl Into<PathBuf>,
        start_id: usize,
        end_id: usize,
        max_tasks: usize,
    ) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(jsonl_path.as_ref())?;
        let mut challenges = Vec::with_capacity(max_tasks);
        let mut prompt_to_idx = HashMap::with_capacity(max_tasks);
        for (line_idx, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let p: MbppProblem = serde_json::from_str(line).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("line {line_idx}: {e}"),
                )
            })?;
            if p.task_id < start_id || p.task_id >= end_id {
                continue;
            }
            if let Some(ch) = build_challenge(&p) {
                prompt_to_idx.insert(ch.prompt.clone(), challenges.len());
                challenges.push(ch);
                if challenges.len() >= max_tasks {
                    break;
                }
            }
        }
        let scratch_dir = scratch_dir.into();
        std::fs::create_dir_all(&scratch_dir)?;
        Ok(Self {
            challenges,
            prompt_to_idx,
            scratch_dir,
            timeout: Duration::from_secs(8),
            next_call_id: AtomicU64::new(0),
        })
    }

    /// Phase 17 S3 default: task_id 11-110 (100 tasks).
    pub fn from_jsonl(
        jsonl_path: impl AsRef<Path>,
        scratch_dir: impl Into<PathBuf>,
    ) -> std::io::Result<Self> {
        Self::from_jsonl_range(jsonl_path, scratch_dir, 11, 111, 100)
    }

    pub fn n_problems(&self) -> usize {
        self.challenges.len()
    }

    fn build_program(&self, ch: &MbppChallenge, completion: &str) -> String {
        let mut s =
            String::with_capacity(ch.prompt.len() + completion.len() + ch.suffix.len() + 16);
        s.push_str(&ch.prompt);
        s.push_str(completion);
        s.push_str(&ch.suffix);
        s
    }
}

impl Domain for MbppDomain {
    fn sample_prompt(&self, rng: &mut StdRng) -> String {
        let i = rng.gen_range(0..self.challenges.len());
        self.challenges[i].prompt.clone()
    }

    fn verify(&self, prompt: &str, completion: &str) -> Verdict {
        let idx = match self.prompt_to_idx.get(prompt) {
            Some(i) => *i,
            None => {
                return Verdict::Inconclusive {
                    reason: format!(
                        "unknown MBPP prompt (truncated to first 80 chars): {:?}",
                        &prompt[..prompt.len().min(80)]
                    ),
                };
            }
        };
        let ch = &self.challenges[idx];
        let program = self.build_program(ch, completion);

        // Unique scratch filename per call → concurrent verifies safe
        // without a Mutex. Cleaned up at the end.
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
        ""
    }

    fn n_prompts(&self) -> Option<usize> {
        Some(self.challenges.len())
    }

    fn nth_prompt(&self, i: usize) -> Option<String> {
        self.challenges.get(i).map(|c| c.prompt.clone())
    }

    fn truncate_completion(&self, completion: &str) -> String {
        crate::domain::truncate_python_completion(completion)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn load_default(scratch_tag: &str) -> Option<MbppDomain> {
        let scratch = std::env::temp_dir().join(format!("workllm-mbpp-test-{scratch_tag}"));
        let p = PathBuf::from("../data/mbpp/mbpp.jsonl");
        if !p.exists() {
            let p2 = PathBuf::from("data/mbpp/mbpp.jsonl");
            if !p2.exists() {
                return None;
            }
            return MbppDomain::from_jsonl(&p2, &scratch).ok();
        }
        MbppDomain::from_jsonl(&p, &scratch).ok()
    }

    #[test]
    fn parse_signature_basic() {
        let code = "R = 3\ndef foo(a, b, c):\n    return a + b + c\n";
        let (name, sig) = parse_signature(code).expect("should parse");
        assert_eq!(name, "foo");
        assert_eq!(sig, "def foo(a, b, c):");
    }

    #[test]
    fn parse_signature_skips_indented_nested_def() {
        let code = "def outer(x):\n    def inner(y):\n        return y\n    return inner\n";
        let (name, _) = parse_signature(code).expect("should parse outer");
        assert_eq!(name, "outer");
    }

    #[test]
    fn detect_imports_handles_typing_aliases() {
        let code = "def f(xs: List[int]) -> int:\n    return sum(xs)\n";
        let imps = detect_imports(code);
        assert!(imps.contains(&"from typing import *"));
    }

    #[test]
    fn detect_imports_collections_via_counter_call() {
        let code = "def f(xs):\n    return Counter(xs)\n";
        let imps = detect_imports(code);
        assert!(imps.contains(&"import collections"));
    }

    #[test]
    fn loader_returns_100_problems_by_default() {
        let Some(d) = load_default(function_name!()) else {
            eprintln!("skipping: data/mbpp/mbpp.jsonl not on disk");
            return;
        };
        // 100 max, but some rows may be skipped if unparseable. Phase 17
        // S3 typically lands at exactly 100 since 11-110 all parse.
        assert!(d.n_problems() >= 90, "got {}", d.n_problems());
        assert!(d.n_problems() <= 100, "got {}", d.n_problems());
    }

    #[test]
    fn verify_canonical_completion_passes() {
        let Some(d) = load_default(function_name!()) else {
            eprintln!("skipping: data/mbpp/mbpp.jsonl not on disk");
            return;
        };
        // Use the first challenge's canonical code (minus the prelude
        // imports we re-emit ourselves) as the completion.
        let ch = &d.challenges[0];
        // Re-load the raw problem to recover the full canonical solution.
        let p = PathBuf::from("../data/mbpp/mbpp.jsonl");
        let path = if p.exists() {
            p
        } else {
            PathBuf::from("data/mbpp/mbpp.jsonl")
        };
        let text = std::fs::read_to_string(&path).expect("read");
        let raw: MbppProblem = text
            .lines()
            .filter_map(|l| serde_json::from_str::<MbppProblem>(l).ok())
            .find(|p| p.task_id == ch.task_id)
            .expect("find task by id");
        // The canonical code is a complete program. Use it as a "completion"
        // that replaces both the body of the signature AND any pre-def
        // module state — we strip the signature line itself because the
        // prompt already emits it. Actually the simplest correct
        // completion: just splice the entire canonical code; the prompt's
        // signature is then duplicated but Python allows that (the second
        // `def` rebinds the name). The asserts run against the rebound
        // function — same canonical impl.
        let completion = format!("    pass\n\n{}\n", raw.code);
        let v = d.verify(&ch.prompt, &completion);
        assert!(
            v.is_correct(),
            "canonical completion failed verify for {}: {v:?}",
            ch.entry_point
        );
    }

    #[test]
    fn verify_unknown_prompt_is_inconclusive() {
        let Some(d) = load_default(function_name!()) else {
            eprintln!("skipping: data/mbpp/mbpp.jsonl not on disk");
            return;
        };
        let v = d.verify("def fake_problem():", "    pass");
        assert!(
            matches!(v, Verdict::Inconclusive { .. }),
            "expected Inconclusive, got {v:?}"
        );
    }
}
