//! Phase 22 §6.5 — LiveCodeBench as a **generation-only** `Domain`.
//!
//! Like `bigcodebench.rs`, this domain does NOT score in Rust: LCB solutions
//! run competitive-programming test cases in a sandbox, scored by the official
//! `lcb_runner` eval core (`codegen_metrics`) on our dumped
//! `[{question_id, code_list}]` (`bench_export::write_lcb`). `verify` is an
//! honest stub. The point of LCB is **contamination**: the official harness
//! filters problems by `contest_date`, so the same generations can be scored
//! pre- vs post-cutoff.
//!
//! Prompt format mirrors `lcb_runner`'s code-generation template (the
//! correctness-critical detail), in a **completion** shape for a base code
//! model (no chat markup): problem statement + LCB's FORMATTING instruction +
//! an open ```` ```python ```` fence the model completes. `truncate_completion`
//! extracts the code up to the closing fence — the runnable solution the
//! harness scores.
//!
//! Data: `data/livecodebench/lcb_<version>.jsonl`, one problem per line with
//! `{question_id, question_content, starter_code, contest_date, platform,
//! difficulty}` (see `scripts/phase22_bench/lcb_export_problems.py`).

use std::path::Path;

use rand::rngs::StdRng;
use rand::Rng;
use serde::Deserialize;

use crate::domain::Domain;
use crate::types::Verdict;

// Verbatim from lcb_runner/prompts/code_generation.py::PromptConstants.
const FMT_WITH_STARTER: &str = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters.";
const FMT_WITHOUT_STARTER: &str = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.";
const PREAMBLE: &str = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.";

#[derive(Debug, Clone, Deserialize)]
pub struct LcbProblem {
    pub question_id: String,
    #[serde(default)]
    pub question_content: String,
    #[serde(default)]
    pub starter_code: String,
    #[serde(default)]
    pub contest_date: String,
    #[serde(default)]
    pub platform: String,
    #[serde(default)]
    pub difficulty: String,
}

pub struct LiveCodeBenchDomain {
    pub problems: Vec<LcbProblem>,
}

impl LiveCodeBenchDomain {
    /// Build the completion-style prompt for one problem, mirroring
    /// `lcb_runner`'s template but ending in an open python fence for a base
    /// code model to complete.
    fn build_prompt(p: &LcbProblem) -> String {
        let mut s = String::with_capacity(p.question_content.len() + 512);
        s.push_str(PREAMBLE);
        s.push_str("\n\nQuestion:\n");
        s.push_str(&p.question_content);
        s.push_str("\n\n");
        if p.starter_code.trim().is_empty() {
            s.push_str(FMT_WITHOUT_STARTER);
            s.push_str("\n\n```python\n");
        } else {
            s.push_str(FMT_WITH_STARTER);
            s.push_str("\n```python\n");
            s.push_str(&p.starter_code);
            s.push_str("\n```\n\nWrite the complete solution:\n```python\n");
        }
        s
    }

    pub fn from_lines(text: &str) -> std::io::Result<Self> {
        let mut problems = Vec::new();
        for (line_idx, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let p: LcbProblem = serde_json::from_str(line).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("line {line_idx}: {e}"),
                )
            })?;
            if p.question_content.trim().is_empty() {
                continue;
            }
            problems.push(p);
        }
        Ok(Self { problems })
    }

    pub fn from_jsonl(jsonl_path: impl AsRef<Path>) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(jsonl_path.as_ref())?;
        Self::from_lines(&text)
    }

    pub fn len(&self) -> usize {
        self.problems.len()
    }

    pub fn is_empty(&self) -> bool {
        self.problems.is_empty()
    }
}

impl Domain for LiveCodeBenchDomain {
    fn sample_prompt(&self, rng: &mut StdRng) -> String {
        let i = rng.gen_range(0..self.problems.len());
        Self::build_prompt(&self.problems[i])
    }

    /// Honest stub — LCB is scored by the external `lcb_runner` eval core, not
    /// in Rust. Always `Incorrect` so an accidental in-Rust eval reads 0%.
    fn verify(&self, _prompt: &str, _completion: &str) -> Verdict {
        Verdict::Incorrect {
            reason: "LiveCodeBench is scored by the external lcb_runner eval core \
                     (codegen_metrics), not in Rust — use --dump-completions + \
                     scripts/phase22_bench/lcb_score.py"
                .to_string(),
        }
    }

    fn charset(&self) -> &str {
        ""
    }

    fn n_prompts(&self) -> Option<usize> {
        Some(self.problems.len())
    }

    fn nth_prompt(&self, i: usize) -> Option<String> {
        self.problems.get(i).map(Self::build_prompt)
    }

    fn task_id(&self, i: usize) -> Option<String> {
        self.problems.get(i).map(|p| p.question_id.clone())
    }

    /// Extract the runnable solution: the code up to the closing ``` fence the
    /// model emits after completing the open python fence in the prompt.
    fn truncate_completion(&self, completion: &str) -> String {
        match completion.find("```") {
            Some(idx) => completion[..idx].trim_end().to_string(),
            None => completion.trim_end().to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = concat!(
        r#"{"question_id":"lcb/1","question_content":"Add two numbers.","starter_code":"class Solution:\n    def add(self, a, b):","contest_date":"2024-11-01","platform":"leetcode","difficulty":"easy"}"#,
        "\n",
        r#"{"question_id":"lcb/2","question_content":"Read N then print N.","starter_code":"","contest_date":"2023-06-15","platform":"codeforces","difficulty":"medium"}"#,
        "\n",
    );

    #[test]
    fn loads_and_exposes_question_ids() {
        let d = LiveCodeBenchDomain::from_lines(SAMPLE).unwrap();
        assert_eq!(d.len(), 2);
        assert_eq!(d.task_id(0).as_deref(), Some("lcb/1"));
        assert_eq!(d.task_id(1).as_deref(), Some("lcb/2"));
        assert!(d.task_id(2).is_none());
        assert_eq!(d.n_prompts(), Some(2));
    }

    #[test]
    fn prompt_embeds_content_and_ends_open_fence() {
        let d = LiveCodeBenchDomain::from_lines(SAMPLE).unwrap();
        // starter-code problem: reference block + open fence to complete.
        let p0 = d.nth_prompt(0).unwrap();
        assert!(p0.contains("Add two numbers."));
        assert!(p0.contains(FMT_WITH_STARTER));
        assert!(p0.trim_end().ends_with("```python"));
        // no-starter problem: stdin format instruction.
        let p1 = d.nth_prompt(1).unwrap();
        assert!(p1.contains(FMT_WITHOUT_STARTER));
        assert!(p1.trim_end().ends_with("```python"));
    }

    #[test]
    fn truncate_extracts_code_before_closing_fence() {
        let d = LiveCodeBenchDomain::from_lines(SAMPLE).unwrap();
        let out = d.truncate_completion(
            "class Solution:\n    def add(self,a,b):\n        return a+b\n```\nrest ignored",
        );
        assert!(out.contains("return a+b"));
        assert!(!out.contains("rest ignored"));
        assert!(!out.contains("```"));
    }

    #[test]
    fn verify_is_external_scoring_stub() {
        let d = LiveCodeBenchDomain::from_lines(SAMPLE).unwrap();
        assert!(!d.verify("p", "anything").is_correct());
    }
}
