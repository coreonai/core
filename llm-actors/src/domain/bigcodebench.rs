//! Phase 22 §6.5 — BigCodeBench as a **generation-only** `Domain`.
//!
//! Unlike `HumanEvalDomain`/`MbppDomain`, this domain does NOT score in Rust:
//! BigCodeBench solutions execute arbitrary libraries (numpy, pandas, requests)
//! and are scored in the official Docker sandbox
//! (`bigcodebench.syncheck --samples` → `bigcodebench.evaluate --execution
//! local`). Porting that scorer into Rust would re-create the exact
//! silent-failure surface the `truncate_completion` bug came from
//! (`docs/phase22-c4-c5-rl-vs-sft.md`), so [`verify`](Domain::verify) is an
//! honest stub that always returns `Incorrect` with a pointer to the external
//! flow. The intended path is: generate with `--dump-completions`
//! (`bench_export::write_bigcodebench_jsonl`) → score with the Docker harness.
//!
//! Splits: `Complete` (docstring-based, completion-style — suits a base model)
//! and `Instruct` (natural-language). Subsets (Full 1140 / Hard 148) are chosen
//! by which JSONL you load. See `docs/phase22-bigcodebench-notes.md`.
//!
//! Data: not bundled. Export the HF dataset `bigcode/bigcodebench` to a JSONL
//! with at least `task_id` and `complete_prompt` (and `instruct_prompt` for the
//! Instruct split), one task per line, at e.g.
//! `data/bigcodebench/BigCodeBench.jsonl`.

use std::path::Path;

use rand::rngs::StdRng;
use rand::Rng;
use serde::Deserialize;

use crate::domain::Domain;
use crate::types::Verdict;

/// Which prompt a task exposes for generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BcbSplit {
    /// Docstring-based, completion-style — the base-model-appropriate split.
    Complete,
    /// Natural-language instruction.
    Instruct,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BigCodeBenchProblem {
    /// Already in `"BigCodeBench/N"` form in the dataset.
    pub task_id: String,
    /// Complete-split prompt (function signature + docstring).
    #[serde(default)]
    pub complete_prompt: String,
    /// Instruct-split prompt (natural language). Absent in some exports.
    #[serde(default)]
    pub instruct_prompt: String,
}

impl BigCodeBenchProblem {
    fn prompt_for(&self, split: BcbSplit) -> &str {
        match split {
            BcbSplit::Complete => &self.complete_prompt,
            BcbSplit::Instruct => &self.instruct_prompt,
        }
    }
}

pub struct BigCodeBenchDomain {
    pub problems: Vec<BigCodeBenchProblem>,
    split: BcbSplit,
}

impl BigCodeBenchDomain {
    /// Parse tasks from JSONL text (one task per line). Blank lines skipped;
    /// tasks whose selected-split prompt is empty are dropped (so a Complete
    /// load of an Instruct-only export doesn't yield empty prompts).
    pub fn from_lines(text: &str, split: BcbSplit) -> std::io::Result<Self> {
        let mut problems = Vec::new();
        for (line_idx, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let p: BigCodeBenchProblem = serde_json::from_str(line).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("line {line_idx}: {e}"),
                )
            })?;
            if p.prompt_for(split).is_empty() {
                continue;
            }
            problems.push(p);
        }
        Ok(Self { problems, split })
    }

    /// Load tasks from a JSONL file. Default location:
    /// `data/bigcodebench/BigCodeBench.jsonl` (see module docs for how to
    /// produce it from the HF dataset).
    pub fn from_jsonl(jsonl_path: impl AsRef<Path>, split: BcbSplit) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(jsonl_path.as_ref())?;
        Self::from_lines(&text, split)
    }

    pub fn split(&self) -> BcbSplit {
        self.split
    }

    pub fn len(&self) -> usize {
        self.problems.len()
    }

    pub fn is_empty(&self) -> bool {
        self.problems.is_empty()
    }
}

impl Domain for BigCodeBenchDomain {
    fn sample_prompt(&self, rng: &mut StdRng) -> String {
        // Index-based like HumanEval/Mbpp so `FilteredDomain` identity holds.
        let i = rng.gen_range(0..self.problems.len());
        self.problems[i].prompt_for(self.split).to_string()
    }

    /// Honest stub — BigCodeBench is scored by the external Docker harness, not
    /// in Rust. Always `Incorrect`, so an accidental in-Rust eval reads 0% (a
    /// clear "wrong scorer" signal) rather than a plausible-but-meaningless
    /// number. Use `--dump-completions` + `bigcodebench.evaluate`.
    fn verify(&self, _prompt: &str, _completion: &str) -> Verdict {
        Verdict::Incorrect {
            reason: "BigCodeBench is scored by the external Docker harness \
                     (bigcodebench.evaluate --execution local), not in Rust — \
                     use --dump-completions + bench_export::write_bigcodebench_jsonl"
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
        self.problems
            .get(i)
            .map(|p| p.prompt_for(self.split).to_string())
    }

    fn task_id(&self, i: usize) -> Option<String> {
        self.problems.get(i).map(|p| p.task_id.clone())
    }

    fn truncate_completion(&self, completion: &str) -> String {
        crate::domain::truncate_python_completion(completion)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = concat!(
        r#"{"task_id":"BigCodeBench/0","complete_prompt":"def task_func(a):\n    \"\"\"doc0\"\"\"\n","instruct_prompt":"do thing 0"}"#,
        "\n",
        r#"{"task_id":"BigCodeBench/1","complete_prompt":"def task_func(b):\n    \"\"\"doc1\"\"\"\n","instruct_prompt":"do thing 1"}"#,
        "\n",
    );

    #[test]
    fn loads_complete_split_and_exposes_task_ids() {
        let d = BigCodeBenchDomain::from_lines(SAMPLE, BcbSplit::Complete).unwrap();
        assert_eq!(d.len(), 2);
        assert_eq!(d.task_id(0).as_deref(), Some("BigCodeBench/0"));
        assert_eq!(d.task_id(1).as_deref(), Some("BigCodeBench/1"));
        assert!(d.nth_prompt(0).unwrap().contains("doc0"));
        assert_eq!(d.n_prompts(), Some(2));
        assert!(d.task_id(2).is_none());
    }

    #[test]
    fn instruct_split_selects_the_other_prompt() {
        let d = BigCodeBenchDomain::from_lines(SAMPLE, BcbSplit::Instruct).unwrap();
        assert_eq!(d.nth_prompt(0).as_deref(), Some("do thing 0"));
    }

    #[test]
    fn verify_is_an_external_scoring_stub() {
        let d = BigCodeBenchDomain::from_lines(SAMPLE, BcbSplit::Complete).unwrap();
        let v = d.verify("p", "anything");
        assert!(!v.is_correct(), "in-Rust verify must not claim correctness");
    }

    #[test]
    fn truncate_completion_is_python_aware() {
        let d = BigCodeBenchDomain::from_lines(SAMPLE, BcbSplit::Complete).unwrap();
        // A trailing top-level statement after the function body is cut, same
        // as the other code domains.
        let out = d.truncate_completion("    return 1\n\ndef other():\n    pass\n");
        assert!(
            !out.contains("def other"),
            "trailing def not truncated: {out:?}"
        );
    }
}
