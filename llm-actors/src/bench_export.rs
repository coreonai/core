//! Standard-format export of generated completions for official grading
//! harnesses.
//!
//! Phase 22 §6.5 benchmarking direction: **generate in Rust, score with the
//! official harness.** Porting a scorer into Rust re-creates the exact
//! silent-failure surface the `truncate_completion` bug came from
//! (`docs/phase22-c4-c5-rl-vs-sft.md`); delegating to the upstream harness
//! keeps a single measurement ruler AND makes our numbers directly comparable
//! to the public leaderboard. This module owns the on-disk hand-off format.
//!
//! LiveCodeBench custom evaluation
//! (`python -m lcb_runner.runner.custom_evaluator --custom_output_file X`)
//! ingests a top-level JSON array of `{"question_id", "code_list"}`, where
//! `code_list` holds the k sampled solutions for that problem. Cutoff /
//! contamination filtering is done at the harness with `--start_date` /
//! `--end_date` / `--release_version`, so this format carries only the ids and
//! generations; the harness supplies the dates.
//!
//! BigCodeBench differs: it ingests **JSONL** (one `{task_id, solution}` object
//! per line, k samples of a problem as k lines) rather than LCB's array, and
//! its scoring runs in the official Docker sandbox
//! (`bigcodebench.syncheck --samples` → `--execution local`). See
//! [`write_bigcodebench_jsonl`] and `docs/phase22-bigcodebench-notes.md`.

use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// One LiveCodeBench custom-generation entry:
/// `{"question_id": "...", "code_list": ["gen1", "gen2", ...]}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LcbEntry {
    pub question_id: String,
    pub code_list: Vec<String>,
}

/// Group flat `(question_id, completion)` samples into per-problem
/// `LcbEntry { question_id, code_list }`, preserving first-seen problem order
/// and per-problem sample order. Samples whose `question_id` is `None` (a
/// domain without a stable id) are dropped — a benchmark export needs ids, and
/// silently emitting index-keyed junk would defeat the harness comparison.
pub fn group_lcb_entries<I>(samples: I) -> Vec<LcbEntry>
where
    I: IntoIterator<Item = (Option<String>, String)>,
{
    let mut order: Vec<String> = Vec::new();
    let mut by_id: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for (id, code) in samples {
        let Some(id) = id else { continue };
        if !by_id.contains_key(&id) {
            order.push(id.clone());
        }
        by_id.entry(id).or_default().push(code);
    }
    order
        .into_iter()
        .map(|question_id| {
            let code_list = by_id.remove(&question_id).unwrap_or_default();
            LcbEntry {
                question_id,
                code_list,
            }
        })
        .collect()
}

/// Write LCB custom-generation JSON (a top-level array) to `path`.
pub fn write_lcb(entries: &[LcbEntry], path: &Path) -> Result<()> {
    let f = std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    serde_json::to_writer_pretty(BufWriter::new(f), entries).with_context(|| {
        format!(
            "serialize {} LCB entries to {}",
            entries.len(),
            path.display()
        )
    })?;
    Ok(())
}

/// One BigCodeBench sample. Unlike LCB (which groups the k samples of a problem
/// into `code_list`), BigCodeBench takes **one JSONL line per sample**, so a
/// problem with k generations appears as k `BcbEntry` lines sharing a
/// `task_id`. `raw_solution` (the pre-truncation output) is optional and
/// omitted from the JSON when `None`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BcbEntry {
    /// From `get_bigcodebench()` order, e.g. `"BigCodeBench/12"`.
    pub task_id: String,
    /// The sanitizable solution (our truncated completion).
    pub solution: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub raw_solution: Option<String>,
}

/// Turn flat `(task_id, solution)` samples into `BcbEntry` lines, one per
/// sample (no grouping). Samples whose `task_id` is `None` are dropped — a
/// benchmark export needs ids. `raw_solution` is left `None`; set it explicitly
/// if the un-truncated output is wanted for `bigcodebench.syncheck`.
pub fn bigcodebench_entries<I>(samples: I) -> Vec<BcbEntry>
where
    I: IntoIterator<Item = (Option<String>, String)>,
{
    samples
        .into_iter()
        .filter_map(|(id, solution)| {
            id.map(|task_id| BcbEntry {
                task_id,
                solution,
                raw_solution: None,
            })
        })
        .collect()
}

/// Write BigCodeBench samples as **JSONL** — one compact JSON object per line,
/// the shape `bigcodebench.syncheck --samples` / the `--execution local`
/// harness ingests. (Not a pretty array; not LCB's `code_list` grouping.)
pub fn write_bigcodebench_jsonl(entries: &[BcbEntry], path: &Path) -> Result<()> {
    let f = std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut w = BufWriter::new(f);
    for e in entries {
        let line = serde_json::to_string(e)
            .with_context(|| format!("serialize BigCodeBench entry {}", e.task_id))?;
        writeln!(w, "{line}").with_context(|| format!("write {}", path.display()))?;
    }
    w.flush()
        .with_context(|| format!("flush {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_by_question_id_preserving_order() {
        let samples = vec![
            (Some("HumanEval/0".to_string()), "a0".to_string()),
            (Some("HumanEval/0".to_string()), "a1".to_string()),
            (Some("HumanEval/1".to_string()), "b0".to_string()),
            (None, "dropped".to_string()),
        ];
        let entries = group_lcb_entries(samples);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].question_id, "HumanEval/0");
        assert_eq!(entries[0].code_list, vec!["a0", "a1"]);
        assert_eq!(entries[1].question_id, "HumanEval/1");
        assert_eq!(entries[1].code_list, vec!["b0"]);
    }

    #[test]
    fn serializes_exact_lcb_schema_keys() {
        let entries = vec![LcbEntry {
            question_id: "q1".to_string(),
            code_list: vec!["def f(): pass".to_string()],
        }];
        let json = serde_json::to_value(&entries).unwrap();
        let obj = &json[0];
        // Exact field names the official harness expects — not `id`/`codes`.
        assert!(
            obj.get("question_id").is_some(),
            "missing question_id: {obj}"
        );
        assert!(obj.get("code_list").is_some(), "missing code_list: {obj}");
        assert_eq!(obj.as_object().unwrap().len(), 2, "unexpected extra fields");
    }

    #[test]
    fn roundtrips_through_file() {
        let entries = vec![
            LcbEntry {
                question_id: "a".into(),
                code_list: vec!["x".into(), "y".into()],
            },
            LcbEntry {
                question_id: "b".into(),
                code_list: vec![],
            },
        ];
        let dir = std::env::temp_dir().join("workllm-bench-export-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("gen.json");
        write_lcb(&entries, &path).unwrap();
        let back: Vec<LcbEntry> =
            serde_json::from_reader(std::fs::File::open(&path).unwrap()).unwrap();
        assert_eq!(back, entries);
    }

    #[test]
    fn bcb_builder_drops_none_and_keeps_one_line_per_sample() {
        // Two samples of one task + one of another + a None -> 3 entries, NOT
        // grouped (BigCodeBench is one line per sample).
        let samples = vec![
            (Some("BigCodeBench/0".to_string()), "s0a".to_string()),
            (Some("BigCodeBench/0".to_string()), "s0b".to_string()),
            (Some("BigCodeBench/1".to_string()), "s1".to_string()),
            (None, "dropped".to_string()),
        ];
        let entries = bigcodebench_entries(samples);
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].task_id, "BigCodeBench/0");
        assert_eq!(entries[1].task_id, "BigCodeBench/0");
        assert_eq!(entries[2].task_id, "BigCodeBench/1");
        assert!(entries[0].raw_solution.is_none());
    }

    #[test]
    fn bcb_serializes_exact_schema_and_omits_none_raw() {
        let e = BcbEntry {
            task_id: "BigCodeBench/3".into(),
            solution: "def task_func():\n    return 1".into(),
            raw_solution: None,
        };
        let obj = serde_json::to_value(&e).unwrap();
        let map = obj.as_object().unwrap();
        assert!(map.contains_key("task_id"), "missing task_id: {obj}");
        assert!(map.contains_key("solution"), "missing solution: {obj}");
        // raw_solution omitted when None (skip_serializing_if).
        assert!(
            !map.contains_key("raw_solution"),
            "raw_solution should be omitted: {obj}"
        );
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn bcb_writes_jsonl_one_object_per_line_and_roundtrips() {
        let entries = vec![
            BcbEntry {
                task_id: "BigCodeBench/0".into(),
                solution: "a".into(),
                raw_solution: Some("a-raw".into()),
            },
            BcbEntry {
                task_id: "BigCodeBench/1".into(),
                solution: "b".into(),
                raw_solution: None,
            },
        ];
        let dir = std::env::temp_dir().join("workllm-bench-export-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bcb.jsonl");
        write_bigcodebench_jsonl(&entries, &path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        // Exactly one JSON object per line (JSONL, not a pretty array).
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2, "expected 2 JSONL lines: {text:?}");
        let back: Vec<BcbEntry> = lines
            .iter()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(back, entries);
    }
}
