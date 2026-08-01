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

use std::io::BufWriter;
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
}
