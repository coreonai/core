//! Phase 24 — multi-family Pekko/MSA harvest domain.
//!
//! Multiplexes:
//! - **F0** retention via [`RustCodeDomain`] expression slots
//! - **F1–F5** via isolated scratch crates under `verify_root/{f1_tool,…}`:
//!   write the model completion into `src/student.rs`, then
//!   `cargo test --features student` (exit 0 ⇒ Correct).
//!
//! Each F1–F5 challenge asks for a **full** `student.rs` body (same shape as
//! that crate's `reference.rs`). Gold for format-SFT seeds is the reference
//! file contents.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::Rng;

use crate::domain::rust_code::RustCodeDomain;
use crate::domain::Domain;
use crate::types::Verdict;

/// Harvest family selector (CLI `--families f0,f1,…`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Family {
    F0,
    F1,
    F2,
    F3,
    F4,
    F5,
}

impl Family {
    pub fn as_str(self) -> &'static str {
        match self {
            Family::F0 => "f0",
            Family::F1 => "f1",
            Family::F2 => "f2",
            Family::F3 => "f3",
            Family::F4 => "f4",
            Family::F5 => "f5",
        }
    }

    /// Crate directory name under `scratch-pekko-harvest/` / `_verify/`.
    pub fn crate_dir(self) -> Option<&'static str> {
        match self {
            Family::F0 => None,
            Family::F1 => Some("f1_tool"),
            Family::F2 => Some("f2_domain"),
            Family::F3 => Some("f3_message"),
            Family::F4 => Some("f4_repair"),
            Family::F5 => Some("f5_supervisor"),
        }
    }

    pub fn parse_one(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f0" | "f0_expr" | "0" => Some(Family::F0),
            "f1" | "f1_tool" | "1" => Some(Family::F1),
            "f2" | "f2_domain" | "2" => Some(Family::F2),
            "f3" | "f3_message" | "3" => Some(Family::F3),
            "f4" | "f4_repair" | "4" => Some(Family::F4),
            "f5" | "f5_supervisor" | "5" => Some(Family::F5),
            _ => None,
        }
    }

    /// Parse comma/space-separated list. Empty → all families.
    pub fn parse_list(s: &str) -> Result<Vec<Self>, String> {
        let raw: Vec<_> = s
            .split(|c: char| c == ',' || c.is_whitespace())
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .collect();
        if raw.is_empty() {
            return Ok(vec![
                Family::F0,
                Family::F1,
                Family::F2,
                Family::F3,
                Family::F4,
                Family::F5,
            ]);
        }
        let mut out = Vec::new();
        for t in raw {
            let f = Self::parse_one(t).ok_or_else(|| format!("unknown family: {t:?}"))?;
            if !out.contains(&f) {
                out.push(f);
            }
        }
        Ok(out)
    }
}

/// One F1–F5 slot challenge: NL + API surface → full `student.rs`.
#[derive(Debug, Clone)]
pub struct SlotChallenge {
    pub family: Family,
    pub task_id: &'static str,
    pub prompt: &'static str,
}

/// Marker embedded in every slot prompt so repair wraps still match.
pub const TASK_MARKER_PREFIX: &str = "// pekko-harvest-task: ";

fn marker_line(task_id: &str) -> String {
    format!("{TASK_MARKER_PREFIX}{task_id}")
}

/// Default F1–F5 challenges (paraphrases → same gold = reference.rs).
pub fn default_slot_challenges() -> Vec<SlotChallenge> {
    vec![
        // ---- F1 ----
        SlotChallenge {
            family: Family::F1,
            task_id: "f1_tool/full_v1",
            prompt: concat!(
                "Implement tool stubs for the Tool registry scratch.\n",
                "Write the FULL contents of src/student.rs so EchoTool, PingTool, and UpperTool\n",
                "pass `cargo test --features student`.\n",
                "API already in lib.rs (do not redefine Tool / ToolError / ToolRegistry):\n",
                "  trait Tool: Send + Sync { fn name(&self) -> &str; fn execute(&self, args: &str) -> Result<String, ToolError>; }\n",
                "Echo returns args unchanged; ping always returns \"pong\"; upper returns ASCII uppercase.\n",
                "Output ONLY the student.rs module body (use super::{Tool, ToolError}; + structs/impls).\n",
                "// pekko-harvest-task: f1_tool/full_v1\n"
            ),
        },
        SlotChallenge {
            family: Family::F1,
            task_id: "f1_tool/full_v2",
            prompt: concat!(
                "Add Tools named echo, ping, and upper; register/dispatch must work under student feature.\n",
                "Replace src/student.rs entirely. echo→args, ping→pong, upper→to_ascii_uppercase.\n",
                "use super::{Tool, ToolError};\n",
                "// pekko-harvest-task: f1_tool/full_v2\n"
            ),
        },
        SlotChallenge {
            family: Family::F1,
            task_id: "f1_tool/full_ko",
            prompt: concat!(
                "echo/ping/upper 툴을 student.rs에 전부 구현해. cargo test --features student 가 통과해야 한다.\n",
                "use super::{Tool, ToolError}; 로 시작하고 struct+impl만 작성.\n",
                "// pekko-harvest-task: f1_tool/full_ko\n"
            ),
        },
        // ---- F2 ----
        SlotChallenge {
            family: Family::F2,
            task_id: "f2_domain/full_v1",
            prompt: concat!(
                "Implement OkOnlyDomain, DigitCharsetDomain, and NonEmptyDomain in src/student.rs.\n",
                "OkOnlyDomain: Correct iff completion == \"ok\". DigitCharsetDomain: charset is digits 0-9.\n",
                "NonEmptyDomain: reject empty completions. Traits Domain/Verdict live in lib.rs.\n",
                "Output ONLY the student.rs body starting with `use super::{Domain, Verdict};`.\n",
                "// pekko-harvest-task: f2_domain/full_v1\n"
            ),
        },
        SlotChallenge {
            family: Family::F2,
            task_id: "f2_domain/full_v2",
            prompt: concat!(
                "Toy Domain impls: only \"ok\" passes; charset includes 0-9; empty string is Incorrect.\n",
                "Fill src/student.rs completely for cargo test --features student.\n",
                "// pekko-harvest-task: f2_domain/full_v2\n"
            ),
        },
        SlotChallenge {
            family: Family::F2,
            task_id: "f2_domain/full_ko",
            prompt: concat!(
                "완성이 ok일 때만 통과하는 Domain과 digit charset, non-empty Domain을 student.rs에 구현해.\n",
                "// pekko-harvest-task: f2_domain/full_ko\n"
            ),
        },
        // ---- F3 ----
        SlotChallenge {
            family: Family::F3,
            task_id: "f3_message/full_v1",
            prompt: concat!(
                "Implement CounterActor::handle in src/student.rs.\n",
                "Ping→Pong; Inc bumps n and returns Count(n); Get returns Count(n).\n",
                "Message/Response/Handler are in lib.rs. Start with use super::{Handler, Message, Response};\n",
                "// pekko-harvest-task: f3_message/full_v1\n"
            ),
        },
        SlotChallenge {
            family: Family::F3,
            task_id: "f3_message/full_v2",
            prompt: concat!(
                "When the actor receives Ping, reply Pong; Inc bumps; Get returns the count.\n",
                "Write full student.rs for the message-handler scratch.\n",
                "// pekko-harvest-task: f3_message/full_v2\n"
            ),
        },
        SlotChallenge {
            family: Family::F3,
            task_id: "f3_message/full_ko",
            prompt: concat!(
                "Ping 메시지를 받으면 Pong을 반환하고 Inc/Get을 처리하는 CounterActor 핸들러를 student.rs에 작성해.\n",
                "// pekko-harvest-task: f3_message/full_ko\n"
            ),
        },
        // ---- F4 ----
        SlotChallenge {
            family: Family::F4,
            task_id: "f4_repair/full_v1",
            prompt: concat!(
                "Repair src/student.rs so cargo test --features student passes.\n",
                "Need: count_keys with HashMap (remember `use std::collections::HashMap`),\n",
                "greet(name) -> \"hi {name}\", exhaustive color_name for Red/Blue/Green.\n",
                "Write the FULL fixed student.rs module.\n",
                "// pekko-harvest-task: f4_repair/full_v1\n"
            ),
        },
        SlotChallenge {
            family: Family::F4,
            task_id: "f4_repair/full_v2",
            prompt: concat!(
                "Fix this compile/test surface: HashMap insert+len, greet, Color::Green arm.\n",
                "Replace student.rs entirely with a compiling implementation.\n",
                "// pekko-harvest-task: f4_repair/full_v2\n"
            ),
        },
        // ---- F5 ----
        SlotChallenge {
            family: Family::F5,
            task_id: "f5_supervisor/full_v1",
            prompt: concat!(
                "Implement one_round(gen, ver, prompts) in src/student.rs:\n",
                "for each prompt: push \"generate\", generate, push \"verify\", keep completion if Correct.\n",
                "use super::{Generator, RoundResult, Verdict, Verifier};\n",
                "// pekko-harvest-task: f5_supervisor/full_v1\n"
            ),
        },
        SlotChallenge {
            family: Family::F5,
            task_id: "f5_supervisor/full_v2",
            prompt: concat!(
                "Wire Verifier after Generator for one self-improve round without training.\n",
                "Record order generate/verify and keep only Correct samples. Full student.rs please.\n",
                "// pekko-harvest-task: f5_supervisor/full_v2\n"
            ),
        },
        SlotChallenge {
            family: Family::F5,
            task_id: "f5_supervisor/full_ko",
            prompt: concat!(
                "Generator 다음에 Verifier가 오도록 한 라운드를 student.rs의 one_round에 연결해.\n",
                "// pekko-harvest-task: f5_supervisor/full_ko\n"
            ),
        },
    ]
}

/// Multiplexed harvest domain: F0 expression slots + F1–F5 student.rs slots.
pub struct PekkoHarvestDomain {
    /// Isolated crate copies live here: `{verify_root}/f1_tool`, …
    pub verify_root: PathBuf,
    /// F0 cargo-run scratch (independent `[workspace]` project).
    pub f0: Option<RustCodeDomain>,
    pub slots: Vec<SlotChallenge>,
    pub timeout: Duration,
    write_lock: Mutex<()>,
}

impl PekkoHarvestDomain {
    pub fn new(
        verify_root: impl Into<PathBuf>,
        f0_scratch: impl Into<PathBuf>,
        families: &[Family],
    ) -> Self {
        let include_f0 = families.iter().any(|f| *f == Family::F0);
        let slot_families: Vec<Family> = families
            .iter()
            .copied()
            .filter(|f| *f != Family::F0)
            .collect();
        let slots = default_slot_challenges()
            .into_iter()
            .filter(|c| slot_families.contains(&c.family))
            .collect();
        let f0 = if include_f0 {
            Some(RustCodeDomain::new(f0_scratch))
        } else {
            None
        };
        Self {
            verify_root: verify_root.into(),
            f0,
            slots,
            timeout: Duration::from_secs(120),
            write_lock: Mutex::new(()),
        }
    }

    pub fn ensure_ready(&self) -> io::Result<()> {
        if let Some(f0) = &self.f0 {
            f0.ensure_scratch_project()?;
        }
        self.ensure_verify_crates()
    }

    /// Copy each needed family crate into `verify_root` with an empty `[workspace]`.
    pub fn ensure_verify_crates(&self) -> io::Result<()> {
        fs::create_dir_all(&self.verify_root)?;
        let harvest_root = self
            .verify_root
            .parent()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "verify_root has no parent"))?;
        let needed: Vec<Family> = {
            let mut v = Vec::new();
            for c in &self.slots {
                if !v.contains(&c.family) {
                    v.push(c.family);
                }
            }
            v
        };
        for fam in needed {
            let name = fam.crate_dir().expect("slot family has crate");
            let src = harvest_root.join(name);
            let dst = self.verify_root.join(name);
            if !src.join("Cargo.toml").exists() {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("scaffold missing: {}", src.display()),
                ));
            }
            if !dst.join("Cargo.toml").exists() {
                copy_dir_recursive(&src, &dst)?;
            }
            ensure_empty_workspace_table(&dst.join("Cargo.toml"))?;
            // Always reset student.rs to scaffold todos so a prior pass doesn't leak.
            let scaffold_student = src.join("src/student.rs");
            if scaffold_student.exists() {
                fs::copy(&scaffold_student, dst.join("src/student.rs"))?;
            }
        }
        Ok(())
    }

    fn f0_len(&self) -> usize {
        self.f0.as_ref().map(|d| d.challenges.len()).unwrap_or(0)
    }

    fn slot_for_prompt(&self, prompt: &str) -> Option<&SlotChallenge> {
        if let Some(c) = self.slots.iter().find(|c| c.prompt == prompt) {
            return Some(c);
        }
        // Repair / NL wrap: find marker.
        for c in &self.slots {
            let m = marker_line(c.task_id);
            if prompt.contains(&m) {
                return Some(c);
            }
        }
        // Longest prompt that is a suffix (repair may append after original).
        self.slots
            .iter()
            .filter(|c| prompt.ends_with(c.prompt) || prompt.contains(c.prompt))
            .max_by_key(|c| c.prompt.len())
    }

    fn verify_slot(&self, challenge: &SlotChallenge, completion: &str) -> Verdict {
        let name = match challenge.family.crate_dir() {
            Some(n) => n,
            None => {
                return Verdict::Inconclusive {
                    reason: "slot challenge missing crate".into(),
                }
            }
        };
        let crate_dir = self.verify_root.join(name);
        let student_path = crate_dir.join("src/student.rs");
        let _guard = self.write_lock.lock().expect("write_lock poisoned");
        if let Err(e) = fs::write(&student_path, completion) {
            return Verdict::Inconclusive {
                reason: format!("write student.rs failed: {e}"),
            };
        }
        let start = Instant::now();
        let mut cmd = Command::new("cargo");
        cmd.arg("test")
            .arg("--features")
            .arg("student")
            .arg("--offline")
            .arg("--quiet")
            .current_dir(&crate_dir)
            .env(
                "PATH",
                format!(
                    "{}:{}",
                    "/home/paulyu/.cargo/bin",
                    std::env::var("PATH").unwrap_or_default()
                ),
            );
        // Soft timeout via wait — Command::output blocks; rely on cargo being fast for these.
        let output = match cmd.output() {
            Ok(o) => o,
            Err(e) => {
                return Verdict::Inconclusive {
                    reason: format!("cargo invoke failed: {e}"),
                }
            }
        };
        let elapsed = start.elapsed();
        if elapsed > self.timeout {
            return Verdict::Inconclusive {
                reason: format!("cargo test exceeded {:?}", self.timeout),
            };
        }
        if output.status.success() {
            Verdict::Correct
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            let combined = if stderr.trim().is_empty() {
                stdout.to_string()
            } else {
                stderr.to_string()
            };
            Verdict::Incorrect {
                reason: format!(
                    "cargo test --features student failed in {:?}: {}",
                    elapsed,
                    tail_lines(&combined, 12)
                ),
            }
        }
    }
}

impl Domain for PekkoHarvestDomain {
    fn sample_prompt(&self, rng: &mut StdRng) -> String {
        let n = self.n_prompts().unwrap_or(0);
        assert!(n > 0, "PekkoHarvestDomain has 0 prompts — check --families");
        let i = rng.gen_range(0..n);
        self.nth_prompt(i).expect("in range")
    }

    fn verify(&self, prompt: &str, completion: &str) -> Verdict {
        if let Some(slot) = self.slot_for_prompt(prompt) {
            return self.verify_slot(slot, completion);
        }
        if let Some(f0) = &self.f0 {
            return f0.verify(prompt, completion);
        }
        Verdict::Inconclusive {
            reason: format!("unknown prompt (no F0, no slot match): {prompt:?}"),
        }
    }

    fn charset(&self) -> &str {
        " \n\t!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
    }

    fn n_prompts(&self) -> Option<usize> {
        Some(self.f0_len() + self.slots.len())
    }

    fn nth_prompt(&self, i: usize) -> Option<String> {
        let f0n = self.f0_len();
        if i < f0n {
            return self.f0.as_ref()?.nth_prompt(i);
        }
        self.slots.get(i - f0n).map(|c| c.prompt.to_string())
    }

    fn task_id(&self, i: usize) -> Option<String> {
        let f0n = self.f0_len();
        if i < f0n {
            return self.f0.as_ref()?.task_id(i);
        }
        self.slots.get(i - f0n).map(|c| c.task_id.to_string())
    }

    fn truncate_completion(&self, completion: &str) -> String {
        truncate_pekko_completion(completion)
    }

    fn repair_prompt(&self, prompt: &str, completion: &str, v: &Verdict) -> Option<String> {
        let reason = match v {
            Verdict::Incorrect { reason } => reason.as_str(),
            _ => return None,
        };
        if let Some(slot) = self.slot_for_prompt(prompt) {
            // Feed cargo stderr; ask for a full rewrite of student.rs only.
            return Some(format!(
                "{}\n\n// Previous student.rs attempt:\n{}\n\n// ERR:{}\n// Rewrite the FULL src/student.rs only (no markdown fences):\n{}",
                slot.prompt,
                completion,
                reason,
                marker_line(slot.task_id),
            ));
        }
        if let Some(f0) = &self.f0 {
            return f0.repair_prompt(prompt, completion, v);
        }
        None
    }
}

/// Multi-line Rust slot truncation (F1–F5) with a fallback for F0-style expressions.
///
/// Also rejects FIM/path garbage (`<|repo_name|>`, `<|fim_…|>`, `/src/src/…` spam)
/// so verify gets an empty completion instead of a poison `student.rs`.
pub fn truncate_pekko_completion(completion: &str) -> String {
    let mut s = completion.trim_start();
    // Strip markdown fences if the model wraps the file.
    if let Some(rest) = s.strip_prefix("```rust") {
        s = rest.trim_start_matches('\n');
    } else if let Some(rest) = s.strip_prefix("```rs") {
        s = rest.trim_start_matches('\n');
    } else if let Some(rest) = s.strip_prefix("```") {
        s = rest.trim_start_matches('\n');
    }
    let mut cut = s.len();
    for stop in [
        "\n```",
        "<|fim_",
        "<|endof",
        "<|im_end|>",
        "<|repo_name|>",
        "<|file_sep|>",
        "<|fim_suffix|>",
        "<|fim_middle|>",
        "<|fim_prefix|>",
    ] {
        if let Some(i) = s.find(stop) {
            cut = cut.min(i);
        }
    }
    // Path-repetition spam (`/src/src/...`, `/main/main/...`): cut at first such line.
    if let Some(i) = first_path_spam_offset(s) {
        cut = cut.min(i);
    }
    let body = s[..cut].trim_end();
    if body.is_empty() || is_path_spam_body(body) {
        return String::new();
    }

    // Module-task garbage: no Rust item keywords in the first ~80 chars → empty.
    // (F0 short expressions are exempted below.)
    let head = &body[..body.chars().take(80).map(|c| c.len_utf8()).sum::<usize>().min(body.len())];
    let has_rust_kw = ["fn ", "impl ", "use ", "struct ", "pub ", "enum ", "const ", "type ", "#[", "mod "]
        .iter()
        .any(|k| head.contains(k));

    // Heuristic: module-shaped → keep multi-line; else F0 expression stops.
    let module_shaped = body.lines().next().is_some_and(|l| {
        let t = l.trim_start();
        t.starts_with("use ")
            || t.starts_with("pub ")
            || t.starts_with("struct ")
            || t.starts_with("impl ")
            || t.starts_with("fn ")
            || t.starts_with("#[")
            || t.starts_with("const ")
            || t.starts_with("type ")
            || t.starts_with("enum ")
            || t.starts_with("mod ")
    });
    if module_shaped {
        return body.to_string();
    }

    // Short F0-ish expression: must look like code (ops/calls/literals), not prose.
    if !has_rust_kw && !looks_like_rust_expr(body) {
        // Module-task garbage / FIM residue without Rust items — fail clean.
        return String::new();
    }

    // F0-ish: cut at blank line / next item so a runaway generation doesn't poison cargo.
    let stops = [
        "\npub ",
        "\nfn ",
        "\nuse ",
        "\nstruct ",
        "\nimpl ",
        "\n\n",
        "<|fim_prefix|>",
        "<|repo_name|>",
    ];
    let mut c = body.len();
    for st in stops {
        if let Some(i) = body.find(st) {
            c = c.min(i);
        }
    }
    body[..c].trim_end().to_string()
}

/// Offset of the first line that looks like path-repetition spam, if any.
fn first_path_spam_offset(s: &str) -> Option<usize> {
    let mut offset = 0usize;
    for line in s.split_inclusive('\n') {
        if is_path_spam_line(line.trim_end_matches('\n')) {
            return Some(offset);
        }
        offset += line.len();
    }
    None
}

fn is_path_spam_line(line: &str) -> bool {
    let t = line.trim();
    if t.is_empty() {
        return false;
    }
    // Classic crash mode: `/src/src/src/...` or `/main/main/...`
    if t.matches("/src/").count() >= 2 {
        return true;
    }
    if t.starts_with('/') && t.matches('/').count() >= 3 && !t.contains(' ') {
        let lower = t.to_ascii_lowercase();
        if lower.contains("/src") || lower.contains("/main") || lower.contains("student") {
            return true;
        }
    }
    false
}

fn is_path_spam_body(body: &str) -> bool {
    let lines: Vec<&str> = body.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.is_empty() {
        return true;
    }
    let spam = lines.iter().filter(|l| is_path_spam_line(l)).count();
    spam * 2 >= lines.len() || body.matches("/src/").count() >= 3
}


fn looks_like_rust_expr(body: &str) -> bool {
    let t = body.trim();
    if t.is_empty() || t.len() > 120 || t.lines().count() > 2 {
        return false;
    }
    if t.starts_with('/') || t.contains("/src/") || t.contains("<|") {
        return false;
    }
    if t == "true" || t == "false" || t == "()" || t == "None" {
        return true;
    }
    t.contains('+')
        || t.contains('*')
        || t.contains('(')
        || t.contains('[')
        || t.contains('.')
        || t.contains("::")
        || t.contains("=>")
        || t.contains('"')
        || t.contains('\'')
        || t.chars().any(|c| c.is_ascii_digit())
        || t.contains(" / ")
        || t.contains(" - ")
        || (t.contains('-') && !t.starts_with('-'))
}

fn ensure_empty_workspace_table(cargo_toml: &Path) -> io::Result<()> {
    let cur = fs::read_to_string(cargo_toml)?;
    if cur.contains("[workspace]") {
        return Ok(());
    }
    let mut f = fs::OpenOptions::new().append(true).open(cargo_toml)?;
    writeln!(f)?;
    writeln!(f, "[workspace]")?;
    Ok(())
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> io::Result<()> {
    fs::create_dir_all(dst)?;
    for ent in fs::read_dir(src)? {
        let ent = ent?;
        let ty = ent.file_type()?;
        let to = dst.join(ent.file_name());
        if ty.is_dir() {
            // Skip target/ if present in scaffold (shouldn't be).
            if ent.file_name() == *"target" {
                continue;
            }
            copy_dir_recursive(&ent.path(), &to)?;
        } else if ty.is_file() {
            fs::copy(ent.path(), &to)?;
        }
    }
    Ok(())
}

fn tail_lines(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    let take = lines.len().saturating_sub(n);
    lines[take..].join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_families_default_all() {
        let v = Family::parse_list("").unwrap();
        assert_eq!(v.len(), 6);
    }

    #[test]
    fn parse_families_subset() {
        let v = Family::parse_list("f1,f3,f5").unwrap();
        assert_eq!(v, vec![Family::F1, Family::F3, Family::F5]);
    }

    #[test]
    fn truncate_keeps_module() {
        let raw = "use super::{Tool, ToolError};\n\npub struct EchoTool;\n```\nextra";
        let t = truncate_pekko_completion(raw);
        assert!(t.contains("EchoTool"));
        assert!(!t.contains("```"));
    }

    #[test]
    fn truncate_expression_cuts_runaway() {
        let raw = "2 + 3\n\nfn other() {}";
        let t = truncate_pekko_completion(raw);
        assert_eq!(t, "2 + 3");
    }

    #[test]
    fn truncate_cuts_repo_name_and_fim() {
        let raw = "use super::Tool;\n<|repo_name|>foo\nmore";
        let t = truncate_pekko_completion(raw);
        assert!(t.contains("use super::Tool"));
        assert!(!t.contains("repo_name"));
        let raw2 = "impl Foo {}\n<|fim_prefix|>zzz";
        let t2 = truncate_pekko_completion(raw2);
        assert!(t2.starts_with("impl Foo"));
        assert!(!t2.contains("fim_"));
    }

    #[test]
    fn truncate_rejects_path_spam() {
        let raw = "/src/src/src/student/src/src\n#[derive(Debu";
        let t = truncate_pekko_completion(raw);
        assert_eq!(t, "");
        let raw2 = "/main/main/main/src/src\nfn";
        let t2 = truncate_pekko_completion(raw2);
        assert_eq!(t2, "");
    }

    #[test]
    fn truncate_module_without_keywords_is_empty() {
        let raw = "hello world this is not rust code at all and goes on";
        let t = truncate_pekko_completion(raw);
        assert_eq!(t, "");
    }

    #[test]
    fn slot_markers_unique() {
        let cs = default_slot_challenges();
        let mut seen = std::collections::HashSet::new();
        for c in &cs {
            assert!(seen.insert(c.task_id), "dup {}", c.task_id);
            assert!(
                c.prompt.contains(&marker_line(c.task_id)),
                "missing marker in {}",
                c.task_id
            );
        }
    }
}
