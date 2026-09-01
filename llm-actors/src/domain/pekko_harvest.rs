//! Phase 24 — multi-family Pekko/MSA harvest domain.
//!
//! Multiplexes:
//! - **F0** retention via [`RustCodeDomain`] expression slots
//! - **F1–F5** via isolated scratch crates under `verify_root/{f1_tool,…}`:
//!   keep the structural `student.rs` scaffold and replace **one** targeted
//!   `todo!(...)` with the model completion (short body / expression), then
//!   `cargo test --features student` (exit 0 ⇒ Correct). Other todos in the
//!   same file are filled with gold so the crate still compiles.
//!
//! Completions are **bodies only** (e.g. `Ok(args.to_string())`), not full files.

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

/// One F1–F5 body-slot challenge: NL + stub → replace a single `todo!(...)`.
#[derive(Debug, Clone)]
pub struct SlotChallenge {
    pub family: Family,
    pub task_id: &'static str,
    pub prompt: &'static str,
    /// Exact `todo!("…")` needle in scaffold `student.rs`.
    pub todo_needle: &'static str,
    /// Gold body that replaces the needle (expression / block body only).
    pub gold_body: &'static str,
}

/// Marker embedded in every slot prompt so repair wraps still match.
pub const TASK_MARKER_PREFIX: &str = "// pekko-harvest-task: ";

fn marker_line(task_id: &str) -> String {
    format!("{TASK_MARKER_PREFIX}{task_id}")
}

macro_rules! body_chal {
    ($fam:expr, $id:expr, $needle:expr, $gold:expr, $($prompt:expr),+ $(,)?) => {
        SlotChallenge {
            family: $fam,
            task_id: $id,
            todo_needle: $needle,
            gold_body: $gold,
            prompt: concat!($($prompt),+),
        }
    };
}

/// Default F1–F5 challenges — one `todo!` per challenge (short body completions).
pub fn default_slot_challenges() -> Vec<SlotChallenge> {
    vec![
        // ---- F1 echo ----
        body_chal!(
            Family::F1,
            "f1_tool/echo_v1",
            "todo!(\"return args unchanged\")",
            "Ok(args.to_string())",
            "Fill the todo! body for EchoTool::execute.\n",
            "Return args unchanged as Ok(String).\n",
            "Stub:\n",
            "    fn execute(&self, args: &str) -> Result<String, ToolError> {\n",
            "        todo!(\"return args unchanged\")\n",
            "    }\n",
            "Output ONLY the replacement for todo!(...) (e.g. Ok(args.to_string())), no fn/impl wrappers.\n",
            "// pekko-harvest-task: f1_tool/echo_v1\n"
        ),
        body_chal!(
            Family::F1,
            "f1_tool/echo_v2",
            "todo!(\"return args unchanged\")",
            "Ok(args.to_string())",
            "EchoTool execute body: echo the args string.\n",
            "Replace todo!(\"return args unchanged\") with a one-line Ok(...).\n",
            "// pekko-harvest-task: f1_tool/echo_v2\n"
        ),
        body_chal!(
            Family::F1,
            "f1_tool/echo_ko",
            "todo!(\"return args unchanged\")",
            "Ok(args.to_string())",
            "EchoTool::execute의 todo!만 채워라. args를 그대로 Ok로 반환.\n",
            "본문만 출력 (Ok(args.to_string()) 형태).\n",
            "// pekko-harvest-task: f1_tool/echo_ko\n"
        ),
        // ---- F1 ping ----
        body_chal!(
            Family::F1,
            "f1_tool/ping_v1",
            "todo!(\"return pong\")",
            "let _ = args; Ok(\"pong\".into())",
            "Fill PingTool::execute todo!. Always return Ok(\"pong\").\n",
            "Stub: todo!(\"return pong\") — output ONLY the body replacement.\n",
            "// pekko-harvest-task: f1_tool/ping_v1\n"
        ),
        body_chal!(
            Family::F1,
            "f1_tool/ping_v2",
            "todo!(\"return pong\")",
            "let _ = args; Ok(\"pong\".into())",
            "Ping tool body: ignore args, return the string pong.\n",
            "Replace todo!(\"return pong\") only.\n",
            "// pekko-harvest-task: f1_tool/ping_v2\n"
        ),
        body_chal!(
            Family::F1,
            "f1_tool/ping_ko",
            "todo!(\"return pong\")",
            "let _ = args; Ok(\"pong\".into())",
            "PingTool todo!를 pong 반환으로 채워. 본문만.\n",
            "// pekko-harvest-task: f1_tool/ping_ko\n"
        ),
        // ---- F1 upper ----
        body_chal!(
            Family::F1,
            "f1_tool/upper_v1",
            "todo!(\"ASCII uppercase\")",
            "Ok(args.to_ascii_uppercase())",
            "Fill UpperTool::execute. Return ASCII uppercase of args.\n",
            "Replace todo!(\"ASCII uppercase\") with Ok(args.to_ascii_uppercase()).\n",
            "Body only — no struct/impl.\n",
            "// pekko-harvest-task: f1_tool/upper_v1\n"
        ),
        body_chal!(
            Family::F1,
            "f1_tool/upper_v2",
            "todo!(\"ASCII uppercase\")",
            "Ok(args.to_ascii_uppercase())",
            "upper tool: to_ascii_uppercase the args into Ok(String).\n",
            "todo!(\"ASCII uppercase\") → body only.\n",
            "// pekko-harvest-task: f1_tool/upper_v2\n"
        ),
        body_chal!(
            Family::F1,
            "f1_tool/upper_ko",
            "todo!(\"ASCII uppercase\")",
            "Ok(args.to_ascii_uppercase())",
            "UpperTool todo!를 ASCII 대문자 변환으로 채워. 본문만.\n",
            "// pekko-harvest-task: f1_tool/upper_ko\n"
        ),
        // ---- F2 ok_only ----
        body_chal!(
            Family::F2,
            "f2_domain/ok_only_v1",
            "todo!(\"Correct iff completion == ok\")",
            "if completion == \"ok\" { Verdict::Correct } else { Verdict::Incorrect { reason: \"expected ok\".into() } }",
            "Fill OkOnlyDomain::verify todo!. Correct iff completion == \"ok\".\n",
            "Stub: todo!(\"Correct iff completion == ok\")\n",
            "Output ONLY the body (if/else returning Verdict).\n",
            "// pekko-harvest-task: f2_domain/ok_only_v1\n"
        ),
        body_chal!(
            Family::F2,
            "f2_domain/ok_only_v2",
            "todo!(\"Correct iff completion == ok\")",
            "if completion == \"ok\" { Verdict::Correct } else { Verdict::Incorrect { reason: \"expected ok\".into() } }",
            "OkOnlyDomain verify body: only the literal ok passes.\n",
            "Replace the todo! only.\n",
            "// pekko-harvest-task: f2_domain/ok_only_v2\n"
        ),
        body_chal!(
            Family::F2,
            "f2_domain/ok_only_ko",
            "todo!(\"Correct iff completion == ok\")",
            "if completion == \"ok\" { Verdict::Correct } else { Verdict::Incorrect { reason: \"expected ok\".into() } }",
            "OkOnlyDomain::verify todo!만 채워. completion==\"ok\"일 때만 Correct.\n",
            "// pekko-harvest-task: f2_domain/ok_only_ko\n"
        ),
        // ---- F2 digit_charset ----
        body_chal!(
            Family::F2,
            "f2_domain/digits_v1",
            "todo!(\"return digits 0-9\")",
            "\"0123456789\"",
            "Fill DigitCharsetDomain::charset todo! with the digit string 0-9.\n",
            "Replace todo!(\"return digits 0-9\") — body only (a &str literal).\n",
            "// pekko-harvest-task: f2_domain/digits_v1\n"
        ),
        body_chal!(
            Family::F2,
            "f2_domain/digits_v2",
            "todo!(\"return digits 0-9\")",
            "\"0123456789\"",
            "charset should be \"0123456789\". Replace the todo! only.\n",
            "// pekko-harvest-task: f2_domain/digits_v2\n"
        ),
        body_chal!(
            Family::F2,
            "f2_domain/digits_ko",
            "todo!(\"return digits 0-9\")",
            "\"0123456789\"",
            "DigitCharsetDomain charset todo!를 0-9 문자열로 채워. 본문만.\n",
            "// pekko-harvest-task: f2_domain/digits_ko\n"
        ),
        // ---- F2 non_empty ----
        body_chal!(
            Family::F2,
            "f2_domain/nonempty_v1",
            "todo!(\"reject empty\")",
            "if completion.is_empty() { Verdict::Incorrect { reason: \"empty\".into() } } else { Verdict::Correct }",
            "Fill NonEmptyDomain::verify. Reject empty completions.\n",
            "Replace todo!(\"reject empty\") with if/else Verdict body only.\n",
            "// pekko-harvest-task: f2_domain/nonempty_v1\n"
        ),
        body_chal!(
            Family::F2,
            "f2_domain/nonempty_v2",
            "todo!(\"reject empty\")",
            "if completion.is_empty() { Verdict::Incorrect { reason: \"empty\".into() } } else { Verdict::Correct }",
            "NonEmptyDomain: empty → Incorrect, else Correct. Body only.\n",
            "// pekko-harvest-task: f2_domain/nonempty_v2\n"
        ),
        body_chal!(
            Family::F2,
            "f2_domain/nonempty_ko",
            "todo!(\"reject empty\")",
            "if completion.is_empty() { Verdict::Incorrect { reason: \"empty\".into() } } else { Verdict::Correct }",
            "NonEmptyDomain verify todo!만 채워. 빈 문자열 거부.\n",
            "// pekko-harvest-task: f2_domain/nonempty_ko\n"
        ),
        // ---- F3 Ping / Inc / Get arms (one todo per harvest) ----
        body_chal!(
            Family::F3,
            "f3_message/ping_v1",
            "todo!(\"Ping arm\")",
            "Response::Pong",
            "Fill ONLY the Ping match arm todo!. Return Response::Pong.\n",
            "Stub: Message::Ping => todo!(\"Ping arm\"),\n",
            "Output ONLY the arm expression (e.g. Response::Pong), no match/fn wrappers.\n",
            "// pekko-harvest-task: f3_message/ping_v1\n"
        ),
        body_chal!(
            Family::F3,
            "f3_message/ping_v2",
            "todo!(\"Ping arm\")",
            "Response::Pong",
            "Ping arm body: Response::Pong. Replace todo!(\"Ping arm\") only.\n",
            "// pekko-harvest-task: f3_message/ping_v2\n"
        ),
        body_chal!(
            Family::F3,
            "f3_message/ping_ko",
            "todo!(\"Ping arm\")",
            "Response::Pong",
            "Ping 팔 todo!만 Response::Pong으로 채워. 표현식만.\n",
            "// pekko-harvest-task: f3_message/ping_ko\n"
        ),
        body_chal!(
            Family::F3,
            "f3_message/inc_v1",
            "todo!(\"Inc arm\")",
            "{ self.n += 1; Response::Count(self.n) }",
            "Fill ONLY the Inc match arm todo!. Bump self.n then return Count(n).\n",
            "Stub: Message::Inc => todo!(\"Inc arm\"),\n",
            "Output ONLY the arm body, e.g. { self.n += 1; Response::Count(self.n) }\n",
            "// pekko-harvest-task: f3_message/inc_v1\n"
        ),
        body_chal!(
            Family::F3,
            "f3_message/inc_v2",
            "todo!(\"Inc arm\")",
            "{ self.n += 1; Response::Count(self.n) }",
            "Inc arm: increment n and return Response::Count(self.n). Body only.\n",
            "// pekko-harvest-task: f3_message/inc_v2\n"
        ),
        body_chal!(
            Family::F3,
            "f3_message/inc_ko",
            "todo!(\"Inc arm\")",
            "{ self.n += 1; Response::Count(self.n) }",
            "Inc 팔 todo!만 n+=1 후 Count. 본문만.\n",
            "// pekko-harvest-task: f3_message/inc_ko\n"
        ),
        body_chal!(
            Family::F3,
            "f3_message/get_v1",
            "todo!(\"Get arm\")",
            "Response::Count(self.n)",
            "Fill ONLY the Get match arm todo!. Return Response::Count(self.n).\n",
            "Stub: Message::Get => todo!(\"Get arm\"),\n",
            "Output ONLY the arm expression. No match/fn wrappers.\n",
            "// pekko-harvest-task: f3_message/get_v1\n"
        ),
        body_chal!(
            Family::F3,
            "f3_message/get_v2",
            "todo!(\"Get arm\")",
            "Response::Count(self.n)",
            "Get arm body: Response::Count(self.n). Replace todo!(\"Get arm\") only.\n",
            "// pekko-harvest-task: f3_message/get_v2\n"
        ),
        body_chal!(
            Family::F3,
            "f3_message/get_ko",
            "todo!(\"Get arm\")",
            "Response::Count(self.n)",
            "Get 팔 todo!만 Count(self.n). 표현식만.\n",
            "// pekko-harvest-task: f3_message/get_ko\n"
        ),
        // ---- F4 count_keys ----
        body_chal!(
            Family::F4,
            "f4_repair/count_keys_v1",
            "todo!(\"HashMap insert + len — remember the import\")",
            "{\n    let mut m = std::collections::HashMap::new();\n    for (k, v) in pairs {\n        m.insert(*k, *v);\n    }\n    m.len()\n}",
            "Fill count_keys todo!. Insert pairs into a HashMap and return len.\n",
            "Use std::collections::HashMap (fully qualified is fine).\n",
            "Replace todo!(\"HashMap insert + len — remember the import\") — body only.\n",
            "// pekko-harvest-task: f4_repair/count_keys_v1\n"
        ),
        body_chal!(
            Family::F4,
            "f4_repair/count_keys_v2",
            "todo!(\"HashMap insert + len — remember the import\")",
            "{\n    let mut m = std::collections::HashMap::new();\n    for (k, v) in pairs {\n        m.insert(*k, *v);\n    }\n    m.len()\n}",
            "count_keys body: HashMap insert each pair, return m.len(). Body only.\n",
            "// pekko-harvest-task: f4_repair/count_keys_v2\n"
        ),
        // ---- F4 greet ----
        body_chal!(
            Family::F4,
            "f4_repair/greet_v1",
            "todo!(\"return hi <name>\")",
            "format!(\"hi {name}\")",
            "Fill greet todo!. Return format!(\"hi {name}\").\n",
            "Replace todo!(\"return hi <name>\") — expression body only.\n",
            "// pekko-harvest-task: f4_repair/greet_v1\n"
        ),
        body_chal!(
            Family::F4,
            "f4_repair/greet_v2",
            "todo!(\"return hi <name>\")",
            "format!(\"hi {name}\")",
            "greet(name) should yield \"hi {name}\". Body only.\n",
            "// pekko-harvest-task: f4_repair/greet_v2\n"
        ),
        // ---- F4 green ----
        body_chal!(
            Family::F4,
            "f4_repair/green_v1",
            "todo!(\"repair: name for Green\")",
            "\"green\"",
            "Repair Color::Green arm: replace todo!(\"repair: name for Green\") with \"green\".\n",
            "Output ONLY the match-arm expression.\n",
            "// pekko-harvest-task: f4_repair/green_v1\n"
        ),
        body_chal!(
            Family::F4,
            "f4_repair/green_v2",
            "todo!(\"repair: name for Green\")",
            "\"green\"",
            "Green color name is the string green. Replace the todo! only.\n",
            "// pekko-harvest-task: f4_repair/green_v2\n"
        ),
        body_chal!(
            Family::F4,
            "f4_repair/green_ko",
            "todo!(\"repair: name for Green\")",
            "\"green\"",
            "Color::Green 팔의 todo!를 \"green\"으로 고쳐. 표현식만.\n",
            "// pekko-harvest-task: f4_repair/green_ko\n"
        ),
        // ---- F5 generate vs verify-keep (short bodies) ----
        body_chal!(
            Family::F5,
            "f5_supervisor/gen_v1",
            "todo!(\"push generate then gen.generate\")",
            "{\n    out.order.push(\"generate\");\n    gen.generate(p)\n}",
            "Fill record_generate todo!. Push \"generate\" then return gen.generate(p).\n",
            "Replace todo!(\"push generate then gen.generate\") — 2–4 line body only.\n",
            "// pekko-harvest-task: f5_supervisor/gen_v1\n"
        ),
        body_chal!(
            Family::F5,
            "f5_supervisor/gen_v2",
            "todo!(\"push generate then gen.generate\")",
            "{\n    out.order.push(\"generate\");\n    gen.generate(p)\n}",
            "record_generate body: order.push generate, then gen.generate(p). Body only.\n",
            "// pekko-harvest-task: f5_supervisor/gen_v2\n"
        ),
        body_chal!(
            Family::F5,
            "f5_supervisor/gen_ko",
            "todo!(\"push generate then gen.generate\")",
            "{\n    out.order.push(\"generate\");\n    gen.generate(p)\n}",
            "record_generate todo!만: order에 generate 푸시 후 gen.generate. 본문만.\n",
            "// pekko-harvest-task: f5_supervisor/gen_ko\n"
        ),
        body_chal!(
            Family::F5,
            "f5_supervisor/keep_v1",
            "todo!(\"push verify; keep Correct\")",
            "{\n    out.order.push(\"verify\");\n    if ver.verify(p, &c) == Verdict::Correct { out.kept.push(c); }\n}",
            "Fill record_verify_keep todo!. Push \"verify\"; keep completion iff Correct.\n",
            "Replace todo!(\"push verify; keep Correct\") — short body only.\n",
            "// pekko-harvest-task: f5_supervisor/keep_v1\n"
        ),
        body_chal!(
            Family::F5,
            "f5_supervisor/keep_v2",
            "todo!(\"push verify; keep Correct\")",
            "{\n    out.order.push(\"verify\");\n    if ver.verify(p, &c) == Verdict::Correct { out.kept.push(c); }\n}",
            "record_verify_keep: push verify, then kept.push if Verdict::Correct. Body only.\n",
            "// pekko-harvest-task: f5_supervisor/keep_v2\n"
        ),
        body_chal!(
            Family::F5,
            "f5_supervisor/keep_ko",
            "todo!(\"push verify; keep Correct\")",
            "{\n    out.order.push(\"verify\");\n    if ver.verify(p, &c) == Verdict::Correct { out.kept.push(c); }\n}",
            "record_verify_keep todo!만: verify 기록, Correct만 kept. 본문만.\n",
            "// pekko-harvest-task: f5_supervisor/keep_ko\n"
        ),
        body_chal!(
            Family::F5,
            "f5_supervisor/round_v1",
            "todo!(\"for each prompt: generate then verify-keep\")",
            "{\n    let mut out = RoundResult::default();\n    for p in prompts { let c = record_generate(&mut out, gen, p); record_verify_keep(&mut out, ver, p, c); }\n    out\n}",
            "Fill one_round todo!. For each prompt call record_generate then record_verify_keep.\n",
            "Replace todo!(\"for each prompt: generate then verify-keep\") — short body only.\n",
            "// pekko-harvest-task: f5_supervisor/round_v1\n"
        ),
        body_chal!(
            Family::F5,
            "f5_supervisor/round_v2",
            "todo!(\"for each prompt: generate then verify-keep\")",
            "{\n    let mut out = RoundResult::default();\n    for p in prompts { let c = record_generate(&mut out, gen, p); record_verify_keep(&mut out, ver, p, c); }\n    out\n}",
            "one_round body: loop prompts; generate then verify-keep helpers. Return RoundResult.\n",
            "// pekko-harvest-task: f5_supervisor/round_v2\n"
        ),
        body_chal!(
            Family::F5,
            "f5_supervisor/round_ko",
            "todo!(\"for each prompt: generate then verify-keep\")",
            "{\n    let mut out = RoundResult::default();\n    for p in prompts { let c = record_generate(&mut out, gen, p); record_verify_keep(&mut out, ver, p, c); }\n    out\n}",
            "one_round todo!만: 프롬프트마다 generate 다음 verify-keep. 본문만.\n",
            "// pekko-harvest-task: f5_supervisor/round_ko\n"
        ),
    ]
}

/// Apply gold bodies for non-target slots, then put `completion` in the target todo.
pub fn apply_body_slot(
    scaffold: &str,
    family_slots: &[&SlotChallenge],
    target: &SlotChallenge,
    completion: &str,
) -> Result<String, String> {
    let body = truncate_pekko_completion(completion);
    if body.trim().is_empty() {
        return Err("empty body after truncate".into());
    }
    let mut text = scaffold.to_string();
    for c in family_slots {
        // Paraphrases share a needle; only gold-fill *other* unique todos.
        if c.todo_needle == target.todo_needle {
            continue;
        }
        if text.contains(c.todo_needle) {
            text = text.replacen(c.todo_needle, c.gold_body, 1);
        }
    }
    if !text.contains(target.todo_needle) {
        return Err(format!(
            "target needle missing after gold fill: {}",
            target.todo_needle
        ));
    }
    Ok(text.replacen(target.todo_needle, &body, 1))
}

/// Multiplexed harvest domain: F0 expression slots + F1–F5 body slots.
pub struct PekkoHarvestDomain {
    /// Isolated crate copies live here: `{verify_root}/f1_tool`, …
    pub verify_root: PathBuf,
    /// Parent harvest root holding scaffolds (sibling of verify_root).
    harvest_root: PathBuf,
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
        let verify_root = verify_root.into();
        let harvest_root = verify_root
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
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
            verify_root,
            harvest_root,
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
            let src = self.harvest_root.join(name);
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
        for c in &self.slots {
            let m = marker_line(c.task_id);
            if prompt.contains(&m) {
                return Some(c);
            }
        }
        self.slots
            .iter()
            .filter(|c| prompt.ends_with(c.prompt) || prompt.contains(c.prompt))
            .max_by_key(|c| c.prompt.len())
    }

    fn family_slots(&self, family: Family) -> Vec<&SlotChallenge> {
        // Dedup by todo_needle so multiple paraphrases don't re-apply gold.
        // Constructor keeps all paraphrases for each selected family, so every
        // unique needle in that family is present in `self.slots`.
        let mut out: Vec<&SlotChallenge> = Vec::new();
        for c in &self.slots {
            if c.family != family {
                continue;
            }
            if out.iter().any(|x| x.todo_needle == c.todo_needle) {
                continue;
            }
            out.push(c);
        }
        out
    }

    fn scaffold_student(&self, family: Family) -> io::Result<String> {
        let name = family.crate_dir().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "F0 has no student scaffold")
        })?;
        let p = self.harvest_root.join(name).join("src/student.rs");
        fs::read_to_string(&p)
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
        let scaffold = match self.scaffold_student(challenge.family) {
            Ok(s) => s,
            Err(e) => {
                return Verdict::Inconclusive {
                    reason: format!("read scaffold: {e}"),
                }
            }
        };
        let family_slots = self.family_slots(challenge.family);
        let filled = match apply_body_slot(&scaffold, &family_slots, challenge, completion) {
            Ok(t) => t,
            Err(e) => {
                return Verdict::Incorrect {
                    reason: format!("body apply failed: {e}"),
                }
            }
        };

        let _guard = self.write_lock.lock().expect("write_lock poisoned");
        if let Err(e) = fs::write(&student_path, &filled) {
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
            return Some(format!(
                "{}\n\n// Previous body attempt:\n{}\n\n// ERR:{}\n// Rewrite ONLY the todo! body replacement (no markdown, no fn/impl wrappers):\n{}",
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

/// Truncate model completion for F0 expressions **or** F1–F5 short bodies.
///
/// Body-shaped completions (`Ok(…)`, `match …`, `format!(…)`, string lits, blocks)
/// keep multi-line content and only cut on FIM/fence/path-spam — they must NOT
/// use F0 stops like `\\npub `/`\\nfn ` which would chop valid match arms.
pub fn truncate_pekko_completion(completion: &str) -> String {
    let mut s = completion.trim_start();
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
    if let Some(i) = first_path_spam_offset(s) {
        cut = cut.min(i);
    }
    let body = s[..cut].trim_end();
    if body.is_empty() || is_path_spam_body(body) {
        return String::new();
    }

    // Legacy full-module completions (still accepted if a model emits them).
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

    // Body-slot shaped: keep a balanced fragment. Do NOT use F0 stops
    // (`\npub `/`\nfn `/`\nuse `) or `\n}` — those chop short match/Inc bodies.
    if is_body_shaped(body) {
        if let Some(end) = balanced_fragment_end(body) {
            return body[..end].trim_end().to_string();
        }
        let soft = ["\n\npub ", "\n\nfn ", "\n\nstruct ", "\n\nimpl ", "\n\nuse "];
        let mut c = body.len();
        for st in soft {
            if let Some(i) = body.find(st) {
                c = c.min(i);
            }
        }
        let trimmed = body[..c].trim_end();
        if let Some(one) = first_complete_oneliner(trimmed) {
            return one.to_string();
        }
        return trimmed.to_string();
    }

    let head = &body[..body
        .chars()
        .take(80)
        .map(|c| c.len_utf8())
        .sum::<usize>()
        .min(body.len())];
    let has_rust_kw = [
        "fn ", "impl ", "use ", "struct ", "pub ", "enum ", "const ", "type ", "#[", "mod ",
    ]
    .iter()
    .any(|k| head.contains(k));

    if !has_rust_kw && !looks_like_rust_expr(body) {
        return String::new();
    }

    // F0-ish expression stops.
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

fn balanced_fragment_end(body: &str) -> Option<usize> {
    let skip = body.len() - body.trim_start().len();
    let t = &body[skip..];
    let use_paren = t.starts_with("Ok(")
        || t.starts_with("Err(")
        || t.starts_with("format!")
        || t.starts_with('(');
    let use_brace = t.starts_with('{')
        || t.starts_with("match ")
        || t.starts_with("if ")
        || t.starts_with("for ")
        || t.starts_with("while ")
        || t.starts_with("loop ");
    if !use_paren && !use_brace {
        return None;
    }
    let mut depth = 0i32;
    let mut seen = false;
    let mut in_str = false;
    let mut esc = false;
    for (i, ch) in t.char_indices() {
        if in_str {
            if esc {
                esc = false;
                continue;
            }
            if ch == '\\' {
                esc = true;
                continue;
            }
            if ch == '"' {
                in_str = false;
            }
            continue;
        }
        let opener = (use_brace && ch == '{') || (use_paren && (ch == '(' || ch == '['));
        let closer = (use_brace && ch == '}') || (use_paren && (ch == ')' || ch == ']'));
        match ch {
            '"' => in_str = true,
            _ if opener => {
                depth += 1;
                seen = true;
            }
            _ if closer => {
                depth -= 1;
                if depth < 0 {
                    return None;
                }
                if seen && depth == 0 {
                    let end = skip + i + ch.len_utf8();
                    let rest = body[end..].trim_start();
                    if rest.starts_with("else") {
                        continue;
                    }
                    return Some(end);
                }
            }
            _ => {}
        }
    }
    None
}

fn first_complete_oneliner(body: &str) -> Option<&str> {
    // Skip blank / comment / empty-string noise lines (model often emits "" then later gold).
    let mut chosen: Option<&str> = None;
    for line in body.lines() {
        let first = line.trim_end();
        let t = first.trim_start();
        if t.is_empty() || t.starts_with("//") || t == "\"\"" {
            continue;
        }
        if t.starts_with("match ")
            || t.starts_with("if ")
            || t.starts_with("{")
            || t.starts_with("for ")
            || t.starts_with("while ")
            || t.starts_with("loop ")
            || t.ends_with('{')
            || t.ends_with(',')
            || t.ends_with('(')
        {
            return None; // multi-line body — keep soft-cut result
        }
        let mut bal = 0i32;
        for ch in first.chars() {
            match ch {
                '(' | '{' | '[' => bal += 1,
                ')' | '}' | ']' => bal -= 1,
                _ => {}
            }
            if bal < 0 {
                return None;
            }
        }
        if bal != 0 {
            return None;
        }
        chosen = Some(first);
        break;
    }
    chosen
}

fn is_body_shaped(body: &str) -> bool {

    let t = body.trim_start();
    t.starts_with("match ")
        || t.starts_with("if ")
        || t.starts_with("Ok(")
        || t.starts_with("Err(")
        || t.starts_with("let ")
        || t.starts_with("format!")
        || t.starts_with('"')
        || t.starts_with('{')
        || t.starts_with("for ")
        || t.starts_with("while ")
        || t.starts_with("loop ")
        || t.starts_with("return ")
        || t.starts_with("self.")
        || t.starts_with("std::")
        || t.starts_with("Verdict::")
        || t.starts_with("Response::")
        || t.starts_with("Message::")
        || t.starts_with("RoundResult")
}

fn first_path_spam_offset(s: &str) -> Option<usize> {
    let mut offset = 0usize;
    for line in s.split_inclusive('\n') {
        let raw = line.trim_end_matches('\n');
        if let Some(rel) = path_spam_start_in_line(raw) {
            return Some(offset + rel);
        }
        offset += line.len();
    }
    None
}

/// If a line contains path-repetition spam, return byte offset within the line
/// where spam begins (0 if the whole line is spam). Allows salvaging a good
/// prefix like `Ok(args` before `/src/src/...`.
fn path_spam_start_in_line(line: &str) -> Option<usize> {
    if let Some(i) = line.find("/src/src") {
        return Some(i);
    }
    if let Some(i) = line.find("/main/main") {
        return Some(i);
    }
    if is_path_spam_line(line) {
        return Some(0);
    }
    None
}

fn is_path_spam_line(line: &str) -> bool {
    let t = line.trim();
    if t.is_empty() {
        return false;
    }
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
    fn truncate_keeps_body_ok() {
        let raw = "Ok(args.to_string())\n<|repo_name|>junk";
        let t = truncate_pekko_completion(raw);
        assert_eq!(t, "Ok(args.to_string())");
    }

    #[test]
    fn truncate_oneliner_before_path_spam() {
        let raw = "Ok(args.to_ascii_uppercase())/src/src/src\nuse";
        let t = truncate_pekko_completion(raw);
        assert_eq!(t, "Ok(args.to_ascii_uppercase())");
        let raw2 = "\"0123456789\"\n// pekko\n/src/src";
        let t2 = truncate_pekko_completion(raw2);
        assert_eq!(t2, "\"0123456789\"");
    }

    #[test]
    fn truncate_keeps_match_body() {
        let raw = "match msg {\n            Message::Ping => Response::Pong,\n            Message::Get => Response::Count(self.n),\n        }\n\n\npub fn other() {}";
        let t = truncate_pekko_completion(raw);
        assert!(t.contains("Message::Ping"));
        assert!(!t.contains("pub fn other"));
    }

    #[test]
    fn truncate_expression_cuts_runaway() {
        let raw = "2 + 3\n\nfn other() {}";
        let t = truncate_pekko_completion(raw);
        assert_eq!(t, "2 + 3");
    }

    #[test]
    fn truncate_cuts_repo_name_and_fim() {
        let raw = "Ok(1)\n<|repo_name|>foo\nmore";
        let t = truncate_pekko_completion(raw);
        assert_eq!(t, "Ok(1)");
    }

    #[test]
    fn truncate_rejects_path_spam() {
        let raw = "/src/src/src/student/src/src\n#[derive(Debu";
        let t = truncate_pekko_completion(raw);
        assert_eq!(t, "");
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

    #[test]
    fn apply_body_replaces_one_todo() {
        let scaffold = concat!(
            "fn a() { todo!(\"return args unchanged\") }\n",
            "fn b() { todo!(\"return pong\") }\n",
        );
        let slots = default_slot_challenges();
        let echo = slots.iter().find(|c| c.task_id == "f1_tool/echo_v1").unwrap();
        let fam: Vec<&SlotChallenge> = slots
            .iter()
            .filter(|c| c.family == Family::F1)
            .collect::<Vec<_>>()
            .into_iter()
            .fold(Vec::new(), |mut acc, c| {
                if !acc.iter().any(|x: &&SlotChallenge| x.todo_needle == c.todo_needle) {
                    acc.push(c);
                }
                acc
            });
        let out = apply_body_slot(scaffold, &fam, echo, "Ok(args.to_string())").unwrap();
        assert!(out.contains("Ok(args.to_string())"));
        assert!(!out.contains("todo!(\"return args unchanged\")"));
        // other todo filled with gold
        assert!(out.contains("Ok(\"pong\".into())") || out.contains("let _ = args"));
        assert!(!out.contains("todo!(\"return pong\")"));
    }

    #[test]
    fn apply_body_does_not_gold_fill_same_needle_paraphrase() {
        let scaffold = "fn a() { todo!(\"return args unchanged\") }\n";
        let slots = default_slot_challenges();
        let echo_v2 = slots.iter().find(|c| c.task_id == "f1_tool/echo_v2").unwrap();
        let fam: Vec<&SlotChallenge> = slots
            .iter()
            .filter(|c| c.family == Family::F1)
            .fold(Vec::new(), |mut acc, c| {
                if !acc.iter().any(|x: &&SlotChallenge| x.todo_needle == c.todo_needle) {
                    acc.push(c);
                }
                acc
            });
        let out = apply_body_slot(scaffold, &fam, echo_v2, "Ok(args.to_string())").unwrap();
        assert!(out.contains("Ok(args.to_string())"));
        assert!(!out.contains("todo!(\"return args unchanged\")"));
    }

    #[test]
    fn truncate_does_not_chop_inc_block_on_brace() {
        let raw = "{\n    self.n += 1;\n    Response::Count(self.n)\n}\nfn other() {}";
        let t = truncate_pekko_completion(raw);
        assert!(t.contains("self.n += 1"), "{t:?}");
        assert!(t.contains("Response::Count(self.n)"), "{t:?}");
        assert!(t.trim().ends_with('}'), "{t:?}");
        assert!(!t.contains("fn other"), "{t:?}");
    }

    #[test]
    fn truncate_keeps_if_else_verdict() {
        let raw = "if completion == \"ok\" { Verdict::Correct } else { Verdict::Incorrect { reason: \"expected ok\".into() } }\npub fn leak() {}";
        let t = truncate_pekko_completion(raw);
        assert!(t.contains("else"), "{t:?}");
        assert!(t.contains("Verdict::Incorrect"), "{t:?}");
        assert!(!t.contains("fn leak"), "{t:?}");
        let raw2 = "if completion.is_empty() { Verdict::Incorrect { reason: \"empty\".into() } } else { Verdict::Correct }";
        let t2 = truncate_pekko_completion(raw2);
        assert!(t2.contains("else"), "{t2:?}");
        assert!(t2.contains("is_empty"), "{t2:?}");
        assert!(t2.contains("Verdict::Correct"), "{t2:?}");
    }

    #[test]
    fn truncate_keeps_short_ping_arm() {
        let raw = "Response::Pong\npub fn leak() {}";
        let t = truncate_pekko_completion(raw);
        assert_eq!(t, "Response::Pong");
    }

    #[test]
    fn f3_f5_needles_are_split() {
        let cs = default_slot_challenges();
        let f3: Vec<_> = cs.iter().filter(|c| c.family == Family::F3).collect();
        let needles: std::collections::HashSet<_> = f3.iter().map(|c| c.todo_needle).collect();
        assert!(needles.contains("todo!(\"Ping arm\")"));
        assert!(needles.contains("todo!(\"Inc arm\")"));
        assert!(needles.contains("todo!(\"Get arm\")"));
        assert_eq!(needles.len(), 3);
        let f5: Vec<_> = cs.iter().filter(|c| c.family == Family::F5).collect();
        let n5: std::collections::HashSet<_> = f5.iter().map(|c| c.todo_needle).collect();
        assert_eq!(n5.len(), 3);
        assert!(f3.iter().all(|c| c.gold_body.lines().count() <= 6));
        assert!(f5.iter().all(|c| c.gold_body.lines().count() <= 6));
    }

    #[test]
    fn all_slot_golds_pass_cargo_student() {
        let harvest = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../scratch-pekko-harvest");
        if !harvest.join("f1_tool/Cargo.toml").exists() {
            return;
        }
        let verify = harvest.join("_verify_gold_v8");
        let scratch = harvest.join("_cargo_scratch_gold_v8");
        let families = Family::parse_list("f0,f1,f2,f3,f4,f5").unwrap();
        let d = PekkoHarvestDomain::new(&verify, &scratch, &families);
        d.ensure_ready().expect("ensure_ready");
        let mut seen = std::collections::HashSet::new();
        for c in &d.slots {
            if !seen.insert(c.task_id) {
                continue;
            }
            let v = d.verify(c.prompt, c.gold_body);
            assert!(
                matches!(v, Verdict::Correct),
                "gold failed {}: {v:?}",
                c.task_id
            );
        }
        // F0 expression golds
        if let Some(f0) = &d.f0 {
            let golds = [
                ("equals_5", "2 + 3"),
                ("equals_14_via_doubling", "7"),
                ("len_5_string", "\"hello\""),
                ("equals_10", "10"),
                ("equals_zero", "0"),
                ("bool_true", "true"),
                ("bool_false", "false"),
                ("len_3_string", "\"abc\""),
                ("vec_sum_6", "[1, 2, 3]"),
                ("option_some_5", "Some(5)"),
            ];
            for (i, ch) in f0.challenges.iter().enumerate() {
                let g = golds.iter().find(|(n, _)| *n == ch.name).map(|(_, g)| *g);
                let Some(g) = g else { continue };
                let prompt = f0.nth_prompt(i).expect("f0 prompt");
                let v = d.verify(&prompt, g);
                assert!(
                    matches!(v, Verdict::Correct),
                    "F0 gold failed {}: {v:?}",
                    ch.name
                );
            }
        }
    }
}
