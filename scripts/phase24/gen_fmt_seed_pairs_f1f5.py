#!/usr/bin/env python3
"""Build format-SFT seed pairs for F1–F5: prompt → reference.rs body.

Keeps F0 seeds untouched; writes scripts/phase24/fmt_seed_pairs_f1f5.jsonl
and optionally a merged file for a short fmt continue.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/raid/users/paul/workLLM")
HARVEST = ROOT / "scratch-pekko-harvest"
OUT = ROOT / "scripts/phase24/fmt_seed_pairs_f1f5.jsonl"
MERGED = ROOT / "scripts/phase24/fmt_seed_pairs_f0f5.jsonl"
F0 = ROOT / "scripts/phase24/fmt_seed_pairs.jsonl"

# Must match llm-actors/src/domain/pekko_harvest.rs default_slot_challenges prompts
# (marker line included). Kept here as a thin duplicate for seed gen only.
PROMPTS = {
    "f1_tool/full_v1": (
        "Implement tool stubs for the Tool registry scratch.\n"
        "Write the FULL contents of src/student.rs so EchoTool, PingTool, and UpperTool\n"
        "pass `cargo test --features student`.\n"
        "API already in lib.rs (do not redefine Tool / ToolError / ToolRegistry):\n"
        "  trait Tool: Send + Sync { fn name(&self) -> &str; fn execute(&self, args: &str) -> Result<String, ToolError>; }\n"
        "Echo returns args unchanged; ping always returns \"pong\"; upper returns ASCII uppercase.\n"
        "Output ONLY the student.rs module body (use super::{Tool, ToolError}; + structs/impls).\n"
        "// pekko-harvest-task: f1_tool/full_v1\n"
    ),
    "f1_tool/full_v2": (
        "Add Tools named echo, ping, and upper; register/dispatch must work under student feature.\n"
        "Replace src/student.rs entirely. echo→args, ping→pong, upper→to_ascii_uppercase.\n"
        "use super::{Tool, ToolError};\n"
        "// pekko-harvest-task: f1_tool/full_v2\n"
    ),
    "f1_tool/full_ko": (
        "echo/ping/upper 툴을 student.rs에 전부 구현해. cargo test --features student 가 통과해야 한다.\n"
        "use super::{Tool, ToolError}; 로 시작하고 struct+impl만 작성.\n"
        "// pekko-harvest-task: f1_tool/full_ko\n"
    ),
    "f2_domain/full_v1": (
        "Implement OkOnlyDomain, DigitCharsetDomain, and NonEmptyDomain in src/student.rs.\n"
        "OkOnlyDomain: Correct iff completion == \"ok\". DigitCharsetDomain: charset is digits 0-9.\n"
        "NonEmptyDomain: reject empty completions. Traits Domain/Verdict live in lib.rs.\n"
        "Output ONLY the student.rs body starting with `use super::{Domain, Verdict};`.\n"
        "// pekko-harvest-task: f2_domain/full_v1\n"
    ),
    "f2_domain/full_v2": (
        "Toy Domain impls: only \"ok\" passes; charset includes 0-9; empty string is Incorrect.\n"
        "Fill src/student.rs completely for cargo test --features student.\n"
        "// pekko-harvest-task: f2_domain/full_v2\n"
    ),
    "f2_domain/full_ko": (
        "완성이 ok일 때만 통과하는 Domain과 digit charset, non-empty Domain을 student.rs에 구현해.\n"
        "// pekko-harvest-task: f2_domain/full_ko\n"
    ),
    "f3_message/full_v1": (
        "Implement CounterActor::handle in src/student.rs.\n"
        "Ping→Pong; Inc bumps n and returns Count(n); Get returns Count(n).\n"
        "Message/Response/Handler are in lib.rs. Start with use super::{Handler, Message, Response};\n"
        "// pekko-harvest-task: f3_message/full_v1\n"
    ),
    "f3_message/full_v2": (
        "When the actor receives Ping, reply Pong; Inc bumps; Get returns the count.\n"
        "Write full student.rs for the message-handler scratch.\n"
        "// pekko-harvest-task: f3_message/full_v2\n"
    ),
    "f3_message/full_ko": (
        "Ping 메시지를 받으면 Pong을 반환하고 Inc/Get을 처리하는 CounterActor 핸들러를 student.rs에 작성해.\n"
        "// pekko-harvest-task: f3_message/full_ko\n"
    ),
    "f4_repair/full_v1": (
        "Repair src/student.rs so cargo test --features student passes.\n"
        "Need: count_keys with HashMap (remember `use std::collections::HashMap`),\n"
        "greet(name) -> \"hi {name}\", exhaustive color_name for Red/Blue/Green.\n"
        "Write the FULL fixed student.rs module.\n"
        "// pekko-harvest-task: f4_repair/full_v1\n"
    ),
    "f4_repair/full_v2": (
        "Fix this compile/test surface: HashMap insert+len, greet, Color::Green arm.\n"
        "Replace student.rs entirely with a compiling implementation.\n"
        "// pekko-harvest-task: f4_repair/full_v2\n"
    ),
    "f5_supervisor/full_v1": (
        "Implement one_round(gen, ver, prompts) in src/student.rs:\n"
        "for each prompt: push \"generate\", generate, push \"verify\", keep completion if Correct.\n"
        "use super::{Generator, RoundResult, Verdict, Verifier};\n"
        "// pekko-harvest-task: f5_supervisor/full_v1\n"
    ),
    "f5_supervisor/full_v2": (
        "Wire Verifier after Generator for one self-improve round without training.\n"
        "Record order generate/verify and keep only Correct samples. Full student.rs please.\n"
        "// pekko-harvest-task: f5_supervisor/full_v2\n"
    ),
    "f5_supervisor/full_ko": (
        "Generator 다음에 Verifier가 오도록 한 라운드를 student.rs의 one_round에 연결해.\n"
        "// pekko-harvest-task: f5_supervisor/full_ko\n"
    ),
}

CRATE_FOR = {
    "f1_tool": "f1_tool",
    "f2_domain": "f2_domain",
    "f3_message": "f3_message",
    "f4_repair": "f4_repair",
    "f5_supervisor": "f5_supervisor",
}


def main() -> None:
    rows = []
    for task_id, prompt in PROMPTS.items():
        crate = task_id.split("/")[0]
        ref = (HARVEST / crate / "src" / "reference.rs").read_text()
        if not ref.endswith("\n"):
            ref += "\n"
        # Repeat each pair a few times so SFT sees the slot shape more often.
        for k in range(8):
            rows.append(
                {
                    "prompt": prompt,
                    "completion": ref,
                    "task": task_id,
                    "family": crate,
                    "rep": k,
                }
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} pairs -> {OUT}")

    merged = []
    if F0.exists():
        for line in F0.read_text().splitlines():
            if line.strip():
                merged.append(json.loads(line))
    merged.extend(rows)
    with MERGED.open("w") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(merged)} merged pairs -> {MERGED}")


if __name__ == "__main__":
    main()
