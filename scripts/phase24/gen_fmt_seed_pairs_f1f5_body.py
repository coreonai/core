#!/usr/bin/env python3
"""Build format-SFT seed pairs for F1–F5 **body slots** (todo! fill).

Writes:
  scripts/phase24/fmt_seed_pairs_f1f5_body.jsonl
  scripts/phase24/fmt_seed_pairs_f0f5_body.jsonl  (F0 + body, shuffled)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path("/raid/users/paul/workLLM")
OUT = ROOT / "scripts/phase24/fmt_seed_pairs_f1f5_body.jsonl"
MERGED = ROOT / "scripts/phase24/fmt_seed_pairs_f0f5_body.jsonl"
F0 = ROOT / "scripts/phase24/fmt_seed_pairs.jsonl"

# (family, task_id, todo_needle, gold_body, paraphrases[])
# Markers must match pekko_harvest.rs
SLOTS = [
    (
        "f1_tool",
        "f1_tool/echo_v1",
        'todo!("return args unchanged")',
        "Ok(args.to_string())",
        [
            "Fill the todo! body for EchoTool::execute.\nReturn args unchanged as Ok(String).\nStub:\n    fn execute(&self, args: &str) -> Result<String, ToolError> {\n        todo!(\"return args unchanged\")\n    }\nOutput ONLY the replacement for todo!(...) (e.g. Ok(args.to_string())), no fn/impl wrappers.\n",
            "EchoTool execute body: echo the args string.\nReplace todo!(\"return args unchanged\") with a one-line Ok(...).\n",
            "EchoTool::execute의 todo!만 채워라. args를 그대로 Ok로 반환.\n본문만 출력 (Ok(args.to_string()) 형태).\n",
            "Body for echo tool execute — return Ok(args.to_string()).\nSLOT: todo!(\"return args unchanged\")\n",
            "Complete only the todo! in EchoTool::execute so args are returned unchanged.\n",
            "RHS/body that makes EchoTool pass: unchanged args in Ok.\n",
        ],
    ),
    (
        "f1_tool",
        "f1_tool/ping_v1",
        'todo!("return pong")',
        'let _ = args; Ok("pong".into())',
        [
            "Fill PingTool::execute todo!. Always return Ok(\"pong\").\nStub: todo!(\"return pong\") — output ONLY the body replacement.\n",
            "Ping tool body: ignore args, return the string pong.\nReplace todo!(\"return pong\") only.\n",
            "PingTool todo!를 pong 반환으로 채워. 본문만.\n",
            "todo!(\"return pong\") → let _ = args; Ok(\"pong\".into())\nBody only.\n",
            "Write the PingTool::execute body that always yields pong.\n",
            "Constant ping response body for the Tool trait stub.\n",
        ],
    ),
    (
        "f1_tool",
        "f1_tool/upper_v1",
        'todo!("ASCII uppercase")',
        "Ok(args.to_ascii_uppercase())",
        [
            "Fill UpperTool::execute. Return ASCII uppercase of args.\nReplace todo!(\"ASCII uppercase\") with Ok(args.to_ascii_uppercase()).\nBody only — no struct/impl.\n",
            "upper tool: to_ascii_uppercase the args into Ok(String).\ntodo!(\"ASCII uppercase\") → body only.\n",
            "UpperTool todo!를 ASCII 대문자 변환으로 채워. 본문만.\n",
            "SLOT todo!(\"ASCII uppercase\") → Ok(args.to_ascii_uppercase())\n",
            "UpperTool execute body only: ASCII upper.\n",
            "Make upper tool pass cargo tests — fill the todo! body.\n",
        ],
    ),
    (
        "f2_domain",
        "f2_domain/ok_only_v1",
        'todo!("Correct iff completion == ok")',
        'if completion == "ok" { Verdict::Correct } else { Verdict::Incorrect { reason: "expected ok".into() } }',
        [
            "Fill OkOnlyDomain::verify todo!. Correct iff completion == \"ok\".\nStub: todo!(\"Correct iff completion == ok\")\nOutput ONLY the body (if/else returning Verdict).\n",
            "OkOnlyDomain verify body: only the literal ok passes.\nReplace the todo! only.\n",
            "OkOnlyDomain::verify todo!만 채워. completion==\"ok\"일 때만 Correct.\n",
            "verify body: Correct when completion is exactly ok else Incorrect.\n",
            "Domain verify slot — accept only \"ok\". Body only.\n",
            "Replace todo!(\"Correct iff completion == ok\") with if/else Verdict.\n",
        ],
    ),
    (
        "f2_domain",
        "f2_domain/digits_v1",
        'todo!("return digits 0-9")',
        '"0123456789"',
        [
            "Fill DigitCharsetDomain::charset todo! with the digit string 0-9.\nReplace todo!(\"return digits 0-9\") — body only (a &str literal).\n",
            "charset should be \"0123456789\". Replace the todo! only.\n",
            "DigitCharsetDomain charset todo!를 0-9 문자열로 채워. 본문만.\n",
            "Return \"0123456789\" for charset todo!.\n",
            "SLOT: todo!(\"return digits 0-9\") → \"0123456789\"\n",
            "Digit charset body only.\n",
        ],
    ),
    (
        "f2_domain",
        "f2_domain/nonempty_v1",
        'todo!("reject empty")',
        'if completion.is_empty() { Verdict::Incorrect { reason: "empty".into() } } else { Verdict::Correct }',
        [
            "Fill NonEmptyDomain::verify. Reject empty completions.\nReplace todo!(\"reject empty\") with if/else Verdict body only.\n",
            "NonEmptyDomain: empty → Incorrect, else Correct. Body only.\n",
            "NonEmptyDomain verify todo!만 채워. 빈 문자열 거부.\n",
            "todo!(\"reject empty\") body: is_empty check.\n",
            "Reject empty completion in NonEmptyDomain::verify — body only.\n",
            "Write the verify body that fails on empty strings.\n",
        ],
    ),
    (
        "f3_message",
        "f3_message/handle_v1",
        'todo!("Ping->Pong, Inc bumps, Get returns count")',
        "match msg {\n            Message::Ping => Response::Pong,\n            Message::Inc => {\n                self.n += 1;\n                Response::Count(self.n)\n            }\n            Message::Get => Response::Count(self.n),\n        }",
        [
            "Fill CounterActor::handle todo!.\nPing→Pong; Inc bumps n and returns Count(n); Get returns Count(n).\nReplace todo!(\"Ping->Pong, Inc bumps, Get returns count\") with a match body only.\n",
            "CounterActor handle: match Ping/Inc/Get. Output the match expression only.\n",
            "CounterActor::handle todo!를 match로 채워. Ping→Pong, Inc/Get 처리. 본문만.\n",
            "Actor handle body — Ping/Inc/Get match. No fn wrapper.\n",
            "Replace the CounterActor todo! with match msg { ... }.\n",
            "Message handler body for Ping→Pong and counter Inc/Get.\n",
        ],
    ),
    (
        "f4_repair",
        "f4_repair/count_keys_v1",
        'todo!("HashMap insert + len — remember the import")',
        "{\n    let mut m = std::collections::HashMap::new();\n    for (k, v) in pairs {\n        m.insert(*k, *v);\n    }\n    m.len()\n}",
        [
            "Fill count_keys todo!. Insert pairs into a HashMap and return len.\nUse std::collections::HashMap (fully qualified is fine).\nReplace todo!(\"HashMap insert + len — remember the import\") — body only.\n",
            "count_keys body: HashMap insert each pair, return m.len(). Body only.\n",
            "count_keys todo!를 HashMap insert+len으로 채워. 본문만.\n",
            "Repair count_keys: build HashMap from pairs, return length.\n",
            "SLOT HashMap insert + len body.\n",
            "Implement count_keys todo! body with std::collections::HashMap.\n",
        ],
    ),
    (
        "f4_repair",
        "f4_repair/greet_v1",
        'todo!("return hi <name>")',
        'format!("hi {name}")',
        [
            "Fill greet todo!. Return format!(\"hi {name}\").\nReplace todo!(\"return hi <name>\") — expression body only.\n",
            "greet(name) should yield \"hi {name}\". Body only.\n",
            "greet todo!를 format!(\"hi {name}\")으로 채워. 표현식만.\n",
            "todo!(\"return hi <name\") → format!(\"hi {name}\")\n",
            "Greet body: hi plus name.\n",
            "Return a greeting string hi {name} from the todo! slot.\n",
        ],
    ),
    (
        "f4_repair",
        "f4_repair/green_v1",
        'todo!("repair: name for Green")',
        '"green"',
        [
            "Repair Color::Green arm: replace todo!(\"repair: name for Green\") with \"green\".\nOutput ONLY the match-arm expression.\n",
            "Green color name is the string green. Replace the todo! only.\n",
            "Color::Green 팔의 todo!를 \"green\"으로 고쳐. 표현식만.\n",
            "Match arm body for Green → \"green\".\n",
            "todo!(\"repair: name for Green\") → \"green\"\n",
            "Exhaustive color_name: Green arm is green.\n",
        ],
    ),
    (
        "f5_supervisor",
        "f5_supervisor/one_round_v1",
        'todo!("generate then verify each prompt; keep Correct only; record order")',
        "{\n    let mut out = RoundResult::default();\n    for p in prompts {\n        out.order.push(\"generate\");\n        let c = gen.generate(p);\n        out.order.push(\"verify\");\n        if ver.verify(p, &c) == Verdict::Correct {\n            out.kept.push(c);\n        }\n    }\n    out\n}",
        [
            "Fill one_round todo!. For each prompt: push \"generate\", generate, push \"verify\",\nkeep completion if Verdict::Correct. Return RoundResult.\nReplace the todo!(...) with the function body only (a block is fine).\n",
            "Wire generate then verify for one self-improve round; keep Correct only; record order.\nBody only for the one_round todo!.\n",
            "one_round todo!만 채워. generate→verify 순서, Correct만 kept.\n",
            "Supervisor one_round body: generate/verify order + filter Correct.\n",
            "Replace todo!(\"generate then verify...\") with the loop body.\n",
            "Implement one_round slot — keep Correct samples only.\n",
        ],
    ),
]


def main() -> None:
    rows = []
    for family, task_id, _needle, gold, phrasings in SLOTS:
        for i, ph in enumerate(phrasings):
            prompt = ph
            if f"// pekko-harvest-task: {task_id}" not in prompt:
                prompt = prompt.rstrip() + f"\n// pekko-harvest-task: {task_id}\n"
            # Several reps so SFT sees bodies often
            for rep in range(4):
                rows.append(
                    {
                        "prompt": prompt,
                        "completion": gold if gold.endswith("\n") else gold + "\n",
                        "task": task_id,
                        "family": family,
                        "rep": rep,
                        "ph": i,
                    }
                )

    # Shuffle with fixed seed so holdout isn't all one family
    rng = random.Random(24)
    rng.shuffle(rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} body pairs -> {OUT}")

    merged = []
    if F0.exists():
        for line in F0.read_text().splitlines():
            if line.strip():
                merged.append(json.loads(line))
    merged.extend(rows)
    rng.shuffle(merged)
    with MERGED.open("w") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(merged)} merged pairs -> {MERGED}")


if __name__ == "__main__":
    main()
