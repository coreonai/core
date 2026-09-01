#!/usr/bin/env python3
"""Augment F3/F5 (+ F1 upper, F4 greet) body-slot seeds and build v7 boost mix.

Writes:
  scripts/phase24/fmt_seed_pairs_f1f5_body.jsonl   (regenerated with more paraphrases)
  scripts/phase24/fmt_seed_pairs_f0f5_body.jsonl
  scripts/phase24/fmt_seed_pairs_v7_boost.jsonl    (weak-heavy + F0/F2 retention)
  scripts/phase24/fmt_seed_pairs_f3f5_probe.jsonl  (F3+F5 only for probe)
Also patches gen_fmt_seed_pairs_f1f5_body.py paraphrase lists in-place.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path("/raid/users/paul/workLLM")
GEN = ROOT / "scripts/phase24/gen_fmt_seed_pairs_f1f5_body.py"
OUT_F1F5 = ROOT / "scripts/phase24/fmt_seed_pairs_f1f5_body.jsonl"
OUT_MERGED = ROOT / "scripts/phase24/fmt_seed_pairs_f0f5_body.jsonl"
OUT_BOOST = ROOT / "scripts/phase24/fmt_seed_pairs_v7_boost.jsonl"
OUT_PROBE = ROOT / "scripts/phase24/fmt_seed_pairs_f3f5_probe.jsonl"
F0 = ROOT / "scripts/phase24/fmt_seed_pairs.jsonl"

# Extra paraphrases keyed by task_id (appended to existing lists in gen script).
EXTRA = {
    "f1_tool/upper_v1": [
        "UpperTool::execute — body only: Ok(args.to_ascii_uppercase()).\nReplace todo!(\"ASCII uppercase\").\n",
        "Fill the uppercase tool stub. Output just Ok(args.to_ascii_uppercase()).\n",
        "ASCII upper body for UpperTool execute slot. No wrappers.\n",
        "todo!(\"ASCII uppercase\") must become Ok(args.to_ascii_uppercase()).\nBody expression only.\n",
        "upper 툴 execute 본문: args를 to_ascii_uppercase 해서 Ok로.\n",
        "Student slot UpperTool: return ASCII-uppercased args.\n",
        "One-liner body: Ok(args.to_ascii_uppercase()) for the upper tool todo!.\n",
        "Replace UpperTool todo! with the Ok(to_ascii_uppercase) expression.\n",
    ],
    "f3_message/handle_v1": [
        "CounterActor::handle body only — match on Message::{Ping,Inc,Get}.\nPing=>Pong; Inc bumps self.n then Count(n); Get=>Count(n).\n",
        "Write the match msg { ... } that implements Ping→Pong / Inc / Get.\nNo fn/impl wrappers — match expression only.\n",
        "Handler slot: respond to Ping with Pong; Inc increments; Get returns count.\nOutput only the match body for the todo!.\n",
        "todo!(\"Ping->Pong, Inc bumps, Get returns count\") → full match expression.\n",
        "Actor message dispatch body. Match Ping/Inc/Get exactly as the gold handler.\n",
        "메시지 핸들러 본문만: Ping→Pong, Inc는 n+=1 후 Count, Get은 Count.\nmatch만 출력.\n",
        "Fill CounterActor handle todo! with an exhaustive match on Message.\n",
        "Body for handle(&mut self, msg: Message) -> Response — match only.\n",
        "pekko CounterActor: implement Ping/Inc/Get arms; leave surrounding code alone.\n",
        "Replace the handle todo! with:\nmatch msg { Message::Ping => Response::Pong, ... }\n",
        "Self-improve student.rs slot for f3_message — gold is the match expression.\n",
        "Inc arm must do self.n += 1 then Response::Count(self.n). Ping→Pong. Get→Count.\n",
    ],
    "f4_repair/greet_v1": [
        "greet todo! body: format!(\"hi {name}\") — expression only.\n",
        "Return hi {name} from greet. Replace todo!(\"return hi <name>\").\n",
        "greet 슬롯을 format!(\"hi {name}\")으로 채워. 표현식만.\n",
        "One-expression body for greet: format!(\"hi {name}\").\n",
        "SLOT greet → format!(\"hi {name}\")\n",
        "Fix greet stub so it formats a hi greeting with name.\n",
        "Student body for f4_repair/greet_v1: format!(\"hi {name}\").\n",
        "todo!(\"return hi <name>\") becomes format!(\"hi {name}\").\n",
    ],
    "f5_supervisor/one_round_v1": [
        "one_round body: for each prompt push \"generate\", gen.generate, push \"verify\",\nif Correct then kept.push. Return RoundResult block.\n",
        "Implement the supervisor one_round todo! as a block that loops prompts.\nOrder must be generate then verify; keep only Verdict::Correct.\n",
        "Fill one_round slot — RoundResult with order + kept. Body block only.\n",
        "todo!(\"generate then verify each prompt; keep Correct only; record order\")\n→ loop body that records order and filters Correct.\n",
        "Self-improve round without training: Gen then Verify; record order strings.\n",
        "Supervisor topology body: generate/verify alternating; push Correct into kept.\n",
        "one_round 본문만: prompts 루프, order에 generate/verify, Correct만 kept.\n",
        "Wire Generator then Verifier once per prompt; return RoundResult.\nNo fn signature — function body block only.\n",
        "Student.rs f5_supervisor slot: implement one_round loop body.\n",
        "Keep Correct completions only; push \"generate\" then \"verify\" into out.order.\n",
        "RoundResult::default(); for p in prompts { ... } ; out\nFill the todo! with that pattern.\n",
        "Harvest family f5: gold is the generate→verify RoundResult block.\n",
    ],
}

# Golds / needles / existing paraphrases mirrored from gen script (canonical).
SLOTS = [
    (
        "f1_tool",
        "f1_tool/echo_v1",
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


def with_marker(prompt: str, task_id: str) -> str:
    if f"// pekko-harvest-task: {task_id}" not in prompt:
        return prompt.rstrip() + f"\n// pekko-harvest-task: {task_id}\n"
    return prompt if prompt.endswith("\n") else prompt + "\n"


def build_rows(reps: int = 4):
    rows = []
    for family, task_id, gold, base_ph in SLOTS:
        phrasings = list(base_ph) + list(EXTRA.get(task_id, []))
        # de-dupe while preserving order
        seen = set()
        uniq = []
        for ph in phrasings:
            if ph not in seen:
                seen.add(ph)
                uniq.append(ph)
        for i, ph in enumerate(uniq):
            prompt = with_marker(ph, task_id)
            for rep in range(reps):
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
    return rows


def patch_gen_script() -> None:
    """Append EXTRA paraphrases into the gen script's list literals for weak tasks."""
    if not GEN.exists():
        print(f"skip patch: {GEN} missing")
        return
    text = GEN.read_text()
    # For each task, find the paraphrases list after task_id and inject missing strings.
    for task_id, extras in EXTRA.items():
        # locate the task_id occurrence then the next [ ... ], that is paraphrases
        m = re.search(
            rf'("{re.escape(task_id)}",\s*\n\s*\'[^\']+\',\s*\n\s*"[^"]*",\s*\n\s*)(\[.*?\n\s*\],)',
            text,
            flags=re.S,
        )
        if not m:
            # try gold that starts with { or match
            m = re.search(
                rf'("{re.escape(task_id)}",\s*\n\s*\'[^\']+\',\s*\n\s*(?:"[^"]*"|""".*?"""|\'[^\']*\'),\s*\n\s*)(\[.*?\n\s*\],)',
                text,
                flags=re.S,
            )
        if not m:
            print(f"WARN: could not locate paraphrases for {task_id} in gen script")
            continue
        prefix, bracket = m.group(1), m.group(2)
        # parse existing string literals roughly
        existing = set(re.findall(r'"((?:\\.|[^"\\])*)"', bracket))
        # also handle multi-line via the raw block — simpler: check if extra already present as substring
        to_add = []
        for ex in extras:
            # escape for search
            if ex[:40] in bracket or json.dumps(ex)[1:-1][:40] in bracket:
                continue
            to_add.append(ex)
        if not to_add:
            print(f"gen script already has extras for {task_id}")
            continue
        insert = "".join("            " + json.dumps(ex, ensure_ascii=False) + ",\n" for ex in to_add)
        # insert before closing ]
        new_bracket = bracket.rstrip()
        # new_bracket ends with ],
        assert new_bracket.endswith("],"), new_bracket[-20:]
        new_bracket = new_bracket[:-2] + "\n" + insert + "        ],"
        text = text[: m.start()] + prefix + new_bracket + text[m.end() :]
        print(f"patched gen script: +{len(to_add)} paraphrases for {task_id}")
    GEN.write_text(text)


def main() -> None:
    patch_gen_script()

    rows = build_rows(reps=4)
    rng = random.Random(24)
    rng.shuffle(rows)
    OUT_F1F5.parent.mkdir(parents=True, exist_ok=True)
    with OUT_F1F5.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} -> {OUT_F1F5}")
    print("by family:", dict(Counter(r["family"] for r in rows)))
    print("by task:", dict(Counter(r["task"] for r in rows)))

    merged = []
    if F0.exists():
        for line in F0.read_text().splitlines():
            if line.strip():
                o = json.loads(line)
                if "family" not in o:
                    o["family"] = "f0_expr"
                merged.append(o)
    merged.extend(rows)
    rng.shuffle(merged)
    with OUT_MERGED.open("w") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(merged)} -> {OUT_MERGED}")

    # Boost mix for continue-SFT: overweight F3/F5/upper/greet, keep some F0+F2 (+a little F1 echo/ping, F4 green/count)
    weak_tasks = {
        "f3_message/handle_v1",
        "f5_supervisor/one_round_v1",
        "f1_tool/upper_v1",
        "f4_repair/greet_v1",
    }
    retain_fams = {"f0_expr", "f2_domain"}
    light_tasks = {
        "f1_tool/echo_v1",
        "f1_tool/ping_v1",
        "f4_repair/green_v1",
        "f4_repair/count_keys_v1",
    }

    # Unique (prompt,completion,task,family) from rows (drop rep)
    uniq = {}
    for r in rows:
        key = (r["task"], r["ph"])
        uniq[key] = {k: r[k] for k in ("prompt", "completion", "task", "family", "ph")}

    boost = []
    # Weak: 8 reps each unique paraphrase
    for u in uniq.values():
        if u["task"] in weak_tasks:
            for rep in range(8):
                boost.append({**u, "rep": rep, "mix": "weak"})
    # F2 retention: 2 reps
    for u in uniq.values():
        if u["family"] == "f2_domain":
            for rep in range(2):
                boost.append({**u, "rep": rep, "mix": "retain_f2"})
    # Light other body slots: 1 rep
    for u in uniq.values():
        if u["task"] in light_tasks:
            boost.append({**u, "rep": 0, "mix": "light"})
    # F0 retention: take up to 80 F0 pairs
    f0_pool = [o for o in merged if o.get("family") in ("f0_expr", "unknown") or o.get("family") == "f0_expr"]
    # prefer explicit f0_expr
    f0_pool = [o for o in merged if o.get("family") == "f0_expr"]
    rng2 = random.Random(7)
    rng2.shuffle(f0_pool)
    for o in f0_pool[:80]:
        boost.append(
            {
                "prompt": o["prompt"],
                "completion": o["completion"] if o["completion"].endswith("\n") else o["completion"] + "\n",
                "task": o.get("task", "f0_expr"),
                "family": "f0_expr",
                "rep": 0,
                "mix": "retain_f0",
            }
        )

    rng2.shuffle(boost)
    # Holdout-first: put ~15% aside at front by stratified-ish shuffle already random
    with OUT_BOOST.open("w") as f:
        for r in boost:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(boost)} -> {OUT_BOOST}")
    print("boost by family:", dict(Counter(r["family"] for r in boost)))
    print("boost by mix:", dict(Counter(r.get("mix", "?") for r in boost)))

    # F3+F5 probe file: unique paraphrases, holdout-friendly order (shuffle then write)
    probe = []
    for u in uniq.values():
        if u["family"] in ("f3_message", "f5_supervisor"):
            probe.append({**u, "rep": 0})
    rng3 = random.Random(42)
    rng3.shuffle(probe)
    with OUT_PROBE.open("w") as f:
        for r in probe:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(probe)} -> {OUT_PROBE} families={dict(Counter(r['family'] for r in probe))}")


if __name__ == "__main__":
    main()
