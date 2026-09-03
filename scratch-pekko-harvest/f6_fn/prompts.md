# F6 — function-level Rust (curriculum: only parse_kv grows)

Prompt shows crate API + failing tests. Completion is a **function body**
that replaces a single `todo!(...)` inside `src/student.rs`. Gold lives in
`src/reference.rs`. Sibling todos are gold-filled at verify.

Challenges (distinct prompts / needles):
1. `shout` — v11 one-liner `s.to_uppercase()` (no trim)
2. `parse_kv` — ONLY this one grows: `s.split_once('=').map(|(k, v)| (k.to_string(), v.to_string()))` → `Option<(String, String)>`
3. `grade` — keep v13 A/P/F `if score >= 90 { "A" } else if score >= 60 { "P" } else { "F" }`
4. `sum_evens` — v11 one-liner `xs.iter().sum()` (tests use even-only slices)

F5 remains transfer-only / not harvested. F0–F4 stay as optional retention.
Not a general Rust coder.
