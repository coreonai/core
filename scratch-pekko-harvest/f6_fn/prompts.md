# F6 — function-level Rust (short 1–3 line golds)

Prompt shows crate API + failing tests. Completion is a **function body**
(1–3 statements) that replaces a single `todo!(...)` inside
`src/student.rs`. Gold lives in `src/reference.rs`. Sibling todos are
gold-filled at verify.

Challenges (distinct prompts / needles):
1. `shout` — `s.trim().to_uppercase()`
2. `parse_kv` — `s.split_once('=').map(|(k,v)| (k.to_string(), v.to_string()))` → `Option<(String, String)>`
3. `grade` — A/P/F: `if score >= 90 { "A" } else if score >= 60 { "P" } else { "F" }`
4. `sum_evens` — `xs.iter().copied().filter(|x| x % 2 == 0).sum()` (mixed odds/evens)

F5 remains transfer-only / not harvested. F0–F4 stay as optional retention.
Not a general Rust coder.
