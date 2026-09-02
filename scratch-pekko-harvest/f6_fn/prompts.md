# F6 — function-level Rust (short golds)

Prompt shows crate API + failing tests. Completion is a **function body**
(~1 expression / few tokens) that replaces a single `todo!(...)` inside
`src/student.rs`. Gold lives in `src/reference.rs`. Sibling todos are
gold-filled at verify.

Challenges (distinct prompts / needles):
1. `shout` — `s.to_uppercase()`
2. `parse_kv` — `s.contains('=')` (bool)
3. `grade` — pass/fail: `if score >= 90 { "A" } else { "F" }`
4. `sum_evens` — `xs.iter().sum()` (tests use even-only slices)

F5 remains transfer-only / not harvested. F0–F4 stay as optional retention.
