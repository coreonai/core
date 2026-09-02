# F6 — function-level Rust (not one-line todo, not a full crate)

Prompt shows crate API + failing tests. Completion is a **function body**
(tens of tokens) that replaces a single `todo!(...)` inside `src/student.rs`.
Gold lives in `src/reference.rs`. Sibling todos are gold-filled at verify.

Challenges (distinct prompts / needles):
1. `shout` — echo-like: trim, ASCII uppercase, bang (empty stays empty)
2. `parse_kv` — parse `k=v` into Option<(String, String)>
3. `grade` — small match on score → letter
4. `sum_evens` — iterator / loop over i32 slice

F5 remains transfer-only / not harvested. F0–F4 stay as optional retention.
