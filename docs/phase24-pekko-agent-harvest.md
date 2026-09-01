# Phase 24 (draft) — Pekko / MSA agent harvest families

Status: **plan** — first six cargo-graded task families for a specialized
model on this repo's Pekko agent stack (`llm-actors` + path-dep `pekko-actor`).

## Principle (from Phase 23)

Do **not** train "knows Rust" or "knows Pekko". Train **phrase-level task
templates** with a free verifier. Here the verifier is `cargo check` /
`cargo test` / a tiny scratch binary — the analogue of Phase 23's python
exec.

Vary **phrasing and surface form** inside each family, or the model will
only fire on one prompt wording (Phase 23 digit-product / numpy finding).

## Scope of "the domain"

In: actor messages, `Domain` / `Tool` contracts, supervisor one-round
wiring, compile/test repair on `llm-actors` patterns.
Out (for v1): open-ended MSA design, novel crate ecosystems, LCB/BCB.

## Shared harness

| piece | role |
|---|---|
| `RustCodeDomain` (extend) or new `PekkoAgentDomain` | `sample_prompt` / `verify` / optional `repair_prompt` |
| scratch crate per family under `scratch-pekko-harvest/fN/` | isolated `cargo check`/`test` — never the main workspace |
| gold = `cargo test` green **or** exact stdout / trait object smoke | free verifier |
| harvest rule | store **repaired final code** with the **original** prompt (Phase 23) |

`--init-dir` starts from a format-SFT'd code checkpoint (not raw base), same
bootstrap lesson as Phase 23.

## Family 0 — expression slot (baseline / retention)

**Why:** already exists as `DEFAULT_CHALLENGES` in `domain/rust_code.rs`.
Keeps a ruler for "did we destroy basic Rust fill-in?"

| | |
|---|---|
| Prompt shape | `fn main() { assert_eq!(` / `let x: i32 = ` / … |
| Target | expression that makes `cargo run` succeed |
| Verifier | existing `RustCodeDomain` |
| Phrasing variants | keep ≥10 distinct prompt prefixes (already the rule) |

## Family 1 — `Tool` stub that registers and dispatches

**Why:** core MSA agent boundary in `tools/mod.rs`.

| | |
|---|---|
| Prompt | "Implement tool `echo` that returns its args" / "add a Tool named ping" / Korean+English paraphrases |
| Skeleton | trait + `ToolRegistry::from_tools` smoke in scratch |
| Gold | `registry.dispatch(&ToolCall{name, args}) == Ok(expected)` under `cargo test` |
| Failure modes to harvest | wrong `name()`, BadArgs, forgetting `Send+Sync` |

## Family 2 — minimal `Domain` impl

**Why:** every self-improve loop is `Domain`-shaped (`sample_prompt`/`verify`/`charset`).

| | |
|---|---|
| Prompt | "Domain that accepts only completion `ok`" / "always-correct toy domain" / "charset must include digits" |
| Gold | `verify` returns Correct iff completion matches contract; `cargo test` |
| Explicit anti-pattern | wrapper that forgets `repair_prompt` / `task_id` (see `delegation_probe`) — optional later family |

## Family 3 — actor `Message` enum + one handler arm

**Why:** Pekko/MSA unit of work is message → behavior.

| | |
|---|---|
| Prompt | given a stub actor, "handle `Ping` with `Pong`" / "add Restart to Message" |
| Skeleton | `enum Message { Ping, … }` + `match` in `handle` |
| Gold | unit test sends Ping, expects Pong (or state bump) via existing test harness style in scratch |
| Phrasing variants | different enum/method names so it is not one template |

## Family 4 — cargo stderr repair (compile loop)

**Why:** Phase 23 `repair_prompt` analogue — capability that sampling never finds.

| | |
|---|---|
| Prompt | broken snippet + **verbatim** `cargo check` stderr tail |
| Gold | patched file `cargo check` clean |
| Seed bugs | missing `use`, wrong type, non-exhaustive match, `String` vs `&str` at actor boundary |
| Harvest | repaired source paired with **original broken prompt**, not the stderr turn |

## Family 5 — one-round supervisor wiring smoke

**Why:** product-shaped MSA: Gen→Verify→Curate (minimal) without full training.

| | |
|---|---|
| Prompt | "wire Verifier after Generator for domain X" / "Supervisor runs one round and returns eval score" |
| Skeleton | fake Generator/Verifier with trait objects or channel stubs in scratch |
| Gold | `cargo test` asserts order and that Correct samples reach curator buffer |
| Keep tiny | no CUDA, no real model — topology only |

## Transfer / retention probes (do not harvest)

| probe | measures |
|---|---|
| Novel Tool name + different grammar wording | template vs rule (Phase 23 Collatz/`itertools`) |
| Domain wrapper missing one delegated method | silent-failure surface they already document |
| Message handler for untrained variant | enum exhaustiveness generalization |

## Scaffold location

Created at `scratch-pekko-harvest/` (workspace members `f0_expr` … `f5_supervisor`).
Each crate has `prompts.md`, `reference.rs` (tests pass), `student.rs` (`--features student`).

```bash
cd scratch-pekko-harvest && cargo test --workspace
```

## Rollout order

1. Extend challenges / add `PekkoAgentDomain` scratch layout under `scratch-pekko-harvest/`
2. Baseline pass@1 per family on format-SFT checkpoint (`--baseline`)
3. Self-improve with `--harvest-repair` on F1–F4 first (F0 retention, F5 last — denser)
4. Measure three axes: targets / retention(F0) / transfer probes
5. Only then consider merging into main `RustCodeDomain` or a phase24 example binary

## Non-goals for this draft

- Beating LiveCodeBench
- Replacing Claude/Cursor for open design
- Training on the full `workLLM` tree (too much contamination, too slow to verify)

## Success bar (v1)

- F1–F4 target families → ≥0.8 pass@1 after ≤2 rounds
- F0 retention drop <0.1 vs pre-loop
- Transfer probe: at least one **rephrased** Tool/Domain task succeeds without that exact harvest phrasing

