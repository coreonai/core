# scratch-pekko-harvest

Phase 24 scaffolds for Pekko/MSA agent harvest families (see
`../docs/phase24-pekko-agent-harvest.md`).

Each crate:

- `prompts.md` — phrasing variants to sample (the training unit)
- `src/reference.rs` — known-good solution (tests pass against this)
- `src/student.rs` — incomplete stub the model should fill
- default tests compile/run against **reference** so the skeleton stays green
- `cargo test -p <crate> --features student` runs tests against **student** (expect fail until filled)

```bash
cd scratch-pekko-harvest
cargo test --workspace
cargo test -p f1_tool --features student   # should fail on stubs
```
