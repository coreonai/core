# F4 — cargo stderr repair

Each task pairs a broken snippet with a stderr tail. Harvest the **fixed
source** with the **original broken prompt** (Phase 23 rule).

Seed bugs:
1. missing `use std::collections::HashMap;`
2. `String` passed where `&str` expected at a toy actor boundary
3. non-exhaustive match after adding enum variant
4. wrong type in `assert_eq!` (i32 vs usize)
5. Paraphrase: "fix this compile error" + paste stderr
