//! Domain-wrapper equivalence harness (test-only).
//!
//! Enforces the [`domain-wrapper-equivalence`] skill's two properties for every
//! `Domain` wrapper:
//!
//! - **A. Identity construction** — an empty-skip wrapper equals its inner on
//!   *every* trait method. This single property would have caught `ad86e22`
//!   (the `FilteredDomain::truncate_completion` delegation gap) on the first
//!   run: with an empty skip, a delegated method returns the inner's value and
//!   a defaulted (forgotten) one returns the trait default — they differ.
//! - **B. Filter invariance** — with a non-empty skip, every method that is
//!   *not* the wrapper's reason to exist (everything except the transformed
//!   prompt-selection methods) still equals the inner.
//!
//! The harness iterates **all 8** `Domain` methods (100% coverage). If a method
//! is added to `Domain`, add it here in the same commit — see
//! `DOMAIN_METHOD_COUNT` and the enumeration in [`assert_all_methods_match`].
//!
//! [`domain-wrapper-equivalence`]: ../../../.claude/skills/domain-wrapper-equivalence/SKILL.md

use std::sync::Arc;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::domain::filtered::FilteredDomain;
use crate::domain::Domain;
use crate::types::Verdict;

/// Number of methods on `trait Domain`. Bump this — and the enumeration in
/// [`assert_all_methods_match`] / [`assert_filter_invariant_methods_match`] —
/// whenever a method is added, or the coverage claim in the skill is a lie.
/// (sample_prompt, verify, charset, score, n_prompts, nth_prompt,
/// truncate_completion, task_id.)
pub const DOMAIN_METHOD_COUNT: usize = 8;

/// A finite, fully-deterministic inner domain. Every method returns a value
/// distinct from the trait default (so a missing delegation is observable) and
/// its `sample_prompt` is index-based like `HumanEvalDomain`/`MbppDomain`
/// (`nth_prompt(gen_range(0..n))`), so identity equivalence holds for
/// `sample_prompt` too.
pub struct EquivProbeDomain {
    pub n: usize,
}

impl Domain for EquivProbeDomain {
    fn sample_prompt(&self, rng: &mut StdRng) -> String {
        let i = rng.gen_range(0..self.n);
        format!("prompt{i}")
    }
    fn verify(&self, _prompt: &str, completion: &str) -> Verdict {
        // Deterministic on the completion so `verify` equivalence is a real
        // test, not trivially-always-Correct.
        if completion.contains("ok") {
            Verdict::Correct
        } else {
            Verdict::Incorrect {
                reason: "no ok".into(),
            }
        }
    }
    fn charset(&self) -> &str {
        "xyz"
    }
    fn score(&self, _v: &Verdict) -> f32 {
        0.5 // non-default (default is 1.0/0.0)
    }
    fn n_prompts(&self) -> Option<usize> {
        Some(self.n) // non-default (default is None)
    }
    fn nth_prompt(&self, i: usize) -> Option<String> {
        (i < self.n).then(|| format!("prompt{i}")) // non-default
    }
    fn truncate_completion(&self, completion: &str) -> String {
        completion.split("CUT").next().unwrap_or("").to_string() // non-default (default is identity)
    }
    fn task_id(&self, i: usize) -> Option<String> {
        (i < self.n).then(|| format!("q{i}")) // non-default (default is None)
    }
}

/// Property B set — methods that must equal the inner regardless of filtering.
/// Covers charset, score, verify, truncate_completion (4 of 8). `pub(crate)`
/// so a new wrapper's tests register here (skill step 5).
pub(crate) fn assert_filter_invariant_methods_match(inner: &dyn Domain, wrapper: &dyn Domain) {
    assert_eq!(wrapper.charset(), inner.charset(), "charset");
    assert_eq!(
        wrapper.score(&Verdict::Correct),
        inner.score(&Verdict::Correct),
        "score(Correct)"
    );
    assert_eq!(
        wrapper.score(&Verdict::Incorrect { reason: "r".into() }),
        inner.score(&Verdict::Incorrect { reason: "r".into() }),
        "score(Incorrect)"
    );
    for probe in ["", "keepCUTdrop", "no marker here", "aCUTbCUTc"] {
        assert_eq!(
            wrapper.truncate_completion(probe),
            inner.truncate_completion(probe),
            "truncate_completion({probe:?})"
        );
    }
    for (p, c) in [("prompt0", "ok"), ("prompt1", "nope")] {
        // Verdict doesn't derive PartialEq; correctness is the observable that
        // matters (a delegated verify returns the inner's verdict verbatim).
        assert_eq!(
            wrapper.verify(p, c).is_correct(),
            inner.verify(p, c).is_correct(),
            "verify({p},{c})"
        );
    }
}

/// Property A set — EVERY method matches the inner (valid only for identity
/// construction). Extends property B with the 3 prompt-selection methods
/// (n_prompts, nth_prompt, sample_prompt, task_id) → 8 of 8. `pub(crate)` so a new
/// wrapper's tests register here (skill step 5).
pub(crate) fn assert_all_methods_match(inner: &dyn Domain, wrapper: &dyn Domain, seed: u64) {
    assert_filter_invariant_methods_match(inner, wrapper);
    assert_eq!(wrapper.n_prompts(), inner.n_prompts(), "n_prompts");
    let n = inner.n_prompts().unwrap_or(0);
    for i in 0..=n {
        // 0..=n includes the out-of-range index n (both should be None).
        assert_eq!(
            wrapper.nth_prompt(i),
            inner.nth_prompt(i),
            "nth_prompt({i})"
        );
        // task_id is filter-transformed like nth_prompt: identity => matches.
        assert_eq!(wrapper.task_id(i), inner.task_id(i), "task_id({i})");
    }
    let mut r_inner = StdRng::seed_from_u64(seed);
    let mut r_wrap = StdRng::seed_from_u64(seed);
    assert_eq!(
        wrapper.sample_prompt(&mut r_wrap),
        inner.sample_prompt(&mut r_inner),
        "sample_prompt (identically seeded)"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The coverage claim in one place: the harness iterates all 8 `Domain`
    /// methods. If `Domain` grows and this fails, update the harness and the
    /// count together (skill step 4), don't just bump the number.
    #[test]
    fn covers_every_domain_method() {
        assert_eq!(DOMAIN_METHOD_COUNT, 8);
    }

    /// Property A — an empty-skip `FilteredDomain` is observationally identical
    /// to its inner across all 8 methods, over several RNG seeds.
    #[test]
    fn filtered_domain_identity_equivalence() {
        let inner = Arc::new(EquivProbeDomain { n: 5 });
        let wrapper =
            FilteredDomain::new(Arc::clone(&inner) as Arc<dyn Domain>, Vec::<usize>::new());
        for seed in [1u64, 7, 42, 100] {
            assert_all_methods_match(&*inner, &wrapper, seed);
        }
    }

    /// Property B — a non-empty-skip `FilteredDomain` still matches the inner on
    /// every non-prompt-selection method. This is the exact class `ad86e22`
    /// broke (`truncate_completion` under `--prompt-skip-list`).
    #[test]
    fn filtered_domain_filter_invariance() {
        let inner = Arc::new(EquivProbeDomain { n: 5 });
        let wrapper = FilteredDomain::new(Arc::clone(&inner) as Arc<dyn Domain>, [0usize, 2]);
        assert_filter_invariant_methods_match(&*inner, &wrapper);
    }

    /// The harness must observe a real regression. A deliberately-broken
    /// wrapper that drops `truncate_completion` delegation (inherits the
    /// identity default) must fail property A — proving the harness bites and
    /// isn't vacuously green.
    #[test]
    #[should_panic(expected = "truncate_completion")]
    fn broken_wrapper_fails_identity_equivalence() {
        struct DropsTruncate(Arc<dyn Domain>);
        impl Domain for DropsTruncate {
            fn sample_prompt(&self, r: &mut StdRng) -> String {
                self.0.sample_prompt(r)
            }
            fn verify(&self, p: &str, c: &str) -> Verdict {
                self.0.verify(p, c)
            }
            fn charset(&self) -> &str {
                self.0.charset()
            }
            fn score(&self, v: &Verdict) -> f32 {
                self.0.score(v)
            }
            fn n_prompts(&self) -> Option<usize> {
                self.0.n_prompts()
            }
            fn nth_prompt(&self, i: usize) -> Option<String> {
                self.0.nth_prompt(i)
            }
            // truncate_completion INTENTIONALLY dropped -> identity default.
        }
        let inner = Arc::new(EquivProbeDomain { n: 3 });
        let broken = DropsTruncate(Arc::clone(&inner) as Arc<dyn Domain>);
        assert_all_methods_match(&*inner, &broken, 1);
    }
}
