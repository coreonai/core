// Domain-wrapper equivalence harness — TEMPLATE.
//
// Copy this pattern into the crate as a `#[cfg(test)]` module (in workLLM it
// lives at `llm-actors/src/domain/equivalence.rs`) and adapt it to the CURRENT
// `trait Domain` signature. This file is a reference, not compiled.
//
// Two properties, both mandatory (SKILL.md step 3):
//   A. Identity construction  — an empty-skip wrapper == inner on EVERY method.
//   B. Filter invariance       — a non-empty-skip wrapper == inner on every
//                                method that is NOT the wrapper's reason to
//                                exist (i.e. everything except the transformed
//                                prompt-selection methods).
//
// The harness MUST iterate every trait method. If you add a `Domain` method,
// add it here in the same commit (SKILL.md step 4).

use std::sync::Arc;

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::domain::Domain;
use crate::types::Verdict;

/// A finite, fully-deterministic inner domain whose every method returns a
/// value distinct from the trait default AND from the other methods, so any
/// missing delegation or accidental transform is observable. Its
/// `sample_prompt` is index-based (`nth_prompt(gen_range(0..n))`) so it matches
/// the standard finite-domain sampling contract (`HumanEvalDomain`,
/// `MbppDomain`), letting identity equivalence hold for `sample_prompt` too.
pub struct EquivProbeDomain {
    pub n: usize,
}

impl Domain for EquivProbeDomain {
    fn sample_prompt(&self, rng: &mut StdRng) -> String {
        use rand::Rng;
        let i = rng.gen_range(0..self.n);
        format!("prompt{i}")
    }
    fn verify(&self, _prompt: &str, completion: &str) -> Verdict {
        // Deterministic on the completion so `verify` equivalence is a real
        // test, not trivially-always-Correct.
        if completion.contains("ok") {
            Verdict::Correct
        } else {
            Verdict::Incorrect { reason: "no ok".into() }
        }
    }
    fn charset(&self) -> &str {
        "xyz"
    }
    fn score(&self, _v: &Verdict) -> f32 {
        0.5 // non-default
    }
    fn n_prompts(&self) -> Option<usize> {
        Some(self.n)
    }
    fn nth_prompt(&self, i: usize) -> Option<String> {
        (i < self.n).then(|| format!("prompt{i}"))
    }
    fn truncate_completion(&self, completion: &str) -> String {
        completion.split("CUT").next().unwrap_or("").to_string()
    }
    fn task_id(&self, i: usize) -> Option<String> {
        (i < self.n).then(|| format!("q{i}"))
    }
}

/// Methods that must equal the inner REGARDLESS of filtering (property B).
/// Every non-prompt-selection method goes here.
fn assert_filter_invariant_methods_match(inner: &dyn Domain, wrapper: &dyn Domain) {
    assert_eq!(wrapper.charset(), inner.charset(), "charset");
    assert_eq!(
        wrapper.score(&Verdict::Correct),
        inner.score(&Verdict::Correct),
        "score(Correct)"
    );
    for probe in ["", "keepCUTdrop", "no marker here", "aCUTbCUTc"] {
        assert_eq!(
            wrapper.truncate_completion(probe),
            inner.truncate_completion(probe),
            "truncate_completion({probe:?})"
        );
    }
    for (p, c) in [("prompt0", "ok"), ("prompt1", "nope")] {
        assert_eq!(wrapper.verify(p, c), inner.verify(p, c), "verify({p},{c})");
    }
}

/// EVERY method matches the inner (property A) — valid only for identity
/// construction (empty skip). Extends the filter-invariant set with the
/// prompt-selection methods.
fn assert_all_methods_match(inner: &dyn Domain, wrapper: &dyn Domain, seed: u64) {
    assert_filter_invariant_methods_match(inner, wrapper);
    assert_eq!(wrapper.n_prompts(), inner.n_prompts(), "n_prompts");
    let n = inner.n_prompts().unwrap_or(0);
    for i in 0..=n {
        // include the out-of-range index n
        assert_eq!(wrapper.nth_prompt(i), inner.nth_prompt(i), "nth_prompt({i})");
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

#[test]
fn filtered_domain_identity_equivalence() {
    let inner = Arc::new(EquivProbeDomain { n: 5 });
    let wrapper = FilteredDomain::new(Arc::clone(&inner) as Arc<dyn Domain>, Vec::<usize>::new());
    for seed in [1, 7, 42] {
        assert_all_methods_match(&*inner, &wrapper, seed);
    }
}

#[test]
fn filtered_domain_filter_invariance() {
    let inner = Arc::new(EquivProbeDomain { n: 5 });
    let wrapper = FilteredDomain::new(Arc::clone(&inner) as Arc<dyn Domain>, [0usize, 2]);
    assert_filter_invariant_methods_match(&*inner, &wrapper);
}
