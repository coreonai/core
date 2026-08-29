//! Phase 22 Stage D follow-up — a `Domain` wrapper that hides a
//! caller-specified set of underlying prompt indices.
//!
//! Use case: Phase 9 S5 cold-start mitigation — exclude HumanEval/MBPP
//! prompts that the base model has 0/k pass-rate on, so they don't
//! dominate the empty-corpus skip path on multi-round runs. The
//! wrapper exposes only the surviving subset; `sample_prompt` and
//! `nth_prompt` re-index into the kept set so callers don't have to
//! know about the filter.
//!
//! Selection-bias warning: the filtered subset is not representative
//! of the full benchmark. Use it for *training* convenience, not for
//! benchmark-aligned eval. The Stage B aggregate measurement
//! (`phase22_humaneval_baseline --sequential --aggregate`) should
//! always be run against the unfiltered Domain.

use std::collections::HashSet;
use std::sync::Arc;

use rand::rngs::StdRng;
use rand::Rng;

use crate::domain::Domain;
use crate::types::Verdict;

/// Wraps an inner `Domain` and exposes only the prompt indices NOT in
/// `skip`. `sample_prompt` and `nth_prompt` operate over the kept set
/// (so `nth_prompt(0)` of the wrapper is the first non-skipped index
/// of the inner, etc.). `verify` delegates straight through — the
/// wrapper doesn't transform prompt strings, so the inner's prompt
/// lookup tables remain valid.
pub struct FilteredDomain {
    inner: Arc<dyn Domain>,
    /// Sorted list of inner indices that survived the filter. These
    /// are the indices we expose to the outside world (re-numbered
    /// 0..valid_indices.len()).
    valid_indices: Vec<usize>,
}

impl FilteredDomain {
    /// Build a filtered view that hides the `skip` indices. If the
    /// inner is infinite (`n_prompts() == None`) the wrapper is a
    /// no-op for `sample_prompt` (delegates to inner) and `nth_prompt`
    /// (returns None as before). Otherwise the wrapper materializes
    /// the surviving index list at construction.
    pub fn new(inner: Arc<dyn Domain>, skip: impl IntoIterator<Item = usize>) -> Self {
        let skip_set: HashSet<usize> = skip.into_iter().collect();
        let valid_indices: Vec<usize> = match inner.n_prompts() {
            Some(n) => (0..n).filter(|i| !skip_set.contains(i)).collect(),
            None => Vec::new(),
        };
        Self {
            inner,
            valid_indices,
        }
    }

    /// How many prompts survived the filter. Equivalent to
    /// `n_prompts().unwrap_or(0)` but cheaper.
    pub fn n_surviving(&self) -> usize {
        self.valid_indices.len()
    }
}

impl Domain for FilteredDomain {
    fn sample_prompt(&self, rng: &mut StdRng) -> String {
        // If inner is infinite, fall back to the inner's sampler.
        if self.valid_indices.is_empty() && self.inner.n_prompts().is_none() {
            return self.inner.sample_prompt(rng);
        }
        // Otherwise pick a random index from the surviving set.
        // valid_indices is guaranteed non-empty here (we'd have nothing
        // to sample if it were empty AND the inner was finite).
        debug_assert!(
            !self.valid_indices.is_empty(),
            "FilteredDomain has 0 surviving prompts"
        );
        let i = rng.gen_range(0..self.valid_indices.len());
        self.inner
            .nth_prompt(self.valid_indices[i])
            .expect("inner.nth_prompt should succeed on valid_indices entries")
    }

    fn verify(&self, prompt: &str, completion: &str) -> Verdict {
        self.inner.verify(prompt, completion)
    }

    /// Delegated for the same reason as `truncate_completion` below: a
    /// wrapper that inherits a trait default silently overrides whatever the
    /// inner domain meant. No domain overrides `score` today, so this is
    /// currently a no-op — it exists so the next one to do so isn't quietly
    /// ignored the way `truncate_completion` was.
    fn score(&self, verdict: &Verdict) -> f32 {
        self.inner.score(verdict)
    }

    fn charset(&self) -> &str {
        self.inner.charset()
    }

    fn n_prompts(&self) -> Option<usize> {
        if self.valid_indices.is_empty() && self.inner.n_prompts().is_none() {
            None
        } else {
            Some(self.valid_indices.len())
        }
    }

    fn nth_prompt(&self, i: usize) -> Option<String> {
        let &inner_idx = self.valid_indices.get(i)?;
        self.inner.nth_prompt(inner_idx)
    }

    // override: re-index into the kept set, exactly like `nth_prompt` — the
    // exported `question_id` must be the inner's id for the surviving prompt,
    // not the renumbered wrapper index. (A straight delegation would return the
    // wrong problem's id; the identity default would return None and silently
    // drop ids from every filtered export.)
    fn task_id(&self, i: usize) -> Option<String> {
        let &inner_idx = self.valid_indices.get(i)?;
        self.inner.task_id(inner_idx)
    }

    /// Phase 22 C5 follow-up — **must** delegate. This wrapper previously
    /// inherited the trait's identity default, so wrapping a code domain
    /// silently switched OFF `truncate_python_completion` at every
    /// generate/verify site. Every hard-tail experiment ran through
    /// `--prompt-skip-list`, so all of them scored completions raw: a
    /// correct function followed by a trailing top-level `def` counted as
    /// wrong. That is what made the hard-tail base measure 0.246 pass@5
    /// where the unfiltered path measures 0.422, and it inflated the
    /// reported SFT gain — the base is penalised harder than a trained
    /// checkpoint, because training teaches the model to stop.
    fn truncate_completion(&self, completion: &str) -> String {
        self.inner.truncate_completion(completion)
    }

    /// Delegates for the same reason as `truncate_completion` above: a
    /// forgotten defaulted method returns `None` here, which silently turns
    /// self-repair off for every filtered run and looks like the mechanism
    /// simply does not work.
    fn repair_prompt(&self, prompt: &str, completion: &str, v: &Verdict) -> Option<String> {
        self.inner.repair_prompt(prompt, completion, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::human_eval::HumanEvalDomain;
    use std::path::PathBuf;

    fn load_human_eval(tag: &str) -> Option<HumanEvalDomain> {
        let scratch = std::env::temp_dir().join(format!("workllm-filtered-{tag}"));
        let p = PathBuf::from("../data/humaneval/HumanEval.jsonl");
        let path = if p.exists() {
            p
        } else {
            let p2 = PathBuf::from("data/humaneval/HumanEval.jsonl");
            if !p2.exists() {
                return None;
            }
            p2
        };
        HumanEvalDomain::from_jsonl(&path, &scratch).ok()
    }

    /// Every defaulted `Domain` method must be delegated, not inherited — the
    /// silent-failure surface that switched `truncate_completion` OFF for the
    /// whole hard-tail series (see `docs/phase22-c4-c5-rl-vs-sft.md`, Lesson
    /// #6). The shared guard wraps a `ProbeDomain` (non-default sentinels) and
    /// checks that no defaulted method falls back to the trait default. This
    /// one call replaces the former per-method `score` / `truncate_completion`
    /// tests and additionally covers `n_prompts` / `nth_prompt`.
    #[test]
    fn all_defaulted_methods_delegate() {
        use crate::domain::delegation_probe::assert_domain_fully_delegates;
        assert_domain_fully_delegates!(|inner| FilteredDomain::new(inner, [0usize]));
    }

    #[test]
    fn empty_skip_is_identity() {
        let Some(inner) = load_human_eval("empty-skip") else {
            eprintln!("skipping: HumanEval.jsonl not on disk");
            return;
        };
        let original_n = inner.n_prompts().unwrap();
        let wrapped = FilteredDomain::new(Arc::new(inner), Vec::<usize>::new());
        assert_eq!(wrapped.n_prompts(), Some(original_n));
        assert_eq!(wrapped.n_surviving(), original_n);
        // nth_prompt round-trip — first prompt should match the
        // wrapper's first prompt.
        assert!(wrapped.nth_prompt(0).is_some());
    }

    #[test]
    fn skip_first_5_renumbers_correctly() {
        let Some(inner) = load_human_eval("skip-first-5") else {
            eprintln!("skipping: HumanEval.jsonl not on disk");
            return;
        };
        let inner_arc: Arc<dyn Domain> = Arc::new(inner);
        let original_5 = inner_arc.nth_prompt(5).expect("inner has index 5");
        let wrapped = FilteredDomain::new(Arc::clone(&inner_arc), 0..5);
        assert_eq!(wrapped.n_surviving(), 164 - 5);
        // wrapper's nth_prompt(0) should equal inner's nth_prompt(5)
        let wrapped_0 = wrapped.nth_prompt(0).expect("wrapped has index 0");
        assert_eq!(wrapped_0, original_5);
    }

    #[test]
    fn out_of_range_returns_none() {
        let Some(inner) = load_human_eval("out-of-range") else {
            eprintln!("skipping: HumanEval.jsonl not on disk");
            return;
        };
        let wrapped = FilteredDomain::new(Arc::new(inner), [0, 1, 2]);
        assert!(wrapped.nth_prompt(wrapped.n_surviving()).is_none());
    }

    #[test]
    fn verify_delegates_to_inner() {
        let Some(inner) = load_human_eval("verify-delegates") else {
            eprintln!("skipping: HumanEval.jsonl not on disk");
            return;
        };
        let p0 = inner.problems[0].prompt.clone();
        let canonical = inner.problems[0].canonical_solution.clone();
        let wrapped = FilteredDomain::new(Arc::new(inner), [1, 2, 3]);
        // The canonical solution at index 0 should still verify Correct
        // even when we skipped some OTHER indices.
        let v = wrapped.verify(&p0, &canonical);
        assert!(
            v.is_correct(),
            "canonical at index 0 should still verify: {v:?}"
        );
    }
}
