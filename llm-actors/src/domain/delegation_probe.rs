//! Test-only delegation guard for [`Domain`](crate::domain::Domain) wrappers.
//!
//! A wrapper type that implements `Domain` by holding another `Domain` and
//! forgetting to delegate one of the **defaulted** methods (`score`,
//! `n_prompts`, `nth_prompt`, `truncate_completion`, `task_id`) silently inherits the
//! trait default — and the compiler cannot catch it, because a defaulted
//! method needs no `impl`. That is exactly the bug that switched
//! `truncate_completion` OFF for the entire Phase 22 hard-tail series and
//! mis-measured the base (see `docs/phase22-c4-c5-rl-vs-sft.md`, Lesson #6).
//!
//! [`ProbeDomain`] returns a deliberately **non-default** value from every
//! defaulted method, so [`assert_domain_fully_delegates!`] can assert that a
//! wrapper, when built around a probe, never falls back to a trait default.
//! This holds for both pure pass-through wrappers (which return the probe's
//! value unchanged) and selective-override wrappers like `FilteredDomain`
//! (which re-index `n_prompts`/`nth_prompt` but must still return a
//! non-default). Every new `Domain` wrapper should call the macro in its tests.

use rand::rngs::StdRng;

use crate::domain::Domain;
use crate::types::Verdict;

/// A `Domain` whose every defaulted method returns a value that differs from
/// the trait default, so a wrapper that forgets to delegate is detectable.
pub struct ProbeDomain;

impl Domain for ProbeDomain {
    fn sample_prompt(&self, _rng: &mut StdRng) -> String {
        "probe-sample".to_string()
    }
    fn verify(&self, _prompt: &str, _completion: &str) -> Verdict {
        Verdict::Correct
    }
    fn charset(&self) -> &str {
        "abc"
    }
    // --- defaulted methods, each returns a NON-default sentinel ---
    /// Default is `1.0` for `Correct`; return `0.5` instead.
    fn score(&self, _verdict: &Verdict) -> f32 {
        0.5
    }
    /// Default is `None`; return `Some(3)`.
    fn n_prompts(&self) -> Option<usize> {
        Some(3)
    }
    /// Default is `None`; return `Some("probe{i}")` for `i < 3`.
    fn nth_prompt(&self, i: usize) -> Option<String> {
        (i < 3).then(|| format!("probe{i}"))
    }
    /// Default is identity; cut at the first `"CUT"` marker.
    fn truncate_completion(&self, completion: &str) -> String {
        completion.split("CUT").next().unwrap_or("").to_string()
    }
    /// Default is `None`; return `Some("q{i}")` for `i < 3`.
    fn task_id(&self, i: usize) -> Option<String> {
        (i < 3).then(|| format!("q{i}"))
    }
}

/// Assert that a `Domain` wrapper delegates **every** defaulted method — i.e.
/// that wrapping a [`ProbeDomain`] never yields a trait default. Takes a
/// constructor closure `|inner: Arc<dyn Domain>| -> impl Domain`.
///
/// ```ignore
/// assert_domain_fully_delegates!(|inner| FilteredDomain::new(inner, [0usize]));
/// ```
///
/// A forgotten delegation falls back to the default and trips exactly one of
/// the five checks below.
macro_rules! assert_domain_fully_delegates {
    ($ctor:expr) => {{
        let inner: std::sync::Arc<dyn $crate::domain::Domain> =
            std::sync::Arc::new($crate::domain::delegation_probe::ProbeDomain);
        let wrapper = ($ctor)(std::sync::Arc::clone(&inner));
        // score: trait default is 1.0 for Correct; the probe returns 0.5.
        let s = $crate::domain::Domain::score(&wrapper, &$crate::types::Verdict::Correct);
        assert!(
            (s - 1.0).abs() > 1e-6,
            "Domain::score not delegated — wrapper returned the trait default {s} \
             instead of the inner's value (probe = 0.5)"
        );
        // truncate_completion: trait default is identity; the probe cuts at CUT.
        let t = $crate::domain::Domain::truncate_completion(&wrapper, "keepCUTdrop");
        assert_ne!(
            t, "keepCUTdrop",
            "Domain::truncate_completion not delegated — wrapper fell back to the \
             identity default instead of the inner's truncation"
        );
        // n_prompts: trait default is None; the probe returns Some(_).
        assert!(
            $crate::domain::Domain::n_prompts(&wrapper).is_some(),
            "Domain::n_prompts not delegated — wrapper fell back to the None default"
        );
        // nth_prompt: trait default is None; the probe returns Some(_).
        assert!(
            $crate::domain::Domain::nth_prompt(&wrapper, 0).is_some(),
            "Domain::nth_prompt not delegated — wrapper fell back to the None default"
        );
        // task_id: trait default is None; the probe returns Some(_).
        assert!(
            $crate::domain::Domain::task_id(&wrapper, 0).is_some(),
            "Domain::task_id not delegated — wrapper fell back to the None default"
        );
    }};
}
pub(crate) use assert_domain_fully_delegates;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// The probe itself must expose non-default values, or the guard is vacuous.
    #[test]
    fn probe_returns_non_defaults() {
        let p = ProbeDomain;
        assert_eq!(p.score(&Verdict::Correct), 0.5);
        assert_eq!(p.n_prompts(), Some(3));
        assert_eq!(p.nth_prompt(0).as_deref(), Some("probe0"));
        assert_eq!(p.truncate_completion("keepCUTdrop"), "keep");
        assert_eq!(p.task_id(0).as_deref(), Some("q0"));
    }

    /// A correct pass-through wrapper passes the guard.
    #[test]
    fn full_delegator_passes() {
        struct FullDelegator(Arc<dyn Domain>);
        impl Domain for FullDelegator {
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
            fn truncate_completion(&self, c: &str) -> String {
                self.0.truncate_completion(c)
            }
            fn task_id(&self, i: usize) -> Option<String> {
                self.0.task_id(i)
            }
        }
        assert_domain_fully_delegates!(FullDelegator);
    }

    /// A wrapper that forgets `truncate_completion` must trip the guard.
    #[test]
    #[should_panic(expected = "truncate_completion not delegated")]
    fn missing_delegation_trips_guard() {
        struct ForgetsTruncate(Arc<dyn Domain>);
        impl Domain for ForgetsTruncate {
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
            // truncate_completion INTENTIONALLY omitted -> inherits identity default.
        }
        assert_domain_fully_delegates!(ForgetsTruncate);
    }
}
