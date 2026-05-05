//! Phase 6 Shape C scaffolding: a `Critic` is a cheap proxy that
//! predicts whether a candidate `(prompt, completion)` will pass the
//! domain's expensive verifier (cargo, in K9's case).
//!
//! Used as a pre-filter: rank-and-keep top-K candidates per prompt
//! before sending them to cargo. The expensive verifier remains the
//! ultimate label source for the curator; the critic only decides
//! who gets to ASK cargo, not who gets accepted.
//!
//! See `docs/phase6-shape-c.md` for the design rationale, the
//! sessions plan, and the acceptance criteria for promoting Shape C
//! out of "scaffolding only" status.

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::sync::Mutex;

/// A `Critic` produces a real-valued score for `(prompt, completion)`
/// pairs; higher = more likely to pass the actual verifier.
///
/// Scores are NOT required to be probabilities — they're for ranking.
/// Convention: score in `[0.0, 1.0]` when an implementation can
/// produce one cheaply (so callers can apply a threshold), but
/// implementations are free to use any monotone scale.
pub trait Critic: Send + Sync {
    fn score(&self, prompt: &str, completion: &str) -> f32;
}

/// Trivial baseline: every candidate gets the same score, so all
/// pass the threshold. Equivalent to "no filter" and reproduces
/// the current self-improve loop's behavior. Useful as a
/// sanity-check baseline against learned critics.
pub struct AlwaysCorrectCritic;

impl Critic for AlwaysCorrectCritic {
    fn score(&self, _prompt: &str, _completion: &str) -> f32 {
        1.0
    }
}

/// Random scoring (deterministic per `(prompt, completion)` for a
/// given seed via interior mutability of an RNG). Useful as a
/// **negative baseline** — a critic should beat random.
pub struct RandomCritic {
    rng: Mutex<StdRng>,
}

impl RandomCritic {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
        }
    }
}

impl Critic for RandomCritic {
    fn score(&self, _prompt: &str, _completion: &str) -> f32 {
        let mut g = self.rng.lock().expect("RandomCritic mutex poisoned");
        g.gen_range(0.0..1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn always_correct_returns_one() {
        let c = AlwaysCorrectCritic;
        assert_eq!(c.score("any", "thing"), 1.0);
        assert_eq!(c.score("", ""), 1.0);
    }

    #[test]
    fn random_critic_is_deterministic_per_seed() {
        let c1 = RandomCritic::new(42);
        let c2 = RandomCritic::new(42);
        // Same seed, called in the same order → same scores.
        assert_eq!(c1.score("p", "x"), c2.score("p", "x"));
        assert_eq!(c1.score("p", "y"), c2.score("p", "y"));
    }

    #[test]
    fn random_critic_scores_in_unit_interval() {
        let c = RandomCritic::new(0);
        for i in 0..32 {
            let s = c.score(&format!("prompt-{i}"), "completion");
            assert!(
                (0.0..1.0).contains(&s),
                "score {s} out of [0, 1) at iter {i}"
            );
        }
    }

    /// Trait can be passed as `&dyn Critic`. Guards against the trait
    /// accidentally requiring `Sized` etc.
    #[test]
    fn dyn_dispatch_compiles() {
        fn take_dyn(c: &dyn Critic) -> f32 {
            c.score("", "")
        }
        assert_eq!(take_dyn(&AlwaysCorrectCritic), 1.0);
        let r = RandomCritic::new(7);
        let s = take_dyn(&r);
        assert!((0.0..1.0).contains(&s));
    }
}
