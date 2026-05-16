//! Task domains.
//!
//! A `Domain` is anything we can (a) sample prompts for and (b) verify
//! completions of. Phase 2 ships with `arithmetic`; later we'll add
//! Rust-code domains that delegate to `cargo build/test`.

use rand::rngs::StdRng;

use crate::types::Verdict;

pub mod arithmetic;
pub mod filtered;
pub mod human_eval;
pub mod korean_completion;
pub mod mbpp;
pub mod python_code;
pub mod rust_code;
pub mod tool_use;

pub trait Domain: Send + Sync {
    /// Sample a fresh prompt. Caller-owned RNG so domains stay deterministic.
    fn sample_prompt(&self, rng: &mut StdRng) -> String;

    /// Verify a single (prompt, completion) pair.
    fn verify(&self, prompt: &str, completion: &str) -> Verdict;

    /// Score in 0.0..=1.0 (defaults to 1.0 for correct, 0.0 otherwise).
    fn score(&self, verdict: &Verdict) -> f32 {
        if verdict.is_correct() {
            1.0
        } else {
            0.0
        }
    }

    /// Charset that must be present in any tokenizer used with this domain
    /// (for char-level tokenizers). Used to seed CharTokenizer.
    fn charset(&self) -> &str;

    /// Phase 22 Stage B — number of distinct prompts the domain
    /// offers when iterated sequentially. Returns `None` for
    /// infinite-prompt domains (e.g., `ArithmeticDomain` generates
    /// arbitrarily many `(a, b)` pairs at random). Returns `Some(n)`
    /// for fixed-set domains like `HumanEvalDomain` (n=164).
    ///
    /// When `Some(n)`, `nth_prompt(i)` is expected to return prompts
    /// for `i ∈ 0..n` deterministically. `EvaluatorActor::EvalSequential`
    /// uses this to do a no-replacement sweep instead of
    /// `sample_prompt`'s with-replacement sampling.
    fn n_prompts(&self) -> Option<usize> {
        None
    }

    /// Phase 22 Stage B — deterministic indexed accessor. Returns
    /// `None` when the index is out of range or the domain is infinite.
    /// Domains that override `n_prompts` should also override this.
    fn nth_prompt(&self, _i: usize) -> Option<String> {
        None
    }
}
