//! Task domains.
//!
//! A `Domain` is anything we can (a) sample prompts for and (b) verify
//! completions of. Phase 2 ships with `arithmetic`; later we'll add
//! Rust-code domains that delegate to `cargo build/test`.

use rand::rngs::StdRng;

use crate::types::Verdict;

pub mod arithmetic;
pub mod rust_code;
pub mod tool_use;

pub trait Domain: Send + Sync {
    /// Sample a fresh prompt. Caller-owned RNG so domains stay deterministic.
    fn sample_prompt(&self, rng: &mut StdRng) -> String;

    /// Verify a single (prompt, completion) pair.
    fn verify(&self, prompt: &str, completion: &str) -> Verdict;

    /// Score in 0.0..=1.0 (defaults to 1.0 for correct, 0.0 otherwise).
    fn score(&self, verdict: &Verdict) -> f32 {
        if verdict.is_correct() { 1.0 } else { 0.0 }
    }

    /// Charset that must be present in any tokenizer used with this domain
    /// (for char-level tokenizers). Used to seed CharTokenizer.
    fn charset(&self) -> &str;
}
