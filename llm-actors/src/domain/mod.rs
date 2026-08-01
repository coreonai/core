//! Task domains.
//!
//! A `Domain` is anything we can (a) sample prompts for and (b) verify
//! completions of. Phase 2 ships with `arithmetic`; later we'll add
//! Rust-code domains that delegate to `cargo build/test`.

use rand::rngs::StdRng;

use crate::types::Verdict;

pub mod arithmetic;
pub mod bigcodebench;
#[cfg(test)]
pub(crate) mod delegation_probe;
#[cfg(test)]
pub(crate) mod equivalence;
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

    /// Phase 22 Stage D G9 — clean a freshly generated completion before
    /// it is verified, harvested into a training pair, or scored at eval.
    /// Default is identity. Code domains override this with
    /// [`truncate_python_completion`] to match Phase 17's
    /// `truncate_completion`, cutting trailing test/scaffolding code that
    /// otherwise (a) pollutes the SFT target distribution and (b) gets
    /// cut off mid-statement at `max_new_tokens`, yielding syntax errors
    /// at eval. Applied at the GeneratorActor and EvaluatorActor decode
    /// sites so harvest, training, and eval all see the same cleaned text
    /// (exactly like Phase 17, which truncates inside `generate_completion`).
    fn truncate_completion(&self, completion: &str) -> String {
        completion.to_string()
    }

    /// Benchmark-standard identifier for the prompt at index `i` — e.g.
    /// `"HumanEval/3"`, an MBPP task number, a LiveCodeBench `question_id`.
    /// Keys the standard-format export (`crate::bench_export`) that official
    /// grading harnesses ingest, so generation can stay in Rust while scoring
    /// is delegated. Default is `None` (no stable id); fixed-benchmark domains
    /// override, and **wrappers must re-index it exactly like `nth_prompt`** —
    /// a defaulted method a wrapper forgets is a silent-failure surface (see
    /// the `domain-wrapper-equivalence` skill).
    fn task_id(&self, _i: usize) -> Option<String> {
        None
    }
}

/// Phase 22 Stage D G9 — port of Phase 17's `truncate_completion`
/// (`scripts/phase15_s1/self_improve.py`). Keeps the function body and
/// cuts at the first **top-level** (column-0) `def `/`class `/`import `/
/// `from `/`if __name__`/`print(` statement that appears AFTER some body
/// content — i.e., the first sibling statement the model starts emitting
/// once it has finished the requested function. Also cuts at the first
/// `<|` special-token marker (e.g. a leaked `<|fim_middle|>`), then trims
/// trailing blank lines.
pub fn truncate_python_completion(text: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    for line in text.split('\n') {
        if let Some(idx) = line.find("<|") {
            let head = &line[..idx];
            if !head.is_empty() {
                out.push(head);
            }
            break;
        }
        let top_level = line.starts_with("def ")
            || line.starts_with("class ")
            || line.starts_with("import ")
            || line.starts_with("from ")
            || line.starts_with("if __name__")
            || line.starts_with("print(");
        if !out.is_empty() && top_level {
            break;
        }
        out.push(line);
    }
    while let Some(last) = out.last() {
        if last.trim().is_empty() {
            out.pop();
        } else {
            break;
        }
    }
    out.join("\n")
}

/// Phase 22 follow-up C5 — the token-space counterpart of
/// [`truncate_python_completion`].
///
/// `truncate_completion` works on text, but a policy-gradient / SFT step
/// needs token ids. Re-encoding the truncated text would hand the trainer a
/// sequence the model never actually sampled (the tokenizer may pick
/// different merge boundaries), so instead this binary-searches for the
/// longest prefix of `comp_ids` whose decode covers `truncated` — about 8
/// decodes for a 192-token completion.
///
/// `decode` is injected so this is testable without a real tokenizer.
pub fn truncated_token_prefix<F, E>(
    comp_ids: &[u32],
    truncated: &str,
    decode: F,
) -> std::result::Result<Vec<u32>, E>
where
    F: Fn(&[u32]) -> std::result::Result<String, E>,
{
    if truncated.is_empty() {
        return Ok(Vec::new());
    }
    let (mut lo, mut hi) = (0usize, comp_ids.len());
    while lo < hi {
        let mid = lo.midpoint(hi);
        if decode(&comp_ids[..mid])?.len() >= truncated.len() {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    Ok(comp_ids[..lo].to_vec())
}

#[cfg(test)]
mod tests {
    use super::{truncate_python_completion, truncated_token_prefix};

    #[test]
    fn keeps_clean_body_unchanged() {
        let body = "    words = s.split()\n    return ' '.join(words)";
        assert_eq!(truncate_python_completion(body), body);
    }

    #[test]
    fn cuts_trailing_test_def() {
        let raw = "    return n + 1\n\n\ndef test_inc():\n    assert inc(1) == 2";
        assert_eq!(truncate_python_completion(raw), "    return n + 1");
    }

    #[test]
    fn cuts_trailing_print_and_if_main() {
        let raw = "    return lst\n\n# Test\nprint(f(1))\nif __name__ == '__main__':\n    f()";
        // `print(` is the first top-level marker; comment line precedes it
        // and is kept (comments aren't statements), trailing blank trimmed.
        assert_eq!(truncate_python_completion(raw), "    return lst\n\n# Test");
    }

    #[test]
    fn cuts_at_special_token_marker() {
        let raw = "    return x\n    done()<|fim_middle|>\n    return result";
        assert_eq!(truncate_python_completion(raw), "    return x\n    done()");
    }

    #[test]
    fn does_not_cut_indented_def_inside_body() {
        // A nested (indented) def is part of the body — must be kept.
        let raw = "    def helper():\n        return 1\n    return helper()";
        assert_eq!(truncate_python_completion(raw), raw);
    }

    #[test]
    fn trims_trailing_blank_lines() {
        let raw = "    return 1\n\n   \n";
        assert_eq!(truncate_python_completion(raw), "    return 1");
    }

    // ---- Phase 22 C5 — token-space truncation ----

    /// Fake tokenizer: one char per token, so decode(ids[..k]).len() == k.
    fn char_decode(ids: &[u32]) -> std::result::Result<String, std::convert::Infallible> {
        Ok(ids.iter().map(|c| char::from(*c as u8)).collect())
    }

    #[test]
    fn token_prefix_covers_the_truncated_text() {
        let ids: Vec<u32> = "def f():\n    return 1\ndef g():"
            .bytes()
            .map(u32::from)
            .collect();
        let truncated = "def f():\n    return 1";
        let got = truncated_token_prefix(&ids, truncated, char_decode).unwrap();
        assert_eq!(got.len(), truncated.len());
        assert_eq!(char_decode(&got).unwrap(), truncated);
    }

    #[test]
    fn token_prefix_of_empty_truncation_is_empty() {
        let ids: Vec<u32> = "abc".bytes().map(u32::from).collect();
        let got = truncated_token_prefix(&ids, "", char_decode).unwrap();
        assert!(got.is_empty());
    }

    /// Nothing was cut — the prefix must be the whole completion, not a
    /// silently shortened one.
    #[test]
    fn token_prefix_keeps_everything_when_nothing_truncated() {
        let text = "    return 1";
        let ids: Vec<u32> = text.bytes().map(u32::from).collect();
        let got = truncated_token_prefix(&ids, text, char_decode).unwrap();
        assert_eq!(got, ids);
    }
}
