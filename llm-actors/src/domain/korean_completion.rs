//! Korean sentence-completion domain for the self-improvement loop.
//!
//! Pairs naturally with a KoWiki-pretrained model (e.g. the 50M Llama
//! recipe in `train_kowiki`). The agent generates a Korean continuation;
//! a heuristic verifier checks for basic well-formedness:
//!
//!   - non-empty
//!   - contains Korean Hangul characters (filters out pure-Latin/pure-LaTeX
//!     output, the failure mode we observed at loss ~7.0 on the cleaned
//!     KoWiki corpus)
//!   - ends with a Korean sentence terminator (`다.`, `요.`, `까?`, `다`,
//!     `요`, `까`) — the Korean LM's polite/indicative endings
//!   - within a sane character-length window
//!
//! This is a *heuristic* verifier — there's no LLM-judge or grammatical
//! parser involved. It's deliberately strict so that random/noisy output
//! gets filtered. As the model gets better at producing Korean prose, the
//! pass rate should climb; that's the self-improve signal.
//!
//! Prompts come from a small fixed seed list (lookup-style prefixes that
//! are common in Wikipedia-style Korean), making the domain reproducible.
//! Future work can swap in prompts sampled from a held-out corpus.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use super::Domain;
use crate::types::Verdict;

/// Wikipedia-style Korean sentence prefixes. Each prompt is paired with
/// an expected sentence-ending pattern (one of several Korean polite
/// indicative endings).
const SEED_PROMPTS: &[&str] = &[
    "대한민국의 수도는 ",
    "서울특별시는 ",
    "한반도는 ",
    "한국 전쟁은 ",
    "조선은 ",
    "고려는 ",
    "대한민국 정부는 ",
    "한국어는 ",
    "한글은 ",
    "한국의 경제는 ",
];

const MIN_CHARS: usize = 8;
const MAX_CHARS: usize = 400;
const SENTENCE_ENDINGS: &[&str] = &["다.", "요.", "까?", "다", "요", "까"];

pub struct KoreanCompletionDomain {
    prompts: &'static [&'static str],
}

impl Default for KoreanCompletionDomain {
    fn default() -> Self {
        Self { prompts: SEED_PROMPTS }
    }
}

impl KoreanCompletionDomain {
    pub fn new(prompts: &'static [&'static str]) -> Self {
        Self { prompts }
    }

    fn has_hangul(s: &str) -> bool {
        // Hangul Syllables block U+AC00–U+D7A3 covers modern Korean.
        s.chars().any(|c| {
            let c = c as u32;
            (0xAC00..=0xD7A3).contains(&c)
        })
    }

    fn ends_with_sentence_terminator(s: &str) -> bool {
        let trimmed = s.trim_end();
        SENTENCE_ENDINGS.iter().any(|end| trimmed.ends_with(end))
    }
}

impl Domain for KoreanCompletionDomain {
    fn sample_prompt(&self, rng: &mut StdRng) -> String {
        (*self.prompts.choose(rng).expect("non-empty prompt list")).to_string()
    }

    fn verify(&self, _prompt: &str, completion: &str) -> Verdict {
        let trimmed = completion.trim();
        if trimmed.is_empty() {
            return Verdict::Incorrect { reason: "empty completion".into() };
        }
        if trimmed.chars().count() < MIN_CHARS {
            return Verdict::Incorrect {
                reason: format!("too short: {} chars", trimmed.chars().count()),
            };
        }
        if trimmed.chars().count() > MAX_CHARS {
            return Verdict::Incorrect {
                reason: format!("too long: {} chars", trimmed.chars().count()),
            };
        }
        if !Self::has_hangul(trimmed) {
            return Verdict::Incorrect {
                reason: "no Hangul characters — model likely produced LaTeX or noise".into(),
            };
        }
        if !Self::ends_with_sentence_terminator(trimmed) {
            return Verdict::Incorrect {
                reason: "no Korean sentence-ending (다 / 요 / 까)".into(),
            };
        }
        Verdict::Correct
    }

    fn charset(&self) -> &str {
        // A KoWiki BPE tokenizer covers the Hangul block; this is just
        // a hint for non-BPE callers.
        "가나다라마바사아자차카타파하 .,?!"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d() -> KoreanCompletionDomain {
        KoreanCompletionDomain::default()
    }

    #[test]
    fn correct_when_proper_korean_ends_with_da() {
        let v = d().verify("대한민국의 수도는 ", "서울특별시이며 정치, 경제의 중심지이다.");
        assert!(matches!(v, Verdict::Correct), "got {v:?}");
    }

    #[test]
    fn incorrect_on_no_hangul() {
        let v = d().verify("대한민국의 수도는 ", "Seoul is the capital city.");
        assert!(matches!(v, Verdict::Incorrect { .. }), "got {v:?}");
    }

    #[test]
    fn incorrect_on_too_short() {
        let v = d().verify("대한민국의 수도는 ", "서울.");
        assert!(matches!(v, Verdict::Incorrect { .. }), "got {v:?}");
    }

    #[test]
    fn incorrect_on_no_sentence_ending() {
        let v = d().verify(
            "대한민국의 수도는 ",
            "서울특별시이며 정치 경제의 중심지인",
        );
        assert!(matches!(v, Verdict::Incorrect { .. }), "got {v:?}");
    }

    #[test]
    fn incorrect_on_latex_only_output() {
        // Failure mode actually observed at KoWiki loss ~7.0
        let v = d().verify(
            "대한민국의 수도는 ",
            r"\frac{1}{2} \mathcal{matrix} \cos(\theta)",
        );
        assert!(matches!(v, Verdict::Incorrect { .. }), "got {v:?}");
    }
}
