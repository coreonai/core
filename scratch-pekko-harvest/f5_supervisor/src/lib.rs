//! F5: topology-only supervisor smoke (no CUDA, no real model).

pub mod reference;
#[cfg(feature = "student")]
pub mod student;

#[cfg(feature = "student")]
pub use student as impls;
#[cfg(not(feature = "student"))]
pub use reference as impls;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    Correct,
    Incorrect,
}

pub trait Generator {
    fn generate(&mut self, prompt: &str) -> String;
}

pub trait Verifier {
    fn verify(&self, prompt: &str, completion: &str) -> Verdict;
}

#[derive(Debug, Default)]
pub struct RoundResult {
    pub order: Vec<&'static str>,
    pub kept: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedGen;
    impl Generator for FixedGen {
        fn generate(&mut self, prompt: &str) -> String {
            if prompt.ends_with("ok") {
                "ok".into()
            } else {
                "nope".into()
            }
        }
    }

    struct OkVerifier;
    impl Verifier for OkVerifier {
        fn verify(&self, _: &str, completion: &str) -> Verdict {
            if completion == "ok" {
                Verdict::Correct
            } else {
                Verdict::Incorrect
            }
        }
    }

    #[test]
    fn one_round_order_and_filter() {
        let mut gen = FixedGen;
        let ver = OkVerifier;
        let r = impls::one_round(&mut gen, &ver, &["go ok", "go bad"]);
        assert_eq!(r.order, vec!["generate", "verify", "generate", "verify"]);
        assert_eq!(r.kept, vec!["ok".to_string()]);
    }
}
