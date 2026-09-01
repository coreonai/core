//! F2: Domain trait slice used by the self-improve loop.

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
    Incorrect { reason: String },
}

impl Verdict {
    pub fn is_correct(&self) -> bool {
        matches!(self, Verdict::Correct)
    }
}

pub trait Domain: Send + Sync {
    fn sample_prompt(&self) -> String;
    fn verify(&self, prompt: &str, completion: &str) -> Verdict;
    fn charset(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ok_only_domain() {
        let d = impls::OkOnlyDomain;
        assert!(d.verify("x", "ok").is_correct());
        assert!(!d.verify("x", "OK").is_correct());
        assert!(!d.verify("x", "").is_correct());
        assert!(d.charset().contains('o'));
    }

    #[test]
    fn digit_charset_domain() {
        let d = impls::DigitCharsetDomain;
        for ch in "0123456789".chars() {
            assert!(d.charset().contains(ch), "missing {ch}");
        }
    }

    #[test]
    fn reject_empty() {
        let d = impls::NonEmptyDomain;
        assert!(!d.verify("p", "").is_correct());
        assert!(d.verify("p", "a").is_correct());
    }
}
