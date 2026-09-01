use super::{Domain, Verdict};

pub struct OkOnlyDomain;
impl Domain for OkOnlyDomain {
    fn sample_prompt(&self) -> String { "say ok".into() }
    fn verify(&self, _prompt: &str, completion: &str) -> Verdict {
        if completion == "ok" {
            Verdict::Correct
        } else {
            Verdict::Incorrect { reason: "expected ok".into() }
        }
    }
    fn charset(&self) -> &str { "ok" }
}

pub struct DigitCharsetDomain;
impl Domain for DigitCharsetDomain {
    fn sample_prompt(&self) -> String { "n".into() }
    fn verify(&self, _: &str, _: &str) -> Verdict { Verdict::Correct }
    fn charset(&self) -> &str { "0123456789" }
}

pub struct NonEmptyDomain;
impl Domain for NonEmptyDomain {
    fn sample_prompt(&self) -> String { "anything".into() }
    fn verify(&self, _: &str, completion: &str) -> Verdict {
        if completion.is_empty() {
            Verdict::Incorrect { reason: "empty".into() }
        } else {
            Verdict::Correct
        }
    }
    fn charset(&self) -> &str { "abcdefghijklmnopqrstuvwxyz" }
}
