use super::{Domain, Verdict};

pub struct OkOnlyDomain;
impl Domain for OkOnlyDomain {
    fn sample_prompt(&self) -> String { "say ok".into() }
    fn verify(&self, _prompt: &str, _completion: &str) -> Verdict {
        todo!("Correct iff completion == ok")
    }
    fn charset(&self) -> &str { "ok" }
}

pub struct DigitCharsetDomain;
impl Domain for DigitCharsetDomain {
    fn sample_prompt(&self) -> String { "n".into() }
    fn verify(&self, _: &str, _: &str) -> Verdict { Verdict::Correct }
    fn charset(&self) -> &str { todo!("return digits 0-9") }
}

pub struct NonEmptyDomain;
impl Domain for NonEmptyDomain {
    fn sample_prompt(&self) -> String { "anything".into() }
    fn verify(&self, _: &str, _completion: &str) -> Verdict {
        todo!("reject empty")
    }
    fn charset(&self) -> &str { "abcdefghijklmnopqrstuvwxyz" }
}
