use super::{Generator, RoundResult, Verdict, Verifier};

pub fn record_generate(out: &mut RoundResult, gen: &mut dyn Generator, p: &str) -> String {
    todo!("push generate then gen.generate")
}

pub fn record_verify_keep(out: &mut RoundResult, ver: &dyn Verifier, p: &str, c: String) {
    todo!("push verify; keep Correct")
}

pub fn one_round(
    gen: &mut dyn Generator,
    ver: &dyn Verifier,
    prompts: &[&str],
) -> RoundResult {
    todo!("for each prompt: generate then verify-keep")
}
