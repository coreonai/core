use super::{Generator, RoundResult, Verdict, Verifier};

pub fn one_round(
    gen: &mut dyn Generator,
    ver: &dyn Verifier,
    prompts: &[&str],
) -> RoundResult {
    todo!("generate then verify each prompt; keep Correct only; record order")
}
