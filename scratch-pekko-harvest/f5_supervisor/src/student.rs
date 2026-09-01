use super::{Generator, RoundResult, Verifier};

pub fn one_round(
    _gen: &mut dyn Generator,
    _ver: &dyn Verifier,
    _prompts: &[&str],
) -> RoundResult {
    todo!("generate then verify each prompt; keep Correct only; record order")
}
