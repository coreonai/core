use super::{Generator, RoundResult, Verdict, Verifier};

pub fn one_round(
    gen: &mut dyn Generator,
    ver: &dyn Verifier,
    prompts: &[&str],
) -> RoundResult {
    let mut out = RoundResult::default();
    for p in prompts {
        out.order.push("generate");
        let c = gen.generate(p);
        out.order.push("verify");
        if ver.verify(p, &c) == Verdict::Correct {
            out.kept.push(c);
        }
    }
    out
}
