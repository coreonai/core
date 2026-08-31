//! Tool-use domain where the tool is a Python interpreter.
//!
//! Phase 23 established that a format-SFT'd 7B emits a valid `(python ...)`
//! call for task families it has never seen (12/12) but solves only 4/12 of
//! them — the same 4/12 as the base model. It learned the grammar and the
//! discipline of using the tool's result; it did not learn to write correct
//! code for a new problem. This domain exists to close that gap with a
//! self-improve loop.
//!
//! ## Why the verifier is free here
//!
//! Every earlier domain paid for verification: `RustCodeDomain` shells out to
//! `cargo`, `HumanEvalDomain` runs a test harness. Here the answer is an
//! integer we computed in Rust when we generated the question, and the
//! model's snippet is executed by the very tool it is learning to call. There
//! is no verifier to build, and no verifier to be wrong.
//!
//! ## Reward hacking
//!
//! A free verifier that compares numbers invites the obvious cheat:
//! `print(385)`. That would verify, get harvested, and teach the model to
//! emit constants. [`ToolUsePythonDomain`] rejects a snippet that is nothing
//! but `print(<literal>)`.
//!
//! A tighter guard is tempting and wrong. "The snippet must mention `n`"
//! rejects genuinely correct solutions — the base model writes
//! `range(1, 38)` for n=37, which never contains "37". "The snippet must not
//! contain the answer's digits" is worse: for small answers the digits
//! collide with the parameters constantly. So the cheap unambiguous case is
//! rejected and the ambiguous one is *counted* instead — see
//! [`ToolUsePythonDomain::looks_hardcoded`], which the harvest reports rather
//! than silently filters.

use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::Rng;

use super::Domain;
use crate::tools::python_tool::PythonTool;
use crate::tools::{parse_first_tool_call, Tool};
use crate::types::Verdict;

/// One generated problem: the question as asked, and the answer as computed
/// in Rust. `n` is kept so the harvest can report how often a verified
/// snippet never mentions the parameter it was supposed to depend on.
#[derive(Debug, Clone)]
pub struct PyTask {
    pub question: String,
    pub answer: i64,
    pub n: u32,
    pub family: usize,
}

/// The eight task families this domain harvests over.
//
/// Deliberately disjoint from both the three families `phase23_toolcall_sft`
/// trains (sums of squares, multiples of 3 or 5, prime counts — already
/// solved 12/12, so no headroom) and the three `phase23_python_tool_7b
/// --novel` evaluates (divisor counts, Fibonacci, Collatz — kept clean as a
/// transfer set). Mixing either in would make a lift here unreadable.
pub const N_FAMILIES: usize = 10;

// Families 8 and 9 exist to test one thing: whether "import what you
// reference" can be learned as a rule rather than as the single instance
// `import math`.
//
// The 8-family run taught the model to import `math` for `gcd` and nothing
// else — 0/160 imports where none was needed, 49/98 exactly on the phi
// family, and 0 imports across 12 transfer problems, four of which reference
// `itertools`. Of eight families exactly one needed an import and it was
// always `math`, so there was nothing to generalise from.
//
// These two are chosen so the model's own instinct reaches for a DIFFERENT
// module: `functools.reduce` for a digit product (confirmed — it writes
// `reduce(...)` unimported and fails 0/32), and `statistics.median` for a
// median. Nothing forces an import; both are writable with a plain loop, and
// the model taking that route is itself a result — it did exactly that for
// trailing zeros, dropping `math` for Legendre's formula.
//
// `itertools` is deliberately absent. The transfer probe's Collatz problems
// reach for it, so training on it would turn the transfer test into "does
// the same module carry over" instead of "was the rule learned".

fn is_prime(x: i64) -> bool {
    if x < 2 {
        return false;
    }
    let mut d = 2i64;
    while d * d <= x {
        if x % d == 0 {
            return false;
        }
        d += 1;
    }
    true
}

/// `(question, answer)` for a family at parameter `n`. Answers are computed
/// here, in Rust, so the ground truth never depends on the thing being
/// tested.
pub fn task(family: usize, n: u32) -> PyTask {
    let ni = n as i64;
    let (question, answer) = match family {
        0 => (
            format!("sum of the cubes from 1 to {n}?"),
            (1..=ni).map(|i| i * i * i).sum(),
        ),
        1 => (
            format!("sum of the divisors of {n}?"),
            (1..=ni).filter(|d| ni % d == 0).sum(),
        ),
        2 => (
            format!("how many numbers below {n} are divisible by 7?"),
            (1..ni).filter(|i| i % 7 == 0).count() as i64,
        ),
        3 => {
            // Trailing zeros of n! — Legendre's formula, so no bignum.
            let mut z = 0i64;
            let mut p = 5i64;
            while p <= ni {
                z += ni / p;
                p *= 5;
            }
            (format!("how many trailing zeros does {n}! have?"), z)
        }
        4 => {
            let cube = ni * ni * ni;
            let mut s = 0i64;
            let mut x = cube;
            while x > 0 {
                s += x % 10;
                x /= 10;
            }
            (format!("what is the sum of the digits of {n} cubed?"), s)
        }
        5 => {
            // Euler's totient by definition, matching how a model would
            // most plausibly write it.
            let gcd = |mut a: i64, mut b: i64| {
                while b != 0 {
                    let t = a % b;
                    a = b;
                    b = t;
                }
                a
            };
            (
                format!("how many numbers from 1 to {n} are coprime to {n}?"),
                (1..=ni).filter(|&k| gcd(k, ni) == 1).count() as i64,
            )
        }
        6 => {
            let mut best = 1i64;
            let mut x = ni;
            let mut d = 2i64;
            while d * d <= x {
                while x % d == 0 {
                    best = d;
                    x /= d;
                }
                d += 1;
            }
            if x > 1 {
                best = x;
            }
            (format!("what is the largest prime factor of {n}?"), best)
        }
        7 => (
            format!("what is the sum of all primes below {n}?"),
            (2..ni).filter(|&x| is_prime(x)).sum(),
        ),
        8 => {
            // Nonzero digits only: `n**3` hits a 0 digit constantly, and a
            // family whose answer is 0 for half its inputs teaches the model
            // to print 0.
            let mut prod = 1i64;
            let mut x = ni * ni * ni;
            while x > 0 {
                let d = x % 10;
                if d != 0 {
                    prod *= d;
                }
                x /= 10;
            }
            (
                format!("what is the product of the nonzero digits of {n} cubed?"),
                prod,
            )
        }
        _ => {
            // Doubled so the answer stays an integer for an even divisor
            // count. The first version of this family asked for the most
            // common digit's COUNT; its answers were 2..5, and the model
            // scored 9/32 by hardcoding a digit
            // (`sum(int(c)==1 for c in str(21**5))`) that happened to be the
            // most common. A wide answer space is what makes a free verifier
            // hard to hit by luck.
            let divs: Vec<i64> = (1..=ni).filter(|d| ni % d == 0).collect();
            let m = divs.len();
            let twice_median = if m % 2 == 1 {
                2 * divs[m / 2]
            } else {
                divs[m / 2 - 1] + divs[m / 2]
            };
            (
                format!("what is twice the median of the divisors of {n}?"),
                twice_median,
            )
        }
    };
    PyTask {
        question,
        answer,
        n,
        family: family.min(N_FAMILIES - 1),
    }
}

pub struct ToolUsePythonDomain {
    tasks: Vec<PyTask>,
    /// Prompt text → task index. `verify` receives the prompt as a string,
    /// not an index, so the mapping has to be reconstructible from it.
    by_prompt: HashMap<String, usize>,
    tool: PythonTool,
}

impl ToolUsePythonDomain {
    /// Build the full cross product of families × `n_range`.
    pub fn new(n_lo: u32, n_hi: u32) -> Self {
        Self::with_families(n_lo, n_hi, &(0..N_FAMILIES).collect::<Vec<_>>())
    }

    /// Restrict to a subset of families.
    //
    /// Needed because the eight are not equally hard: some are structurally
    /// near-identical to what the format SFT already trained (counting
    /// multiples of 7 is the trained "multiples of 3 or 5" with one constant
    /// changed) and sit at ceiling, while others have real headroom. Pooling
    /// them hides the signal in an aggregate — the saturated-substrate
    /// problem Phase 14 C1 ran into. Measure per family, then harvest where
    /// the headroom is.
    pub fn with_families(n_lo: u32, n_hi: u32, families: &[usize]) -> Self {
        let mut tasks = Vec::new();
        for &family in families {
            for n in n_lo..=n_hi {
                tasks.push(task(family, n));
            }
        }
        let by_prompt = tasks
            .iter()
            .enumerate()
            .map(|(i, t)| (render_prompt(&t.question), i))
            .collect();
        Self {
            tasks,
            by_prompt,
            tool: PythonTool::new(),
        }
    }

    pub fn n_tasks(&self) -> usize {
        self.tasks.len()
    }

    pub fn task_at(&self, i: usize) -> Option<&PyTask> {
        self.tasks.get(i)
    }

    /// A snippet that is only `print(<literal>)` — the one unambiguous cheat.
    /// Rejected outright, because harvesting it teaches the model to emit
    /// constants.
    pub fn is_bare_literal(code: &str) -> bool {
        let c = code.trim();
        let Some(inner) = c.strip_prefix("print(").and_then(|r| r.strip_suffix(')')) else {
            return false;
        };
        let inner = inner.trim();
        !inner.is_empty() && inner.chars().all(|ch| ch.is_ascii_digit() || ch == '-')
    }

    /// Weaker signal, reported rather than enforced: the snippet mentions
    /// neither `n` nor `n + 1`, so it probably does not depend on the
    /// parameter. Not a rejection rule — a correct solution can legitimately
    /// write `range(1, 38)` for n=37 and mention neither.
    pub fn looks_hardcoded(code: &str, n: u32) -> bool {
        !code.contains(&n.to_string()) && !code.contains(&(n + 1).to_string())
    }

    /// Extract the snippet from a completion, if it holds a dispatchable call.
    pub fn snippet_of(completion: &str) -> Option<String> {
        let (_, call) = parse_first_tool_call(completion)?;
        (call.name == "python").then_some(call.args)
    }
}

/// `"Q: <question>\n"` — must match what `phase23_toolcall_sft` trains, or the
/// harvested pairs teach a prompt shape the eval never uses.
pub fn render_prompt(question: &str) -> String {
    format!("Q: {question}\n")
}

impl Domain for ToolUsePythonDomain {
    fn sample_prompt(&self, rng: &mut StdRng) -> String {
        let i = rng.gen_range(0..self.tasks.len());
        render_prompt(&self.tasks[i].question)
    }

    fn verify(&self, prompt: &str, completion: &str) -> Verdict {
        let Some(&idx) = self.by_prompt.get(prompt) else {
            return Verdict::Incorrect {
                reason: "prompt not from this domain".into(),
            };
        };
        let t = &self.tasks[idx];
        let Some(code) = Self::snippet_of(completion) else {
            return Verdict::Incorrect {
                reason: "no dispatchable (python ...) call".into(),
            };
        };
        if Self::is_bare_literal(&code) {
            return Verdict::Incorrect {
                reason: "bare print(<literal>) — hardcoded answer".into(),
            };
        }
        match self.tool.execute(&code) {
            Ok(out) if out.trim() == t.answer.to_string() => Verdict::Correct,
            Ok(out) => Verdict::Incorrect {
                reason: format!("got {:?}, want {}", out.trim(), t.answer),
            },
            Err(e) => Verdict::Incorrect {
                reason: format!("{e}"),
            },
        }
    }

    fn charset(&self) -> &str {
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ \n?()[]{}:,.+-*/%=<>!'\"_|&^"
    }

    fn n_prompts(&self) -> Option<usize> {
        Some(self.tasks.len())
    }

    fn nth_prompt(&self, i: usize) -> Option<String> {
        self.tasks.get(i).map(|t| render_prompt(&t.question))
    }

    /// Reduce the completion to exactly the first call — dropping text on
    /// BOTH sides of it.
    //
    /// Cutting only the tail is not enough, and the shortfall is not
    /// cosmetic. Repair-harvested completions arrived as
    //
    /// ```text
    /// A: 2
    /// (python print(sum([12//5**i for i in range(1,10)])))
    /// ```
    //
    /// The answer line made sense in the two-turn context it was generated
    /// in, but the harvested pair uses the ORIGINAL prompt, so training on it
    /// teaches the model to state an answer *before* computing one. The
    /// snippet still verifies — the executor only ever sees the call — so
    /// nothing downstream catches it. It compounded from ~0 to 17% after five
    /// such pairs and to 82% two rounds later, because from round 1 on the
    /// loop was harvesting its own contaminated output.
    //
    /// That also destroys the property `--sabotage` was built to check: an
    /// answer written before the tool ran cannot have come from the tool.
    fn truncate_completion(&self, completion: &str) -> String {
        let Some((range, _)) = parse_first_tool_call(completion) else {
            // No call: hand it back untouched so verification still fails
            // with an honest reason instead of an empty string.
            return completion.to_string();
        };
        completion[range].to_string()
    }

    /// Hand the tool's own error back, spliced exactly where the result
    /// would have gone — the same shape `agentic_generator_actor` produces.
    //
    /// This is the only channel through which the missing information can
    /// reach the model: it never writes an import on its own (0 in 576
    /// samples), because it is confident `math` is preloaded. Shown the
    /// `NameError`, it does not add the import either — it drops `math` and
    /// writes the arithmetic directly (`sum(18//5**i for i in range(1,10))`
    /// for trailing zeros). What it learns from the error is the tool's
    /// contract, not a syntax fix.
    fn repair_prompt(&self, prompt: &str, completion: &str, v: &Verdict) -> Option<String> {
        let reason = match v {
            Verdict::Incorrect { reason } => reason,
            _ => return None,
        };
        let code = Self::snippet_of(completion)?;
        Some(format!(
            "{prompt}(python {code}{}ERR:{reason})\n",
            crate::tools::RESOLVED_MARKER
        ))
    }

    fn task_id(&self, i: usize) -> Option<String> {
        self.tasks
            .get(i)
            .map(|t| format!("ToolUsePython/f{}/n{}", t.family, t.n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn dom() -> ToolUsePythonDomain {
        ToolUsePythonDomain::new(10, 30)
    }

    fn have_python() -> bool {
        std::process::Command::new("python3")
            .arg("-c")
            .arg("pass")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    /// Ground truth is computed in Rust, so it is worth checking against
    /// values a reader can verify by hand.
    /// Hand-checkable values for the two families added to test whether
    /// importing generalises past `math`.
    #[test]
    fn known_answers_for_the_module_probe_families() {
        // 12^3 = 1728 -> 1*7*2*8
        assert_eq!(task(8, 12).answer, 112);
        // 60^3 = 216000 -> 2*1*6 (zeros skipped)
        assert_eq!(task(8, 60).answer, 12);
        // divisors of 12: 1 2 3 4 6 12 -> median 3.5 -> 7
        assert_eq!(task(9, 12).answer, 7);
        // divisors of 16: 1 2 4 8 16 -> median 4 -> 8
        assert_eq!(task(9, 16).answer, 8);
    }

    #[test]
    fn known_answers() {
        assert_eq!(task(0, 3).answer, 36); // 1 + 8 + 27
        assert_eq!(task(1, 12).answer, 28); // 1+2+3+4+6+12
        assert_eq!(task(2, 30).answer, 4); // 7, 14, 21, 28
        assert_eq!(task(3, 25).answer, 6); // 25/5 + 25/25
        assert_eq!(task(4, 10).answer, 1); // 1000 -> 1
        assert_eq!(task(5, 12).answer, 4); // 1, 5, 7, 11
        assert_eq!(task(6, 60).answer, 5); // 60 = 2^2 * 3 * 5
        assert_eq!(task(7, 10).answer, 17); // 2 + 3 + 5 + 7
    }

    #[test]
    fn every_family_is_reachable_and_distinct() {
        let qs: Vec<String> = (0..N_FAMILIES).map(|f| task(f, 42).question).collect();
        let uniq: std::collections::HashSet<&String> = qs.iter().collect();
        assert_eq!(uniq.len(), N_FAMILIES, "families collapsed: {qs:?}");
    }

    #[test]
    fn indexed_access_covers_the_whole_pool() {
        let d = dom();
        let n = d.n_prompts().unwrap();
        assert_eq!(n, N_FAMILIES * 21);
        assert_eq!(
            N_FAMILIES, 10,
            "the module-probe families must be in the pool"
        );
        assert!(d.nth_prompt(n - 1).is_some());
        assert!(d.nth_prompt(n).is_none());
    }

    #[test]
    fn verifies_a_correct_snippet() {
        if !have_python() {
            return;
        }
        let d = dom();
        let prompt = render_prompt(&task(0, 10).question);
        let comp = "(python print(sum(i**3 for i in range(1,11))))\n";
        assert!(d.verify(&prompt, comp).is_correct());
    }

    #[test]
    fn rejects_a_wrong_snippet() {
        if !have_python() {
            return;
        }
        let d = dom();
        let prompt = render_prompt(&task(0, 10).question);
        assert!(!d
            .verify(&prompt, "(python print(sum(range(11))))\n")
            .is_correct());
    }

    /// The cheat the free verifier invites. `task(0, 10).answer` is 3025, and
    /// printing it must NOT count as solving the problem.
    #[test]
    fn rejects_a_hardcoded_answer() {
        if !have_python() {
            return;
        }
        let d = dom();
        let prompt = render_prompt(&task(0, 10).question);
        assert_eq!(task(0, 10).answer, 3025);
        let v = d.verify(&prompt, "(python print(3025))\n");
        assert!(!v.is_correct(), "hardcoded answer verified as correct");
    }

    #[test]
    fn bare_literal_detector_does_not_fire_on_real_code() {
        assert!(ToolUsePythonDomain::is_bare_literal("print(3025)"));
        assert!(ToolUsePythonDomain::is_bare_literal(" print(-7) "));
        assert!(!ToolUsePythonDomain::is_bare_literal(
            "print(sum(i for i in range(10)))"
        ));
        assert!(!ToolUsePythonDomain::is_bare_literal("print(2*3)"));
    }

    /// The rule deliberately NOT enforced: `range(1, 38)` for n=37 is a
    /// correct solution that mentions neither 37 nor... well, it mentions 38.
    /// The point is that the signal is noisy, so it is reported, not enforced.
    #[test]
    fn hardcode_heuristic_is_only_a_signal() {
        assert!(!ToolUsePythonDomain::looks_hardcoded(
            "print(sum(i*i for i in range(1, 38)))",
            37
        ));
        assert!(ToolUsePythonDomain::looks_hardcoded("print(17575)", 37));
    }

    #[test]
    fn repair_prompt_hands_back_the_error() {
        let d = dom();
        let prompt = render_prompt(&task(5, 12).question);
        let comp = "(python print(math.gcd(1,12)))\n";
        let v = d.verify(&prompt, comp);
        assert!(!v.is_correct());
        let next = d.repair_prompt(&prompt, comp, &v).expect("repair prompt");
        assert!(next.starts_with(&prompt), "must extend the original prompt");
        assert!(
            next.contains("ERR:"),
            "the error must reach the model: {next}"
        );
        assert!(next.contains(crate::tools::RESOLVED_MARKER));
        // ...and the failed call must read as resolved, or the loop would
        // dispatch it a second time instead of letting the model retry.
        assert!(parse_first_tool_call(&next).is_none());
    }

    #[test]
    fn repair_prompt_is_none_when_there_was_no_call() {
        let d = dom();
        let prompt = render_prompt(&task(5, 12).question);
        let v = d.verify(&prompt, "A: 4\n");
        assert!(d.repair_prompt(&prompt, "A: 4\n", &v).is_none());
    }

    #[test]
    fn truncation_keeps_only_the_call() {
        let d = dom();
        let raw = "(python print(1+1))\nA: 2\nQ: next problem\n";
        assert_eq!(d.truncate_completion(raw), "(python print(1+1))\n");
    }

    /// The contamination that ran the first self-improve loop off the rails:
    /// an answer stated BEFORE the call. Cutting only the tail leaves it in
    /// the harvested training pair.
    #[test]
    fn truncation_drops_text_before_the_call() {
        let d = dom();
        let raw = "A: 2\n(python print(sum([12//5**i for i in range(1,10)])))\nA: 2\n";
        assert_eq!(
            d.truncate_completion(raw),
            "(python print(sum([12//5**i for i in range(1,10)])))\n"
        );
    }

    #[test]
    fn truncation_leaves_a_callless_completion_alone() {
        let d = dom();
        assert_eq!(d.truncate_completion("A: 2\n"), "A: 2\n");
    }

    #[test]
    fn completion_without_a_call_is_incorrect() {
        let d = dom();
        let prompt = render_prompt(&task(0, 10).question);
        assert!(!d.verify(&prompt, "A: 3025\n").is_correct());
    }

    #[test]
    fn foreign_prompt_is_incorrect_not_a_panic() {
        let d = dom();
        assert!(!d
            .verify("Q: not mine\n", "(python print(1))\n")
            .is_correct());
    }
}
