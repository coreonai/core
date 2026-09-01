//! F0: tiny expression-fill challenges graded by evaluating the expression.

pub mod reference;
#[cfg(feature = "student")]
pub mod student;

#[cfg(feature = "student")]
pub use student as impls;
#[cfg(not(feature = "student"))]
pub use reference as impls;

pub struct Challenge {
    pub name: &'static str,
    pub prompt: &'static str,
    pub suffix: &'static str,
}

pub const CHALLENGES: &[Challenge] = &[
    Challenge { name: "equals_5", prompt: "fn main() { assert_eq!(", suffix: ", 5); }\n" },
    Challenge { name: "equals_14_via_doubling", prompt: "fn main() { assert_eq!(2 * (", suffix: "), 14); }\n" },
    Challenge { name: "len_5_string", prompt: "fn main() { let s: &str = ", suffix: "; assert_eq!(s.len(), 5); }\n" },
    Challenge { name: "equals_10", prompt: "fn main() { let x: i32 = ", suffix: "; assert_eq!(x, 10); }\n" },
    Challenge { name: "option_some_5", prompt: "fn main() { let o: Option<i32> = ", suffix: "; assert_eq!(o, Some(5)); }\n" },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_reference_completions_defined() {
        for c in CHALLENGES {
            let body = impls::completion_for(c.name);
            assert!(!body.trim().is_empty(), "missing completion for {}", c.name);
            // Assemble source — grading in the real loop uses cargo; here we
            // only check the slot is non-empty and pairs with a known prompt.
            let src = format!("{}{}{}", c.prompt, body, c.suffix);
            assert!(src.contains("fn main"), "{}", c.name);
        }
    }
}
