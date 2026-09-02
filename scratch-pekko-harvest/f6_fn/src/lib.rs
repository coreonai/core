//! F6: function-level bodies (API + failing tests in the prompt; student fills one fn).
//! Golds are F1-sized (one expression / few tokens), not tens-of-token loops.

pub mod reference;
#[cfg(feature = "student")]
pub mod student;

#[cfg(feature = "student")]
pub use student as impls;
#[cfg(not(feature = "student"))]
pub use reference as impls;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shout_upper() {
        assert_eq!(impls::shout("hi"), "HI");
        assert_eq!(impls::shout("Ab"), "AB");
        assert_eq!(impls::shout(""), "");
    }

    #[test]
    fn parse_kv_has_eq() {
        assert!(impls::parse_kv("name=ada"));
        assert!(impls::parse_kv("a=b=c"));
        assert!(!impls::parse_kv("nope"));
        assert!(!impls::parse_kv(""));
    }

    #[test]
    fn grade_pass_fail() {
        assert_eq!(impls::grade(95), "A");
        assert_eq!(impls::grade(90), "A");
        assert_eq!(impls::grade(89), "F");
        assert_eq!(impls::grade(-3), "F");
    }

    #[test]
    fn sum_evens_all_even() {
        assert_eq!(impls::sum_evens(&[2, 4]), 6);
        assert_eq!(impls::sum_evens(&[]), 0);
        assert_eq!(impls::sum_evens(&[-2, 0, 8]), 6);
    }
}
