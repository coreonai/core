//! F6: function-level bodies (API + failing tests in the prompt; student fills one fn).
//! Golds are short 1–3 statement bodies, not v10-sized loops/matches.

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
        assert_eq!(impls::shout(" Ab "), "AB");
        assert_eq!(impls::shout(""), "");
    }

    #[test]
    fn parse_kv_split_once() {
        assert_eq!(
            impls::parse_kv("name=ada"),
            Some(("name".into(), "ada".into()))
        );
        assert_eq!(
            impls::parse_kv("a=b=c"),
            Some(("a".into(), "b=c".into()))
        );
        assert_eq!(impls::parse_kv("nope"), None);
        assert_eq!(impls::parse_kv(""), None);
    }

    #[test]
    fn grade_three_bands() {
        assert_eq!(impls::grade(95), "A");
        assert_eq!(impls::grade(90), "A");
        assert_eq!(impls::grade(89), "P");
        assert_eq!(impls::grade(60), "P");
        assert_eq!(impls::grade(59), "F");
        assert_eq!(impls::grade(-3), "F");
    }

    #[test]
    fn sum_evens_mixed() {
        assert_eq!(impls::sum_evens(&[1, 2, 3, 4]), 6);
        assert_eq!(impls::sum_evens(&[]), 0);
        assert_eq!(impls::sum_evens(&[-2, 1, 0, 8]), 6);
    }
}
