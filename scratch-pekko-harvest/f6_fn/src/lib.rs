//! F6: function-level bodies (API + failing tests in the prompt; student fills one fn).
//! Curriculum: shout / sum_evens stay v11 one-liners; grade stays v13 A/P/F; ONLY parse_kv grows to split_once.

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
    fn parse_kv_split_once() {
        assert_eq!(
            impls::parse_kv("name=ada"),
            Some(("name".into(), "ada".into()))
        );
        assert_eq!(impls::parse_kv("k="), Some(("k".into(), "".into())));
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
    fn sum_evens_all_even() {
        assert_eq!(impls::sum_evens(&[2, 4]), 6);
        assert_eq!(impls::sum_evens(&[]), 0);
        assert_eq!(impls::sum_evens(&[-2, 0, 8]), 6);
    }
}
