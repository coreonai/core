//! F6: function-level bodies (API + failing tests in the prompt; student fills one fn).

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
    fn shout_trims_and_bangs() {
        assert_eq!(impls::shout(" hi "), "HI!");
        assert_eq!(impls::shout("Ab"), "AB!");
        assert_eq!(impls::shout("  "), "");
        assert_eq!(impls::shout(""), "");
    }

    #[test]
    fn parse_kv_split_once() {
        assert_eq!(
            impls::parse_kv("name=ada"),
            Some(("name".into(), "ada".into()))
        );
        assert_eq!(
            impls::parse_kv(" a = b=c "),
            Some(("a".into(), "b=c".into()))
        );
        assert_eq!(impls::parse_kv("=x"), None);
        assert_eq!(impls::parse_kv("nope"), None);
        assert_eq!(impls::parse_kv(""), None);
    }

    #[test]
    fn grade_bands() {
        assert_eq!(impls::grade(95), "A");
        assert_eq!(impls::grade(90), "A");
        assert_eq!(impls::grade(80), "B");
        assert_eq!(impls::grade(70), "C");
        assert_eq!(impls::grade(60), "D");
        assert_eq!(impls::grade(59), "F");
        assert_eq!(impls::grade(-3), "F");
    }

    #[test]
    fn sum_evens_loop() {
        assert_eq!(impls::sum_evens(&[1, 2, 3, 4]), 6);
        assert_eq!(impls::sum_evens(&[]), 0);
        assert_eq!(impls::sum_evens(&[-2, -1, 0, 5]), -2);
        assert_eq!(impls::sum_evens(&[1, 3, 5]), 0);
    }
}
