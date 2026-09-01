//! F4: broken → fixed pairs. `broken` modules fail to typecheck if compiled
//! alone; `reference` holds repairs. Student starts as copy of broken APIs
//! with todos.

pub mod broken;
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
    fn hashmap_insert_works() {
        assert_eq!(impls::count_keys(&[("a", 1), ("b", 2)]), 2);
    }

    #[test]
    fn actor_boundary_accepts_str() {
        assert_eq!(impls::greet("paul"), "hi paul");
    }

    #[test]
    fn match_is_exhaustive_for_color() {
        assert_eq!(impls::color_name(impls::Color::Red), "red");
        assert_eq!(impls::color_name(impls::Color::Blue), "blue");
        assert_eq!(impls::color_name(impls::Color::Green), "green");
    }
}
