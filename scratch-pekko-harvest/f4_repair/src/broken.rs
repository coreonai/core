//! Illustrative broken shapes (documentation + compile-fail intent).
//! These functions are not linked into tests; they show the seed bugs.

// Bug 1: missing HashMap import (shown as comment for the skeleton).
// use std::collections::HashMap;  // REQUIRED
// pub fn count_keys(pairs: &[(&str, i32)]) -> usize {
//     let mut m = HashMap::new();
//     for (k, v) in pairs { m.insert(*k, *v); }
//     m.len()
// }

// Bug 2: String vs &str
// pub fn greet(name: &str) -> String { format!("hi {name}") }
// pub fn call() { let s = String::from("x"); greet(s); } // should be &s or as_str

// Bug 3: non-exhaustive after adding Green
// pub enum Color { Red, Blue, Green }
// pub fn color_name(c: Color) -> &'static str {
//     match c { Color::Red => "red", Color::Blue => "blue" }
// }
