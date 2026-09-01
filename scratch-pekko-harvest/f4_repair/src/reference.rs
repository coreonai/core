use std::collections::HashMap;

pub fn count_keys(pairs: &[(&str, i32)]) -> usize {
    let mut m = HashMap::new();
    for (k, v) in pairs {
        m.insert(*k, *v);
    }
    m.len()
}

pub fn greet(name: &str) -> String {
    format!("hi {name}")
}

#[derive(Debug, Clone, Copy)]
pub enum Color {
    Red,
    Blue,
    Green,
}

pub fn color_name(c: Color) -> &'static str {
    match c {
        Color::Red => "red",
        Color::Blue => "blue",
        Color::Green => "green",
    }
}
