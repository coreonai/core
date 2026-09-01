pub fn count_keys(_pairs: &[(&str, i32)]) -> usize {
    todo!("HashMap insert + len — remember the import")
}

pub fn greet(_name: &str) -> String {
    todo!("return hi <name>")
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
        Color::Green => todo!("repair: name for Green"),
    }
}
