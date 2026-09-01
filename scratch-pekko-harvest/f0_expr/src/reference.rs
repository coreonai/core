pub fn completion_for(name: &str) -> &'static str {
    match name {
        "equals_5" => "2 + 3",
        "equals_14_via_doubling" => "7",
        "len_5_string" => "\"hello\"",
        "equals_10" => "5 + 5",
        "option_some_5" => "Some(5)",
        _ => "",
    }
}
