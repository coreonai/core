pub fn shout(s: &str) -> String {
    s.to_uppercase()
}

pub fn parse_kv(s: &str) -> bool {
    s.contains('=')
}

pub fn grade(score: i32) -> &'static str {
    if score >= 90 {
        "A"
    } else if score >= 60 {
        "P"
    } else {
        "F"
    }
}

pub fn sum_evens(xs: &[i32]) -> i32 {
    xs.iter().sum()
}
