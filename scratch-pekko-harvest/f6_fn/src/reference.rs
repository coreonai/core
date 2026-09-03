pub fn shout(s: &str) -> String {
    s.trim().to_uppercase()
}

pub fn parse_kv(s: &str) -> Option<(String, String)> {
    s.split_once('=').map(|(k, v)| (k.to_string(), v.to_string()))
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
    xs.iter().copied().filter(|x| x % 2 == 0).sum()
}
