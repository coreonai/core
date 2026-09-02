pub fn shout(s: &str) -> String {
    let t = s.trim();
    if t.is_empty() {
        String::new()
    } else {
        format!("{}!", t.to_ascii_uppercase())
    }
}

pub fn parse_kv(s: &str) -> Option<(String, String)> {
    let (k, v) = s.split_once('=')?;
    let k = k.trim();
    let v = v.trim();
    if k.is_empty() {
        None
    } else {
        Some((k.to_string(), v.to_string()))
    }
}

pub fn grade(score: i32) -> &'static str {
    match score {
        s if s >= 90 => "A",
        s if s >= 80 => "B",
        s if s >= 70 => "C",
        s if s >= 60 => "D",
        _ => "F",
    }
}

pub fn sum_evens(xs: &[i32]) -> i32 {
    let mut acc = 0;
    for n in xs {
        if n % 2 == 0 {
            acc += *n;
        }
    }
    acc
}
