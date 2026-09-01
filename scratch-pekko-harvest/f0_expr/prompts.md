# F0 — expression slot (retention baseline)

Mirror of `llm-actors` `RustCodeDomain` challenges. Distinct prompt prefixes.

1. `fn main() { assert_eq!(` … `, 5); }`
2. `fn main() { assert_eq!(2 * (` … `), 14); }`
3. `fn main() { let s: &str = ` … `; assert_eq!(s.len(), 5); }`
4. `fn main() { let x: i32 = ` … `; assert_eq!(x, 10); }`
5. `fn main() { let o: Option<i32> = ` … `; assert_eq!(o, Some(5)); }`

Paraphrases for harvest sampling (same slots, different comments/names):
- "fill so the assert passes"
- "complete the RHS"
- "what expression makes this compile and succeed?"
