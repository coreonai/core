//! Reference Tool implementation: integer arithmetic.
//!
//! Accepts `add a b`, `mul a b`, `sub a b`, `div a b` (integer division,
//! errors on /0). Used in tests + the Phase 4 agentic_arithmetic example.

use super::{Tool, ToolError};

pub struct ArithmeticTool;

impl Tool for ArithmeticTool {
    fn name(&self) -> &str {
        "arith"
    }

    fn execute(&self, args: &str) -> Result<String, ToolError> {
        let mut iter = args.split_whitespace();
        let op = iter.next().ok_or(ToolError::BadArgs {
            tool: "arith".into(),
            reason: "missing op".into(),
        })?;
        let a: i64 = iter
            .next()
            .ok_or(ToolError::BadArgs { tool: "arith".into(), reason: "missing a".into() })?
            .parse()
            .map_err(|e: std::num::ParseIntError| ToolError::BadArgs {
                tool: "arith".into(),
                reason: format!("bad a: {e}"),
            })?;
        let b: i64 = iter
            .next()
            .ok_or(ToolError::BadArgs { tool: "arith".into(), reason: "missing b".into() })?
            .parse()
            .map_err(|e: std::num::ParseIntError| ToolError::BadArgs {
                tool: "arith".into(),
                reason: format!("bad b: {e}"),
            })?;
        let result = match op {
            "add" => a + b,
            "sub" => a - b,
            "mul" => a * b,
            "div" => {
                if b == 0 {
                    return Err(ToolError::ExecutionFailed("division by zero".into()));
                }
                a / b
            }
            other => {
                return Err(ToolError::BadArgs {
                    tool: "arith".into(),
                    reason: format!("unknown op {other}"),
                })
            }
        };
        Ok(result.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_works() {
        let t = ArithmeticTool;
        assert_eq!(t.execute("add 3 4").unwrap(), "7");
    }

    #[test]
    fn sub_works() {
        assert_eq!(ArithmeticTool.execute("sub 10 4").unwrap(), "6");
    }

    #[test]
    fn mul_works() {
        assert_eq!(ArithmeticTool.execute("mul 6 7").unwrap(), "42");
    }

    #[test]
    fn div_by_zero_errors() {
        assert!(matches!(
            ArithmeticTool.execute("div 5 0"),
            Err(ToolError::ExecutionFailed(_))
        ));
    }

    #[test]
    fn bad_args_error() {
        assert!(matches!(
            ArithmeticTool.execute("add hello"),
            Err(ToolError::BadArgs { .. })
        ));
    }
}
