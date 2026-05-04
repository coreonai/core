//! Tool-use foundation.
//!
//! Phase 4 introduces an agentic loop: the model emits a tool call, an
//! executor dispatches to a registered handler, and the result is fed back
//! into the next generation step. This module defines the contract; actual
//! orchestration lives in `tool_executor_actor` and `agentic_generator_actor`.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

pub mod arithmetic_tool;

/// A parsed tool call. The grammar shipped here is intentionally minimal —
/// `(name arg1 arg2 ...)\n`. Newline anchors the call so streaming
/// generation can detect completion. Caller is responsible for stripping
/// the surrounding parens before passing `args`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCall {
    pub name: String,
    pub args: String,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum ToolError {
    #[error("unknown tool: {0}")]
    UnknownTool(String),
    #[error("bad arguments for {tool}: {reason}")]
    BadArgs { tool: String, reason: String },
    #[error("tool execution failed: {0}")]
    ExecutionFailed(String),
}

/// Synchronous tool handler. Async tools can wrap a oneshot internally.
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn execute(&self, args: &str) -> Result<String, ToolError>;
}

/// Registry of tools by name. Cloneable (Arc-shared inside).
#[derive(Clone, Default)]
pub struct ToolRegistry {
    inner: Arc<HashMap<String, Arc<dyn Tool>>>,
}

impl ToolRegistry {
    pub fn from_tools(tools: Vec<Arc<dyn Tool>>) -> Self {
        let mut inner = HashMap::with_capacity(tools.len());
        for t in tools {
            inner.insert(t.name().to_string(), t);
        }
        Self { inner: Arc::new(inner) }
    }

    pub fn dispatch(&self, call: &ToolCall) -> Result<String, ToolError> {
        match self.inner.get(&call.name) {
            Some(tool) => tool.execute(&call.args),
            None => Err(ToolError::UnknownTool(call.name.clone())),
        }
    }

    pub fn names(&self) -> Vec<&str> {
        self.inner.keys().map(|s| s.as_str()).collect()
    }
}

/// Parse the FIRST complete, *unresolved* tool call inside `text`. Returns
/// `(byte_range_in_text, ToolCall)` — caller splices the result back in.
/// `None` if no complete `( ... )\n` is found.
///
/// "Unresolved" means the body contains no `=`. After dispatch we splice
/// `(name args=result)\n` back in; the `=` marker keeps subsequent scans
/// from re-firing on the same call (avoiding an infinite loop). It also
/// means args containing `=` (e.g. `key=value`) cannot be passed — fine for
/// the minimal grammar shipped here.
pub fn parse_first_tool_call(text: &str) -> Option<(std::ops::Range<usize>, ToolCall)> {
    let bytes = text.as_bytes();
    let mut search_from = 0;
    loop {
        let open = bytes[search_from..].iter().position(|b| *b == b'(')?;
        let open = open + search_from;
        let mut close = None;
        for i in (open + 1)..bytes.len() {
            if bytes[i] == b')' && i + 1 < bytes.len() && bytes[i + 1] == b'\n' {
                close = Some(i);
                break;
            }
        }
        let close = close?;
        let body = &text[open + 1..close];
        if body.contains('=') {
            // Already-resolved call. Skip past it and keep looking.
            search_from = close + 2;
            continue;
        }
        let mut parts = body.splitn(2, char::is_whitespace);
        let name = parts.next()?.trim().to_string();
        if name.is_empty() {
            search_from = close + 2;
            continue;
        }
        let args = parts.next().unwrap_or("").trim().to_string();
        return Some((open..(close + 2), ToolCall { name, args }));
    }
}

/// Replace the first tool call with `(call_text=result)\n` so subsequent
/// generation reads the completed call inline. Idempotent on `text` with
/// no tool call.
pub fn splice_result(text: &str, range: std::ops::Range<usize>, result: &str) -> String {
    let original = &text[range.clone()];
    // strip the trailing "\n" then append the result and put back the newline.
    let trimmed = original.trim_end_matches('\n');
    let trimmed = trimmed.strip_suffix(')').unwrap_or(trimmed);
    let mut out = String::with_capacity(text.len() + result.len() + 4);
    out.push_str(&text[..range.start]);
    out.push_str(trimmed);
    out.push('=');
    out.push_str(result);
    out.push_str(")\n");
    out.push_str(&text[range.end..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_complete_call() {
        let s = "thinking...\n(add 3 4)\nrest";
        let (range, call) = parse_first_tool_call(s).unwrap();
        assert_eq!(call.name, "add");
        assert_eq!(call.args, "3 4");
        assert_eq!(&s[range], "(add 3 4)\n");
    }

    #[test]
    fn parse_returns_none_when_unclosed() {
        assert!(parse_first_tool_call("(add 3 4").is_none());
        assert!(parse_first_tool_call("(add 3 4)").is_none()); // no newline
    }

    #[test]
    fn parse_returns_none_for_empty_name() {
        assert!(parse_first_tool_call("( 3 4)\n").is_none());
    }

    #[test]
    fn splice_result_inline() {
        let s = "(add 3 4)\nmore";
        let (range, _) = parse_first_tool_call(s).unwrap();
        let out = splice_result(s, range, "7");
        assert_eq!(out, "(add 3 4=7)\nmore");
    }

    #[test]
    fn registry_dispatches_known_tool() {
        struct Echo;
        impl Tool for Echo {
            fn name(&self) -> &str { "echo" }
            fn execute(&self, args: &str) -> Result<String, ToolError> {
                Ok(args.to_string())
            }
        }
        let r = ToolRegistry::from_tools(vec![Arc::new(Echo)]);
        let call = ToolCall { name: "echo".into(), args: "hi".into() };
        assert_eq!(r.dispatch(&call).unwrap(), "hi");
    }

    #[test]
    fn parse_skips_resolved_call() {
        // Already-resolved calls (containing `=` in body) must NOT match.
        let s = "(arith add 3 4=7)\nrest";
        assert!(parse_first_tool_call(s).is_none());
    }

    #[test]
    fn parse_picks_unresolved_after_resolved() {
        // Resolved first, fresh second — parser should skip the resolved
        // one and surface the second.
        let s = "(arith add 3 4=7)\n(arith add 5 6)\n";
        let (_, call) = parse_first_tool_call(s).unwrap();
        assert_eq!(call.name, "arith");
        assert_eq!(call.args, "add 5 6");
    }

    #[test]
    fn registry_errors_on_unknown_tool() {
        let r = ToolRegistry::default();
        let call = ToolCall { name: "missing".into(), args: "".into() };
        assert!(matches!(r.dispatch(&call), Err(ToolError::UnknownTool(_))));
    }
}
