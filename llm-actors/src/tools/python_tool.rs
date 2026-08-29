//! Code-execution tool: run a Python snippet, return what it printed.
//!
//! `ArithmeticTool` can do one thing. This is the first tool whose output the
//! model cannot shortcut by guessing, which is what makes it useful as a
//! training signal: `(python print(sum(range(101))))` is cheap to execute and
//! expensive to fake.
//!
//! ## Why this could not exist before
//!
//! Phase 4 marked a dispatched call by writing `=` into its body, and
//! `parse_first_tool_call` skips any body containing that marker. `=` is the
//! most common character in source code, so every realistic snippet looked
//! "already resolved" and was silently skipped — the grammar could not
//! express a code tool at all. The marker moved to `→`
//! ([`super::RESOLVED_MARKER`]), which does not occur in Python source.
//!
//! ## Grammar constraints on the snippet
//!
//! A call is `(name args)\n`, and the parser closes at the first `)` that is
//! *followed by a newline*. Two consequences:
//!
//!   - **The snippet is one line.** Use `;` to sequence statements. Blocks
//!     that need real indentation have to go through `exec("...")`.
//!   - Internal parens are fine (`print(len(x))`), because none of them is
//!     followed by a newline.
//!
//! The result is spliced back inline as `(python ...→result)\n`, so it must
//! also be one line: newlines in the output are escaped to `\n` rather than
//! dropped, and the whole thing is capped (see [`PythonTool::max_output`]).
//!
//! ## This is not a sandbox
//!
//! The snippet runs as the current user with the current filesystem. `-I`
//! (isolated mode) only stops `PYTHON*` environment variables and the user
//! site directory from leaking in; it stops nothing else. There is a wall
//! clock timeout so a runaway loop cannot wedge the agentic loop, and that is
//! the extent of the containment. It is the same posture as `RustCodeDomain`,
//! which shells out to `cargo` — appropriate for driving a local research
//! loop over code you generated yourself, not for untrusted input.

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use super::{Tool, ToolError};

pub struct PythonTool {
    /// Interpreter to invoke. `python3` on PATH by default.
    pub interpreter: String,
    /// Wall clock budget for one snippet. A snippet that overruns is killed
    /// and reported as an error; the agentic loop's own per-request timeout
    /// is much longer, so without this a `while True:` would hang the turn.
    pub timeout: Duration,
    /// Cap on the returned string. The result is spliced back into the
    /// prompt, so an unbounded print would blow the context window.
    pub max_output: usize,
}

impl Default for PythonTool {
    fn default() -> Self {
        Self {
            interpreter: "python3".to_string(),
            timeout: Duration::from_secs(5),
            max_output: 512,
        }
    }
}

impl PythonTool {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_max_output(mut self, max_output: usize) -> Self {
        self.max_output = max_output;
        self
    }

    pub fn with_interpreter(mut self, interpreter: impl Into<String>) -> Self {
        self.interpreter = interpreter.into();
        self
    }
}

/// Fold captured output into something that can live inside a one-line tool
/// call: escape newlines, collapse the tail, and cap the length.
///
/// Truncation is marked with `…` rather than silently cutting, so a model
/// reading its own transcript can tell a short answer from a clipped one.
fn one_line(raw: &str, max: usize) -> String {
    let escaped = raw.trim_end_matches('\n').replace('\n', "\\n");
    if escaped.chars().count() <= max {
        return escaped;
    }
    let kept: String = escaped.chars().take(max).collect();
    format!("{kept}…")
}

impl Tool for PythonTool {
    fn name(&self) -> &str {
        "python"
    }

    fn execute(&self, args: &str) -> Result<String, ToolError> {
        let code = args.trim();
        if code.is_empty() {
            return Err(ToolError::BadArgs {
                tool: "python".into(),
                reason: "empty snippet".into(),
            });
        }

        let mut child = Command::new(&self.interpreter)
            .arg("-I") // ignore PYTHON* env vars and the user site dir
            .arg("-c")
            .arg(code)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| ToolError::ExecutionFailed(format!("spawn {}: {e}", self.interpreter)))?;

        // `Tool::execute` is synchronous, so this polls rather than awaiting.
        // The interval trades idle CPU against latency on fast snippets;
        // most arithmetic-scale snippets finish inside the first tick.
        let deadline = Instant::now() + self.timeout;
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) => {
                    if Instant::now() >= deadline {
                        let _ = child.kill();
                        let _ = child.wait();
                        return Err(ToolError::ExecutionFailed(format!(
                            "timed out after {:?}",
                            self.timeout
                        )));
                    }
                    std::thread::sleep(Duration::from_millis(2));
                }
                Err(e) => {
                    let _ = child.kill();
                    return Err(ToolError::ExecutionFailed(format!("wait: {e}")));
                }
            }
        };

        let mut stdout = String::new();
        if let Some(mut h) = child.stdout.take() {
            let _ = h.read_to_string(&mut stdout);
        }
        let mut stderr = String::new();
        if let Some(mut h) = child.stderr.take() {
            let _ = h.read_to_string(&mut stderr);
        }

        if !status.success() {
            // Report the *last* stderr line: a traceback's final line is the
            // exception, which is the part worth feeding back to the model.
            let msg = stderr
                .trim_end()
                .lines()
                .next_back()
                .unwrap_or("non-zero exit")
                .to_string();
            return Err(ToolError::ExecutionFailed(one_line(&msg, self.max_output)));
        }

        // A snippet that ran clean but printed nothing is a likely modelling
        // error (`sum(...)` instead of `print(sum(...))`), so say so rather
        // than splicing an empty result the model cannot interpret.
        if stdout.trim().is_empty() {
            return Ok("<no output>".to_string());
        }
        Ok(one_line(&stdout, self.max_output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::{parse_first_tool_call, splice_result, ToolRegistry};
    use std::sync::Arc;

    /// Skip the process-spawning tests where there is no interpreter, rather
    /// than failing the suite on a machine that has none.
    fn have_python() -> bool {
        Command::new("python3")
            .arg("-c")
            .arg("pass")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    #[test]
    fn runs_a_snippet() {
        if !have_python() {
            return;
        }
        let out = PythonTool::new().execute("print(sum(range(101)))").unwrap();
        assert_eq!(out, "5050");
    }

    /// The whole reason the resolved marker moved off `=`. An assignment is
    /// the most ordinary thing a snippet can contain.
    #[test]
    fn runs_a_snippet_containing_equals() {
        if !have_python() {
            return;
        }
        let out = PythonTool::new()
            .execute("x = 6; y = 7; print(x * y)")
            .unwrap();
        assert_eq!(out, "42");
    }

    /// End to end through the grammar: parse a call carrying code, dispatch
    /// it through the registry, splice the result, and confirm the resolved
    /// call does not re-fire — which is what would loop forever.
    #[test]
    fn round_trips_through_the_grammar() {
        if !have_python() {
            return;
        }
        let registry = ToolRegistry::from_tools(vec![Arc::new(PythonTool::new()) as Arc<dyn Tool>]);
        let text = "(python a = 2; print(a ** 10))\nrest";
        let (range, call) = parse_first_tool_call(text).expect("code call must parse");
        assert_eq!(call.name, "python");
        let result = registry.dispatch(&call).expect("dispatch");
        assert_eq!(result, "1024");
        let spliced = splice_result(text, range, &result);
        assert_eq!(spliced, "(python a = 2; print(a ** 10)\u{2192}1024)\nrest");
        assert!(parse_first_tool_call(&spliced).is_none());
    }

    #[test]
    fn reports_the_exception_line() {
        if !have_python() {
            return;
        }
        let err = PythonTool::new().execute("print(1/0)").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("ZeroDivisionError"), "got {msg:?}");
        assert!(!msg.contains('\n'), "must stay one line: {msg:?}");
    }

    #[test]
    fn kills_a_runaway_snippet() {
        if !have_python() {
            return;
        }
        let t0 = Instant::now();
        let err = PythonTool::new()
            .with_timeout(Duration::from_millis(300))
            .execute("while True: pass")
            .unwrap_err();
        assert!(err.to_string().contains("timed out"), "{err}");
        assert!(t0.elapsed() < Duration::from_secs(3), "kill was too slow");
    }

    #[test]
    fn empty_snippet_is_bad_args() {
        let err = PythonTool::new().execute("   ").unwrap_err();
        assert!(matches!(err, ToolError::BadArgs { .. }));
    }

    #[test]
    fn silent_snippet_says_so() {
        if !have_python() {
            return;
        }
        assert_eq!(PythonTool::new().execute("x = 1").unwrap(), "<no output>");
    }

    /// Output has to survive being spliced back into a one-line call.
    #[test]
    fn multiline_output_is_flattened_and_capped() {
        assert_eq!(one_line("a\nb\n", 100), "a\\nb");
        let capped = one_line(&"x".repeat(50), 10);
        assert_eq!(capped, format!("{}…", "x".repeat(10)));
    }

    #[test]
    fn long_output_stays_parseable_after_splicing() {
        if !have_python() {
            return;
        }
        let tool = PythonTool::new().with_max_output(32);
        let out = tool.execute("print('ab' * 500)").unwrap();
        assert!(out.chars().count() <= 33, "cap not applied: {}", out.len());
    }
}
