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
//! *followed by a newline*. So the real constraint is narrow: **no internal
//! `)` may be immediately followed by a newline.** Everything else is fair
//! game.
//!
//!   - Internal parens are fine (`print(len(x))`) — none is followed by a
//!     newline.
//!   - **Multi-line snippets work**, as long as no line ends in `)`. The
//!     self-improve loop discovered this on its own and uses it:
//!
//!     ```text
//!     (python import math
//!     print(sum(1 for i in range(1,12+1) if math.gcd(i,12)==1)))
//!     ```
//!
//!     `python3 -c` accepts embedded newlines, so this runs. An earlier
//!     version of this comment claimed the snippet had to be one line and
//!     told the reader to use `;`. That was a description of the author's
//!     assumption, not of the parser — the model read the grammar more
//!     carefully than the docs did.
//!   - A line that *does* end in `)` closes the call early and leaves the
//!     rest dangling, which surfaces as `SyntaxError: unexpected EOF`.
//!
//! The result is spliced back inline as `(python ...→result)\n`, so it must
//! also be one line: newlines in the output are escaped to `\n` rather than
//! dropped, and the whole thing is capped (see [`PythonTool::max_output`]).
//!
//! ## Sandboxing
//!
//! The snippet is code a language model wrote. By default it runs under
//! [`bubblewrap`](https://github.com/containers/bubblewrap) with no network,
//! no view of the real filesystem, its own PID and IPC namespaces, and hard
//! resource limits. Each property below was verified against a snippet that
//! tries to break it, in the tests at the bottom of this file — a sandbox
//! nobody attacked is a claim, not a boundary.
//!
//! | property | how | verified by |
//! |---|---|---|
//! | no network | `--unshare-all` | `sandbox_blocks_network` |
//! | no `$HOME`, no `/raid` | only `/usr`, `/lib`, `/lib64`, `/bin` are bound | `sandbox_hides_the_filesystem` |
//! | system dirs read-only | `--ro-bind` | `sandbox_blocks_writes_outside_tmp` |
//! | scratch space | `--tmpfs /tmp`, discarded per call | `sandbox_gives_a_private_tmp` |
//! | memory bounded | `ulimit -v` | `sandbox_blocks_a_memory_bomb` |
//! | processes bounded | `ulimit -u` | `sandbox_blocks_a_fork_bomb` |
//! | file size bounded | `ulimit -f` | `sandbox_blocks_a_huge_write` |
//! | wall clock bounded | kill after [`PythonTool::timeout`] | `kills_a_runaway_snippet` |
//!
//! **It fails closed.** If `bwrap` is missing the call errors rather than
//! running unconfined, because the alternative is a security boundary that
//! disappears silently on a machine that happens not to have it installed.
//! Dropping the sandbox is possible but has to be said out loud:
//! [`PythonTool::without_sandbox`].
//!
//! The interpreter must live under a bound system prefix. The default
//! `python3` on a developer machine is often a pyenv shim under `$HOME`,
//! which the sandbox deliberately cannot see, so the sandboxed default is
//! `/usr/bin/python3`.
//!
//! What this does **not** do: it is one process boundary, not a VM. It does
//! not defend against a kernel exploit, and it does not stop the snippet
//! burning a core for [`PythonTool::timeout`].

use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use super::{Tool, ToolError};

/// Isolation applied to a snippet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sandbox {
    /// bubblewrap namespaces plus `ulimit`s. The default.
    Bubblewrap {
        /// Memory ceiling in KB (`ulimit -v`). 500 MB by default: enough for
        /// ordinary arithmetic and string work, far below anything that
        /// threatens a host also holding a 28 GB model.
        address_space_kb: u64,
        /// Process ceiling (`ulimit -u`).
        max_processes: u64,
        /// File size ceiling in KB (`ulimit -f`).
        max_file_kb: u64,
    },
    /// No isolation. Never a default, and named so it cannot be selected by
    /// accident.
    Disabled,
}

impl Default for Sandbox {
    fn default() -> Self {
        Self::Bubblewrap {
            address_space_kb: 500_000,
            max_processes: 32,
            max_file_kb: 1024,
        }
    }
}

pub struct PythonTool {
    /// Interpreter to invoke. Under the sandbox this must be a path the
    /// sandbox can see — `$HOME` is deliberately not bound, so a pyenv shim
    /// will not resolve.
    pub interpreter: String,
    /// Isolation applied to every snippet. See the module docs.
    pub sandbox: Sandbox,
    /// Wall clock budget for one snippet. A snippet that overruns is killed
    /// and reported as an error; the agentic loop's own per-request timeout
    /// is much longer, so without this a `while True:` would hang the turn.
    pub timeout: Duration,
    /// Cap on the returned string. The result is spliced back into the
    /// prompt, so an unbounded print would blow the context window.
    pub max_output: usize,
}

/// Interpreter used when sandboxed. `python3` on PATH is frequently a pyenv
/// shim under `$HOME`, which the sandbox cannot see by design.
const SANDBOX_INTERPRETER: &str = "/usr/bin/python3";

impl Default for PythonTool {
    fn default() -> Self {
        Self {
            interpreter: SANDBOX_INTERPRETER.to_string(),
            sandbox: Sandbox::default(),
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

    /// Run snippets with **no isolation**, as the current user, with the real
    /// filesystem and network.
    ///
    /// Only defensible when the input cannot come from outside — a local
    /// research loop over prompts you wrote. Anything reachable by a user is
    /// remote code execution. Named in full so it cannot be reached by
    /// fiddling with a boolean.
    pub fn without_sandbox(mut self) -> Self {
        self.sandbox = Sandbox::Disabled;
        self.interpreter = "python3".to_string();
        self
    }

    pub fn with_sandbox(mut self, sandbox: Sandbox) -> Self {
        self.sandbox = sandbox;
        self
    }

    /// Build the command. Under the sandbox this is
    /// `bwrap … -- bash -c 'ulimit …; exec <interp> -I -c <code>'`.
    ///
    /// `bash`, not `sh`: dash's `ulimit` has no `-u`, so on a `sh` that is
    /// dash the process limit silently does not apply and a fork bomb runs
    /// unbounded. That is exactly the kind of quiet gap a sandbox must not
    /// have, so the shell is pinned.
    fn build_command(&self, code: &str) -> Result<Command, ToolError> {
        let (asz, nproc, fsz) = match &self.sandbox {
            Sandbox::Disabled => {
                let mut c = Command::new(&self.interpreter);
                c.arg("-I").arg("-c").arg(code);
                return Ok(c);
            }
            Sandbox::Bubblewrap {
                address_space_kb,
                max_processes,
                max_file_kb,
            } => (*address_space_kb, *max_processes, *max_file_kb),
        };
        if which_bwrap().is_none() {
            return Err(ToolError::ExecutionFailed(
                "bwrap (bubblewrap) not found; refusing to run model-written code \
                 unconfined. Install bubblewrap, or opt out explicitly with \
                 PythonTool::without_sandbox()."
                    .into(),
            ));
        }
        let mut c = Command::new("bwrap");
        c.arg("--unshare-all") // net, pid, ipc, uts, cgroup, user
            .arg("--die-with-parent")
            .arg("--new-session"); // no controlling tty to inject into
                                   // Only what the interpreter needs to start. Notably absent: /home,
                                   // /raid, /etc, and every mount the host has.
        for dir in ["/usr", "/lib", "/lib64", "/bin"] {
            if std::path::Path::new(dir).exists() {
                c.arg("--ro-bind").arg(dir).arg(dir);
            }
        }
        c.arg("--proc")
            .arg("/proc")
            .arg("--dev")
            .arg("/dev")
            .arg("--tmpfs")
            .arg("/tmp")
            .arg("--chdir")
            .arg("/tmp")
            .arg("/bin/bash")
            .arg("-c")
            .arg(format!(
                "ulimit -v {asz}; ulimit -u {nproc}; ulimit -f {fsz}; exec {} -I -c \"$0\"",
                shell_quote(&self.interpreter)
            ))
            // Passed as $0 rather than interpolated, so no amount of quoting
            // in the snippet can escape into the shell command.
            .arg(code);
        Ok(c)
    }
}

fn which_bwrap() -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|p| p.join("bwrap"))
            .find(|p| p.is_file())
    })
}

/// Single-quote for `sh`. Only ever applied to the interpreter path, which is
/// operator-supplied; the model's snippet is passed as an argv entry and is
/// never interpolated into a shell string.
fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
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

        let mut child = self
            .build_command(code)?
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

    /// The sandbox cannot be exercised where bubblewrap is absent. Skipping
    /// is the right call for portability, but it must be visible: a silent
    /// skip turns "the sandbox holds" into "nobody checked".
    fn sandboxed() -> Option<PythonTool> {
        if which_bwrap().is_none() {
            eprintln!("SKIP: bwrap not installed — sandbox properties NOT verified here");
            return None;
        }
        if !std::path::Path::new(SANDBOX_INTERPRETER).exists() {
            eprintln!("SKIP: {SANDBOX_INTERPRETER} missing — sandbox NOT verified here");
            return None;
        }
        Some(PythonTool::new())
    }

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

    // --- sandbox: each of these is an attempt to break out ---

    #[test]
    fn sandbox_blocks_network() {
        let Some(t) = sandboxed() else { return };
        let out = t
            .with_timeout(Duration::from_secs(15))
            .execute(
                "import socket\n\
                 try:\n \
                    socket.create_connection(('1.1.1.1',53),timeout=3); print('REACHED')\n\
                 except Exception as e: print('blocked')",
            )
            .unwrap();
        assert_eq!(out, "blocked", "network was reachable from the sandbox");
    }

    #[test]
    fn sandbox_hides_the_filesystem() {
        let Some(t) = sandboxed() else { return };
        // $HOME and this repo's own tree must not exist inside.
        let out = t
            .execute("import os; print(os.path.exists('/home'), os.path.exists('/raid'))")
            .unwrap();
        assert_eq!(out, "False False", "host filesystem visible: {out}");
    }

    #[test]
    fn sandbox_blocks_writes_outside_tmp() {
        let Some(t) = sandboxed() else { return };
        let out = t
            .execute(
                "try:\n open('/usr/pwned','w'); print('WROTE')\n\
                 except Exception: print('blocked')",
            )
            .unwrap();
        assert_eq!(out, "blocked");
    }

    /// The scratch dir is a fresh tmpfs per call, so one snippet cannot leave
    /// anything for the next.
    #[test]
    fn sandbox_gives_a_private_tmp() {
        let Some(t) = sandboxed() else { return };
        assert_eq!(
            t.execute("open('/tmp/x','w').write('1'); print('ok')")
                .unwrap(),
            "ok"
        );
        assert_eq!(
            t.execute("import os; print(os.path.exists('/tmp/x'))")
                .unwrap(),
            "False",
            "tmp leaked between calls"
        );
    }

    #[test]
    fn sandbox_blocks_a_memory_bomb() {
        let Some(t) = sandboxed() else { return };
        let out = t
            .with_timeout(Duration::from_secs(20))
            .execute(
                "try:\n x='a'*(10**10); print('ALLOCATED')\nexcept Exception: print('blocked')",
            )
            .unwrap();
        assert_eq!(out, "blocked");
    }

    #[test]
    fn sandbox_blocks_a_fork_bomb() {
        let Some(t) = sandboxed() else { return };
        let out = t
            .with_timeout(Duration::from_secs(20))
            .execute(
                "import os\n\
                 n=0\n\
                 try:\n \
                    for _ in range(300):\n  \
                        if os.fork()==0: os._exit(0)\n  \
                        n+=1\n\
                 except Exception: print('blocked')\n\
                 else: print('FORKED', n)",
            )
            .unwrap();
        assert_eq!(out, "blocked", "process limit did not apply");
    }

    #[test]
    fn sandbox_blocks_a_huge_write() {
        let Some(t) = sandboxed() else { return };
        let out = t
            .with_timeout(Duration::from_secs(20))
            .execute(
                "try:\n open('/tmp/big','w').write('x'*(200*1024*1024)); print('WROTE')\n\
                 except Exception: print('blocked')",
            )
            .unwrap();
        assert_eq!(out, "blocked");
    }

    /// Ordinary work must still succeed inside all of the above.
    #[test]
    fn sandbox_still_runs_real_code() {
        let Some(t) = sandboxed() else { return };
        assert_eq!(
            t.execute("import math; print(sum(1 for i in range(1,46) if math.gcd(i,45)==1))")
                .unwrap(),
            "24"
        );
    }

    /// Fail closed: with no sandbox binary available the call must error, not
    /// quietly run model-written code as the server user.
    #[test]
    fn missing_sandbox_binary_fails_closed() {
        let t = PythonTool::new().with_sandbox(Sandbox::Bubblewrap {
            address_space_kb: 1,
            max_processes: 1,
            max_file_kb: 1,
        });
        // Only meaningful where bwrap is absent; where present, the limits
        // above are so small that execution fails anyway — either way the
        // call must not succeed.
        assert!(t.execute("print(1)").is_err());
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
