//! AgenticGeneratorActor: multi-turn loop with tool dispatch.
//!
//! The classic agent loop:
//!   1. ask the model to generate up to N tokens
//!   2. scan the new text for a complete tool call
//!   3. if found: dispatch to ToolExecutor, splice result into the buffer,
//!      loop with the spliced text as the new prompt
//!   4. if not found, a stop sequence is hit, the tool keeps erroring, or
//!      the step budget is exhausted, return the buffer
//!
//! ## Termination
//!
//! Step 4 originally had one exit: "the model emitted no tool call". That is
//! not enough once the model has actually been trained to emit calls. A
//! format-SFT'd 7B answers correctly and then keeps going, emitting malformed
//! calls (`(arith/8 /5)`) that the executor rejects — 36 dispatch errors over
//! 20 problems at `max_steps=4`. Nothing was wrong with the answer; the loop
//! simply had no way to notice it was finished. Two exits close that:
//!
//!   - **truncation at the call boundary** (step 3): the model does not stop
//!     when it finishes a call — it emits the call *and then guesses what
//!     follows*, in the same chunk, without having seen the tool result.
//!     Keeping that tail was the actual defect: the guess usually contained a
//!     second copy of the call, which the next step then dispatched for real.
//!     Everything past the call is now dropped, which is what "run the tool,
//!     then continue from its result" has to mean.
//!   - **stop sequences** (`with_stop_sequences`): the chunk is cut at the
//!     first stop string appearing *after* the incoming buffer, so trailing
//!     junk is never parsed and never dispatched. A stop ends the
//!     *generation*, not the loop: if the cut text still holds an unresolved
//!     call, it is dispatched and the loop continues from the result. That is
//!     what makes `"\n"` a sensible stop — it turns the loop into
//!     one-line-at-a-time, which is exactly the granularity of a format whose
//!     lines are "the call" and "the answer".
//!   - **consecutive tool errors** (`with_max_consecutive_errors`): a model
//!     stuck emitting a call the registry cannot serve makes no progress by
//!     being asked again.
//!
//! `AgenticReport::stop_reason` records which one fired, so a caller can tell
//! a clean finish from a truncated one instead of inferring it from step
//! counts.
//!
//! The model has no built-in concept of tools; the parser detects
//! `(name args)\n` patterns. Phase 4 doesn't require a model that's
//! actually been trained to emit them — the wiring is what matters.

use std::sync::Arc;
use std::time::Duration;

use nanogpt_rs::generate::GenerateConfig;
use nanogpt_rs::Tokenizer;
use pekko_actor::{Actor, ActorContext, ActorRef};
use tokio::sync::oneshot;
use tokio::time::timeout;
use tracing::{info, warn};

use crate::model_actor::{ModelActor, ModelMessage};
use crate::tool_executor_actor::{ToolExecutorActor, ToolExecutorMessage};
use crate::tools::{parse_first_tool_call, splice_result};

pub enum AgenticMessage {
    /// Run the generate↔tool-execute loop starting from `prompt`.
    Run {
        prompt: String,
        sampling: GenerateConfig,
        max_steps: usize,
        reply: oneshot::Sender<anyhow::Result<AgenticReport>>,
    },
}

#[derive(Debug, Clone)]
pub struct AgenticReport {
    pub final_text: String,
    pub tool_calls: usize,
    pub steps: usize,
    /// Per-step records — useful for debugging the loop.
    pub trace: Vec<StepRecord>,
    /// Why the loop ended. `StepBudget` is the one to watch: it means the
    /// loop never concluded on its own and `final_text` may be mid-thought.
    pub stop_reason: StopReason,
}

/// Why `run_loop` returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// A configured stop sequence appeared in the new text. The clean exit.
    StopSequence,
    /// The model generated no tool call, so there was nothing left to do.
    NoToolCall,
    /// `max_consecutive_errors` dispatches in a row failed.
    ToolErrors,
    /// `max_steps` exhausted — the loop was cut off, not finished.
    StepBudget,
}

#[derive(Debug, Clone)]
pub struct StepRecord {
    pub step: usize,
    pub generated_tokens: usize,
    pub tool_called: Option<String>,
    pub tool_result: Option<Result<String, String>>,
}

/// Generic over the model actor so the same multi-turn loop drives either
/// backend: the nanogpt `ModelActor` (Phase 4) or `QwenModelActor` (Phase 21+).
/// The default type parameter keeps every existing call site compiling
/// unchanged. Mirrors how `EvaluatorActor<M>` was opened up for the same
/// reason — `ActorRef<QwenModelActor>` is a different type from
/// `ActorRef<ModelActor>`, so without this the 7B model cannot be driven
/// agentically at all.
pub struct AgenticGeneratorActor<M = ModelActor>
where
    M: Actor<Message = ModelMessage>,
{
    pub model: ActorRef<M>,
    pub executor: ActorRef<ToolExecutorActor>,
    pub tokenizer: Arc<Tokenizer>,
    pub per_request_timeout: Duration,
    /// Strings that end a generation chunk when they appear in new text.
    /// Empty by default: a wrong stop sequence truncates good output, so it
    /// is opt-in per task. `["\n"]` suits any line-oriented format.
    pub stop_sequences: Vec<String>,
    /// Consecutive failed dispatches tolerated before giving up.
    pub max_consecutive_errors: usize,
}

impl<M> AgenticGeneratorActor<M>
where
    M: Actor<Message = ModelMessage>,
{
    pub fn new(
        model: ActorRef<M>,
        executor: ActorRef<ToolExecutorActor>,
        tokenizer: Arc<Tokenizer>,
    ) -> Self {
        Self {
            model,
            executor,
            tokenizer,
            per_request_timeout: Duration::from_secs(60),
            stop_sequences: Vec::new(),
            max_consecutive_errors: 2,
        }
    }

    /// Stop generating once any of these appears in the *new* text.
    ///
    /// Matched against the region past the incoming buffer only, so a stop
    /// string already present in the prompt (the arithmetic prompt ends in
    /// `Q: a+b=`) does not fire before the model has written anything.
    ///
    /// Ends the chunk, not the loop — a call inside the cut text is still
    /// dispatched and still continues.
    pub fn with_stop_sequences<I, S>(mut self, seqs: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.stop_sequences = seqs.into_iter().map(Into::into).collect();
        self
    }

    /// Consecutive failed dispatches tolerated before the loop gives up.
    pub fn with_max_consecutive_errors(mut self, n: usize) -> Self {
        self.max_consecutive_errors = n;
        self
    }

    fn apply_stop(&self, full: &str, prefix_len: usize) -> Option<String> {
        cut_at_stop(full, prefix_len, &self.stop_sequences)
    }

    async fn generate_chunk(
        &self,
        text: &str,
        sampling: &GenerateConfig,
    ) -> anyhow::Result<(String, usize)> {
        let prompt_ids = self.tokenizer.encode(text)?;
        let (tx, rx) = oneshot::channel();
        self.model
            .tell(ModelMessage::GenerateTokens {
                prompt_ids: prompt_ids.clone(),
                cfg: sampling.clone(),
                reply: tx,
            })
            .map_err(|e| anyhow::anyhow!("send GenerateTokens: {e:?}"))?;
        let tokens = timeout(self.per_request_timeout, rx).await???;
        let new_token_count = tokens.len().saturating_sub(prompt_ids.len());
        let full = self.tokenizer.decode(&tokens)?;
        Ok((full, new_token_count))
    }

    async fn dispatch_tool(
        &self,
        call: &crate::tools::ToolCall,
    ) -> anyhow::Result<Result<String, String>> {
        let (tx, rx) = oneshot::channel();
        self.executor
            .tell(ToolExecutorMessage::Execute {
                call: call.clone(),
                reply: tx,
            })
            .map_err(|e| anyhow::anyhow!("send Execute: {e:?}"))?;
        let result = timeout(self.per_request_timeout, rx).await??;
        match result {
            Ok(s) => Ok(Ok(s)),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn run_loop(
        &self,
        prompt: String,
        sampling: GenerateConfig,
        max_steps: usize,
    ) -> anyhow::Result<AgenticReport> {
        let mut buffer = prompt;
        let mut tool_calls = 0usize;
        let mut trace = Vec::with_capacity(max_steps);
        let mut consecutive_errors = 0usize;
        let mut stop_reason = StopReason::StepBudget;

        for step in 0..max_steps {
            // Generate continuation against the running buffer. The parser
            // skips already-resolved calls (those carrying the marker),
            // so scanning the full text is safe and lets us catch tool
            // calls that arrived inside the prompt.
            let (full, new_tokens) = self.generate_chunk(&buffer, &sampling).await?;

            // Cut trailing junk BEFORE parsing, so a malformed call the model
            // rambled out after finishing is never dispatched.
            let (full, hit_stop) = match self.apply_stop(&full, buffer.len()) {
                Some(cut) => (cut, true),
                None => (full, false),
            };

            if let Some((range, call)) = parse_first_tool_call(&full) {
                let res = self.dispatch_tool(&call).await?;
                let inserted = match &res {
                    Ok(s) => s.clone(),
                    Err(e) => format!("ERR:{e}"),
                };
                buffer = resolve_call(&full, range, &inserted, buffer.len());
                tool_calls += 1;
                let failed = res.is_err();
                trace.push(StepRecord {
                    step,
                    generated_tokens: new_tokens,
                    tool_called: Some(call.name.clone()),
                    tool_result: Some(res),
                });
                info!(step, tool = %call.name, inserted = %inserted, "tool call resolved");

                if failed {
                    consecutive_errors += 1;
                    if consecutive_errors >= self.max_consecutive_errors {
                        warn!(step, consecutive_errors, "tool keeps failing; loop done");
                        stop_reason = StopReason::ToolErrors;
                        break;
                    }
                } else {
                    consecutive_errors = 0;
                }
                // Deliberately does NOT break on `hit_stop`. The stop ended
                // the chunk; the call inside it still needs its result fed
                // back, which is the whole point of the loop.
                continue;
            }

            // No tool call this step — accept the new tokens and stop.
            buffer = full;
            trace.push(StepRecord {
                step,
                generated_tokens: new_tokens,
                tool_called: None,
                tool_result: None,
            });
            stop_reason = if hit_stop {
                StopReason::StopSequence
            } else {
                StopReason::NoToolCall
            };
            info!(step, new_tokens, ?stop_reason, "loop done");
            break;
        }

        Ok(AgenticReport {
            final_text: buffer,
            tool_calls,
            steps: trace.len(),
            trace,
            stop_reason,
        })
    }
}

/// Splice a tool result in, dropping whatever the model generated after the
/// call.
///
/// That tail was produced *without* the tool result, so it is a guess about
/// the very thing we just went and computed. In practice it is usually a
/// duplicate of the call itself, which the next step then dispatches for
/// real — the source of 21 spurious dispatches over 20 arithmetic problems.
///
/// Only newly generated calls are truncated. A call that arrived inside the
/// prompt (Phase 4's `agentic_arithmetic` plants one) sits before
/// `prefix_len` and has a legitimate tail that must survive.
fn resolve_call(
    full: &str,
    range: std::ops::Range<usize>,
    inserted: &str,
    prefix_len: usize,
) -> String {
    let base = if range.end > prefix_len {
        &full[..range.end]
    } else {
        full
    };
    splice_result(base, range, inserted)
}

/// Cut `full` at the first stop sequence occurring past `prefix_len`.
///
/// The stop sequence itself is kept: for `"\nQ:"` the newline that ends the
/// answer line belongs to the answer, and dropping it would corrupt text a
/// verifier then parses line-wise.
///
/// Searching only past `prefix_len` is what makes stop sequences usable at
/// all here — the loop re-generates against the whole running buffer, so a
/// stop string that already occurred earlier in the conversation would
/// otherwise terminate every subsequent step immediately.
fn cut_at_stop(full: &str, prefix_len: usize, stops: &[String]) -> Option<String> {
    let search_from = prefix_len.min(full.len());
    if !full.is_char_boundary(search_from) {
        return None;
    }
    stops
        .iter()
        .filter(|s| !s.is_empty())
        .filter_map(|s| {
            full[search_from..]
                .find(s.as_str())
                .map(|i| search_from + i + s.len())
        })
        .min()
        .filter(|&end| end < full.len())
        .map(|end| full[..end].to_string())
}

impl<M> Actor for AgenticGeneratorActor<M>
where
    M: Actor<Message = ModelMessage>,
{
    type Message = AgenticMessage;

    fn receive(
        &mut self,
        msg: Self::Message,
        _ctx: &mut ActorContext<Self>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + '_>> {
        Box::pin(async move {
            match msg {
                AgenticMessage::Run {
                    prompt,
                    sampling,
                    max_steps,
                    reply,
                } => {
                    let r = self.run_loop(prompt, sampling, max_steps).await;
                    if let Err(e) = &r {
                        warn!(error = %e, "agentic loop failed");
                    }
                    let _ = reply.send(r);
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen_model_actor::QwenModelActor;

    /// The point of the generic parameter is that the *7B* backend can be
    /// driven agentically. A build-only check is not enough: the struct must
    /// actually instantiate at `QwenModelActor`, which is a different type
    /// from the default `ModelActor`. This is a compile-time assertion — it
    /// needs no GPU and no weights, and it fails the moment someone
    /// re-hardcodes the model type.
    #[test]
    fn agentic_generator_accepts_both_backends() {
        fn assert_is_actor<A: Actor<Message = AgenticMessage>>() {}
        assert_is_actor::<AgenticGeneratorActor<ModelActor>>();
        assert_is_actor::<AgenticGeneratorActor<QwenModelActor>>();
        // The bare form must keep resolving to the historical default, so
        // existing call sites are unaffected.
        assert_is_actor::<AgenticGeneratorActor>();
    }

    /// The bug the stop sequence exists to fix: the SFT'd 7B answers, then
    /// rambles a malformed call. Without a stop, that junk is what the parser
    /// sees and the executor is asked to dispatch `arith/8`.
    #[test]
    fn stop_sequence_cuts_trailing_junk() {
        let prompt = "Q: 8+5=\n";
        let full = "Q: 8+5=\nA: 13\nQ: 1+1=\n(arith/8 /5)\n";
        let stops = vec!["\nQ:".to_string()];
        let cut = cut_at_stop(full, prompt.len(), &stops).expect("stop must fire");
        assert_eq!(cut, "Q: 8+5=\nA: 13\nQ:");
        assert!(
            parse_first_tool_call(&cut).is_none(),
            "the malformed call must be cut away, not dispatched"
        );
    }

    /// The prompt itself ends in `Q: a+b=`, so a naive search would match at
    /// offset 0 and end the loop before the model ever generated anything.
    #[test]
    fn stop_sequence_ignores_occurrences_inside_the_prompt() {
        let prompt = "Q: 8+5=\n";
        let full = "Q: 8+5=\n(arith add 8 5)\n";
        let stops = vec!["Q:".to_string()];
        assert!(cut_at_stop(full, prompt.len(), &stops).is_none());
        // ...and the turn-1 call therefore still parses and dispatches.
        let (_, call) = parse_first_tool_call(full).unwrap();
        assert_eq!(call.args, "add 8 5");
    }

    #[test]
    fn no_stop_sequences_configured_never_cuts() {
        let full = "Q: 8+5=\nA: 13\nQ: 1+1=\n";
        assert!(cut_at_stop(full, 8, &[]).is_none());
    }

    /// Multi-byte text must not panic the byte-offset search. The resolved
    /// marker is itself 3 bytes, so this is on the live path.
    #[test]
    fn stop_search_is_utf8_safe() {
        let full = "Q: 8+5=\n(arith add 8 5\u{2192}13)\nA: 13\nQ: x\n";
        let stops = vec!["\nQ:".to_string()];
        let cut = cut_at_stop(full, "Q: 8+5=\n".len(), &stops).expect("stop must fire");
        assert!(cut.ends_with("A: 13\nQ:"));
    }

    /// The defect behind the 21 spurious dispatches: the model emits the call
    /// and then, in the same chunk, guesses what comes after it — including a
    /// second copy of the call. Keeping that tail hands the next step a real
    /// call to dispatch.
    #[test]
    fn generated_tail_after_a_call_is_dropped() {
        let prompt = "Q: 5+5=\n";
        let full = "Q: 5+5=\n(arith add 5 5)\nA: \n(arith add 5 5)\n# 10\n";
        let (range, _) = parse_first_tool_call(full).unwrap();
        let next = resolve_call(full, range, "10", prompt.len());
        assert_eq!(next, "Q: 5+5=\n(arith add 5 5\u{2192}10)\n");
        assert!(
            parse_first_tool_call(&next).is_none(),
            "the duplicate call must be gone, not queued for dispatch: {next:?}"
        );
    }

    /// Phase 4's `agentic_arithmetic` plants a call in the prompt and expects
    /// the text around it to survive. Truncating there would delete the
    /// caller's own input.
    #[test]
    fn tail_of_a_call_planted_in_the_prompt_survives() {
        let prompt = "use the tool: (arith add 5 5)\nthen answer.\n";
        let (range, _) = parse_first_tool_call(prompt).unwrap();
        let next = resolve_call(prompt, range, "10", prompt.len());
        assert_eq!(
            next,
            "use the tool: (arith add 5 5\u{2192}10)\nthen answer.\n"
        );
    }

    #[test]
    fn default_type_parameter_is_model_actor() {
        fn same_type<T>(_: &T, _: &T) {}
        // If the default ever changes, this stops compiling.
        let a: Option<AgenticGeneratorActor> = None;
        let b: Option<AgenticGeneratorActor<ModelActor>> = None;
        same_type(&a, &b);
    }
}
