//! InferenceServerActor: external-facing request/reply for inference.
//!
//! The contract is transport-neutral, so the same actor is reachable over an
//! actor-internal channel and over HTTP ([`crate::inference_http`]).
//!
//! ## Generic over the model, and tool-aware
//!
//! Phase 4 hardcoded `ActorRef<ModelActor>` here, which meant the serving
//! path could only ever drive the small nanoGPT model — the 7B built in Phase
//! 22/23 was reachable from example binaries and nowhere else. It is now
//! generic over any actor speaking [`ModelMessage`], so `QwenModelActor`
//! plugs in unchanged. The default type parameter keeps every existing call
//! site compiling; this mirrors what `AgenticGeneratorActor` and
//! `EvaluatorActor` already needed.
//!
//! Tool use is opt-in per request. Wire an [`AgenticGeneratorActor`] with
//! [`InferenceServerActor::with_agent`] and set `tools: true` on a request to
//! route it through generate → dispatch → splice → continue instead of a
//! single generation. Without an agent wired, a request asking for tools is
//! refused rather than silently answered without them — a caller that asked
//! for a tool-backed answer must not receive an ungrounded one and be unable
//! to tell.
//!
//! ## Grounding is checked, not assumed
//!
//! A tool running is not the same as the answer coming from it. Measured in
//! this repo: on a Collatz problem the python call raised `NameError` and the
//! model stated `A: 111` anyway — the right number, produced by the model,
//! not by the tool. Nothing in the completion text distinguishes that from a
//! computed answer.
//!
//! So the server compares the stated answer against what the tools actually
//! returned and reports [`InferenceResponse::grounded`]. With
//! `require_grounded` set, an ungrounded answer is an error instead of a
//! response, which is the setting to use when "tool-backed" is a promise
//! being made to someone.
//!
//! **This check is format-bound.** It knows one convention — the answer is
//! the last `A: <value>` line, and a grounded value appears verbatim in a
//! tool result. That fits the tool-use format trained in Phase 23. A model
//! that derives its answer from a tool result (sums a returned list, say)
//! would be marked ungrounded despite using the tool correctly, so this is a
//! guard for a known format, not a general-purpose truth check.

use std::time::{Duration, Instant};

use nanogpt_rs::generate::GenerateConfig;
use pekko_actor::{Actor, ActorContext, ActorRef};
use tokio::sync::oneshot;
use tokio::time::timeout;
use tracing::{info, warn};

use crate::agentic_generator_actor::{AgenticGeneratorActor, AgenticMessage};
use crate::model_actor::{ModelActor, ModelMessage};

#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub prompt: String,
    pub sampling: GenerateConfig,
    /// Optional client-supplied id (for logging / tracing).
    pub request_id: Option<String>,
    /// Run the agentic loop (generate → dispatch → splice → continue) rather
    /// than a single generation. Requires an agent wired via
    /// [`InferenceServerActor::with_agent`]; the request fails if none is.
    pub tools: bool,
    /// Loop budget when `tools` is set. `0` uses the server default.
    pub max_steps: usize,
    /// Fail the request when the stated answer did not come from a tool.
    /// Only meaningful together with `tools`.
    pub require_grounded: bool,
}

impl InferenceRequest {
    /// A plain single-shot request — the Phase 4 behaviour.
    pub fn new(prompt: String, sampling: GenerateConfig) -> Self {
        Self {
            prompt,
            sampling,
            request_id: None,
            tools: false,
            max_steps: 0,
            require_grounded: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InferenceResponse {
    pub request_id: Option<String>,
    pub completion: String,
    pub tokens: Vec<u32>,
    pub elapsed_ms: u128,
    /// Tools dispatched while answering. `0` on the single-shot path.
    pub tool_calls: usize,
    /// Each `(tool, args, result)` the loop ran, in order. A caller checking
    /// whether an answer is grounded needs to see what actually executed —
    /// this repo has measured models stating an answer the tool never
    /// produced, so "it used a tool" cannot be inferred from the text.
    pub tool_trace: Vec<ToolCallRecord>,
    /// Why the agentic loop stopped, when it ran.
    pub stop_reason: Option<String>,
    /// Whether the stated answer was actually produced by a tool.
    /// `None` when tools were not used, or no answer line was found.
    pub grounded: Option<bool>,
}

/// Outcome of comparing a stated answer against what the tools returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Grounding {
    /// The stated answer appears verbatim in a tool result.
    Grounded,
    /// An answer was stated that no tool produced. The failure mode this
    /// exists to catch.
    Ungrounded,
    /// No answer line to check.
    NoAnswer,
}

/// Compare the last `A: <value>` line against successful tool results.
///
/// Exact match on the trimmed value: for the Phase 23 format an answer is a
/// single scalar that the model copies from the tool, so anything else is
/// either invention or arithmetic the tool did not do. See the module docs on
/// why that is a format-bound rule rather than a general one.
pub fn assess_grounding(final_text: &str, trace: &[ToolCallRecord]) -> Grounding {
    let Some(answer) = final_text
        .lines()
        .rfind(|l| l.trim_start().starts_with("A:"))
        .map(|l| l.trim_start().trim_start_matches("A:").trim())
        .filter(|a| !a.is_empty())
    else {
        return Grounding::NoAnswer;
    };
    let produced = trace
        .iter()
        .filter_map(|c| c.result.as_ref().ok())
        .any(|r| r.trim() == answer);
    if produced {
        Grounding::Grounded
    } else {
        Grounding::Ungrounded
    }
}

#[derive(Debug, Clone)]
pub struct ToolCallRecord {
    pub tool: String,
    pub args: String,
    pub result: Result<String, String>,
}

pub enum InferenceMessage {
    Serve {
        req: InferenceRequest,
        reply: oneshot::Sender<anyhow::Result<InferenceResponse>>,
    },
}

pub struct InferenceServerActor<M = ModelActor>
where
    M: Actor<Message = ModelMessage>,
{
    pub model: ActorRef<M>,
    pub per_request_timeout: Duration,
    /// Optional agentic loop, for requests that ask for tools.
    pub agent: Option<ActorRef<AgenticGeneratorActor<M>>>,
    /// Loop budget when a request does not specify one.
    pub default_max_steps: usize,
}

impl<M> InferenceServerActor<M>
where
    M: Actor<Message = ModelMessage>,
{
    pub fn new(model: ActorRef<M>) -> Self {
        Self {
            model,
            per_request_timeout: Duration::from_secs(60),
            agent: None,
            default_max_steps: 4,
        }
    }

    /// Enable tool use. Requests with `tools: true` are then routed through
    /// the agentic loop.
    pub fn with_agent(mut self, agent: ActorRef<AgenticGeneratorActor<M>>) -> Self {
        self.agent = Some(agent);
        self
    }

    pub fn with_default_max_steps(mut self, n: usize) -> Self {
        self.default_max_steps = n;
        self
    }

    async fn handle_agentic(&self, req: InferenceRequest) -> anyhow::Result<InferenceResponse> {
        let t0 = Instant::now();
        let Some(agent) = &self.agent else {
            anyhow::bail!(
                "request asked for tools but no agent is wired \
                 (build the server with InferenceServerActor::with_agent)"
            );
        };
        let max_steps = if req.max_steps == 0 {
            self.default_max_steps
        } else {
            req.max_steps
        };
        let (tx, rx) = oneshot::channel();
        agent
            .tell(AgenticMessage::Run {
                prompt: req.prompt,
                sampling: req.sampling,
                max_steps,
                reply: tx,
            })
            .map_err(|e| anyhow::anyhow!("send Run: {e:?}"))?;
        let report = timeout(self.per_request_timeout, rx).await???;
        let tool_trace: Vec<ToolCallRecord> = report
            .trace
            .iter()
            .filter_map(|s| {
                let tool = s.tool_called.clone()?;
                Some(ToolCallRecord {
                    tool,
                    args: s.tool_args.clone().unwrap_or_default(),
                    result: s.tool_result.clone()?,
                })
            })
            .collect();
        let grounding = assess_grounding(&report.final_text, &tool_trace);
        if req.require_grounded && grounding != Grounding::Grounded {
            // Deliberately an error, not a response with a flag the caller
            // might not read. The request asked for a tool-backed answer.
            anyhow::bail!(
                "answer is not grounded in a tool result ({grounding:?}); tools ran: {}",
                report.tool_calls
            );
        }
        Ok(InferenceResponse {
            request_id: req.request_id,
            completion: report.final_text,
            // The loop re-generates against the running buffer, so there is
            // no single token sequence to hand back.
            tokens: Vec::new(),
            elapsed_ms: t0.elapsed().as_millis(),
            tool_calls: report.tool_calls,
            tool_trace,
            stop_reason: Some(format!("{:?}", report.stop_reason)),
            grounded: match grounding {
                Grounding::Grounded => Some(true),
                Grounding::Ungrounded => Some(false),
                Grounding::NoAnswer => None,
            },
        })
    }

    async fn handle(&self, req: InferenceRequest) -> anyhow::Result<InferenceResponse> {
        let t0 = Instant::now();
        let (tx, rx) = oneshot::channel();
        self.model
            .tell(ModelMessage::Generate {
                prompt: req.prompt,
                cfg: req.sampling,
                reply: tx,
            })
            .map_err(|e| anyhow::anyhow!("send Generate: {e:?}"))?;
        let reply = timeout(self.per_request_timeout, rx).await???;
        Ok(InferenceResponse {
            request_id: req.request_id,
            completion: reply.text,
            tokens: reply.tokens,
            elapsed_ms: t0.elapsed().as_millis(),
            tool_calls: 0,
            tool_trace: Vec::new(),
            stop_reason: None,
            grounded: None,
        })
    }
}

impl<M> Actor for InferenceServerActor<M>
where
    M: Actor<Message = ModelMessage>,
{
    type Message = InferenceMessage;

    fn receive(
        &mut self,
        msg: Self::Message,
        _ctx: &mut ActorContext<Self>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + '_>> {
        Box::pin(async move {
            match msg {
                InferenceMessage::Serve { req, reply } => {
                    info!(
                        req_id = ?req.request_id,
                        prompt_len = req.prompt.len(),
                        tools = req.tools,
                        "serving inference"
                    );
                    let r = if req.tools {
                        self.handle_agentic(req).await
                    } else {
                        self.handle(req).await
                    };
                    if let Err(e) = &r {
                        warn!(error = %e, "inference failed");
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

    fn ok(tool: &str, args: &str, result: &str) -> ToolCallRecord {
        ToolCallRecord {
            tool: tool.into(),
            args: args.into(),
            result: Ok(result.into()),
        }
    }
    fn err(tool: &str, args: &str, e: &str) -> ToolCallRecord {
        ToolCallRecord {
            tool: tool.into(),
            args: args.into(),
            result: Err(e.into()),
        }
    }

    /// The measured failure this check exists for. On Collatz n=27 the python
    /// call raised `NameError` and the model stated `A: 111` — the correct
    /// answer, produced by the model rather than the tool. Being right is not
    /// being grounded.
    #[test]
    fn the_collatz_case_is_ungrounded() {
        let trace = [err(
            "python",
            "print(sum(1 for i in itertools.takewhile(...)))",
            "NameError: name 'itertools' is not defined",
        )];
        let text = "Q: how many steps ...?\n(python ...\u{2192}ERR:...)\nA: 111\n";
        assert_eq!(assess_grounding(text, &trace), Grounding::Ungrounded);
    }

    #[test]
    fn an_answer_copied_from_the_tool_is_grounded() {
        let trace = [ok("python", "print(sum(...))", "30")];
        let text = "Q: how many divisors does 720 have?\n(python ...\u{2192}30)\nA: 30\n";
        assert_eq!(assess_grounding(text, &trace), Grounding::Grounded);
    }

    /// The loop can revise: a first call answers the wrong question, a second
    /// answers the right one. Grounding must follow the LAST answer, not the
    /// first — taking the first would mark a successful self-correction as
    /// ungrounded.
    #[test]
    fn grounding_follows_the_final_answer() {
        let trace = [
            ok("python", "count primes below 1000", "168"),
            ok("python", "[...][14]", "47"),
        ];
        let text = "Q: what is the 15th prime number?\nA: 168\nA: 47\n";
        assert_eq!(assess_grounding(text, &trace), Grounding::Grounded);
    }

    /// A value no tool returned, even though tools ran and succeeded.
    #[test]
    fn an_invented_value_is_ungrounded() {
        let trace = [ok("python", "print(sum(...))", "17575")];
        let text = "A: 20826\n";
        assert_eq!(assess_grounding(text, &trace), Grounding::Ungrounded);
    }

    #[test]
    fn no_answer_line_is_neither() {
        let trace = [ok("python", "x", "1")];
        assert_eq!(
            assess_grounding("(python x\u{2192}1)\n", &trace),
            Grounding::NoAnswer
        );
        // An empty `A:` is not an answer either.
        assert_eq!(assess_grounding("A:   \n", &trace), Grounding::NoAnswer);
    }

    #[test]
    fn no_tools_at_all_cannot_be_grounded() {
        assert_eq!(assess_grounding("A: 30\n", &[]), Grounding::Ungrounded);
    }

    /// Float formatting is preserved end to end, so an answer the tool really
    /// produced still matches.
    #[test]
    fn float_results_match_verbatim() {
        let trace = [ok("python", "print(60*(60/45))", "80.0")];
        assert_eq!(assess_grounding("A: 80.0\n", &trace), Grounding::Grounded);
        // ...but a reformatted value is not a verbatim match, and the module
        // docs say so: this is a format-bound check, not a numeric one.
        assert_eq!(assess_grounding("A: 80\n", &trace), Grounding::Ungrounded);
    }
}
