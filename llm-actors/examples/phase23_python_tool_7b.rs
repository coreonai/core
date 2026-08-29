//! Phase 23 — the 7B driving `PythonTool` through the agentic loop.
//!
//! `phase23_agentic_7b` proved the loop works, but on `arith`, whose answers
//! the model can also just guess: a 7B knows 8+5. That makes it a wiring
//! test, not a capability test. This uses tasks the model cannot shortcut —
//! `sum(i*i for i in range(1, 97))` — so a correct final answer is evidence
//! the tool ran, not evidence the model memorised arithmetic.
//!
//! Measured separately, because they can come apart:
//!
//!   - **emitted a python call** — did the format transfer few-shot at all
//!   - **tool computed the right value** — the executor's own output
//!   - **model stated it** — did it then use what the tool returned
//!
//! The gap between the last two is the interesting one: a model that ignores
//! the tool result and answers from its own guess scores on the middle line
//! and fails the last.
//!
//! ## `--sabotage` — does the answer actually come from the tool?
//!
//! "The tool computed it and the model said it" is consistent with two very
//! different stories: the model read the result, or the model reached the
//! same number on its own and the tool was decoration. `--sabotage N`
//! separates them by wrapping the tool so it returns `true + N`. Then:
//!
//!   - the model states the **sabotaged** value → it is reading the tool
//!   - the model states the **true** value → it computed it unaided, and the
//!     unsabotaged 12/12 was not evidence of tool use at all
//!
//! Measured on the held-out sizes, F32, 0-shot:
//!
//! ```text
//!   sabotage      stated the tool's value   stated the TRUE value
//!   +0                    12/12                    12/12   (identical)
//!   +1                    12/12                     0/12
//!   +100000               10/12                     0/12
//! ```
//!
//! `0/12` true in both sabotaged conditions: the answer is tool-derived, not
//! recomputed. The two misses at +100000 did not state the true value either
//! — they state a *truncated* copy (`A: 10` for a delivered `100009`), so the
//! failure is in copying a wildly implausible magnitude, not a fallback to
//! the model's own arithmetic. It is not a tokenizer artifact: Qwen splits
//! all of 100006/100009/100013/100017 digit by digit, yet two copy cleanly
//! and two truncate. With 4 problems in that family the split is not worth a
//! mechanism story.
//!
//! Note what a "reads the tool" result does and does not mean. The turn-2
//! SFT pairs are literally "resolved call in, that number out", so copying is
//! trained behaviour, not reasoning. What the test establishes is that the
//! loop's data path — execute, splice, continue — is what determines the
//! answer, which is the property the whole mechanism depends on.
//!
//! ## Held-out
//!
//! The problem sizes below must match `PYTHON_EVAL_N` in
//! `phase23_toolcall_sft`, which excludes them from training. As with the
//! `arith` split, the rule is written out in both files rather than shared,
//! because it is the one thing a reader has to check before trusting a
//! held-out number.
//!
//! Run (base, 2-shot — measures whether prompting alone is enough):
//!   cargo run -p llm-actors --example phase23_python_tool_7b --features cuda --release
//!
//! Run (after `--tool python` SFT — 0-shot, matching the training condition):
//!   ... --checkpoint scratch-7b-sft/p23_py_sft.safetensors --shots false

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device};
use clap::Parser;
use llm_actors::{
    tools::{python_tool::PythonTool, Tool, ToolRegistry},
    AgenticGeneratorActor, AgenticMessage, QwenModelActor, StopReason, ToolExecutorActor,
};
use nanogpt_rs::{generate::GenerateConfig, Tokenizer as NgptTokenizer};
use pekko_actor::ActorSystem;
use tokio::sync::oneshot;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    model_dir: Option<PathBuf>,
    #[arg(long, default_value = "Qwen2.5-Coder-7B")]
    model_id: String,
    /// Optional format-SFT'd checkpoint. The base model is the default
    /// because "does this need SFT too?" is the question.
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    #[arg(long, default_value_t = 12)]
    n_problems: usize,
    /// Evaluate held-IN sizes instead. Mirrors `phase23_toolcall_probe`'s flag
    /// of the same name: if the held-out number is bad, this separates "the
    /// checkpoint or the inference path is broken" (fails here too) from "it
    /// memorised but did not generalise" (passes here).
    #[arg(long)]
    train_side: bool,
    #[arg(long, default_value_t = 4)]
    max_steps: usize,
    #[arg(long, default_value_t = 48)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,
    #[arg(long, default_value_t = 6)]
    show: usize,
    /// Drop the two worked examples. Few-shot measurably *hurts* a fine-tuned
    /// model here (the arith SFT went 100% → 0% exact calls when shots were
    /// added), so pass `--no-shots` with a checkpoint. Written as a negative
    /// flag because clap's derive turns a `bool` into a valueless switch —
    /// `--shots false` does not parse, and a `bool` defaulting to true can
    /// never be turned off.
    #[arg(long)]
    no_shots: bool,
    /// Inference precision. A python call is ~25 tokens of dense syntax, far
    /// more than `arith`'s ~8, and CLAUDE.md gotcha #10 records BF16 corrupting
    /// long generations outright. `f32` is the safe default here; 7B F32 is
    /// 28 GB, which fits a 40 GB card for inference.
    #[arg(long, default_value = "f32")]
    dtype: String,
    /// Perturb every tool result by this much before it reaches the model.
    /// `0` disables. See the module docs — this is the falsifier for "the
    /// answer came from the tool".
    #[arg(long, default_value_t = 0)]
    sabotage: i64,
    /// Keep Qwen's control tokens in play. They are suppressed by default:
    /// without that the base model derails into `(python<|fim_prefix|>`, the
    /// FIM tokens outranking the code it should be writing. Same finding and
    /// same fix as `phase23_toolcall_probe`.
    #[arg(long)]
    keep_special: bool,
}

fn pick_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        if let Ok(d) = Device::new_cuda(0) {
            return d;
        }
    }
    Device::Cpu
}

fn resolve_snapshot(dir: Option<&std::path::Path>, id: &str) -> Result<PathBuf> {
    if let Some(d) = dir {
        return Ok(d.to_path_buf());
    }
    let home = std::env::var("HOME").context("HOME unset")?;
    let snaps = PathBuf::from(format!(
        "{home}/.cache/huggingface/hub/models--Qwen--{id}/snapshots"
    ));
    std::fs::read_dir(&snaps)
        .with_context(|| format!("read_dir {snaps:?}"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.is_dir() && p.join("config.json").exists())
        .ok_or_else(|| anyhow!("no snapshot under {snaps:?}"))
}

/// Two worked examples, with their calls shown **resolved** (`...\u{2192}3025`).
///
/// Not a stylistic choice — the loop scans the whole running buffer, and by
/// design it dispatches a call found in the prompt (Phase 4's
/// `agentic_arithmetic` relies on exactly that). Unresolved shots are
/// therefore executed as real calls: the first version of this example
/// reported "12/12 emitted a python call" while every one of those
/// dispatches was the *shot*, not the model. Resolved shots are skipped by
/// the parser, so a dispatch here can only have come from the model.
///
/// The cost is the known trap in the other direction: a model shown resolved
/// calls tends to write `\u{2192}result` itself, which the parser then skips,
/// and nothing dispatches (2.8% vs 72% grammar in `phase23_toolcall_probe`).
/// Whether few-shot can thread that needle is what this measures — and if it
/// cannot, the answer is the same as for `arith`: SFT the format in.
const SHOTS: &str = "\
Q: sum of the first 10 cubes?
(python print(sum(i**3 for i in range(1, 11)))\u{2192}3025)
A: 3025

Q: how many primes below 50?
(python print(sum(1 for n in range(2,50) if all(n%d for d in range(2,n))))\u{2192}15)
A: 15

";

/// Tasks whose answers are not memorisable at a glance, each paired with the
/// value a correct computation yields. The sizes are `PYTHON_EVAL_N` from
/// `phase23_toolcall_sft`, held out of training.
fn problems(train_side: bool) -> Vec<(String, i64)> {
    if train_side {
        // Sizes that WERE trained on, one per family.
        let mut v: Vec<(String, i64)> = Vec::new();
        for n in [10u32, 20, 30, 40] {
            v.push((
                format!("sum of squares from 1 to {n}?"),
                (1..=n as i64).map(|i| i * i).sum(),
            ));
        }
        for n in [12u32, 22, 32, 42] {
            v.push((
                format!("sum of numbers below {n} divisible by 3 or 5?"),
                (1..n as i64).filter(|i| i % 3 == 0 || i % 5 == 0).sum(),
            ));
        }
        for n in [14u32, 24, 34, 44] {
            v.push((
                format!("how many primes below {n}?"),
                (2..n as i64)
                    .filter(|&d| (2..d).all(|k| d % k != 0))
                    .count() as i64,
            ));
        }
        return v;
    }
    let mut v: Vec<(String, i64)> = Vec::new();
    for n in [37u32, 53, 71, 96] {
        let want: i64 = (1..=n as i64).map(|i| i * i).sum();
        v.push((format!("sum of squares from 1 to {n}?"), want));
    }
    for n in [23u32, 41, 67, 88] {
        let want: i64 = (1..n as i64).filter(|i| i % 3 == 0 || i % 5 == 0).sum();
        v.push((
            format!("sum of numbers below {n} divisible by 3 or 5?"),
            want,
        ));
    }
    for n in [17u32, 29, 43, 61] {
        let want: i64 = (2..n as i64)
            .filter(|&d| (2..d).all(|k| d % k != 0))
            .count() as i64;
        v.push((format!("how many primes below {n}?"), want));
    }
    v
}

/// Wraps a tool and shifts its numeric result. Non-numeric output (an error
/// string, `<no output>`) passes through untouched — perturbing that would
/// test the model's error handling, which is a different question.
///
/// `Tool` has no defaulted methods, so this delegates everything there is to
/// delegate. If that ever changes, `assert_domain_fully_delegates!`-style
/// scrutiny applies here too: a wrapper that silently drops a defaulted
/// method is the failure mode `rust-guardrails` exists for.
struct SabotagedTool {
    inner: PythonTool,
    delta: i64,
}

impl Tool for SabotagedTool {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn execute(&self, args: &str) -> Result<String, llm_actors::tools::ToolError> {
        let out = self.inner.execute(args)?;
        match out.trim().parse::<i64>() {
            Ok(v) => Ok((v + self.delta).to_string()),
            Err(_) => Ok(out),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .init();
    let args = Args::parse();
    let device = pick_device();
    println!("[Phase23Py] device = {device:?}");
    if !device.is_cuda() && std::env::var("PHASE22_ALLOW_CPU").is_err() {
        anyhow::bail!("Refusing to run on CPU. Rebuild with `--features cuda`.");
    }

    let dtype = match args.dtype.as_str() {
        "f32" => DType::F32,
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        other => anyhow::bail!("unknown --dtype {other:?} (expected f32, f16 or bf16)"),
    };
    println!("[Phase23Py] dtype = {dtype:?}");

    let snapshot = resolve_snapshot(args.model_dir.as_deref(), &args.model_id)?;
    let tk = Arc::new(NgptTokenizer::from_hf_file(
        snapshot.join("tokenizer.json"),
    )?);

    let model = match &args.checkpoint {
        Some(ckpt) => {
            println!("[Phase23Py] checkpoint = {}", ckpt.display());
            let cfg_text = std::fs::read_to_string(snapshot.join("config.json"))?;
            let config: candle_transformers::models::qwen2::Config =
                serde_json::from_str(&cfg_text)?;
            let hf = tokenizers::Tokenizer::from_file(snapshot.join("tokenizer.json"))
                .map_err(|e| anyhow!("tokenizer: {e}"))?;
            QwenModelActor::new(ckpt.clone(), Arc::new(hf), config, device.clone(), dtype)?
        }
        None => {
            println!("[Phase23Py] BASE model");
            QwenModelActor::from_snapshot_dir(&snapshot, device.clone(), dtype)?
        }
    };

    let model = if !args.keep_special {
        let tj: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(snapshot.join("tokenizer.json"))?)?;
        // NOT filtered on the `special` flag: Qwen2.5-Coder marks the FIM and
        // repo tokens special=false, and those are precisely the ones that
        // displace the code here. Every added_token is a control token, so
        // suppress the lot and keep EOS so generation can still terminate.
        let ids: Vec<u32> = tj["added_tokens"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter(|t| t["content"].as_str().unwrap_or("") != "<|endoftext|>")
                    .filter_map(|t| t["id"].as_u64().map(|x| x as u32))
                    .collect()
            })
            .unwrap_or_default();
        println!(
            "[Phase23Py] suppressing {} control tokens (EOS kept)",
            ids.len()
        );
        model.with_suppressed_tokens(ids)
    } else {
        model
    };

    let tool: Arc<dyn Tool> = if args.sabotage != 0 {
        println!(
            "[Phase23Py] SABOTAGE: every tool result shifted by {:+}",
            args.sabotage
        );
        Arc::new(SabotagedTool {
            inner: PythonTool::new(),
            delta: args.sabotage,
        })
    } else {
        Arc::new(PythonTool::new())
    };
    let registry = ToolRegistry::from_tools(vec![tool]);
    let system = ActorSystem::new("phase23-py");
    let model_ref = system.spawn(model, "qwen-model").await?;
    let exec_ref = system
        .spawn(ToolExecutorActor::new(registry), "tool-exec")
        .await?;
    let agent_ref = system
        .spawn(
            AgenticGeneratorActor::<QwenModelActor>::new(
                model_ref.clone(),
                exec_ref.clone(),
                tk.clone(),
            )
            // Line-oriented format: one line is the call, the next the answer.
            .with_stop_sequences(["\n"]),
            "agentic",
        )
        .await?;

    let probes: Vec<(String, i64)> = problems(args.train_side)
        .into_iter()
        .take(args.n_problems)
        .collect();
    println!("[Phase23Py] {} problems", probes.len());

    let (mut emitted, mut tool_ok, mut answered) = (0usize, 0usize, 0usize);
    let (mut tool_errors, mut budget_exits, mut shown) = (0usize, 0usize, 0usize);
    let mut said_true_n = 0usize;

    for (q, want) in &probes {
        let cfg = GenerateConfig {
            max_new_tokens: args.max_new_tokens,
            temperature: args.temperature,
            top_k: if args.temperature > 0.0 {
                Some(40)
            } else {
                None
            },
            top_p: None,
            seed: Some(0),
        };
        let (tx, rx) = oneshot::channel();
        agent_ref
            .tell(AgenticMessage::Run {
                prompt: if !args.no_shots {
                    format!("{SHOTS}Q: {q}\n")
                } else {
                    format!("Q: {q}\n")
                },
                sampling: cfg,
                max_steps: args.max_steps,
                reply: tx,
            })
            .map_err(|e| anyhow!("{e:?}"))?;
        let report = tokio::time::timeout(Duration::from_secs(180), rx).await???;

        // Shots are pre-resolved, so any dispatch is the model's own call.
        let called_python = report
            .trace
            .iter()
            .any(|s| s.tool_called.as_deref() == Some("python"));
        if called_python {
            emitted += 1;
        }
        // What the tool actually handed back. Under `--sabotage` this is
        // deliberately not `want`.
        let delivered = want + args.sabotage;
        let exec_right = report
            .trace
            .iter()
            .any(|s| matches!(&s.tool_result, Some(Ok(r)) if r.trim() == delivered.to_string()));
        if exec_right {
            tool_ok += 1;
        }
        // Did the model state what the TOOL said, or what is actually TRUE?
        // Identical without sabotage; the whole point of the flag is to pull
        // them apart.
        let said_tool = report.final_text.contains(&format!("A: {delivered}"));
        let said_true = report.final_text.contains(&format!("A: {want}"));
        if said_tool {
            answered += 1;
        }
        if said_true {
            said_true_n += 1;
        }
        let said = said_tool;
        tool_errors += report
            .trace
            .iter()
            .filter(|s| matches!(&s.tool_result, Some(Err(_))))
            .count();
        if report.stop_reason == StopReason::StepBudget {
            budget_exits += 1;
        }

        if shown < args.show {
            // Only the model's own continuation matters; the shots are ours.
            let tail = report
                .final_text
                .strip_prefix(SHOTS)
                .unwrap_or(&report.final_text);
            println!(
                "  [{}] want={want} exec_ok={exec_right} said={said} stop={:?}\n      {}",
                if exec_right && said { "ok " } else { "MISS" },
                report.stop_reason,
                tail.replace('\n', "\\n")
                    .chars()
                    .take(150)
                    .collect::<String>()
            );
            shown += 1;
        }
    }

    let n = probes.len() as f64;
    println!("\n[Phase23Py] === PythonTool in the loop ===");
    println!(
        "  emitted a python call = {emitted:3}/{} ({:.0}%)",
        probes.len(),
        100.0 * emitted as f64 / n
    );
    println!(
        "  tool computed it      = {tool_ok:3}/{} ({:.0}%)",
        probes.len(),
        100.0 * tool_ok as f64 / n
    );
    println!(
        "  model stated it       = {answered:3}/{} ({:.0}%)",
        probes.len(),
        100.0 * answered as f64 / n
    );
    if args.sabotage != 0 {
        println!(
            "  ...stated TRUE value  = {said_true_n:3}/{} ({:.0}%)  <- would mean the tool was ignored",
            probes.len(),
            100.0 * said_true_n as f64 / n
        );
    }
    println!("  tool dispatch errors  = {tool_errors}");
    println!("  ran out of steps      = {budget_exits}/{}", probes.len());
    println!("\nphase23_python_tool_7b: PASS");
    Ok(())
}
