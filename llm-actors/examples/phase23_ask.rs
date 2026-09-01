//! Ask the Phase 23 model a question and watch it use its tool.
//!
//! Every other Phase 23 example runs a fixed problem list to produce a
//! number. This one exists to *use* the thing: type a question, see the call
//! it writes, the value the interpreter returns, and the answer it gives.
//!
//! ```text
//! $ phase23_ask --checkpoint scratch-7b-sft/p23_si_all10/r0_merged.r1.safetensors
//! ask> how many trailing zeros does 100! have?
//!   call   (python print(sum([100//5**i for i in range(1,10)])))
//!   tool   24
//!   answer A: 24
//! ```
//!
//! ## What this model actually is
//!
//! Qwen2.5-Coder-7B base, format-SFT'd to emit `(python <code>)` and then to
//! use what comes back, then self-improved on ten families of small
//! integer-valued questions. Scope is narrow and worth stating plainly:
//!
//!   - It expects one short question with a numeric answer, phrased like the
//!     training distribution (`how many …?`, `what is the … of N?`).
//!   - It is strong on counting/number-theory shapes it was harvested on and
//!     on close neighbours (divisor counts transfer at 4/4).
//!   - It is **not** a chat model. No instruction following, no multi-turn
//!     conversation, no prose.
//!   - Two documented weaknesses show up immediately if you go looking:
//!     it invents closed forms for things like Fibonacci (the code runs and
//!     the mathematics is wrong), and it references modules it was never
//!     trained to import — `itertools.takewhile` without `import itertools`.
//!     See `docs/phase23-tooluse-self-improve.md`.
//!
//! ## Reading the output
//!
//! `tool` is what the interpreter actually printed. `answer` is what the
//! model then said. When those disagree, the model is not using its tool —
//! the property `phase23_python_tool_7b --sabotage` was built to check, and
//! it is worth knowing when it fails.
//!
//! F32 by default: a dense code completion corrupts at F16 even on inputs the
//! model has memorised (CLAUDE.md gotcha #11).

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device};
use clap::Parser;
use llm_actors::{
    tools::{arithmetic_tool::ArithmeticTool, python_tool::PythonTool, Tool, ToolRegistry},
    AgenticGeneratorActor, AgenticMessage, QwenModelActor, StopReason, ToolExecutorActor,
};
use nanogpt_rs::{generate::GenerateConfig, Tokenizer as NgptTokenizer};
use pekko_actor::ActorSystem;
use tokio::sync::oneshot;

#[derive(Parser, Debug)]
struct Args {
    /// Self-improved checkpoint. Without one you get the base model, which
    /// does not emit dispatchable calls 0-shot.
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    #[arg(long, default_value = "Qwen2.5-Coder-7B")]
    model_id: String,
    #[arg(long)]
    model_dir: Option<PathBuf>,
    /// Ask one question and exit. Without it, reads questions from stdin.
    #[arg(long)]
    question: Option<String>,
    #[arg(long, default_value = "f32")]
    dtype: String,
    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,
    /// 0.0 is greedy. The model is sharply peaked after self-improve, so
    /// sampling mostly costs correctness here.
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,
    #[arg(long, default_value_t = 4)]
    max_steps: usize,
    /// Stop sequences, matched against newly generated text only.
    ///
    /// Default is the CALL BOUNDARY, not a bare newline. A newline stop cuts
    /// a multi-line snippet in half — the self-improved model writes
    ///
    /// ```text
    /// (python import math
    /// print(sum(1 for i in range(1,46) if math.gcd(i,45)==1)))
    /// ```
    ///
    /// and `"\n"` truncates it to `(python import math\n`, which never closes
    /// and so never dispatches. That is a property of the stop sequence, not
    /// of the model, and it silently converts a working call into "emitted no
    /// call". `")\n"` ends the chunk exactly where a call completes.
    #[arg(long, num_args = 1.., default_values_t = vec![")\n".to_string()])]
    stop: Vec<String>,
    /// Print the raw trajectory as well as the parsed view.
    #[arg(long)]
    raw: bool,
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

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::ERROR)
        .init();
    let args = Args::parse();
    let device = pick_device();
    if !device.is_cuda() && std::env::var("PHASE22_ALLOW_CPU").is_err() {
        anyhow::bail!("Refusing to run on CPU. Rebuild with `--features cuda`.");
    }
    let dtype = match args.dtype.as_str() {
        "f32" => DType::F32,
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        other => anyhow::bail!("unknown --dtype {other:?}"),
    };

    let snapshot = resolve_snapshot(args.model_dir.as_deref(), &args.model_id)?;
    let tk = Arc::new(NgptTokenizer::from_hf_file(
        snapshot.join("tokenizer.json"),
    )?);

    eprint!("loading 7B ({dtype:?})... ");
    let model = match &args.checkpoint {
        Some(ckpt) => {
            let cfg_text = std::fs::read_to_string(snapshot.join("config.json"))?;
            let config: candle_transformers::models::qwen2::Config =
                serde_json::from_str(&cfg_text)?;
            let hf = tokenizers::Tokenizer::from_file(snapshot.join("tokenizer.json"))
                .map_err(|e| anyhow!("tokenizer: {e}"))?;
            QwenModelActor::new(ckpt.clone(), Arc::new(hf), config, device.clone(), dtype)?
        }
        None => {
            eprintln!("\nWARNING: no --checkpoint; the base model does not emit calls 0-shot.");
            QwenModelActor::from_snapshot_dir(&snapshot, device.clone(), dtype)?
        }
    };
    // Control tokens outrank code here — the base derails into
    // `(python<|fim_prefix|>` without this. EOS is kept so generation ends.
    let tj: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(snapshot.join("tokenizer.json"))?)?;
    let suppress: Vec<u32> = tj["added_tokens"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter(|t| t["content"].as_str().unwrap_or("") != "<|endoftext|>")
                .filter_map(|t| t["id"].as_u64().map(|x| x as u32))
                .collect()
        })
        .unwrap_or_default();
    let model = model.with_suppressed_tokens(suppress);

    let registry = ToolRegistry::from_tools(vec![
        Arc::new(PythonTool::new()) as Arc<dyn Tool>,
        Arc::new(ArithmeticTool) as Arc<dyn Tool>,
    ]);
    let system = ActorSystem::new("phase23-ask");
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
            .with_stop_sequences(args.stop.iter().filter(|s| !s.is_empty()).cloned()),
            "agentic",
        )
        .await?;
    eprintln!("ready.");

    let ask = |q: String| {
        let agent_ref = agent_ref.clone();
        let raw = args.raw;
        let (max_steps, max_new_tokens, temperature) =
            (args.max_steps, args.max_new_tokens, args.temperature);
        async move {
            let prompt = format!("Q: {}\n", q.trim());
            let (tx, rx) = oneshot::channel();
            agent_ref
                .tell(AgenticMessage::Run {
                    prompt: prompt.clone(),
                    sampling: GenerateConfig {
                        max_new_tokens,
                        temperature,
                        top_k: (temperature > 0.0).then_some(40),
                        top_p: (temperature > 0.0).then_some(0.95),
                        seed: Some(0),
                    },
                    max_steps,
                    reply: tx,
                })
                .map_err(|e| anyhow!("{e:?}"))?;
            let report = tokio::time::timeout(Duration::from_secs(300), rx).await???;

            for step in &report.trace {
                if let (Some(name), Some(a)) = (&step.tool_called, &step.tool_args) {
                    println!("  call   ({name} {a})");
                }
                match &step.tool_result {
                    Some(Ok(v)) => println!("  tool   {v}"),
                    Some(Err(e)) => println!("  tool   ERROR: {e}"),
                    None => {}
                }
            }
            // The answer line the format trains, if the model produced one.
            let answer = report
                .final_text
                .strip_prefix(prompt.as_str())
                .unwrap_or(&report.final_text)
                .lines()
                .find(|l| l.trim_start().starts_with("A:"))
                .unwrap_or("(no answer line)");
            println!("  answer {}", answer.trim());
            if report.stop_reason == StopReason::StepBudget {
                println!("  note   ran out of steps — the answer may be unfinished");
            }
            if raw {
                println!("  raw    {:?}", report.final_text);
            }
            Ok::<(), anyhow::Error>(())
        }
    };

    if let Some(q) = args.question {
        return ask(q).await;
    }

    println!("Ask a question with a numeric answer. Ctrl-D to quit.");
    let stdin = std::io::stdin();
    loop {
        print!("ask> ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            println!();
            return Ok(());
        }
        if line.trim().is_empty() {
            continue;
        }
        if let Err(e) = ask(line).await {
            println!("  error  {e}");
        }
    }
}
