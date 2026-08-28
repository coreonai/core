//! Phase 23 gate — can the 7B BASE model emit the stack's tool-call syntax?
//!
//! Before any agentic training is worth starting, we need the same number this
//! repo asks for every time: **is there headroom to harvest?** Self-improve
//! sharpens latent ability; it cannot create ability that isn't there
//! (risk #11 — cold-start prompts with 0 base pass rate never recover).
//!
//! What this measures, and what it deliberately does not:
//!
//!   - The grammar (`(name args)\n`, body must not contain `=`) is a
//!     project-specific DSL that Phase 4 *trained into* a 1M model. A base
//!     model has never seen it, so a zero-shot probe would measure "doesn't
//!     know our format", not "can't do tool use". We therefore prompt
//!     **few-shot** and ask whether the ability is *elicitable*.
//!   - This is single-turn: generate once, parse once. It does NOT run the
//!     multi-turn dispatch loop, because `AgenticGeneratorActor` is still
//!     hardcoded to `ActorRef<ModelActor>` and does not accept
//!     `QwenModelActor`. Generic-ifying it is the next step and is not needed
//!     to answer the gate question.
//!
//! Reported, in increasing strictness:
//!   grammar   — `parse_first_tool_call` returns Some (the syntax is right)
//!   tool      — ...and the tool name matches the registry
//!   args      — ...and the arguments are the right ones for the question
//!   pass@k    — at least one of k samples reached `args`
//!
//! `grammar` is the gate. If it is ~0 the format must be SFT'd in first; if it
//! is 20-50% the existing self-improve loop can bootstrap from it.
//!
//! Run (needs one free 40GB card):
//!   cargo run -p llm-actors --example phase23_toolcall_probe --features cuda --release -- \
//!       --model-id Qwen2.5-Coder-7B --n-problems 40 --passk 10

use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device};
use clap::Parser;
use llm_actors::{
    domain::tool_use::ToolUseArithmeticDomain,
    tools::{arithmetic_tool::ArithmeticTool, parse_first_tool_call, Tool, ToolRegistry},
    ModelMessage, QwenModelActor,
};
use nanogpt_rs::{generate::GenerateConfig, Tokenizer as NgptTokenizer};
use pekko_actor::ActorSystem;
use tokio::sync::oneshot;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    model_dir: Option<std::path::PathBuf>,
    #[arg(long, default_value = "Qwen2.5-Coder-7B")]
    model_id: String,
    /// Held-out (a, b) problems to probe.
    #[arg(long, default_value_t = 40)]
    n_problems: usize,
    /// Samples per problem.
    #[arg(long, default_value_t = 10)]
    passk: usize,
    /// In-context examples shown before the question.
    #[arg(long, default_value_t = 4)]
    n_shot: usize,
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,
    #[arg(long, default_value_t = 48)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 7)]
    seed: u64,
    /// Mask every *special* token except EOS out of sampling. The base model
    /// emits `<|fim_prefix|>` where ` add` belongs; this tests whether the
    /// exact-call rate is recoverable by decoding alone. EOS is left alone so
    /// the model can still terminate.
    #[arg(long, default_value_t = false)]
    suppress_special: bool,
    /// Print the first few raw completions — the failure mode matters as much
    /// as the rate (wrong syntax vs right syntax wrong args are different
    /// problems with different fixes).
    #[arg(long, default_value_t = 6)]
    show: usize,
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

fn resolve_snapshot(dir: Option<&std::path::Path>, id: &str) -> Result<std::path::PathBuf> {
    if let Some(d) = dir {
        return Ok(d.to_path_buf());
    }
    let home = std::env::var("HOME").context("HOME unset")?;
    let snaps = std::path::PathBuf::from(format!(
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
        .with_max_level(tracing::Level::WARN)
        .init();
    let args = Args::parse();
    let device = pick_device();
    let on_cuda = device.is_cuda();
    println!("[Phase23] device = {device:?}, on_cuda = {on_cuda}");
    if !on_cuda && std::env::var("PHASE22_ALLOW_CPU").is_err() {
        anyhow::bail!("Refusing to run on CPU. Rebuild with `--features cuda`.");
    }

    let snapshot = resolve_snapshot(args.model_dir.as_deref(), &args.model_id)?;
    let tk = Arc::new(NgptTokenizer::from_hf_file(
        snapshot.join("tokenizer.json"),
    )?);
    let mut model = QwenModelActor::from_snapshot_dir(&snapshot, device.clone(), DType::F16)?;
    if args.suppress_special {
        let tj: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(snapshot.join("tokenizer.json"))?)?;
        // NOT filtered on `special`. Qwen2.5-Coder marks the FIM/repo tokens
        // (<|fim_prefix|>, <|repo_name|>, ...) as special=false, and those are
        // exactly the ones the model emits where ` add` belongs — filtering on
        // the flag suppressed 14 tokens that were never the problem and left
        // all 6 that were, reproducing the unsuppressed numbers bit for bit.
        // Every added_token is a control token here, so suppress the lot and
        // keep only EOS so generation can still terminate.
        let ids: Vec<u32> = tj["added_tokens"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter(|t| t["content"].as_str() != Some("<|endoftext|>"))
                    .filter_map(|t| t["id"].as_u64().map(|x| x as u32))
                    .collect()
            })
            .unwrap_or_default();
        println!(
            "[Phase23] suppressing {} special tokens (EOS kept)",
            ids.len()
        );
        model = model.with_suppressed_tokens(ids);
    }
    let system = ActorSystem::new("phase23-probe");
    let model_ref = system.spawn(model, "qwen-model").await?;

    let domain = ToolUseArithmeticDomain::default();
    let registry = ToolRegistry::from_tools(vec![Arc::new(ArithmeticTool) as Arc<dyn Tool>]);
    let known: Vec<String> = registry.names().iter().map(|s| s.to_string()).collect();
    println!("[Phase23] registry tools = {known:?}");

    // Few-shot preamble.
    //
    // ⚠ It must demonstrate what the model is supposed to EMIT, which is the
    // *unresolved* call `(arith add a b)`. `render_full_trajectory` renders
    // the post-dispatch line `(arith add a b=r)` — that is what the model
    // *reads back* after the executor splices a result in, not what it should
    // produce. Prompting with the resolved form makes the model imitate it
    // faithfully, and `parse_first_tool_call` then skips the output by design
    // ("body contains `=`" means already-resolved). The first version of this
    // probe made exactly that mistake and measured 2.8% grammar, which said
    // more about the prompt than about the model.
    let mut shots = String::new();
    for i in 0..args.n_shot {
        let a = (i as u32 * 3 + 1) % 10;
        let b = (i as u32 * 5 + 2) % 10;
        let r = a + b;
        shots.push_str(&format!("Q: {a}+{b}=\n(arith add {a} {b})\nA: {r}\n"));
    }

    // Held-out problems, disjoint from the shot pairs by construction
    // (shots use a small deterministic set; probes are drawn from the rest).
    let mut probes: Vec<(u32, u32)> = Vec::new();
    let mut a = 2u32;
    let mut b = 7u32;
    while probes.len() < args.n_problems {
        a = (a * 7 + 3) % 10;
        b = (b * 3 + 5) % 10;
        probes.push((a, b));
    }

    let (mut n_grammar, mut n_tool, mut n_args, mut n_total) = (0usize, 0usize, 0usize, 0usize);
    let mut pass_prompts = 0usize;
    let mut shown = 0usize;

    for (pi, &(a, b)) in probes.iter().enumerate() {
        let prompt = format!("{shots}{}", domain.render_prompt(a, b));
        let prompt_ids = tk.encode(&prompt)?;
        let mut any = false;
        for k in 0..args.passk {
            let cfg = GenerateConfig {
                max_new_tokens: args.max_new_tokens,
                temperature: args.temperature,
                top_k: Some(40),
                top_p: Some(0.95),
                seed: Some(args.seed.wrapping_add(((pi as u64) << 8) ^ k as u64)),
            };
            let (tx, rx) = oneshot::channel();
            model_ref
                .tell(ModelMessage::GenerateTokens {
                    prompt_ids: prompt_ids.clone(),
                    cfg,
                    reply: tx,
                })
                .map_err(|e| anyhow!("{e:?}"))?;
            let full = rx.await??;
            let comp_ids = if full.len() > prompt_ids.len() {
                &full[prompt_ids.len()..]
            } else {
                &[][..]
            };
            let text = tk.decode(comp_ids)?;
            n_total += 1;

            match parse_first_tool_call(&text) {
                None => {
                    if shown < args.show {
                        println!(
                            "  [no-call] {a}+{b} -> {:?}",
                            text.chars().take(70).collect::<String>()
                        );
                        shown += 1;
                    }
                }
                Some((_, call)) => {
                    n_grammar += 1;
                    let tool_ok = known.iter().any(|n| n == &call.name);
                    if tool_ok {
                        n_tool += 1;
                    }
                    // The domain's convention is `(arith add A B)`.
                    let want = format!("add {a} {b}");
                    if tool_ok && call.args.trim() == want {
                        n_args += 1;
                        any = true;
                    } else if shown < args.show {
                        println!(
                            "  [call ok, args off] {a}+{b} -> ({} {}) want ({} {})",
                            call.name, call.args, "arith", want
                        );
                        shown += 1;
                    }
                }
            }
        }
        if any {
            pass_prompts += 1;
        }
    }

    let pct = |x: usize| 100.0 * x as f64 / n_total as f64;
    println!("\n[Phase23] === base tool-call probe ===");
    println!("  model      = {}", args.model_id);
    println!(
        "  prompting  = {}-shot, temp {}, k={}",
        args.n_shot, args.temperature, args.passk
    );
    println!(
        "  samples    = {n_total} ({} problems x {})",
        probes.len(),
        args.passk
    );
    println!(
        "  grammar    = {n_grammar:5} ({:.1}%)   <- THE GATE",
        pct(n_grammar)
    );
    println!("  tool ok    = {n_tool:5} ({:.1}%)", pct(n_tool));
    println!("  args ok    = {n_args:5} ({:.1}%)", pct(n_args));
    println!(
        "  pass@{}    = {:.3} ({}/{} problems)",
        args.passk,
        pass_prompts as f64 / probes.len() as f64,
        pass_prompts,
        probes.len()
    );
    println!("\n  reading: grammar ~0 -> the format must be SFT'd in before any");
    println!("  self-improve loop can harvest. grammar 20-50% -> the ability is");
    println!("  latent and the existing loop can bootstrap it (risk #11 clear).");
    println!("\nphase23_toolcall_probe: PASS");
    Ok(())
}
