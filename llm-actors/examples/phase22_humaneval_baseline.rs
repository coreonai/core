//! Phase 22 Stage A — HumanEval baseline against Qwen2.5-Coder-0.5B
//! via the Pekko pipeline.
//!
//! Reproduces Phase 17 S6's Python-side measurement (base pass@1 ≈
//! 0.216, pass@10 ≈ 0.524 on the full 164 HumanEval set) **but
//! through the Rust actor stack instead of standalone Python**:
//!
//!   QwenModelActor (Stage D inference)
//!     ↑ ModelMessage::GenerateTokens
//!   EvaluatorActor::<QwenModelActor> (Stage E generic eval, A pass@k)
//!     ↑ EvaluatorMessage::Eval { passk }
//!   HumanEvalDomain (Stage 22-A — python3 subprocess verify)
//!
//! No Python is in the GENERATION path. Python is only invoked
//! by `HumanEvalDomain::verify` for each completion — same as
//! Phase 15+'s scripts.
//!
//! Run on a small subset:
//!   cargo run -p llm-actors --example phase22_humaneval_baseline \
//!       --features cuda --release -- --n-problems 8 --passk 1
//!
//! Run the full 164-problem benchmark:
//!   cargo run -p llm-actors --example phase22_humaneval_baseline \
//!       --features cuda --release -- --n-problems 164 --passk 1
//!
//! pass@k > 1 sweeps:
//!   --passk 5      # pass@5 (~5× wallclock)
//!   --passk 10     # pass@10 (matches Phase 17 S6 baseline at 0.524)
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device};
use clap::Parser;
use llm_actors::{
    domain::{human_eval::HumanEvalDomain, Domain},
    EvaluatorActor, EvaluatorMessage, QwenModelActor,
};
use nanogpt_rs::{generate::GenerateConfig, Tokenizer as NgptTokenizer};
use pekko_actor::ActorSystem;
use tokio::sync::oneshot;

#[derive(Parser, Debug)]
struct Args {
    /// Number of HumanEval problems to sample. Use 164 to run the full
    /// benchmark; smaller values for quick smoke runs.
    #[arg(long, default_value_t = 8)]
    n_problems: usize,
    /// pass@k value (1 = greedy baseline).
    #[arg(long, default_value_t = 1)]
    passk: usize,
    /// max tokens per completion. Phase 17 used 200.
    #[arg(long, default_value_t = 200)]
    max_new_tokens: usize,
    /// HumanEval JSONL path. Defaults to workspace's `data/humaneval/HumanEval.jsonl`.
    #[arg(long)]
    jsonl: Option<PathBuf>,
    /// Eval RNG seed (determines which problems are sampled).
    #[arg(long, default_value_t = 7)]
    seed: u64,
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

fn resolve_default_snapshot() -> Result<PathBuf> {
    let home = std::env::var("HOME").context("HOME unset")?;
    let snapshots_dir = PathBuf::from(format!(
        "{home}/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B/snapshots"
    ));
    let entries = std::fs::read_dir(&snapshots_dir)
        .with_context(|| format!("read_dir {snapshots_dir:?}"))?
        .collect::<Result<Vec<_>, _>>()?;
    entries
        .into_iter()
        .map(|e| e.path())
        .find(|p| p.is_dir() && p.join("config.json").exists())
        .ok_or_else(|| anyhow!("no snapshot under {snapshots_dir:?} has a config.json"))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .init();

    let args = Args::parse();
    let device = pick_device();
    let on_cuda = device.is_cuda();
    println!("[Phase22A] device = {device:?}, on_cuda = {on_cuda}");
    // Use BF16 on CUDA to match Qwen2.5-Coder's native HF format and
    // Phase 17 S6's measurement. F16 has same exponent range but less
    // mantissa precision; some completions silently drift.
    let dtype = if on_cuda { DType::BF16 } else { DType::F32 };

    let snapshot = resolve_default_snapshot()?;
    println!("[Phase22A] snapshot = {}", snapshot.display());
    let qwen = QwenModelActor::from_snapshot_dir(&snapshot, device, dtype)?;
    let tk = Arc::new(NgptTokenizer::from_hf_file(
        snapshot.join("tokenizer.json"),
    )?);

    let jsonl = args
        .jsonl
        .unwrap_or_else(|| PathBuf::from("data/humaneval/HumanEval.jsonl"));
    let scratch = std::env::temp_dir().join("workllm-humaneval-baseline-scratch");
    let domain = HumanEvalDomain::from_jsonl(&jsonl, &scratch)
        .with_context(|| format!("loading HumanEval from {}", jsonl.display()))?;
    let n_total = domain.n_problems();
    println!(
        "[Phase22A] HumanEvalDomain loaded {} problems (using {} for this run)",
        n_total, args.n_problems
    );
    let domain: Arc<dyn Domain> = Arc::new(domain);

    let system = ActorSystem::new("phase22-a");
    let model_ref = system.spawn(qwen, "qwen-model").await?;
    let evaluator = EvaluatorActor::<QwenModelActor>::new(model_ref.clone(), tk, domain, None);
    let evaluator_ref = system.spawn(evaluator, "evaluator").await?;

    // For pass@1: greedy (temp=0). For pass@k>1: sample with the Phase
    // 17 hyperparameters (temp=0.8, top_p=0.95).
    let (temperature, top_k, top_p) = if args.passk > 1 {
        (0.8, Some(40usize), Some(0.95f64))
    } else {
        (0.0, Some(1usize), None)
    };
    let sampling = GenerateConfig {
        max_new_tokens: args.max_new_tokens,
        temperature,
        top_k,
        top_p,
        seed: Some(args.seed),
    };

    println!(
        "[Phase22A] starting eval n={} passk={} temperature={} top_k={:?} top_p={:?}",
        args.n_problems, args.passk, temperature, top_k, top_p
    );
    let t0 = std::time::Instant::now();
    let (tx, rx) = oneshot::channel();
    evaluator_ref
        .tell(EvaluatorMessage::Eval {
            n: args.n_problems,
            seed: args.seed,
            sampling,
            passk: args.passk,
            reply: tx,
        })
        .map_err(|e| anyhow!("{e:?}"))?;
    let report = rx.await??;
    let elapsed = t0.elapsed();
    println!(
        "\n[Phase22A] pass@{} = {:.4}  ({}/{})  elapsed={:.1}s  wallclock_per_problem={:.2}s",
        args.passk,
        report.pass_rate(),
        report.correct,
        report.total,
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() / args.n_problems.max(1) as f64,
    );
    // Sample dump
    for (i, s) in report.samples.iter().take(3).enumerate() {
        let first_line = s
            .prompt
            .lines()
            .next()
            .unwrap_or("")
            .chars()
            .take(60)
            .collect::<String>();
        println!(
            "  sample {i}  prompt[0]={:?}  completion[:80]={:?}",
            first_line,
            s.completion.chars().take(80).collect::<String>(),
        );
    }
    println!("\nphase22_humaneval_baseline: PASS");
    Ok(())
}
