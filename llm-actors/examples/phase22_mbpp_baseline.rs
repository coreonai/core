//! Phase 22 Stage C — MBPP-100 baseline against Qwen2.5-Coder-0.5B
//! via the Pekko pipeline.
//!
//! Cross-substrate companion of Stage A's `phase22_humaneval_baseline`.
//! Same actor stack, different `Domain` impl: `MbppDomain` synthesizes
//! HumanEval-style prompts from MBPP's (text, code, test_list) triples
//! exactly the way Phase 17 S3's `problems.py` did, and verifies via
//! python3 subprocess (assertions in `test_list` run at module
//! top-level — no `check(...)` call needed).
//!
//! Phase 17 S9 reference (single seed, base Qwen):
//!   pass@1 raw ≈ 0.36, pass@10 ≈ 0.66, Δ ≈ +0.30  (similar dynamics
//!   to S6's HumanEval pass@1 0.216 → pass@10 0.524 lift).
//!
//! Stage B closed the metric mismatch on HumanEval — `--sequential
//! --aggregate` here reports `total_passes / total_attempts` matching
//! Phase 17 S9's "pass@1 raw" measurement directly.
//!
//! Run on a small smoke (n=8, k=1, ~30s):
//!   cargo run -p llm-actors --example phase22_mbpp_baseline \
//!       --features cuda --release -- --n-problems 8 --passk 1
//!
//! Full MBPP-100 aggregate baseline (~30 min):
//!   cargo run -p llm-actors --example phase22_mbpp_baseline \
//!       --features cuda --release -- --n-problems 100 --passk 10 \
//!       --sequential --aggregate --max-new-tokens 200
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device};
use clap::Parser;
use llm_actors::{
    domain::{mbpp::MbppDomain, Domain},
    EvaluatorActor, EvaluatorMessage, QwenModelActor,
};
use nanogpt_rs::{generate::GenerateConfig, Tokenizer as NgptTokenizer};
use pekko_actor::ActorSystem;
use tokio::sync::oneshot;

#[derive(Parser, Debug)]
struct Args {
    /// Qwen model snapshot directory (must contain config.json,
    /// tokenizer.json, and model.safetensors or its shard index).
    /// Overrides --model-id. Point this at a 7B snapshot to run 7B.
    #[arg(long)]
    model_dir: Option<PathBuf>,
    /// HF repo suffix under `models--Qwen--<id>` used when --model-dir
    /// is absent; globs the cache snapshot that has a config.json.
    #[arg(long, default_value = "Qwen2.5-Coder-0.5B")]
    model_id: String,
    /// Number of MBPP problems to evaluate. Use 100 for the full
    /// MBPP-100 subset; smaller values for quick smokes.
    #[arg(long, default_value_t = 8)]
    n_problems: usize,
    /// pass@k value (1 = greedy baseline).
    #[arg(long, default_value_t = 1)]
    passk: usize,
    /// max tokens per completion. Phase 17 S9 used 200.
    #[arg(long, default_value_t = 200)]
    max_new_tokens: usize,
    /// MBPP JSONL path. Defaults to workspace's `data/mbpp/mbpp.jsonl`.
    #[arg(long)]
    jsonl: Option<PathBuf>,
    /// Eval RNG seed (determines which problems are sampled when
    /// `--sequential` is OFF; per-(prompt, k) seed when it's ON).
    #[arg(long, default_value_t = 7)]
    seed: u64,
    /// Phase 22 Stage B-style sequential / no-replacement sweep over
    /// `domain.nth_prompt(0..n)`. Required for bit-exact Phase 17 S9
    /// reproduction.
    #[arg(long, default_value_t = false)]
    sequential: bool,
    /// Phase 22 Stage B-style aggregate mode for `--sequential`. When
    /// `true`, exhaust all `passk` samples per prompt and report
    /// aggregate pass-rate `total_passes / total_attempts` (matches
    /// Phase 17 S9's "pass@1 raw at temp=0.8" number).
    #[arg(long, default_value_t = false)]
    aggregate: bool,
    /// Override the model.safetensors path (config + tokenizer still
    /// come from the snapshot). Used to eval trained checkpoints from
    /// `phase22_mbpp_mr_sft`'s SaveMergedCheckpoint.
    #[arg(long)]
    checkpoint: Option<PathBuf>,
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

fn resolve_snapshot(model_dir: Option<&std::path::Path>, model_id: &str) -> Result<PathBuf> {
    if let Some(d) = model_dir {
        if !d.join("config.json").exists() {
            anyhow::bail!("--model-dir {d:?} has no config.json");
        }
        return Ok(d.to_path_buf());
    }
    let home = std::env::var("HOME").context("HOME unset")?;
    let snapshots_dir = PathBuf::from(format!(
        "{home}/.cache/huggingface/hub/models--Qwen--{model_id}/snapshots"
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
    println!("[Phase22C] device = {device:?}, on_cuda = {on_cuda}");
    // Match Stage A: BF16 on CUDA for Qwen2.5-Coder's native HF format.
    let dtype = if on_cuda { DType::BF16 } else { DType::F32 };

    let snapshot = resolve_snapshot(args.model_dir.as_deref(), &args.model_id)?;
    println!("[Phase22C] snapshot = {}", snapshot.display());
    let qwen = if let Some(ckpt) = args.checkpoint.as_ref() {
        println!("[Phase22C] checkpoint override = {}", ckpt.display());
        let cfg_text = std::fs::read_to_string(snapshot.join("config.json"))?;
        let config: candle_transformers::models::qwen2::Config = serde_json::from_str(&cfg_text)?;
        let tokenizer = tokenizers::Tokenizer::from_file(snapshot.join("tokenizer.json"))
            .map_err(|e| anyhow!("tokenizer: {e}"))?;
        QwenModelActor::new(ckpt.clone(), Arc::new(tokenizer), config, device, dtype)?
    } else {
        QwenModelActor::from_snapshot_dir(&snapshot, device, dtype)?
    };
    let tk = Arc::new(NgptTokenizer::from_hf_file(
        snapshot.join("tokenizer.json"),
    )?);

    let jsonl = args
        .jsonl
        .unwrap_or_else(|| PathBuf::from("data/mbpp/mbpp.jsonl"));
    let scratch = std::env::temp_dir().join("workllm-mbpp-baseline-scratch");
    let domain = MbppDomain::from_jsonl(&jsonl, &scratch)
        .with_context(|| format!("loading MBPP from {}", jsonl.display()))?;
    let n_total = domain.n_problems();
    println!(
        "[Phase22C] MbppDomain loaded {} problems (using {} for this run)",
        n_total, args.n_problems
    );
    let domain: Arc<dyn Domain> = Arc::new(domain);

    let system = ActorSystem::new("phase22-c");
    let model_ref = system.spawn(qwen, "qwen-model").await?;
    let evaluator = EvaluatorActor::<QwenModelActor>::new(model_ref.clone(), tk, domain, None);
    let evaluator_ref = system.spawn(evaluator, "evaluator").await?;

    // Same hyperparameter scheme as Stage A: greedy for k=1, Phase 17
    // sampler for k>1.
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
        "[Phase22C] starting eval n={} passk={} temperature={} top_k={:?} top_p={:?} sequential={} aggregate={}",
        args.n_problems, args.passk, temperature, top_k, top_p, args.sequential, args.aggregate
    );
    let t0 = std::time::Instant::now();
    let (tx, rx) = oneshot::channel();
    let msg = if args.sequential {
        EvaluatorMessage::EvalSequential {
            n: args.n_problems,
            offset: 0,
            sampling,
            passk: args.passk,
            aggregate: args.aggregate,
            reply: tx,
        }
    } else {
        EvaluatorMessage::Eval {
            n: args.n_problems,
            seed: args.seed,
            sampling,
            passk: args.passk,
            reply: tx,
        }
    };
    evaluator_ref.tell(msg).map_err(|e| anyhow!("{e:?}"))?;
    let report = rx.await??;
    let elapsed = t0.elapsed();
    println!(
        "\n[Phase22C] per-prompt pass@{} = {:.4}  ({}/{})  elapsed={:.1}s  wallclock_per_problem={:.2}s",
        args.passk,
        report.pass_rate(),
        report.correct,
        report.total,
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() / args.n_problems.max(1) as f64,
    );
    if let (Some(att), Some(passes)) = (report.total_attempts, report.total_passes) {
        let p1_raw = passes as f64 / att.max(1) as f64;
        println!(
            "[Phase22C] aggregate pass@1 (raw, all samples) = {:.4}  ({}/{})  \
             — comparable to Phase 17 S9's pass@1 raw on MBPP-100",
            p1_raw, passes, att
        );
    }
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
    println!("\nphase22_mbpp_baseline: PASS");
    Ok(())
}
