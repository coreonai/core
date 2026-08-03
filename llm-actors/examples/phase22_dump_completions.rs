//! Phase 22 §6.5 — benchmark-agnostic completion dumper.
//!
//! "Generate in Rust, score with the official harness." Generates completions
//! for a chosen benchmark and writes them in that harness's custom-generation
//! format (`bench_export`), then exits — scoring is delegated:
//!   - `bigcodebench` -> JSONL `{task_id, solution}` for
//!     `bigcodebench.syncheck` + `bigcodebench.evaluate --execution local`
//!     (Docker sandbox).
//!   - `humaneval` / `mbpp` -> LiveCodeBench-style array `[{question_id,
//!     code_list}]` (also the shape `lcb_runner.custom_evaluator` ingests).
//!
//! Uses the same per-(prompt, k) seed scheme as
//! `phase22_humaneval_baseline --sequential --aggregate`, so a dump is the same
//! generations the eval would score.
//!
//! Build (CUDA):
//!   CUDA_HOME=/usr/local/cuda-12.5 PATH=/usr/local/cuda-12.5/bin:$PATH \
//!     cargo build -p llm-actors --example phase22_dump_completions \
//!     --features cuda --release
//! Run (BigCodeBench Complete/Hard, greedy):
//!   CUDA_VISIBLE_DEVICES=0 ./target/release/examples/phase22_dump_completions \
//!     --benchmark bigcodebench --model-id Qwen2.5-Coder-7B \
//!     --jsonl data/bigcodebench/BigCodeBench-Hard.jsonl \
//!     --n-problems 148 --passk 1 --max-new-tokens 512 \
//!     --dump /tmp/bcb_hard_base.jsonl

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device};
use clap::{Parser, ValueEnum};
use llm_actors::{
    bench_export::{bigcodebench_entries, group_lcb_entries, write_bigcodebench_jsonl, write_lcb},
    domain::{
        bigcodebench::{BcbSplit, BigCodeBenchDomain},
        human_eval::HumanEvalDomain,
        livecodebench::LiveCodeBenchDomain,
        mbpp::MbppDomain,
        Domain,
    },
    ModelMessage, QwenModelActor,
};
use nanogpt_rs::{generate::GenerateConfig, Tokenizer as NgptTokenizer};
use pekko_actor::ActorSystem;
use tokio::sync::oneshot;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Benchmark {
    Bigcodebench,
    Humaneval,
    Mbpp,
    Livecodebench,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Format {
    /// JSONL `{task_id, solution}` (BigCodeBench harness).
    Bigcodebench,
    /// Array `[{question_id, code_list}]` (LiveCodeBench custom_evaluator).
    Lcb,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Split {
    Complete,
    Instruct,
}

/// Inference dtype. **BF16 silently corrupts generation on long prompts**
/// (~500+ total tokens): its 7-bit mantissa loses too much rotary/attention
/// precision, and the output degenerates into token-doubling garbage after a
/// few dozen clean tokens. Short-prompt benchmarks (HumanEval/MBPP, ~150-token
/// prompts) never hit it, which is why all prior Phase 22 work ran BF16 fine.
/// The long-prompt benchmarks (LiveCodeBench, BigCodeBench) MUST use F32 (2×
/// memory, still fits a 40GB card for 7B inference). See
/// `docs/phase22-livecodebench-notes.md`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum DtypeArg {
    Bf16,
    F16,
    F32,
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, value_enum)]
    benchmark: Benchmark,
    /// Output format. Defaults to the benchmark's native format
    /// (bigcodebench -> JSONL, others -> LCB array).
    #[arg(long, value_enum)]
    format: Option<Format>,
    /// Split for BigCodeBench (docstring `complete` vs `instruct`). Ignored for
    /// other benchmarks.
    #[arg(long, value_enum, default_value = "complete")]
    split: Split,
    /// Benchmark JSONL. Defaults per benchmark (see code).
    #[arg(long)]
    jsonl: Option<PathBuf>,

    #[arg(long)]
    model_dir: Option<PathBuf>,
    #[arg(long, default_value = "Qwen2.5-Coder-0.5B")]
    model_id: String,
    /// Override the model.safetensors path (evaluate a trained checkpoint).
    #[arg(long)]
    checkpoint: Option<PathBuf>,

    #[arg(long, default_value_t = 0)]
    offset: usize,
    #[arg(long, default_value_t = 8)]
    n_problems: usize,
    /// Samples per problem. `1` = greedy; `>1` = temp 0.8 / top_k 40 /
    /// top_p 0.95 sampling (Phase 17 hyperparameters).
    #[arg(long, default_value_t = 1)]
    passk: usize,
    #[arg(long, default_value_t = 512)]
    max_new_tokens: usize,
    /// Inference dtype. Defaults to `f32` because this dumper targets
    /// long-prompt benchmarks that BF16 corrupts (see `DtypeArg`). Use `bf16`
    /// only for short-prompt benchmarks where speed/memory matter.
    #[arg(long, value_enum, default_value = "f32")]
    dtype: DtypeArg,
    /// Output path for the generations.
    #[arg(long)]
    dump: PathBuf,
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

fn default_jsonl(b: Benchmark) -> PathBuf {
    PathBuf::from(match b {
        Benchmark::Bigcodebench => "data/bigcodebench/BigCodeBench.jsonl",
        Benchmark::Humaneval => "data/humaneval/HumanEval.jsonl",
        Benchmark::Mbpp => "data/mbpp/mbpp.jsonl",
        Benchmark::Livecodebench => "data/livecodebench/lcb_release_v5.jsonl",
    })
}

fn build_domain(args: &Args, jsonl: &std::path::Path) -> Result<Arc<dyn Domain>> {
    Ok(match args.benchmark {
        Benchmark::Bigcodebench => {
            let split = match args.split {
                Split::Complete => BcbSplit::Complete,
                Split::Instruct => BcbSplit::Instruct,
            };
            Arc::new(BigCodeBenchDomain::from_jsonl(jsonl, split)?)
        }
        Benchmark::Humaneval => {
            let scratch = std::env::temp_dir().join("workllm-dump-humaneval");
            Arc::new(HumanEvalDomain::from_jsonl(jsonl, &scratch)?)
        }
        Benchmark::Mbpp => {
            let scratch = std::env::temp_dir().join("workllm-dump-mbpp");
            Arc::new(MbppDomain::from_jsonl(jsonl, &scratch)?)
        }
        Benchmark::Livecodebench => Arc::new(LiveCodeBenchDomain::from_jsonl(jsonl)?),
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .init();

    let args = Args::parse();
    let device = pick_device();
    let on_cuda = device.is_cuda();
    println!("[dump] device = {device:?}, on_cuda = {on_cuda}");
    if !on_cuda && std::env::var("PHASE22_ALLOW_CPU").is_err() {
        anyhow::bail!(
            "Refusing to run on CPU. Rebuild with `--features cuda`. \
             Set PHASE22_ALLOW_CPU=1 to override."
        );
    }
    let dtype = if on_cuda {
        match args.dtype {
            DtypeArg::Bf16 => DType::BF16,
            DtypeArg::F16 => DType::F16,
            DtypeArg::F32 => DType::F32,
        }
    } else {
        DType::F32
    };
    println!("[dump] dtype = {dtype:?}");

    let snapshot = resolve_snapshot(args.model_dir.as_deref(), &args.model_id)?;
    println!("[dump] snapshot = {}", snapshot.display());
    let qwen = if let Some(ckpt) = args.checkpoint.as_ref() {
        println!("[dump] checkpoint override = {}", ckpt.display());
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
        .clone()
        .unwrap_or_else(|| default_jsonl(args.benchmark));
    let domain = build_domain(&args, &jsonl)
        .with_context(|| format!("loading {:?} from {}", args.benchmark, jsonl.display()))?;
    let total = domain.n_prompts().unwrap_or(0);
    println!(
        "[dump] {:?} loaded {} prompts; dumping idx {}..{} x passk={}",
        args.benchmark,
        total,
        args.offset,
        args.offset + args.n_problems,
        args.passk
    );

    let format = args.format.unwrap_or(match args.benchmark {
        Benchmark::Bigcodebench => Format::Bigcodebench,
        _ => Format::Lcb,
    });

    let system = ActorSystem::new("phase22-dump");
    let model_ref = system.spawn(qwen, "qwen-model").await?;

    let (temperature, top_k, top_p) = if args.passk > 1 {
        (0.8, Some(40usize), Some(0.95f64))
    } else {
        (0.0, Some(1usize), None)
    };
    let mut samples: Vec<(Option<String>, String)> = Vec::new();
    for prompt_idx in args.offset..args.offset + args.n_problems {
        let Some(prompt) = domain.nth_prompt(prompt_idx) else {
            break;
        };
        let question_id = domain.task_id(prompt_idx);
        let prompt_ids = tk.encode(&prompt).map_err(|e| anyhow!("encode: {e}"))?;
        for k in 0..args.passk {
            let k_seed = (prompt_idx as u64)
                .wrapping_mul(args.passk as u64)
                .wrapping_add(k as u64);
            let cfg = GenerateConfig {
                max_new_tokens: args.max_new_tokens,
                temperature,
                top_k,
                top_p,
                seed: Some(k_seed),
            };
            let (tx, rx) = oneshot::channel();
            model_ref
                .tell(ModelMessage::GenerateTokens {
                    prompt_ids: prompt_ids.clone(),
                    cfg,
                    reply: tx,
                })
                .map_err(|e| anyhow!("{e:?}"))?;
            let tokens = rx.await??;
            let comp_ids = if tokens.len() > prompt_ids.len() {
                &tokens[prompt_ids.len()..]
            } else {
                &[][..]
            };
            let raw = tk.decode(comp_ids).map_err(|e| anyhow!("decode: {e}"))?;
            let code = domain.truncate_completion(&raw);
            samples.push((question_id.clone(), code));
        }
        if (prompt_idx - args.offset + 1) % 20 == 0 {
            println!("[dump]   {} prompts done", prompt_idx - args.offset + 1);
        }
    }

    let n_written = match format {
        Format::Bigcodebench => {
            let entries = bigcodebench_entries(samples);
            let n = entries.len();
            write_bigcodebench_jsonl(&entries, &args.dump)?;
            n
        }
        Format::Lcb => {
            let entries = group_lcb_entries(samples);
            let n = entries.len();
            write_lcb(&entries, &args.dump)?;
            n
        }
    };
    println!(
        "[dump] wrote {} entries in {:?} format to {}",
        n_written,
        format,
        args.dump.display()
    );
    println!("phase22_dump_completions: PASS");
    Ok(())
}
