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
    domain::{filtered::FilteredDomain, human_eval::HumanEvalDomain, Domain},
    EvaluatorActor, EvaluatorMessage, ModelMessage, QwenModelActor,
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
    /// Phase 22 Stage B — when `true`, use `EvalSequential` (no
    /// replacement) over `domain.nth_prompt(0..n)` instead of
    /// `Eval`'s with-replacement RNG sampling. Required for
    /// bit-exact Phase 17 baseline reproduction (Phase 17 evaluates
    /// each of 164 problems exactly once).
    #[arg(long, default_value_t = false)]
    sequential: bool,
    /// Phase 22 Stage E follow-up — start index for `--sequential`.
    /// Evaluates `nth_prompt(offset..offset+n_problems)` instead of
    /// `0..n_problems`. Used to score a held-out tail (e.g. `--offset 64
    /// --n-problems 100` = task 64..164) after RL trained on task 0..64,
    /// for clean generalization measurement. Per-task seeds key off the
    /// absolute index, so the slice is bit-identical to that window of a
    /// full 0..164 run. Default 0.
    #[arg(long, default_value_t = 0)]
    offset: usize,
    /// Phase 22 Stage B — aggregate mode for `--sequential`. When
    /// `true`, exhaust all `passk` samples per prompt (no short-circuit)
    /// and report aggregate pass-rate `total_passes / total_attempts`
    /// alongside per-prompt pass@k. Matches Phase 17 S6's "pass@1 raw"
    /// number (0.216 at temp=0.8, k=10).
    #[arg(long, default_value_t = false)]
    aggregate: bool,
    /// Override the model.safetensors path. Useful for evaluating
    /// trained checkpoints produced by `phase22_he_mr_sft`'s
    /// `SaveMergedCheckpoint`. When set, `tokenizer.json` and
    /// `config.json` are still loaded from the snapshot dir, only
    /// the weights come from this path. Output is drop-in compatible
    /// with the upstream `candle_transformers::models::qwen2` loader.
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    /// Hide these prompt indices behind a `FilteredDomain`, renumbering
    /// the rest. Present so the eval path used by the multi-round SFT
    /// runs (which always wrap in FilteredDomain) is reproducible here.
    #[arg(long, value_delimiter = ',')]
    prompt_skip_list: Vec<usize>,
    /// Phase 22 §6.5 — in the canonical greedy full-set base config, compare
    /// the measured base pass@1 against the published Qwen2.5-Coder number
    /// (`eval_sanity`) and exit non-zero on a drift beyond tolerance. Off by
    /// default (the check still prints an informational `[SANITY]` line).
    #[arg(long, default_value_t = false)]
    sanity_strict: bool,
    /// Phase 22 §6.5 — instead of scoring in Rust, GENERATE completions and
    /// write them in LiveCodeBench custom-eval format
    /// (`[{question_id, code_list}]`, `bench_export`) to this path, then exit.
    /// Scoring is delegated to the official harness (`generate in Rust, score
    /// with the official harness`). Uses the same per-(prompt, k) seed scheme
    /// as `--sequential --aggregate`, so the dumped generations match the eval.
    #[arg(long)]
    dump_completions: Option<PathBuf>,
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
    println!("[Phase22A] device = {device:?}, on_cuda = {on_cuda}");
    if !on_cuda && std::env::var("PHASE22_ALLOW_CPU").is_err() {
        anyhow::bail!(
            "Refusing to run on CPU. Rebuild with `--features cuda`. \
             Set PHASE22_ALLOW_CPU=1 to override."
        );
    }
    // Use BF16 on CUDA to match Qwen2.5-Coder's native HF format and
    // Phase 17 S6's measurement. F16 has same exponent range but less
    // mantissa precision; some completions silently drift.
    let dtype = if on_cuda { DType::BF16 } else { DType::F32 };

    let snapshot = resolve_snapshot(args.model_dir.as_deref(), &args.model_id)?;
    println!("[Phase22A] snapshot = {}", snapshot.display());
    let qwen = if let Some(ckpt) = args.checkpoint.as_ref() {
        println!("[Phase22A] checkpoint override = {}", ckpt.display());
        // Use the snapshot's config + tokenizer but the override's weights.
        // QwenModelActor::new takes (model_path, tokenizer, config, device, dtype).
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
        .unwrap_or_else(|| PathBuf::from("data/humaneval/HumanEval.jsonl"));
    let scratch = std::env::temp_dir().join("workllm-humaneval-baseline-scratch");
    let domain = HumanEvalDomain::from_jsonl(&jsonl, &scratch)
        .with_context(|| format!("loading HumanEval from {}", jsonl.display()))?;
    let n_total = domain.n_problems();
    println!(
        "[Phase22A] HumanEvalDomain loaded {} problems (using {} for this run)",
        n_total, args.n_problems
    );
    let domain: Arc<dyn Domain> = if args.prompt_skip_list.is_empty() {
        Arc::new(domain)
    } else {
        // Phase 22 C5 follow-up — reproduce the SFT-era hard-tail eval
        // condition (which wrapped the domain in FilteredDomain) so the
        // 0.246-vs-0.422 base discrepancy can be measured directly rather
        // than argued about. With the skip list set, indices are renumbered:
        // the hard tail is `--prompt-skip-list 0..99 --offset 0`.
        let filtered = FilteredDomain::new(Arc::new(domain), args.prompt_skip_list.iter().copied());
        println!(
            "[Phase22A] FilteredDomain: hiding {} indices, {} surviving",
            args.prompt_skip_list.len(),
            filtered.n_surviving()
        );
        Arc::new(filtered)
    };

    let system = ActorSystem::new("phase22-a");
    let model_ref = system.spawn(qwen, "qwen-model").await?;

    // Phase 22 §6.5 — standard-format export. Generate completions in Rust and
    // write LiveCodeBench custom-eval JSON; scoring is delegated to the
    // official harness (bench_export). Mirrors EvalSequential's per-(prompt, k)
    // seed so the dump matches what `--sequential --aggregate` would score.
    if let Some(dump_path) = args.dump_completions.clone() {
        use llm_actors::bench_export::{group_lcb_entries, write_lcb};
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
                // Same truncation the eval applies, so the dumped code is what
                // was (or would be) scored — one ruler.
                let code = domain.truncate_completion(&raw);
                samples.push((question_id.clone(), code));
            }
        }
        let entries = group_lcb_entries(samples);
        write_lcb(&entries, &dump_path)?;
        println!(
            "[Phase22A] dumped {} problems x passk={} to {} (LiveCodeBench custom-eval format)",
            entries.len(),
            args.passk,
            dump_path.display()
        );
        return Ok(());
    }

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
        "[Phase22A] starting eval n={} offset={} passk={} temperature={} top_k={:?} top_p={:?} sequential={}",
        args.n_problems, args.offset, args.passk, temperature, top_k, top_p, args.sequential
    );
    let t0 = std::time::Instant::now();
    let (tx, rx) = oneshot::channel();
    let msg = if args.sequential {
        EvaluatorMessage::EvalSequential {
            n: args.n_problems,
            offset: args.offset,
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
        "\n[Phase22A] per-prompt pass@{} = {:.4}  ({}/{})  elapsed={:.1}s  wallclock_per_problem={:.2}s",
        args.passk,
        report.pass_rate(),
        report.correct,
        report.total,
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() / args.n_problems.max(1) as f64,
    );
    // Phase 17 S6-style "pass@1 raw" reporting when in aggregate mode.
    if let (Some(att), Some(passes)) = (report.total_attempts, report.total_passes) {
        let p1_raw = passes as f64 / att.max(1) as f64;
        println!(
            "[Phase22A] aggregate pass@1 (raw, all samples) = {:.4}  ({}/{})  \
             — comparable to Phase 17 S6's 0.216 at temp=0.8/k=10",
            p1_raw, passes, att
        );
    }
    // Phase 22 §6.5 — public-baseline sanity check. Only the canonical greedy
    // full-set BASE config is comparable to the published number; a filtered,
    // subset, sampled, or trained-checkpoint run is not, and saying so is half
    // the lesson (the FilteredDomain bug measured a non-comparable number and
    // compared it anyway).
    let sanity_fail = {
        use llm_actors::eval_sanity::check_public_baseline;
        let model_label = args
            .model_dir
            .as_deref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| args.model_id.clone());
        let p1 = report.pass_rate() as f64;
        if !args.prompt_skip_list.is_empty() {
            println!(
                "[SANITY] WARN filtered domain ({} skipped) — a subset is NOT \
                 benchmark-comparable; do not compare this pass@1 to any public \
                 or unfiltered baseline (docs/phase22-c4-c5-rl-vs-sft.md).",
                args.prompt_skip_list.len()
            );
            false
        } else if args.checkpoint.is_some() {
            false // trained checkpoint, not the base — nothing to compare
        } else if args.passk == 1 && args.offset == 0 && args.sequential && args.n_problems >= 164 {
            let verdict = check_public_baseline(&model_label, "HumanEval", p1);
            println!("{}", verdict.describe(&model_label, "HumanEval", p1));
            args.sanity_strict && verdict.is_drift()
        } else {
            println!(
                "[SANITY] skipped — not the canonical greedy full-set base config \
                 (need --passk 1 --offset 0 --sequential --n-problems 164, no \
                 --checkpoint / --prompt-skip-list); this pass@1 is not \
                 public-comparable."
            );
            false
        }
    };

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
    if sanity_fail {
        anyhow::bail!(
            "sanity check failed (--sanity-strict): base pass@1 drifted from the \
             published baseline — see the [SANITY] DRIFT line above"
        );
    }
    println!("\nphase22_humaneval_baseline: PASS");
    Ok(())
}
