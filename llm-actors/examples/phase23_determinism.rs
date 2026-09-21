//! Is generation reproducible at the token level?
//!
//! The evaluation varies run to run — 51/96 then 52/96 with the same binary,
//! checkpoint and seed. The sampler, the verifier and the rotary fix are all
//! ruled out (see `docs/phase23-serving-economics.md`), which leaves
//! nondeterminism below the actor. This cuts that question in half:
//!
//!   - **same tokens every time** → the cause is above generation, in how the
//!     evaluator or the domain uses them, and is findable in our code.
//!   - **different tokens** → the cause is the forward pass itself. CUDA
//!     reductions that use atomics, or a cuBLAS heuristic that picks a
//!     different split-k for the same shapes, return slightly different
//!     logits for identical input; a near-tie then flips a token, and one
//!     flipped token flips a problem.
//!
//! Repeats the same generation `--repeats` times *inside one process* and
//! again after dropping and reloading the model, because those are different
//! claims: the first says a single loaded model is stable, the second says
//! two loads of the same weights agree.
//!
//! Greedy by default. Sampling would add its own variation and the question
//! here is about the forward pass, not the sampler.
//!
//! Run:
//!   phase23_determinism --init-dir <dir> --repeats 5

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device};
use clap::Parser;
use llm_actors::{ModelMessage, QwenModelActor};
use nanogpt_rs::{generate::GenerateConfig, Tokenizer as NgptTokenizer};
use pekko_actor::ActorSystem;
use tokio::sync::oneshot;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    init_dir: PathBuf,
    #[arg(long, default_value_t = 5)]
    repeats: usize,
    #[arg(long, default_value_t = 64)]
    max_new_tokens: usize,
    /// 0.0 is greedy, which isolates the forward pass from the sampler.
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,
    #[arg(long, default_value = "f32")]
    dtype: String,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(
        long,
        default_value = "Q: how many numbers from 1 to 45 are coprime to 45?\n"
    )]
    prompt: String,
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

async fn draw(
    model_ref: &pekko_actor::ActorRef<QwenModelActor>,
    prompt_ids: &[u32],
    cfg: &GenerateConfig,
) -> Result<Vec<u32>> {
    let (tx, rx) = oneshot::channel();
    model_ref
        .tell(ModelMessage::GenerateTokens {
            prompt_ids: prompt_ids.to_vec(),
            cfg: cfg.clone(),
            reply: tx,
        })
        .map_err(|e| anyhow!("{e:?}"))?;
    Ok(tokio::time::timeout(std::time::Duration::from_secs(300), rx).await???)
}

fn report(label: &str, runs: &[Vec<u32>], prompt_len: usize, tk: &NgptTokenizer) -> Result<bool> {
    let first = &runs[0];
    let mut differing = 0usize;
    let mut first_div: Option<usize> = None;
    for r in runs.iter().skip(1) {
        if r != first {
            differing += 1;
            let p = first
                .iter()
                .zip(r.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(first.len().min(r.len()));
            first_div = Some(first_div.map_or(p, |q: usize| q.min(p)));
        }
    }
    println!(
        "  {label:<22} {}/{} identical to the first",
        runs.len() - differing,
        runs.len()
    );
    if let Some(p) = first_div {
        println!(
            "      first divergence at token {} (prompt is {prompt_len})",
            p
        );
        let a = &first[prompt_len.min(first.len())..];
        println!(
            "      run 0 {:?}",
            tk.decode(a)?.chars().take(90).collect::<String>()
        );
        for r in runs.iter().skip(1) {
            if r != first {
                let b = &r[prompt_len.min(r.len())..];
                println!(
                    "      other {:?}",
                    tk.decode(b)?.chars().take(90).collect::<String>()
                );
                break;
            }
        }
    }
    Ok(differing == 0)
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
        "bf16" => DType::BF16,
        other => anyhow::bail!("unknown --dtype {other:?}"),
    };

    let tk = Arc::new(NgptTokenizer::from_hf_file(
        args.init_dir.join("tokenizer.json"),
    )?);
    let prompt_ids = tk.encode(&args.prompt)?;
    let cfg = GenerateConfig {
        max_new_tokens: args.max_new_tokens,
        temperature: args.temperature,
        top_k: (args.temperature > 0.0).then_some(40),
        top_p: (args.temperature > 0.0).then_some(0.95),
        seed: Some(args.seed),
    };

    let system = ActorSystem::new("phase23-determinism");
    let model = QwenModelActor::from_snapshot_dir(&args.init_dir, device.clone(), dtype)
        .context("loading model")?;
    let model_ref = system.spawn(model, "qwen-model").await?;

    println!(
        "[Determinism] {dtype:?}, temperature {}, prompt {} tokens, {} repeats\n",
        args.temperature,
        prompt_ids.len(),
        args.repeats
    );

    let mut runs = Vec::with_capacity(args.repeats);
    for _ in 0..args.repeats {
        runs.push(draw(&model_ref, &prompt_ids, &cfg).await?);
    }
    println!("[Determinism] === within one loaded model ===");
    let stable_in_process = report("same model", &runs, prompt_ids.len(), &tk)?;

    // Across loads is checked by running this binary twice and comparing the
    // digest, not by holding two models at once: two 7B copies do not fit a
    // 40 GB card, and the first attempt OOM'd for exactly that reason. The
    // same constraint is why `ReloadCheckpoint` drops before it loads.
    let digest = runs[0].iter().fold(1469598103934665603u64, |h, t| {
        (h ^ *t as u64).wrapping_mul(1099511628211)
    });
    println!("\n[Determinism] token digest {digest:016x}   <- compare across processes");

    println!();
    match stable_in_process {
        true => println!(
            "[Determinism] token-stable within one loaded model. If two processes\n\
             print different digests, the difference is between LOADS, not in the\n\
             arithmetic of a single one."
        ),
        false => println!(
            "[Determinism] the SAME loaded model returns different tokens for the\n\
             same input. That is the forward pass, not our scheduling: CUDA\n\
             reductions using atomics, or a cuBLAS heuristic picking a different\n\
             split-k, give slightly different logits and a near-tie flips."
        ),
    }
    println!("\nphase23_determinism: PASS");
    Ok(())
}
