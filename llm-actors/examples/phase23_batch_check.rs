//! Does batched decode agree with sequential decode, and how much faster?
//!
//! `GenerateBatch` draws K samples of one prompt in a single batched pass.
//! Speed is the point, but a faster path that returns different tokens is not
//! an optimisation — it is a second implementation to keep correct. So this
//! checks agreement first and reports the speedup second.
//!
//! Agreement is exact, not statistical: each sample `i` is seeded
//! `cfg.seed + i` in both paths, so batch sample `i` must equal the `i`-th
//! sequential draw token for token. Anything else means the batched forward
//! or the per-row sampling diverged, and a "mostly the same" batched decoder
//! would silently change every harvest and every pass@k number downstream.
//!
//! Run:
//!   phase23_batch_check --init-dir <dir> --k 8

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

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
    #[arg(long, default_value_t = 8)]
    k: usize,
    #[arg(long, default_value_t = 64)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,
    #[arg(long, default_value = "bf16")]
    dtype: String,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value = "Q: how many divisors does 720 have?\n")]
    prompt: String,
    /// Also check the VARIABLE-LENGTH path: several different prompts, left
    /// padded, in one batch. That is the harder case — padding means a mask,
    /// and a mask that drops causality produces fluent wrong text rather than
    /// an error.
    #[arg(long)]
    varlen: bool,
    /// Same as --varlen but with prompts chosen to tokenize to EQUAL length,
    /// so nothing is padded. Separates two candidate causes of a late
    /// divergence: the pad mask, or the rotary position shift that left
    /// padding imposes. If this is exact and --varlen is not, the shift is
    /// the cause and padded batching cannot be bit-identical by construction.
    #[arg(long)]
    varlen_equal: bool,
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
    let model = QwenModelActor::from_snapshot_dir(&args.init_dir, device, dtype)
        .context("loading model")?;
    let system = ActorSystem::new("phase23-batch-check");
    let model_ref = system.spawn(model, "qwen-model").await?;

    let prompt_ids = tk.encode(&args.prompt)?;
    let cfg = GenerateConfig {
        max_new_tokens: args.max_new_tokens,
        temperature: args.temperature,
        top_k: Some(40),
        top_p: Some(0.95),
        seed: Some(args.seed),
    };

    // sequential: K calls, sample i seeded seed+i
    let t0 = Instant::now();
    let mut seq_out = Vec::with_capacity(args.k);
    for i in 0..args.k {
        let mut c = cfg.clone();
        c.seed = Some(args.seed.wrapping_add(i as u64));
        let (tx, rx) = oneshot::channel();
        model_ref
            .tell(ModelMessage::GenerateTokens {
                prompt_ids: prompt_ids.clone(),
                cfg: c,
                reply: tx,
            })
            .map_err(|e| anyhow!("{e:?}"))?;
        seq_out.push(tokio::time::timeout(std::time::Duration::from_secs(300), rx).await???);
    }
    let seq_ms = t0.elapsed().as_millis();

    // batched: one call
    let t1 = Instant::now();
    let (tx, rx) = oneshot::channel();
    model_ref
        .tell(ModelMessage::GenerateBatch {
            prompt_ids: prompt_ids.clone(),
            n_samples: args.k,
            cfg: cfg.clone(),
            reply: tx,
        })
        .map_err(|e| anyhow!("{e:?}"))?;
    let batch_out = tokio::time::timeout(std::time::Duration::from_secs(600), rx).await???;
    let batch_ms = t1.elapsed().as_millis();

    println!(
        "[BatchCheck] prompt {} tokens, k={}",
        prompt_ids.len(),
        args.k
    );
    let mut mismatches = 0usize;
    for i in 0..args.k {
        let a = &seq_out[i];
        let b = batch_out.get(i).map(|v| v.as_slice()).unwrap_or(&[]);
        if a.as_slice() != b {
            mismatches += 1;
            if mismatches <= 2 {
                let first = a
                    .iter()
                    .zip(b.iter())
                    .position(|(x, y)| x != y)
                    .map(|p| p.to_string())
                    .unwrap_or_else(|| "length only".into());
                println!(
                    "  sample {i}: DIFFERS at {first}  seq_len={} batch_len={}",
                    a.len(),
                    b.len()
                );
                println!("    seq   {:?}", tk.decode(&a[prompt_ids.len()..])?);
                println!(
                    "    batch {:?}",
                    tk.decode(&b[prompt_ids.len().min(b.len())..])?
                );
            }
        }
    }
    println!("\n[BatchCheck] === agreement ===");
    println!("  identical samples   {}/{}", args.k - mismatches, args.k);
    println!("\n[BatchCheck] === speed ===");
    println!("  sequential  {seq_ms:>6} ms");
    println!("  batched     {batch_ms:>6} ms");
    if batch_ms > 0 {
        println!("  speedup     {:.2}x", seq_ms as f64 / batch_ms as f64);
    }
    if mismatches > 0 {
        anyhow::bail!(
            "batched decode disagrees with sequential on {mismatches}/{} samples — \
             a faster path that returns different tokens is a second \
             implementation, not an optimisation",
            args.k
        );
    }
    if args.varlen || args.varlen_equal {
        // Deliberately different lengths, and deliberately different content:
        // equal-length prompts would pass even if the pad mask were ignored.
        let prompts: [&str; 4] = if args.varlen_equal {
            // Same shape, same digit count -> same token count, no padding.
            [
                "Q: how many divisors does 720 have?\n",
                "Q: how many divisors does 360 have?\n",
                "Q: how many divisors does 480 have?\n",
                "Q: how many divisors does 840 have?\n",
            ]
        } else {
            [
                "Q: how many divisors does 720 have?\n",
                "Q: what is the sum of all primes below 200?\n",
                "Q: how many numbers from 1 to 45 are coprime to 45?\n",
                "Q: how many trailing zeros does 100! have?\n",
            ]
        };
        let ids: Vec<Vec<u32>> = prompts
            .iter()
            .map(|p| tk.encode(p))
            .collect::<Result<_, _>>()?;
        let lens: Vec<usize> = ids.iter().map(|v| v.len()).collect();
        let seeds: Vec<u64> = (0..prompts.len())
            .map(|i| args.seed.wrapping_add(i as u64))
            .collect();

        let t2 = Instant::now();
        let mut seq2 = Vec::new();
        for (p, sd) in ids.iter().zip(seeds.iter()) {
            let mut c = cfg.clone();
            c.seed = Some(*sd);
            let (tx, rx) = oneshot::channel();
            model_ref
                .tell(ModelMessage::GenerateTokens {
                    prompt_ids: p.clone(),
                    cfg: c,
                    reply: tx,
                })
                .map_err(|e| anyhow!("{e:?}"))?;
            seq2.push(tokio::time::timeout(std::time::Duration::from_secs(300), rx).await???);
        }
        let seq2_ms = t2.elapsed().as_millis();

        let t3 = Instant::now();
        let (tx, rx) = oneshot::channel();
        model_ref
            .tell(ModelMessage::GenerateBatchPrompts {
                prompts: ids.clone(),
                seeds: seeds.clone(),
                cfg: cfg.clone(),
                reply: tx,
            })
            .map_err(|e| anyhow!("{e:?}"))?;
        let bat2 = tokio::time::timeout(std::time::Duration::from_secs(600), rx).await???;
        let bat2_ms = t3.elapsed().as_millis();

        println!("\n[BatchCheck] === variable length ===");
        println!("  prompt lengths {lens:?}");
        let mut bad = 0usize;
        for i in 0..ids.len() {
            let a = &seq2[i];
            let b = bat2.get(i).map(|v| v.as_slice()).unwrap_or(&[]);
            if a.as_slice() != b {
                bad += 1;
                println!(
                    "  prompt {i}: DIFFERS  seq_len={} batch_len={}",
                    a.len(),
                    b.len()
                );
                println!("    seq   {:?}", tk.decode(&a[ids[i].len()..])?);
                println!(
                    "    batch {:?}",
                    tk.decode(&b[ids[i].len().min(b.len())..])?
                );
            }
        }
        println!("  identical prompts   {}/{}", ids.len() - bad, ids.len());
        println!("  sequential  {seq2_ms:>6} ms");
        println!("  batched     {bat2_ms:>6} ms");
        if bat2_ms > 0 {
            println!("  speedup     {:.2}x", seq2_ms as f64 / bat2_ms as f64);
        }
        if bad > 0 {
            anyhow::bail!(
                "variable-length batch disagrees on {bad}/{} prompts",
                ids.len()
            );
        }
    }
    println!("\nphase23_batch_check: PASS");
    Ok(())
}
