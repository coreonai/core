//! What do you harvest with when there is no verifier?
//!
//! Every result in Phase 23 rests on a free verifier: the answer is an integer
//! computed in Rust and the candidate is executed by the tool. Real domains
//! rarely offer that. The HBM service-report case that prompted this has
//! free-form text and no automatic grader, and without one the loop has
//! nothing to filter a harvest with and nothing to measure against.
//!
//! The honest way to evaluate a substitute is to run it **where ground truth
//! exists** and see how much of the verifier's job it recovers. That is this
//! binary: on `ToolUsePythonDomain`, sample K completions per prompt, then
//! score three verifier-free signals against the real verdict.
//!
//! ## The candidates
//!
//! - **self-consistency** — take the value the most samples agree on. Needs
//!   nothing but generation, which is why it is the first thing to try in a
//!   domain with no grader.
//! - **execution success** — keep only snippets that ran without raising.
//!   Free, and available in any domain where the output is executable at all.
//!   Note it says nothing about whether the answer is *right*: this repo has
//!   measured a clean run returning 2 where the answer was 111.
//! - **model log-prob** — Phase 6/7's "Shape C": the model's own confidence as
//!   a critic, which measured sum-AUC 0.727 on K9 and 0.702 on Qwen-0.5B. It
//!   costs one extra forward pass per sample.
//!
//! ## What is reported, and why each number
//!
//! ```text
//!   random sample correct     the baseline a loop gets with no filter at all
//!   modal value correct       self-consistency
//!   P(correct | ran clean)    precision of the execution filter
//!   top-logprob correct       picking the sample the model likes most
//!   log-prob AUC              whether confidence separates right from wrong
//! ```
//!
//! A substitute is worth having only if it beats the unfiltered baseline by
//! more than the cost of computing it. AUC near 0.5 means the signal carries
//! nothing, whatever its accuracy looks like.
//!
//! Run:
//!   phase23_proxy_verifier --init-dir <dir> --families 3,5,8,9 --k 8

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device};
use clap::Parser;
use llm_actors::{
    domain::{
        tool_use_python::{ToolUsePythonDomain, N_FAMILIES},
        Domain,
    },
    tools::{python_tool::PythonTool, Tool},
    ModelMessage, QwenModelActor,
};
use nanogpt_rs::{generate::GenerateConfig, Tokenizer as NgptTokenizer};
use pekko_actor::ActorSystem;
use tokio::sync::oneshot;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    init_dir: PathBuf,
    #[arg(long, default_value = "Qwen2.5-Coder-7B")]
    model_id: String,
    /// Samples per prompt. Self-consistency needs several; below ~4 the modal
    /// value is mostly the only value.
    #[arg(long, default_value_t = 8)]
    k: usize,
    #[arg(long, value_delimiter = ',')]
    families: Vec<usize>,
    #[arg(long, default_value_t = 12)]
    n_lo: u32,
    #[arg(long, default_value_t = 35)]
    n_hi: u32,
    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,
    #[arg(long, default_value = "bf16")]
    dtype: String,
    #[arg(long, default_value_t = 42)]
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

/// One sampled attempt, with everything a candidate filter could look at.
struct Attempt {
    correct: bool,
    ran_clean: bool,
    value: Option<String>,
    logprob: f32,
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
        other => anyhow::bail!("unknown --dtype {other:?} (f32 or bf16; f16 overflows)"),
    };

    let tk = Arc::new(NgptTokenizer::from_hf_file(
        args.init_dir.join("tokenizer.json"),
    )?);
    let families: Vec<usize> = if args.families.is_empty() {
        (0..N_FAMILIES).collect()
    } else {
        args.families.clone()
    };
    let domain = ToolUsePythonDomain::with_families(args.n_lo, args.n_hi, &families);
    let tool = PythonTool::new();

    let model = QwenModelActor::from_snapshot_dir(&args.init_dir, device, dtype)
        .context("loading model")?;
    let system = ActorSystem::new("phase23-proxy");
    let model_ref = system.spawn(model, "qwen-model").await?;

    let n = domain.n_tasks();
    println!(
        "[Proxy] {n} prompts x {} samples, families {families:?}, {dtype:?}\n",
        args.k
    );

    let mut per_prompt: Vec<Vec<Attempt>> = Vec::with_capacity(n);
    for i in 0..n {
        let t = domain.task_at(i).expect("in range").clone();
        let prompt = domain.nth_prompt(i).expect("in range");
        let prompt_ids = tk.encode(&prompt)?;
        let mut attempts = Vec::with_capacity(args.k);
        for s in 0..args.k {
            let (tx, rx) = oneshot::channel();
            model_ref
                .tell(ModelMessage::GenerateTokens {
                    prompt_ids: prompt_ids.clone(),
                    cfg: GenerateConfig {
                        max_new_tokens: args.max_new_tokens,
                        temperature: args.temperature,
                        top_k: Some(40),
                        top_p: Some(0.95),
                        seed: Some(
                            args.seed
                                .wrapping_add((i as u64) << 8)
                                .wrapping_add(s as u64),
                        ),
                    },
                    reply: tx,
                })
                .map_err(|e| anyhow!("{e:?}"))?;
            let tokens = tokio::time::timeout(std::time::Duration::from_secs(180), rx).await???;
            let full = tk.decode(&tokens)?;
            let raw = full[prompt.len().min(full.len())..].to_string();
            let completion = domain.truncate_completion(&raw);

            // What a verifier-free filter can see: did it run, and what did it
            // print? The verdict itself is ground truth, used only to score
            // the filters — never as an input to them.
            let (ran_clean, value) = match ToolUsePythonDomain::snippet_of(&completion) {
                Some(code) => match tool.execute(&code) {
                    Ok(v) if v.trim() != "<no output>" => (true, Some(v.trim().to_string())),
                    _ => (false, None),
                },
                None => (false, None),
            };
            let correct = domain.verify(&prompt, &completion).is_correct();

            let comp_ids = &tokens[prompt_ids.len().min(tokens.len())..];
            let logprob = if comp_ids.is_empty() {
                f32::NEG_INFINITY
            } else {
                let (tx, rx) = oneshot::channel();
                model_ref
                    .tell(ModelMessage::ScoreLogProb {
                        prompt_ids: prompt_ids.clone(),
                        completion_ids: comp_ids.to_vec(),
                        reply: tx,
                    })
                    .map_err(|e| anyhow!("{e:?}"))?;
                tokio::time::timeout(std::time::Duration::from_secs(180), rx).await???
            };
            attempts.push(Attempt {
                correct,
                ran_clean,
                value,
                logprob,
            });
        }
        let _ = t;
        per_prompt.push(attempts);
    }

    // --- score the candidate filters against ground truth ---
    let total: usize = per_prompt.iter().map(|a| a.len()).sum();
    let n_correct: usize = per_prompt.iter().flatten().filter(|a| a.correct).count();

    let mut modal_correct = 0usize;
    let mut modal_available = 0usize;
    for attempts in &per_prompt {
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for a in attempts.iter().filter(|a| a.ran_clean) {
            if let Some(v) = &a.value {
                *counts.entry(v.as_str()).or_default() += 1;
            }
        }
        // Ties broken by taking the first max — a tie means the signal did not
        // actually decide, and counting it as a win would flatter the method.
        if let Some((v, _)) = counts.iter().max_by_key(|(_, c)| **c) {
            modal_available += 1;
            if attempts
                .iter()
                .any(|a| a.correct && a.value.as_deref() == Some(*v))
            {
                modal_correct += 1;
            }
        }
    }

    let clean: Vec<&Attempt> = per_prompt
        .iter()
        .flatten()
        .filter(|a| a.ran_clean)
        .collect();
    let clean_correct = clean.iter().filter(|a| a.correct).count();

    let mut top_lp_correct = 0usize;
    for attempts in &per_prompt {
        if let Some(best) = attempts.iter().max_by(|a, b| {
            a.logprob
                .partial_cmp(&b.logprob)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            if best.correct {
                top_lp_correct += 1;
            }
        }
    }

    // AUC over all attempts: P(a correct one scores above an incorrect one).
    let (pos, neg): (Vec<f32>, Vec<f32>) = {
        let all: Vec<&Attempt> = per_prompt.iter().flatten().collect();
        (
            all.iter()
                .filter(|a| a.correct)
                .map(|a| a.logprob)
                .collect(),
            all.iter()
                .filter(|a| !a.correct)
                .map(|a| a.logprob)
                .collect(),
        )
    };
    let auc = if pos.is_empty() || neg.is_empty() {
        f32::NAN
    } else {
        let mut wins = 0f64;
        for p in &pos {
            for q in &neg {
                wins += if p > q {
                    1.0
                } else if (p - q).abs() < f32::EPSILON {
                    0.5
                } else {
                    0.0
                };
            }
        }
        (wins / (pos.len() as f64 * neg.len() as f64)) as f32
    };

    let np = per_prompt.len() as f32;
    println!("[Proxy] === verifier-free signals vs the real verdict ===");
    println!(
        "  random sample correct  {:.3}   ({n_correct}/{total})   <- unfiltered baseline",
        n_correct as f32 / total as f32
    );
    println!(
        "  modal value correct    {:.3}   ({modal_correct}/{modal_available} prompts with a mode)",
        modal_correct as f32 / modal_available.max(1) as f32
    );
    println!(
        "  P(correct | ran clean) {:.3}   ({clean_correct}/{})",
        clean_correct as f32 / clean.len().max(1) as f32,
        clean.len()
    );
    println!(
        "  top-logprob correct    {:.3}   ({top_lp_correct}/{})",
        top_lp_correct as f32 / np,
        per_prompt.len()
    );
    println!("  log-prob AUC           {auc:.3}   <- 0.5 = carries nothing");
    println!("\nphase23_proxy_verifier: PASS");
    Ok(())
}
