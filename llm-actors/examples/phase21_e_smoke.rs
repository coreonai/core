//! Phase 21 Stage E — `EvaluatorActor<QwenModelActor>` end-to-end smoke.
//!
//! Demonstrates the genericization: same `EvaluatorActor` code path
//! that drove `ModelActor` (Stage A's pass@k) now drives
//! `QwenModelActor` (Stage D's Candle-native Qwen2.5-Coder-0.5B).
//!
//! Builds a trivial in-binary `Domain` that checks completions
//! contain the `"return"` keyword (so any plausible function-body
//! completion passes — wiring focus, not benchmark realism). Sends
//! `EvaluatorMessage::Eval` with `passk=1` (greedy baseline) and
//! `passk=5` (temp=0.8 sampling) and reports both.
//!
//! Run:
//!   cargo run -p llm-actors --example phase21_e_smoke \
//!       --features cuda --release
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device};
use llm_actors::{
    domain::Domain, EvaluatorActor, EvaluatorMessage, QwenModelActor, Trajectory, Verdict,
};
use nanogpt_rs::{generate::GenerateConfig, Tokenizer as NgptTokenizer};
use pekko_actor::ActorSystem;
use rand::rngs::StdRng;
use tokio::sync::oneshot;

const PROMPTS: &[&str] = &[
    "def fibonacci(n):",
    "def is_prime(n):",
    "def reverse_string(s):",
];

#[derive(Debug)]
struct PythonReturnDomain;

impl Domain for PythonReturnDomain {
    fn sample_prompt(&self, rng: &mut StdRng) -> String {
        use rand::Rng;
        PROMPTS[rng.gen_range(0..PROMPTS.len())].to_string()
    }

    fn verify(&self, _prompt: &str, completion: &str) -> Verdict {
        // Smoke verdict: any completion that includes `return` passes.
        // This is plausible-function-body detection, not benchmark
        // realism — the point is to exercise the actor wiring with
        // genuine Verdict::Correct results.
        if completion.contains("return") {
            Verdict::Correct
        } else {
            Verdict::Incorrect {
                reason: "no `return` keyword".to_string(),
            }
        }
    }

    fn charset(&self) -> &str {
        // Unused for the BPE tokenizer path; required by the trait.
        ""
    }
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
        .with_max_level(tracing::Level::INFO)
        .init();

    let device = pick_device();
    let on_cuda = device.is_cuda();
    println!("[Phase21E] device = {device:?}, on_cuda = {on_cuda}");
    let dtype = if on_cuda { DType::F16 } else { DType::F32 };

    let snapshot = resolve_default_snapshot()?;
    println!("[Phase21E] snapshot = {}", snapshot.display());

    // Build the QwenModelActor with the same loader Stage D uses.
    let qwen = QwenModelActor::from_snapshot_dir(&snapshot, device, dtype)?;
    println!("[Phase21E] QwenModelActor built");

    // The same HF tokenizer the Qwen actor uses, but wrapped in the
    // nanogpt_rs::Tokenizer enum that EvaluatorActor expects.
    let ngpt_tokenizer = Arc::new(NgptTokenizer::from_hf_file(
        snapshot.join("tokenizer.json"),
    )?);

    // Spawn the actor system and the Qwen model actor.
    let system = ActorSystem::new("phase21-e");
    let qwen_ref = system.spawn(qwen, "qwen-model").await?;

    // Build EvaluatorActor<QwenModelActor> — the key Stage E demonstration.
    let domain: Arc<dyn Domain> = Arc::new(PythonReturnDomain);
    let evaluator =
        EvaluatorActor::<QwenModelActor>::new(qwen_ref.clone(), ngpt_tokenizer, domain, None);
    let evaluator_ref = system.spawn(evaluator, "evaluator").await?;
    println!("[Phase21E] EvaluatorActor<QwenModelActor> spawned");

    let prompts_len = PROMPTS.len();
    println!("[Phase21E] domain prompts: {prompts_len}");
    println!("[Phase21E] running eval at multiple passk values...\n");

    for &passk in &[1usize, 5] {
        let (temp, topk) = if passk > 1 { (0.8, 40) } else { (0.0, 1) };
        let sampling = GenerateConfig {
            max_new_tokens: 28,
            temperature: temp,
            top_k: Some(topk),
            top_p: Some(0.95),
            seed: Some(42),
        };
        let (tx, rx) = oneshot::channel();
        evaluator_ref
            .tell(EvaluatorMessage::Eval {
                n: 12,
                seed: 7,
                sampling,
                passk,
                reply: tx,
            })
            .map_err(|e| anyhow!("{e:?}"))?;
        let report = rx.await??;
        println!(
            "[Phase21E] passk={:>2}  pass-rate={:.3}  ({}/{})  eval_sampling(temp={}, topk={})",
            passk,
            report.pass_rate(),
            report.correct,
            report.total,
            temp,
            topk,
        );
        for (i, s) in report.samples.iter().take(2).enumerate() {
            println!(
                "    sample {i}  prompt={:?}  completion={:?}",
                s.prompt,
                s.completion.trim_end()
            );
        }
    }

    println!("\nphase21_e_smoke: PASS");
    let _ = Trajectory {
        prompt: String::new(),
        completion: String::new(),
        source: "_".into(),
    }; // ensure Trajectory import is used
    Ok(())
}
