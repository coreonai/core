//! Phase 21 Stage G — RL with pass@k-style reward.
//!
//! Demonstrates REINFORCE on Qwen2.5-Coder-0.5B LoRA:
//!   1. Pick a small prompt set, generate `k` completions per prompt
//!      via QwenModelActor (temp > 0).
//!   2. Verify each completion via the Domain (PythonReturnDomain).
//!   3. Compute per-prompt baseline-subtracted rewards (RLOO style):
//!      `reward_i = verdict_i − mean(verdicts_for_prompt)`. A sample
//!      that beat its prompt's average gets a positive reward; one
//!      below average gets a negative reward.
//!   4. Send a `TrainPolicyGradient { samples }` to `QwenTrainerActor`.
//!      The trainer applies one AdamW step that ascends
//!      `reward_i * log P(comp_i | prompt_i)`.
//!   5. Repeat for multiple RL steps and log the trajectory.
//!
//! This is the *off-policy approximation* (sampling from QwenModelActor
//! whose weights drift away from the trainer's LoRA-augmented policy
//! as training proceeds). Importance-weight correction is deferred —
//! Stage G ships the loop structure and a working reward-weighted
//! update, not on-policy correctness guarantees.
//!
//! Run:
//!   cargo run -p llm-actors --example phase21_g_smoke \
//!       --features cuda --release
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device};
use llm_actors::{
    domain::Domain, qwen2_lora::LoraConfig, ModelMessage, QwenModelActor, QwenTrainerActor,
    QwenTrainerMessage, Verdict,
};
use nanogpt_rs::{generate::GenerateConfig, Tokenizer as NgptTokenizer};
use pekko_actor::ActorSystem;
use tokio::sync::oneshot;

const PROMPTS: &[&str] = &[
    "def fibonacci(n):",
    "def is_prime(n):",
    "def reverse_string(s):",
];

#[derive(Debug)]
struct PythonReturnDomain;

impl Domain for PythonReturnDomain {
    fn sample_prompt(&self, _rng: &mut rand::rngs::StdRng) -> String {
        unreachable!("Stage G builds prompts directly, not via sample_prompt")
    }
    fn verify(&self, _prompt: &str, completion: &str) -> Verdict {
        if completion.contains("return") {
            Verdict::Correct
        } else {
            Verdict::Incorrect {
                reason: "no return".into(),
            }
        }
    }
    fn charset(&self) -> &str {
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
        .with_max_level(tracing::Level::WARN)
        .init();

    let device = pick_device();
    let on_cuda = device.is_cuda();
    println!("[Phase21G] device = {device:?}, on_cuda = {on_cuda}");

    let snapshot = resolve_default_snapshot()?;
    println!("[Phase21G] snapshot = {}", snapshot.display());
    let inference_dtype = if on_cuda { DType::F16 } else { DType::F32 };
    let train_dtype = DType::F32;

    let tk = Arc::new(NgptTokenizer::from_hf_file(
        snapshot.join("tokenizer.json"),
    )?);
    let domain = PythonReturnDomain;

    let qwen_model = QwenModelActor::from_snapshot_dir(&snapshot, device.clone(), inference_dtype)?;
    let qwen_trainer = QwenTrainerActor::from_snapshot_dir(
        &snapshot,
        device.clone(),
        train_dtype,
        LoraConfig {
            rank: 16,
            alpha: 32.0,
        },
        2e-4,
    )?;

    let system = ActorSystem::new("phase21-g");
    let model_ref = system.spawn(qwen_model, "qwen-model").await?;
    let trainer_ref = system.spawn(qwen_trainer, "qwen-trainer").await?;
    println!("[Phase21G] 2 actors spawned (model + trainer)\n");

    let rl_steps = 3;
    let k_per_prompt = 2;
    let mut losses = Vec::with_capacity(rl_steps);

    for rl_step in 0..rl_steps {
        // 1. Sample k completions per prompt via QwenModelActor.
        let mut samples: Vec<(Vec<u32>, Vec<u32>, f32)> = Vec::new();
        let mut prompt_verdicts: Vec<Vec<bool>> = Vec::with_capacity(PROMPTS.len());

        for (p_idx, prompt) in PROMPTS.iter().enumerate() {
            let prompt_ids = tk.encode(prompt)?;
            let mut verdicts_for_prompt = Vec::with_capacity(k_per_prompt);
            let mut prompt_samples: Vec<(Vec<u32>, f32)> = Vec::with_capacity(k_per_prompt);

            for k in 0..k_per_prompt {
                let cfg = GenerateConfig {
                    max_new_tokens: 16,
                    temperature: 0.8,
                    top_k: Some(40),
                    top_p: Some(0.95),
                    // Per-(prompt, k, rl_step) seed for diversity that's
                    // still reproducible across runs.
                    seed: Some(((rl_step as u64) << 16) ^ ((p_idx as u64) << 8) ^ (k as u64)),
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
                let comp_ids: Vec<u32> = if full.len() > prompt_ids.len() {
                    full[prompt_ids.len()..].to_vec()
                } else {
                    vec![]
                };
                let comp_text = tk.decode(&comp_ids)?;
                let verdict = domain.verify(prompt, &comp_text);
                let v_value: f32 = if verdict.is_correct() { 1.0 } else { 0.0 };
                verdicts_for_prompt.push(verdict.is_correct());
                prompt_samples.push((comp_ids, v_value));
            }

            // 2. RLOO baseline — center rewards on the prompt's mean
            // verdict so high-variance prompts don't dominate.
            let baseline: f32 =
                prompt_samples.iter().map(|(_, v)| *v).sum::<f32>() / prompt_samples.len() as f32;
            for (comp_ids, v) in prompt_samples {
                let reward = v - baseline;
                if !comp_ids.is_empty() {
                    samples.push((prompt_ids.clone(), comp_ids, reward));
                }
            }
            prompt_verdicts.push(verdicts_for_prompt);
        }

        // 3. Send TrainPolicyGradient to the trainer.
        let (tx, rx) = oneshot::channel();
        trainer_ref
            .tell(QwenTrainerMessage::TrainPolicyGradient { samples, reply: tx })
            .map_err(|e| anyhow!("{e:?}"))?;
        let loss = rx.await??.loss;
        losses.push(loss);

        let pass_counts: Vec<usize> = prompt_verdicts
            .iter()
            .map(|v| v.iter().filter(|x| **x).count())
            .collect();
        let total_pass: usize = pass_counts.iter().sum();
        let total_samples = PROMPTS.len() * k_per_prompt;
        println!(
            "[Phase21G] rl_step {rl_step}  loss = {loss:+.4}  \
             pass@1-of-k = {total_pass}/{total_samples}  per-prompt = {:?}",
            pass_counts
        );
    }

    println!("\n[Phase21G] losses across RL steps: {:?}", losses);
    println!(
        "[Phase21G] RL loop ran {} steps × {} prompts × k={} = {} samples total",
        rl_steps,
        PROMPTS.len(),
        k_per_prompt,
        rl_steps * PROMPTS.len() * k_per_prompt
    );
    println!("\nphase21_g_smoke: PASS");
    Ok(())
}
