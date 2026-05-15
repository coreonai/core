//! Phase 21 Stage E.next.next — full Pekko-bridge multi-round demo.
//!
//! Wires three actors in one ActorSystem:
//!   - `QwenModelActor` (Stage D inference)
//!   - `QwenTrainerActor` (Stage E.next training)
//!   - `EvaluatorActor<QwenModelActor>` (Stage E generic eval)
//!
//! Runs a 2-round loop:
//!   1. Eval-before via EvaluatorActor (pre-train baseline)
//!   2. Train via QwenTrainerActor
//!   3. SaveMergedCheckpoint → merged safetensors on disk
//!   4. QwenModelActor::ReloadCheckpoint(merged_path) — inference now
//!      reflects training, no LoRA-awareness on the inference side
//!   5. Eval-after via the SAME EvaluatorActor
//!
//! Demonstrates the Phase 17-20 recipe shape (Gen-Train-Eval round)
//! running end-to-end through Pekko actors against the real
//! Qwen2.5-Coder-0.5B production model — no Python sidecar, no
//! standalone driver functions. The training-side bridge gap from
//! Stage E.next is closed by the merge step.
//!
//! Run:
//!   cargo run -p llm-actors --example phase21_e_next_next_smoke \
//!       --features cuda --release
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device};
use llm_actors::{
    domain::Domain, qwen2_lora::LoraConfig, EvaluatorActor, EvaluatorMessage, ModelMessage,
    QwenModelActor, QwenTrainerActor, QwenTrainerMessage, Verdict,
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
        if completion.contains("return") {
            Verdict::Correct
        } else {
            Verdict::Incorrect {
                reason: "no `return` keyword".to_string(),
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

async fn eval_passk(
    evaluator_ref: &pekko_actor::ActorRef<EvaluatorActor<QwenModelActor>>,
    passk: usize,
    eval_n: usize,
) -> Result<f32> {
    let (temp, topk) = if passk > 1 { (0.8, 40) } else { (0.0, 1) };
    let sampling = GenerateConfig {
        max_new_tokens: 24,
        temperature: temp,
        top_k: Some(topk),
        top_p: Some(0.95),
        seed: Some(42),
    };
    let (tx, rx) = oneshot::channel();
    evaluator_ref
        .tell(EvaluatorMessage::Eval {
            n: eval_n,
            seed: 7,
            sampling,
            passk,
            reply: tx,
        })
        .map_err(|e| anyhow!("{e:?}"))?;
    let report = rx.await??;
    Ok(report.pass_rate())
}

async fn reload_model(
    model_ref: &pekko_actor::ActorRef<QwenModelActor>,
    path: &Path,
) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    model_ref
        .tell(ModelMessage::ReloadCheckpoint {
            path: path.to_path_buf(),
            reply: tx,
        })
        .map_err(|e| anyhow!("{e:?}"))?;
    rx.await??;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .init();

    let device = pick_device();
    let on_cuda = device.is_cuda();
    println!("[Phase21E.next.next] device = {device:?}, on_cuda = {on_cuda}");

    let snapshot = resolve_default_snapshot()?;
    let base_safetensors = snapshot.join("model.safetensors");
    println!("[Phase21E.next.next] snapshot = {}", snapshot.display());

    // The inference actor uses F16 (Stage D default) — production-shape.
    let inference_dtype = if on_cuda { DType::F16 } else { DType::F32 };
    let qwen_model = QwenModelActor::from_snapshot_dir(&snapshot, device.clone(), inference_dtype)?;

    // The trainer actor uses F32 — Stage F documented that rank=16 LoRA
    // gradients at lr=2e-4 are too coarse in F16.
    let train_dtype = DType::F32;
    let lora_cfg = LoraConfig {
        rank: 16,
        alpha: 32.0,
    };
    let qwen_trainer = QwenTrainerActor::from_snapshot_dir(
        &snapshot,
        device.clone(),
        train_dtype,
        lora_cfg,
        2e-4,
    )?;
    println!(
        "[Phase21E.next.next] QwenTrainerActor built  lora_params = {}",
        qwen_trainer.lora_param_count()
    );

    let system = ActorSystem::new("phase21-enn");
    let model_ref = system.spawn(qwen_model, "qwen-model").await?;
    let trainer_ref = system.spawn(qwen_trainer, "qwen-trainer").await?;

    // EvaluatorActor<QwenModelActor> — the Stage E genericization in action.
    let ngpt_tokenizer = Arc::new(NgptTokenizer::from_hf_file(
        snapshot.join("tokenizer.json"),
    )?);
    let domain: Arc<dyn Domain> = Arc::new(PythonReturnDomain);
    let evaluator =
        EvaluatorActor::<QwenModelActor>::new(model_ref.clone(), ngpt_tokenizer, domain, None);
    let evaluator_ref = system.spawn(evaluator, "evaluator").await?;
    println!("[Phase21E.next.next] 3 actors spawned: model + trainer + evaluator\n");

    let train_texts: Vec<String> = vec![
        "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)".to_string(),
        "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0:\n            return False\n    return True".to_string(),
        "def reverse_string(s):\n    return s[::-1]".to_string(),
    ];

    let rounds = 2;
    let train_steps_per_round = 6;
    let eval_n = 9;
    let mut round_log: Vec<(usize, f32, f32, f32, f32)> = Vec::new();

    // Initial baseline — eval before any training.
    let baseline_p1 = eval_passk(&evaluator_ref, 1, eval_n).await?;
    let baseline_p5 = eval_passk(&evaluator_ref, 5, eval_n).await?;
    println!(
        "[Phase21E.next.next] baseline  pass@1={:.3}  pass@5={:.3}",
        baseline_p1, baseline_p5
    );

    for r in 0..rounds {
        println!("\n=== Round {r} ===");

        // 1. Train
        let (tx, rx) = oneshot::channel();
        trainer_ref
            .tell(QwenTrainerMessage::Train {
                texts: train_texts.clone(),
                train_steps: train_steps_per_round,
                reply: tx,
            })
            .map_err(|e| anyhow!("{e:?}"))?;
        let outcome = rx.await??;
        println!(
            "  train loss: {:.3} → {:.3} (Δ={:+.3} over {} steps)",
            outcome.initial_loss,
            outcome.final_loss,
            outcome.final_loss - outcome.initial_loss,
            outcome.losses.len()
        );

        // 2. SaveMergedCheckpoint → drops the trained LoRA delta into a
        // base-compatible safetensors that the upstream qwen2 loader
        // (QwenModelActor) can pick up via ReloadCheckpoint.
        let merged_path: PathBuf =
            format!("checkpoints/phase21_enn_merged_round{r}.safetensors").into();
        if let Some(p) = merged_path.parent() {
            std::fs::create_dir_all(p).ok();
        }
        let (sx, srx) = oneshot::channel();
        trainer_ref
            .tell(QwenTrainerMessage::SaveMergedCheckpoint {
                base_path: base_safetensors.clone(),
                out_path: merged_path.clone(),
                reply: sx,
            })
            .map_err(|e| anyhow!("{e:?}"))?;
        srx.await??;
        let sz = std::fs::metadata(&merged_path)?.len();
        println!(
            "  merged checkpoint saved  path={}  size={} bytes",
            merged_path.display(),
            sz
        );

        // 3. Hot-swap into the inference actor.
        reload_model(&model_ref, &merged_path).await?;
        println!("  QwenModelActor reloaded from merged checkpoint");

        // 4. Eval after.
        let after_p1 = eval_passk(&evaluator_ref, 1, eval_n).await?;
        let after_p5 = eval_passk(&evaluator_ref, 5, eval_n).await?;
        println!(
            "  eval-after  pass@1={:.3}  pass@5={:.3}",
            after_p1, after_p5
        );
        round_log.push((r, outcome.final_loss, after_p1, after_p5, sz as f32));
    }

    println!("\n=== Summary ===");
    println!(
        "baseline      pass@1={:.3}  pass@5={:.3}",
        baseline_p1, baseline_p5
    );
    for (r, loss, p1, p5, _) in &round_log {
        println!(
            "round {r}  loss={:.3}  pass@1={:.3}  pass@5={:.3}",
            loss, p1, p5
        );
    }

    println!("\nphase21_e_next_next_smoke: PASS");
    Ok(())
}
