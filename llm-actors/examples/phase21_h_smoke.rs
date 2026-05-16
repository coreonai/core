//! Phase 21 Stage H — full supervisor pipeline against real Qwen.
//!
//! Closes the supervisor integration gap left by Stage E.next.next.
//! `RoundActors.trainer` is now `Arc<dyn TrainerHandle>`; we wrap the
//! `QwenTrainerActor` in `QwenTrainerActorHandle` and `run_multi_round`
//! drives the complete `Gen → Verify → Curate → Train → Reload → Eval`
//! cycle against the Phase 14-20 production Qwen2.5-Coder-0.5B model.
//!
//! What this exercises (every step a Pekko actor message):
//!   - Generator   = `GeneratorActor::<QwenModelActor>`
//!   - Verifier    = `VerifierActor` (PythonReturnDomain — trivial)
//!   - Curator     = `CuratorActor` (keeps correct trajectories)
//!   - Trainer     = `QwenTrainerActor` via `QwenTrainerActorHandle`
//!     (rendered corpus → Train → SaveMergedCheckpoint)
//!   - Reload      = `ModelMessage::ReloadCheckpoint` on QwenModelActor
//!   - Evaluator   = `EvaluatorActor::<QwenModelActor>` (pass@k)
//!
//! Run:
//!   cargo run -p llm-actors --example phase21_h_smoke \
//!       --features cuda --release
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device};
use llm_actors::{
    curator_actor::SampleMode, domain::Domain, qwen2_lora::LoraConfig, run_multi_round,
    supervisor::MultiRoundConfig, CuratorActor, EvaluatorActor, GeneratorActor, QwenModelActor,
    QwenTrainerActor, QwenTrainerActorHandle, RoundActors, RoundConfig, TrainerHandle, Verdict,
    VerifierActor,
};
use nanogpt_rs::{
    generate::GenerateConfig,
    train::{OptimizerKind, TrainConfig},
    Tokenizer as NgptTokenizer,
};
use pekko_actor::ActorSystem;
use rand::rngs::StdRng;

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

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let device = pick_device();
    let on_cuda = device.is_cuda();
    println!("[Phase21H] device = {device:?}, on_cuda = {on_cuda}");

    let snapshot = resolve_default_snapshot()?;
    let base_safetensors = snapshot.join("model.safetensors");
    println!("[Phase21H] snapshot = {}", snapshot.display());

    let inference_dtype = if on_cuda { DType::F16 } else { DType::F32 };
    let train_dtype = DType::F32;
    let lora_cfg = LoraConfig {
        rank: 16,
        alpha: 32.0,
    };

    // Tokenizer — wrap Qwen HF tokenizer in the nanogpt_rs enum so
    // Generator/Evaluator can use it through the existing API.
    let tk = Arc::new(NgptTokenizer::from_hf_file(
        snapshot.join("tokenizer.json"),
    )?);
    let domain: Arc<dyn Domain> = Arc::new(PythonReturnDomain);

    // ===== Actors =====
    let qwen_model = QwenModelActor::from_snapshot_dir(&snapshot, device.clone(), inference_dtype)?;
    let qwen_trainer = QwenTrainerActor::from_snapshot_dir(
        &snapshot,
        device.clone(),
        train_dtype,
        lora_cfg,
        2e-4,
    )?;

    let system = ActorSystem::new("phase21-h");
    let model_ref = system.spawn(qwen_model, "qwen-model").await?;
    let trainer_ref = system.spawn(qwen_trainer, "qwen-trainer").await?;

    let generator = GeneratorActor::<QwenModelActor>::new(
        model_ref.clone(),
        tk.clone(),
        domain.clone(),
        None, // no stop_char; just respect max_new_tokens
        "qwen".to_string(),
    );
    let generator_ref = system.spawn(generator, "generator").await?;

    let verifier_ref = system
        .spawn(VerifierActor::new(domain.clone()), "verifier")
        .await?;
    let curator_ref = system.spawn(CuratorActor::new(256), "curator").await?;

    let evaluator =
        EvaluatorActor::<QwenModelActor>::new(model_ref.clone(), tk.clone(), domain.clone(), None);
    let evaluator_ref = system.spawn(evaluator, "evaluator").await?;

    let trainer_handle = Arc::new(QwenTrainerActorHandle::new(
        trainer_ref,
        4,                        // train_steps per round (small for smoke)
        base_safetensors.clone(), // for SaveMergedCheckpoint
    )) as Arc<dyn TrainerHandle>;

    let actors = RoundActors::<QwenModelActor> {
        model: model_ref,
        generator: generator_ref,
        verifier: verifier_ref,
        curator: curator_ref,
        trainer: trainer_handle,
        evaluator: evaluator_ref,
    };
    println!("[Phase21H] 6 actors spawned + RoundActors built\n");

    // ===== RoundConfig =====
    // train_cfg is required by RoundConfig but ignored by
    // QwenTrainerActorHandle. The Qwen trainer uses its own
    // construction-time params (lr=2e-4, train_steps=4 per round).
    let mut train_cfg = TrainConfig::smoke();
    train_cfg.max_steps = 4;
    train_cfg.optimizer = OptimizerKind::Adam;

    let base = RoundConfig {
        round: 0,
        gen_n: 4,
        gen_seed: 42,
        gen_sampling: GenerateConfig {
            max_new_tokens: 24,
            temperature: 0.8,
            top_k: Some(40),
            top_p: Some(0.95),
            seed: Some(42),
        },
        eval_n: 6,
        eval_seed: 7,
        eval_sampling: GenerateConfig {
            max_new_tokens: 24,
            temperature: 0.0,
            top_k: Some(1),
            top_p: None,
            seed: Some(7),
        },
        train_cfg,
        init_from: None,
        save_path: PathBuf::from("checkpoints/phase21_h_merged.safetensors"),
        min_corpus_chars: 32,
        sample_mode: SampleMode::Uniform,
        corpus_seed: Some(0),
        anchor: None,
        freeze_base: false,
        gen_oversample: 1,
        dpo_beta: None,
        dpo_reference_path: None,
        dpo_max_pairs_per_prompt: 0,
        dpo_sft_anchor_weight: 0.0,
        eval_passk: 1,
        sft_mask_prompt: true,
    };

    let reports = run_multi_round(&actors, MultiRoundConfig::new(2, base), |r, rep| {
        println!(
            "[Phase21H] round {r}  gen={}/{}  eval_before={}/{}  eval_after={}/{}  elapsed_ms={}",
            rep.correct,
            rep.generated,
            rep.eval_correct_before.unwrap_or(0),
            rep.eval_total,
            rep.eval_correct_after.unwrap_or(0),
            rep.eval_total,
            rep.elapsed_ms,
        );
    })
    .await?;

    assert_eq!(reports.len(), 2, "expected 2 round reports");
    println!("\n[Phase21H] supervisor::run_multi_round drove the full");
    println!("          Gen → Verify → Curate → Train → Reload → Eval cycle");
    println!("          against Qwen2.5-Coder-0.5B end-to-end.\n");
    println!("phase21_h_smoke: PASS");
    Ok(())
}
