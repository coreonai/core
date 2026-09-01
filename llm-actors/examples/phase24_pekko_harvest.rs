//! Phase 24 — Pekko/MSA harvest loop on `RustCodeDomain` (cargo verifier).
//!
//! Gen → Verify(`cargo run`) → optional repair turn → Curate → LoRA SFT → Reload → Eval.
//! Starts from a format-SFT init dir (`model.safetensors` + tokenizer/config).
//!
//! ```text
//! cargo run -p llm-actors --example phase24_pekko_harvest --features cuda --release -- \
//!     --init-dir scratch-7b-sft/p24_fmt_sft_v2_dir \
//!     --scratch-dir scratch-pekko-harvest/_cargo_scratch \
//!     --rounds 2 --gen-n 48 --eval-n 21 --harvest-repair \
//!     --out-dir scratch-7b-sft/p24_harvest
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use candle_core::{DType, Device};
use clap::Parser;
use llm_actors::{
    curator_actor::SampleMode,
    domain::rust_code::RustCodeDomain,
    qwen2_lora::LoraConfig,
    run_multi_round,
    supervisor::MultiRoundConfig,
    CuratorActor, EvaluatorActor, GeneratorActor, QwenModelActor, QwenTrainerActor,
    QwenTrainerActorHandle, RoundActors, RoundConfig, TrainerHandle, VerifierActor,
};
use nanogpt_rs::{
    generate::GenerateConfig,
    train::{OptimizerKind, TrainConfig},
    Tokenizer as NgptTokenizer,
};
use pekko_actor::ActorSystem;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    init_dir: PathBuf,
    #[arg(long, default_value = "scratch-pekko-harvest/_cargo_scratch")]
    scratch_dir: PathBuf,
    #[arg(long, default_value = "scratch-7b-sft/p24_harvest")]
    out_dir: PathBuf,
    #[arg(long, default_value_t = 2)]
    rounds: usize,
    #[arg(long, default_value_t = 48)]
    gen_n: usize,
    #[arg(long, default_value_t = 21)]
    eval_n: usize,
    #[arg(long, default_value_t = 1)]
    eval_passk: usize,
    #[arg(long, default_value_t = 4)]
    samples_per_prompt: usize,
    #[arg(long, default_value_t = 400)]
    train_steps: usize,
    #[arg(long, default_value_t = 2e-4)]
    lr: f64,
    #[arg(long, default_value_t = 16)]
    lora_rank: usize,
    #[arg(long, default_value_t = 32.0)]
    lora_alpha: f32,
    #[arg(long, default_value_t = 4)]
    batch_size: usize,
    #[arg(long, default_value_t = 64)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,
    #[arg(long, default_value_t = 40)]
    top_k: usize,
    #[arg(long, default_value_t = 7)]
    seed: u64,
    #[arg(long, default_value_t = true)]
    harvest_repair: bool,
    #[arg(long, default_value_t = 0)]
    trainer_gpu: usize,
}

fn pick_device(idx: usize) -> Device {
    #[cfg(feature = "cuda")]
    {
        if let Ok(d) = Device::new_cuda(idx) {
            return d;
        }
    }
    let _ = idx;
    Device::Cpu
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .init();
    let args = Args::parse();
    for f in ["model.safetensors", "config.json", "tokenizer.json"] {
        let p = args.init_dir.join(f);
        if !p.exists() {
            anyhow::bail!(
                "--init-dir {:?} missing {f}. Symlink format-SFT weights + snapshot files.",
                args.init_dir
            );
        }
    }
    std::fs::create_dir_all(&args.out_dir)?;
    std::fs::create_dir_all(&args.scratch_dir)?;

    let concrete = RustCodeDomain::new(&args.scratch_dir);
    concrete.ensure_scratch_project()?;
    let domain = Arc::new(concrete);

    let device = pick_device(0);
    let trainer_device = pick_device(args.trainer_gpu);
    if !device.is_cuda() && std::env::var("PHASE22_ALLOW_CPU").is_err() {
        anyhow::bail!("need CUDA");
    }
    let inference_dtype = DType::F16;
    let train_dtype = DType::BF16;
    let tk = Arc::new(NgptTokenizer::from_hf_file(
        args.init_dir.join("tokenizer.json"),
    )?);

    println!(
        "[Phase24Harvest] init={} scratch={} repair={}",
        args.init_dir.display(),
        args.scratch_dir.display(),
        args.harvest_repair
    );

    let qwen_model =
        QwenModelActor::from_snapshot_dir(&args.init_dir, device.clone(), inference_dtype)?;
    let qwen_trainer = QwenTrainerActor::from_snapshot_dir(
        &args.init_dir,
        trainer_device,
        train_dtype,
        LoraConfig {
            rank: args.lora_rank,
            alpha: args.lora_alpha,
        },
        args.lr,
    )?
    .with_sft_batch_size(args.batch_size)
    .with_fresh_optimizer(true);

    let system = ActorSystem::new("phase24-harvest");
    let model_ref = system.spawn(qwen_model, "qwen-model").await?;
    let trainer_ref = system.spawn(qwen_trainer, "qwen-trainer").await?;
    let generator_ref = system
        .spawn(
            GeneratorActor::<QwenModelActor>::new(
                model_ref.clone(),
                tk.clone(),
                domain.clone(),
                Some('\n'),
                "qwen".to_string(),
            )
            .with_repair_failures(args.harvest_repair),
            "generator",
        )
        .await?;
    let verifier_ref = system
        .spawn(VerifierActor::new(domain.clone()), "verifier")
        .await?;
    let curator_ref = system.spawn(CuratorActor::new(2048), "curator").await?;
    let evaluator_ref = system
        .spawn(
            EvaluatorActor::<QwenModelActor>::new(
                model_ref.clone(),
                tk.clone(),
                domain.clone(),
                Some('\n'),
            ),
            "evaluator",
        )
        .await?;
    let trainer_handle = Arc::new(QwenTrainerActorHandle::new(
        trainer_ref,
        args.train_steps,
        args.init_dir.clone(),
    )) as Arc<dyn TrainerHandle>;

    let actors = RoundActors::<QwenModelActor> {
        model: model_ref,
        generator: generator_ref,
        verifier: verifier_ref,
        curator: curator_ref,
        trainer: trainer_handle,
        evaluator: evaluator_ref,
    };

    let mut train_cfg = TrainConfig::smoke();
    train_cfg.max_steps = args.train_steps;
    train_cfg.optimizer = OptimizerKind::Adam;

    let gen_seed = args.seed;
    let eval_seed = args.seed.wrapping_sub(35);
    let base = RoundConfig {
        round: 0,
        gen_n: args.gen_n,
        gen_seed,
        gen_sampling: GenerateConfig {
            max_new_tokens: args.max_new_tokens,
            temperature: args.temperature,
            top_k: (args.top_k > 0).then_some(args.top_k),
            top_p: Some(0.95),
            seed: Some(gen_seed),
        },
        eval_n: args.eval_n,
        eval_seed,
        eval_sampling: GenerateConfig {
            max_new_tokens: args.max_new_tokens,
            temperature: 0.8,
            top_k: Some(40),
            top_p: Some(0.95),
            seed: Some(eval_seed),
        },
        train_cfg,
        init_from: None,
        save_path: args.out_dir.join("r0_merged.safetensors"),
        min_corpus_chars: 1,
        sample_mode: SampleMode::Uniform,
        corpus_seed: Some(args.seed.wrapping_sub(42)),
        anchor: None,
        freeze_base: false,
        gen_oversample: 1,
        dpo_beta: None,
        dpo_reference_path: None,
        dpo_max_pairs_per_prompt: 0,
        dpo_sft_anchor_weight: 0.0,
        eval_passk: args.eval_passk,
        sft_mask_prompt: true,
        samples_per_prompt: Some(args.samples_per_prompt),
    };

    let reports = run_multi_round(
        &actors,
        MultiRoundConfig::new(args.rounds, base),
        |r, rep| {
            let fmt = |c: Option<usize>| match c {
                Some(n) => format!("{:.3}", n as f32 / rep.eval_total.max(1) as f32),
                None => "skipped".to_string(),
            };
            println!(
                "[Phase24Harvest] round {r}: harvested {}/{} | pass {} -> {}",
                rep.correct,
                rep.generated,
                fmt(rep.eval_correct_before),
                fmt(rep.eval_correct_after),
            );
        },
    )
    .await?;

    println!("\n[Phase24Harvest] === summary ===");
    for (r, rep) in reports.iter().enumerate() {
        println!(
            "  round {r}: harvest {}/{}  eval_before={:?} eval_after={:?} / {}",
            rep.correct,
            rep.generated,
            rep.eval_correct_before,
            rep.eval_correct_after,
            rep.eval_total
        );
    }
    Ok(())
}
