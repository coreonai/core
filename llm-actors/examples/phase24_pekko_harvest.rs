//! Phase 24 — Pekko/MSA harvest loop on multi-family `PekkoHarvestDomain`.
//!
//! Gen → Verify(`cargo test --features student` / F0 `cargo run`) → optional
//! repair turn → Curate → LoRA SFT → Reload → Eval.
//!
//! ```text
//! cargo run -p llm-actors --example phase24_pekko_harvest --features cuda --release -- \
//!     --init-dir scratch-7b-sft/p24_fmt_sft_v3_dir \
//!     --families f0,f1,f2,f3,f4,f5 \
//!     --rounds 1 --gen-n 28 --eval-n 14 --samples-per-prompt 2 \
//!     --max-new-tokens 256 --harvest-repair \
//!     --out-dir scratch-7b-sft/p24_harvest_f0f5
//! ```

use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use candle_core::{DType, Device};
use clap::Parser;
use llm_actors::{
    curator_actor::SampleMode,
    domain::pekko_harvest::{format_family_counts, Family, PekkoHarvestDomain},
    domain::Domain,
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
    /// Isolated F1–F5 verify crates (each gets empty `[workspace]`).
    #[arg(long, default_value = "scratch-pekko-harvest/_verify")]
    verify_root: PathBuf,
    /// F0 expression-slot cargo scratch.
    #[arg(long, default_value = "scratch-pekko-harvest/_cargo_scratch")]
    scratch_dir: PathBuf,
    #[arg(long, default_value = "scratch-7b-sft/p24_harvest_f0f5")]
    out_dir: PathBuf,
    /// Comma-separated families: f0,f1,f2,f3,f4,f5 (default all).
    #[arg(long, default_value = "f0,f1,f2,f3,f4,f5")]
    families: String,
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
    #[arg(long, default_value_t = 256)]
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
    #[arg(long, default_value_t = 0)]
    infer_gpu: usize,
    /// Mask FIM/repo control tokens at decode (Phase 23 / fmt_probe lesson). Default on.
    #[arg(long, default_value_t = true)]
    suppress_special: bool,
}

fn flush_stdout() {
    let _ = io::stdout().flush();
}

/// Collect non-EOS added_tokens ids from tokenizer.json (same as phase24_fmt_probe).
fn control_token_ids(tokenizer_json: &PathBuf) -> anyhow::Result<Vec<u32>> {
    let tj: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(tokenizer_json)?)?;
    let ids = tj["added_tokens"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter(|t| {
                    let c = t["content"].as_str().unwrap_or("");
                    c != "<|endoftext|>"
                })
                .filter_map(|t| t["id"].as_u64().map(|x| x as u32))
                .collect()
        })
        .unwrap_or_default();
    Ok(ids)
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
    let families = Family::parse_list(&args.families).map_err(anyhow::Error::msg)?;
    std::fs::create_dir_all(&args.out_dir)?;
    std::fs::create_dir_all(&args.verify_root)?;
    std::fs::create_dir_all(&args.scratch_dir)?;

    let concrete = PekkoHarvestDomain::new(&args.verify_root, &args.scratch_dir, &families);
    concrete.ensure_ready()?;
    let n_prompts = concrete.n_prompts().unwrap_or(0);
    let domain = Arc::new(concrete);

    // Multi-line student.rs slots must not stop at the first newline.
    // F0 expressions rely on truncate_completion instead.
    let stop_char: Option<char> = None;

    let device = pick_device(args.infer_gpu);
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
        "[Phase24Harvest] init={} families={:?} n_prompts={} verify={} f0_scratch={} repair={} stop_char={:?} max_new={} suppress_special={}",
        args.init_dir.display(),
        families.iter().map(|f| f.as_str()).collect::<Vec<_>>(),
        n_prompts,
        args.verify_root.display(),
        args.scratch_dir.display(),
        args.harvest_repair,
        stop_char,
        args.max_new_tokens,
        args.suppress_special
    );
    flush_stdout();

    let mut qwen_model =
        QwenModelActor::from_snapshot_dir(&args.init_dir, device.clone(), inference_dtype)?;
    if args.suppress_special {
        let ids = control_token_ids(&args.init_dir.join("tokenizer.json"))?;
        println!(
            "[Phase24Harvest] suppressing {} control tokens (EOS kept)",
            ids.len()
        );
        flush_stdout();
        qwen_model = qwen_model.with_suppressed_tokens(ids);
    }
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
                stop_char,
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
                stop_char,
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
            let hf = format_family_counts("harvest", &rep.harvest_family);
            if !hf.is_empty() {
                println!("{hf}");
            }
            let eb = format_family_counts("eval-before", &rep.eval_family_before);
            if !eb.is_empty() {
                println!("{eb}");
            }
            let ea = format_family_counts("eval-after", &rep.eval_family_after);
            if !ea.is_empty() {
                println!("{ea}");
            }
            flush_stdout();
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
    flush_stdout();
    Ok(())
}
