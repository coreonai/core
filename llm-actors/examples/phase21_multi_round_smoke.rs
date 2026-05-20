//! Phase 21 Stage C — `run_multi_round` smoke test.
//!
//! Spawns the standard actor set (Model/Generator/Verifier/Curator/Trainer/
//! Evaluator) on `ArithmeticDomain` with a tiny char-GPT, then drives
//! `rounds=3` of self-improvement through the new `run_multi_round`
//! helper instead of a hand-written loop.
//!
//! Wiring smoke only — the model is small enough that absolute pass-rates
//! are noisy. The acceptance criterion is that the helper produces 3
//! `RoundReport`s with correctly chained `init_from → save_path` paths.
//!
//! Run:
//!   cargo run -p llm-actors --example phase21_multi_round_smoke --release \
//!       --features cuda
//!
//! No CUDA needed; CPU is fine for the smoke.
use std::path::PathBuf;
use std::sync::Arc;

use candle_core::Device;
use llm_actors::{
    curator_actor::SampleMode,
    domain::{
        arithmetic::{ArithmeticDomain, SeedMode},
        Domain,
    },
    run_multi_round,
    supervisor::MultiRoundConfig,
    CuratorActor, CuratorMessage, EvaluatorActor, GeneratorActor, ModelActor, RoundActors,
    RoundConfig, TrainerActor, TrainerActorHandle, Trajectory, Verdict, VerifiedTrajectory,
    VerifierActor,
};
use nanogpt_rs::{
    config::GPTConfig,
    data::TokenDataset,
    generate::GenerateConfig,
    tokenizer::Tokenizer,
    train::{train_from, TrainConfig},
};
use pekko_actor::ActorSystem;
use tokio::sync::oneshot;
use tracing::info;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let device = Device::Cpu;
    let domain = Arc::new(ArithmeticDomain::default());

    // Char tokenizer covering the arithmetic charset.
    let pretrain_text = domain.synth_corpus(150, 7);
    let mut seed_chars = String::from(domain.charset());
    seed_chars.push_str(&pretrain_text);
    let tk = Tokenizer::char_from_text(&seed_chars);
    let vocab = tk.vocab_size();

    // Tiny model — wiring focus, not accuracy.
    let gpt_cfg = GPTConfig {
        vocab_size: vocab,
        block_size: 32,
        n_layer: 2,
        n_head: 2,
        n_embd: 64,
        dropout: 0.0,
        bias: false,
        ffn_mult: 4,
        use_rope: false,
        rope_base: 10_000.0,
        n_kv_head: 2,
        n_experts: 1,
        moe_top_k: 0,
        moe_aux_weight: 0.01,
        activation: nanogpt_rs::config::ActivationKind::Gelu,
        weight_tying: true,
        norm_kind: nanogpt_rs::config::NormKind::LayerNorm,
        norm_position: nanogpt_rs::config::NormPosition::Pre,
        lora_rank: 0,
        lora_alpha: 16.0,
    };
    info!(params = gpt_cfg.num_params_estimate(), "model config");

    // Brief pretrain so the round can see *some* signal.
    let ids = tk.encode(&pretrain_text)?;
    let ds = TokenDataset::new(ids, gpt_cfg.block_size);
    let mut pre_cfg = TrainConfig::smoke();
    pre_cfg.max_steps = 200;
    pre_cfg.batch_size = 64;
    pre_cfg.lr = 3e-3;
    pre_cfg.min_lr = 3e-4;
    pre_cfg.warmup_steps = 20;

    let seed_ckpt: PathBuf = "checkpoints/phase21_smoke_seed.safetensors".into();
    train_from(
        &gpt_cfg,
        &ds,
        None,
        &pre_cfg,
        &device,
        Some(&seed_ckpt),
        None,
    )?;

    // Actor wiring.
    let tk = Arc::new(tk);
    let system = ActorSystem::new("phase21-smoke");
    let model =
        ModelActor::from_checkpoint(gpt_cfg.clone(), device.clone(), tk.clone(), &seed_ckpt)?;
    let model_ref = system.spawn(model, "model").await?;

    let curator = CuratorActor::new(512);
    let curator_ref = system.spawn(curator, "curator").await?;
    seed_curator(&curator_ref, &domain).await?;

    let verifier_ref = system
        .spawn(
            VerifierActor::new(domain.clone() as Arc<dyn Domain>),
            "verifier",
        )
        .await?;

    let generator_ref = system
        .spawn(
            GeneratorActor::new(
                model_ref.clone(),
                tk.clone(),
                domain.clone() as Arc<dyn Domain>,
                Some('\n'),
                "model".to_string(),
            ),
            "generator",
        )
        .await?;

    let evaluator_ref = system
        .spawn(
            EvaluatorActor::new(
                model_ref.clone(),
                tk.clone(),
                domain.clone() as Arc<dyn Domain>,
                Some('\n'),
            ),
            "evaluator",
        )
        .await?;

    let trainer_ref = system
        .spawn(
            TrainerActor::new(gpt_cfg.clone(), tk.clone(), device.clone()),
            "trainer",
        )
        .await?;

    let actors = RoundActors {
        model: model_ref,
        generator: generator_ref,
        verifier: verifier_ref,
        curator: curator_ref,
        trainer: Arc::new(TrainerActorHandle::new(trainer_ref)),
        evaluator: evaluator_ref,
    };

    // Build the per-round base config. `run_multi_round` mutates the
    // round / init_from / save_path / seeds per round automatically.
    let mut train_cfg = TrainConfig::smoke();
    train_cfg.max_steps = 150;
    train_cfg.batch_size = 64;
    train_cfg.eval_interval = 150;
    train_cfg.lr = 1e-3;
    train_cfg.min_lr = 1e-4;
    train_cfg.warmup_steps = 10;

    let base = RoundConfig {
        round: 0,
        gen_n: 40,
        gen_seed: 100,
        gen_sampling: GenerateConfig {
            max_new_tokens: 4,
            temperature: 1.0,
            top_k: Some(10),
            top_p: None,
            seed: Some(0),
        },
        eval_n: 24,
        eval_seed: 0xE7A1,
        eval_sampling: GenerateConfig {
            max_new_tokens: 4,
            temperature: 0.0,
            top_k: Some(1),
            top_p: None,
            seed: Some(0xE7A1),
        },
        train_cfg,
        init_from: Some(seed_ckpt.clone()),
        save_path: PathBuf::from("checkpoints/phase21_smoke.safetensors"),
        min_corpus_chars: 4_000,
        sample_mode: SampleMode::Priority {
            recency_decay: 0.95,
        },
        corpus_seed: Some(7),
        anchor: None,
        freeze_base: false,
        gen_oversample: 1,
        dpo_beta: None,
        dpo_reference_path: None,
        dpo_max_pairs_per_prompt: 0,
        dpo_sft_anchor_weight: 0.0,
        // Phase 21 Stage A axis: this smoke also exercises passk > 1
        // through the helper so we get end-to-end coverage in one shot.
        eval_passk: 3,
        sft_mask_prompt: true,
        samples_per_prompt: None,
    };

    let reports = run_multi_round(&actors, MultiRoundConfig::new(3, base), |r, rep| {
        println!(
            "[round {r}] gen={}/{}  eval before={}/{} after={}/{}  Δ={}  loss={:?}",
            rep.correct,
            rep.generated,
            rep.eval_correct_before.unwrap_or(0),
            rep.eval_total,
            rep.eval_correct_after.unwrap_or(0),
            rep.eval_total,
            rep.eval_correct_after.unwrap_or(0) as i64
                - rep.eval_correct_before.unwrap_or(0) as i64,
            rep.last_train_loss,
        );
    })
    .await?;

    println!("\n=== run_multi_round summary ===");
    for (i, rep) in reports.iter().enumerate() {
        println!(
            "round {i}: eval_correct_after={}/{}  elapsed_ms={}",
            rep.eval_correct_after.unwrap_or(0),
            rep.eval_total,
            rep.elapsed_ms,
        );
    }
    assert_eq!(reports.len(), 3, "expected 3 round reports");
    println!("\nphase21_multi_round_smoke: PASS");
    Ok(())
}

async fn seed_curator(
    curator: &pekko_actor::ActorRef<CuratorActor>,
    domain: &Arc<ArithmeticDomain>,
) -> anyhow::Result<()> {
    let mut seeds: Vec<VerifiedTrajectory> = Vec::new();
    for (a, b) in domain
        .enumerate_seed_pairs(SeedMode::NoCarry)
        .into_iter()
        .take(32)
    {
        let prompt = format!("{a}+{b}=");
        let completion = format!("{}\n", a + b);
        seeds.push(VerifiedTrajectory {
            trajectory: Trajectory {
                prompt,
                completion,
                source: "seed".to_string(),
            },
            verdict: Verdict::Correct,
            score: 1.0,
        });
    }
    let (tx, rx) = oneshot::channel();
    curator
        .tell(CuratorMessage::Add {
            items: seeds,
            reply: tx,
        })
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    rx.await?;
    Ok(())
}
