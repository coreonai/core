//! Phase 22 Stage D — Multi-round SFT on MBPP through Pekko.
//!
//! Cross-substrate companion of `phase22_he_mr_sft`. Phase 17 SB
//! measured mean 0.453 ± 0.016 at r=2 on MBPP-100 (Δ=+0.093 over
//! single-round); Phase 18 S3 confirmed at r=3 = 0.457 ± 0.013;
//! Phase 20 S2 extended to r=5 = 0.541 ± 0.014 (cross-substrate
//! saturation, σ tighter than HumanEval's 0.037 at r=5).
//! Phase 22 Stage C ships `MbppDomain`; this binary feeds it into the
//! same `run_multi_round` that the HumanEval variant uses.
//!
//! Architecture (every step a Pekko actor message):
//!   - Generator   = `GeneratorActor::<QwenModelActor>`
//!   - Verifier    = `VerifierActor` (MbppDomain — python3 subprocess)
//!   - Curator     = `CuratorActor` (keeps correct trajectories)
//!   - Trainer     = `QwenTrainerActor` via `QwenTrainerActorHandle`
//!     (rendered corpus → Train → SaveMergedCheckpoint)
//!   - Reload      = `ModelMessage::ReloadCheckpoint` on `QwenModelActor`
//!   - Evaluator   = `EvaluatorActor::<QwenModelActor>` (pass@k random)
//!
//! Note: the per-round eval here is `EvalRandom` (sampling with
//! replacement) — that's what `supervisor::run_round` invokes today.
//! The Phase 17 S9 / Stage C aggregate measurement is a separate
//! benchmark anchor (`phase22_mbpp_baseline --sequential --aggregate`);
//! this binary is the **mechanism reproduction**. Once the saturation
//! curve compounds round-over-round we know the Pekko-side recipe
//! works; numeric calibration to Phase 17 SB r=2 = 0.453 is then a
//! wallclock question (full 100 × passk=10 × multi-round).
//!
//! Smoke recipe (r=2, gen 32, eval 32, train 30 steps/round, ~20 min):
//!   cargo run -p llm-actors --example phase22_mbpp_mr_sft \
//!       --features cuda --release -- --rounds 2
//!
//! Larger smoke (r=3, gen 64, eval 64, train 50 steps/round, ~50 min):
//!   cargo run -p llm-actors --example phase22_mbpp_mr_sft \
//!       --features cuda --release -- --rounds 3 \
//!       --gen-n 64 --eval-n 64 --train-steps 50
//!
//! Best-of-2 quality filter (modest pass-rate lift via model log-prob):
//!   cargo run -p llm-actors --example phase22_mbpp_mr_sft \
//!       --features cuda --release -- --gen-oversample 2

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device};
use clap::Parser;
use llm_actors::{
    curator_actor::SampleMode,
    domain::{filtered::FilteredDomain, mbpp::MbppDomain, Domain},
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
    /// Number of MR rounds. Phase 17 saturation curve covers 1..6.
    #[arg(long, default_value_t = 2)]
    rounds: usize,
    /// Generation count per round (problems sampled with replacement).
    /// Phase 17 used the full 164; Stage D default is 32 — large enough
    /// to keep P(empty 0/N corpus | p≈0.10) ≈ 0.034, so the supervisor's
    /// empty-corpus skip path is hit < 5% of rounds. Bump to 64+ for
    /// long sweeps where any skipped round breaks the saturation chain.
    #[arg(long, default_value_t = 32)]
    gen_n: usize,
    /// Best-of-k filter on the generation step (Phase 6 Shape C).
    /// `oversample = K` → for each of `gen_n` sampled prompts, generate
    /// K candidates with different per-(prompt, k) seeds, score each
    /// via `ModelMessage::ScoreLogProb`, keep the highest-confidence
    /// trajectory. Output count is still `gen_n` — this is a quality
    /// filter, NOT a quantity multiplier. Helps modestly when the
    /// model's log-prob is calibrated to the verifier (Phase 7 found
    /// sum-AUC ~0.55-0.65 for Qwen). Default 1 = off.
    #[arg(long, default_value_t = 1)]
    gen_oversample: usize,
    /// Eval count per round. Phase 17 evaluated all 164 at temp=0.8/k=10;
    /// supervisor's per-round eval is random-with-replacement (Stage B
    /// aggregate is a separate benchmark step). Smoke uses 32 with k=3.
    #[arg(long, default_value_t = 32)]
    eval_n: usize,
    /// passk for the per-round eval inside `run_multi_round`.
    #[arg(long, default_value_t = 3)]
    eval_passk: usize,
    /// AdamW steps per round inside `QwenTrainerActorHandle`. Phase 17
    /// used ~100 steps/round for the full 164-problem set; smoke
    /// uses 30 against gen_n=16.
    #[arg(long, default_value_t = 30)]
    train_steps: usize,
    /// max tokens per generation. Phase 17 used 200.
    #[arg(long, default_value_t = 200)]
    max_new_tokens: usize,
    /// MBPP JSONL path.
    #[arg(long)]
    jsonl: Option<PathBuf>,
    /// Generation sampling temperature. Phase 17 used 0.8 throughout.
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,
    /// LoRA rank (Phase 14-20 recipe = 16).
    #[arg(long, default_value_t = 16)]
    lora_rank: usize,
    /// LoRA alpha (Phase 14-20 recipe = 32, so scale = α/r = 2.0).
    #[arg(long, default_value_t = 32.0)]
    lora_alpha: f32,
    /// AdamW learning rate. Phase 14-20 recipe = 2e-4.
    #[arg(long, default_value_t = 2e-4)]
    lr: f64,
    /// Output dir for per-round merged checkpoints.
    #[arg(long, default_value = "checkpoints/phase22_mbpp_mr_sft")]
    out_dir: PathBuf,
    /// Base seed for all RNGs (gen, gen_sampling, eval, eval_sampling,
    /// corpus). Each is offset deterministically from this base so a
    /// single `--seed N` shift gives a complete reproducible run.
    /// Default 42 preserves the previous hardcoded values bit-exactly.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Scratch dir for `MbppDomain::verify` python3 invocations.
    /// **Required to be unique per concurrent run** — `verify` writes
    /// `solution.py` under this path and an in-process write_lock that
    /// does NOT extend across processes. Two parallel
    /// `phase22_mbpp_mr_sft` runs sharing the same scratch dir will
    /// clobber each other's solution.py.
    #[arg(long)]
    scratch_dir: Option<PathBuf>,
    /// Optional comma-separated list of MBPP task indices to
    /// skip during generation and eval. Useful for the Phase 9 S5
    /// cold-start mitigation: exclude prompts the base Qwen has 0/k
    /// pass-rate on, so they don't dominate the empty-corpus skip
    /// path. Selection bias warning — filtered subset isn't
    /// representative of the full benchmark; use this for *training*
    /// convenience and run benchmark-aligned eval against the
    /// unfiltered domain via `phase22_mbpp_baseline`.
    ///
    /// Wired through `FilteredDomain` (a `Domain` wrapper) — no
    /// supervisor changes required; the wrapped Domain just
    /// renumbers `sample_prompt`/`nth_prompt` to skip the hidden
    /// indices.
    #[arg(long, value_delimiter = ',')]
    prompt_skip_list: Vec<usize>,
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

    let args = Args::parse();
    let device = pick_device();
    let on_cuda = device.is_cuda();
    println!(
        "[Phase22D-MBPP] device = {device:?}, on_cuda = {on_cuda}, rounds = {}",
        args.rounds
    );
    if !on_cuda && std::env::var("PHASE22_ALLOW_CPU").is_err() {
        anyhow::bail!(
            "Refusing to run on CPU. Rebuild with `--features cuda`. \
             Set PHASE22_ALLOW_CPU=1 to override."
        );
    }
    std::fs::create_dir_all(&args.out_dir)?;

    let snapshot = resolve_default_snapshot()?;
    let base_safetensors = snapshot.join("model.safetensors");
    println!("[Phase22D-MBPP] snapshot = {}", snapshot.display());

    // Inference in F16 (Phase 21 D pattern); training in F32 for stable
    // LoRA gradient accumulation. SaveMergedCheckpoint casts down to
    // base dtype on disk so ReloadCheckpoint stays F16.
    let inference_dtype = if on_cuda { DType::F16 } else { DType::F32 };
    let train_dtype = DType::F32;
    let lora_cfg = LoraConfig {
        rank: args.lora_rank,
        alpha: args.lora_alpha,
    };

    let tk = Arc::new(NgptTokenizer::from_hf_file(
        snapshot.join("tokenizer.json"),
    )?);

    let jsonl = args
        .jsonl
        .unwrap_or_else(|| PathBuf::from("data/mbpp/mbpp.jsonl"));
    let scratch = args
        .scratch_dir
        .clone()
        .unwrap_or_else(|| std::env::temp_dir().join("workllm-phase22d-mbpp"));
    let mbpp = MbppDomain::from_jsonl(&jsonl, &scratch)
        .with_context(|| format!("loading MBPP from {}", jsonl.display()))?;
    let total = mbpp.n_problems();
    println!("[Phase22D-MBPP] MbppDomain loaded {} problems", total);
    let inner_domain: Arc<dyn Domain> = Arc::new(mbpp);
    let domain: Arc<dyn Domain> = if args.prompt_skip_list.is_empty() {
        inner_domain
    } else {
        let filtered = FilteredDomain::new(inner_domain, args.prompt_skip_list.iter().copied());
        let surviving = filtered.n_surviving();
        println!(
            "[Phase22D-MBPP] FilteredDomain: hiding {} indices, {} surviving (= {}/{})",
            args.prompt_skip_list.len(),
            surviving,
            surviving,
            total,
        );
        Arc::new(filtered)
    };

    // ===== Actors =====
    let qwen_model = QwenModelActor::from_snapshot_dir(&snapshot, device.clone(), inference_dtype)?;
    let qwen_trainer = QwenTrainerActor::from_snapshot_dir(
        &snapshot,
        device.clone(),
        train_dtype,
        lora_cfg,
        args.lr,
    )?;

    let system = ActorSystem::new("phase22-d");
    let model_ref = system.spawn(qwen_model, "qwen-model").await?;
    let trainer_ref = system.spawn(qwen_trainer, "qwen-trainer").await?;

    let generator = GeneratorActor::<QwenModelActor>::new(
        model_ref.clone(),
        tk.clone(),
        domain.clone(),
        None,
        "qwen".to_string(),
    );
    let generator_ref = system.spawn(generator, "generator").await?;
    let verifier_ref = system
        .spawn(VerifierActor::new(domain.clone()), "verifier")
        .await?;
    let curator_ref = system.spawn(CuratorActor::new(1024), "curator").await?;
    let evaluator =
        EvaluatorActor::<QwenModelActor>::new(model_ref.clone(), tk.clone(), domain.clone(), None);
    let evaluator_ref = system.spawn(evaluator, "evaluator").await?;

    let trainer_handle = Arc::new(QwenTrainerActorHandle::new(
        trainer_ref,
        args.train_steps,
        base_safetensors.clone(),
    )) as Arc<dyn TrainerHandle>;

    let actors = RoundActors::<QwenModelActor> {
        model: model_ref,
        generator: generator_ref,
        verifier: verifier_ref,
        curator: curator_ref,
        trainer: trainer_handle,
        evaluator: evaluator_ref,
    };
    println!("[Phase22D-MBPP] 6 actors spawned + RoundActors built\n");

    // train_cfg is required by RoundConfig but ignored by
    // QwenTrainerActorHandle. We populate it with a sensible-looking
    // smoke config so downstream logging is honest.
    let mut train_cfg = TrainConfig::smoke();
    train_cfg.max_steps = args.train_steps;
    train_cfg.optimizer = OptimizerKind::Adam;

    // All seeds derive from --seed deterministically. The offsets
    // preserve the prior bit-exact defaults at seed=42:
    //   gen_seed       = seed              (was 42)
    //   gen_sampling   = seed              (was 42)
    //   eval_seed      = seed - 35         (so seed=42 → 7)
    //   eval_sampling  = seed - 35         (so seed=42 → 7)
    //   corpus_seed    = seed - 42         (so seed=42 → 0)
    let gen_seed = args.seed;
    let eval_seed = args.seed.wrapping_sub(35);
    let corpus_seed = args.seed.wrapping_sub(42);
    let base = RoundConfig {
        round: 0,
        gen_n: args.gen_n,
        gen_seed,
        gen_sampling: GenerateConfig {
            max_new_tokens: args.max_new_tokens,
            temperature: args.temperature,
            top_k: Some(40),
            top_p: Some(0.95),
            seed: Some(gen_seed),
        },
        eval_n: args.eval_n,
        eval_seed,
        eval_sampling: GenerateConfig {
            max_new_tokens: args.max_new_tokens,
            // Per-round eval at temp=0.8 + passk>1 mirrors Phase 17 S6
            // / Stage B's recipe for "pass@1 (raw)" measurement (which
            // is the correct apples-to-apples Phase 17 metric).
            temperature: 0.8,
            top_k: Some(40),
            top_p: Some(0.95),
            seed: Some(eval_seed),
        },
        train_cfg,
        init_from: None,
        save_path: args.out_dir.join("r0_merged.safetensors"),
        min_corpus_chars: 32,
        sample_mode: SampleMode::Uniform,
        corpus_seed: Some(corpus_seed),
        anchor: None,
        freeze_base: false,
        gen_oversample: args.gen_oversample.max(1),
        dpo_beta: None,
        dpo_reference_path: None,
        dpo_max_pairs_per_prompt: 0,
        dpo_sft_anchor_weight: 0.0,
        eval_passk: args.eval_passk,
    };

    // `run_multi_round` auto-chains init_from and bumps seeds per round.
    // We also rewrite save_path per round so each checkpoint is kept.
    let out_dir = args.out_dir.clone();
    let reports = run_multi_round(
        &actors,
        MultiRoundConfig::new(args.rounds, base),
        |r, rep| {
            // Distinguish "measured 0/N" from "skipped (None)". On
            // empty-corpus rounds, supervisor early-returns BEFORE
            // train/save/reload/eval-after — `eval_correct_after` stays
            // None. Conflating that with a measured zero (as a
            // `unwrap_or(0.0)` does) prints a misleading Δ=-pass_before
            // and looks like the model collapsed when in fact eval-after
            // simply didn't run.
            let fmt_pass = |c: Option<usize>| -> String {
                match c {
                    Some(n) => format!("{:.3}", n as f32 / rep.eval_total.max(1) as f32),
                    None => "N/A".to_string(),
                }
            };
            let delta_str = match (rep.eval_correct_before, rep.eval_correct_after) {
                (Some(b), Some(a)) => {
                    let denom = rep.eval_total.max(1) as f32;
                    format!("Δ={:+.3}", (a as f32 - b as f32) / denom)
                }
                _ => "Δ=N/A".to_string(),
            };
            println!(
                "[Phase22D-MBPP] round {r}  gen={}/{}  pass@{}={}→{}  {}  elapsed_ms={}",
                rep.correct,
                rep.generated,
                args.eval_passk,
                fmt_pass(rep.eval_correct_before),
                fmt_pass(rep.eval_correct_after),
                delta_str,
                rep.elapsed_ms,
            );
            let _ = std::fs::copy(
                out_dir.join("r0_merged.safetensors"),
                out_dir.join(format!("r{r}_merged.safetensors")),
            );
        },
    )
    .await?;

    assert_eq!(reports.len(), args.rounds, "round-count mismatch");

    // Final summary: Phase 17 r=2 reference for context.
    println!("\n[Phase22D-MBPP] === multi-round summary ===");
    for (i, rep) in reports.iter().enumerate() {
        let pa = match rep.eval_correct_after {
            Some(c) => format!("{:.3}", c as f32 / rep.eval_total.max(1) as f32),
            None => "N/A (skipped)".to_string(),
        };
        println!("  round {i}  eval_after pass@{} = {}", args.eval_passk, pa);
    }
    println!("\n[Phase22D-MBPP] Phase 17/18/20 MBPP-100 reference (mean over 5 seeds, passk=10):");
    println!(
        "           r=2 = 0.453 ± 0.016 (Phase 17 SB)   r=3 = 0.457 ± 0.013 (Phase 18 S3)   r=5 = 0.541 ± 0.014 (Phase 20 S2)"
    );
    println!("           This binary's mechanism reproduces the Pekko-side wiring;");
    println!("           numerical match requires gen_n=100, eval_n=100, passk=10,");
    println!("           train_steps=~100 — wallclock ~30 GPU-min per round.");
    println!("\nphase22_mbpp_mr_sft: PASS");
    Ok(())
}
