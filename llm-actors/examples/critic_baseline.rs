//! Phase 6 Shape C Session 2 measurement: does the K9 generator's
//! own log-probability of a candidate `(prompt, completion)` correlate
//! with cargo's verdict?
//!
//! If yes (AUC ≥ 0.6 from `docs/phase6-shape-c.md`), the LM's
//! `lm_head` already encodes enough "this looks plausible" signal
//! to act as a free pre-filter before the expensive cargo verifier.
//! No separate critic model is needed.
//!
//! Pipeline:
//!   1. Pretrain a small char-level K9 model briefly (so it has
//!      *some* signal — pure-random init has no log-prob ranking).
//!   2. Sample N stochastic completions per prompt across the 3
//!      RustCodeDomain challenges.
//!   3. For each (prompt, completion):
//!        - cargo says correct/incorrect (label)
//!        - LogitCritic gives mean-log-prob (score)
//!        - RandomCritic gives noise (negative baseline)
//!        - AlwaysCorrectCritic gives 1.0 (no-filter baseline)
//!   4. Compute roc_auc for each critic; report.
//!
//! Run:
//!   cargo run -p llm-actors --example critic_baseline --features cuda --release

use std::path::PathBuf;
use std::sync::Arc;

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use clap::Parser;
use llm_actors::{
    domain::{rust_code::RustCodeDomain, Domain},
    roc_auc, AlwaysCorrectCritic, Critic, LogitCritic, RandomCritic,
};
use nanogpt_rs::{
    config::GPTConfig,
    data::TokenDataset,
    generate::{generate, GenerateConfig},
    model::GPT,
    tokenizer::Tokenizer,
    train::{train_from, TrainConfig},
};
use tracing::info;

#[derive(Parser, Debug)]
struct Args {
    /// Pretrain steps for the K9 model (the critic uses this same model).
    #[arg(long, default_value_t = 1500)]
    pretrain_steps: usize,
    #[arg(long, default_value_t = 900)]
    pretrain_examples: usize,
    /// Number of stochastic generations per prompt to harvest as
    /// labeled examples.
    #[arg(long, default_value_t = 30)]
    samples_per_prompt: usize,
    /// Generation temperature for harvesting candidates.
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,
    /// Top-k for harvesting.
    #[arg(long, default_value_t = 10)]
    top_k: usize,
    #[arg(long, default_value_t = 16)]
    max_new_tokens: usize,
    /// Where to save / re-load the pretrained checkpoint. The critic
    /// reloads from here so its tensors are independent of the
    /// generator's varmap.
    #[arg(long, default_value = "checkpoints/critic_baseline_seed.safetensors")]
    seed_ckpt: PathBuf,
    #[arg(long, default_value = "/tmp/workllm-rust-scratch-critic")]
    scratch_dir: PathBuf,
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

const CHALLENGES: &[(&str, &[&str])] = &[
    (
        "fn main() { assert_eq!(",
        &[
            "2 + 3", "1 + 4", "5 + 0", "0 + 5", "10 - 5", "5 * 1", "1 * 5", "100 / 20",
        ],
    ),
    (
        "fn main() { assert_eq!(2 * (",
        &["7", "3 + 4", "4 + 3", "1 + 6", "6 + 1", "10 - 3", "14 / 2"],
    ),
    (
        "fn main() { let s: &str = ",
        &[
            r#""hello""#,
            r#""world""#,
            r#""abcde""#,
            r#""12345""#,
            r#""HELLO""#,
        ],
    ),
];

fn synth_pretrain_corpus(n: usize, seed: u64) -> String {
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = String::with_capacity(n * 32);
    for _ in 0..n {
        let (prompt, slots) = CHALLENGES.choose(&mut rng).expect("non-empty");
        let slot = slots.choose(&mut rng).expect("non-empty");
        out.push_str(prompt);
        out.push_str(slot);
        out.push('\n');
    }
    out
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    let args = Args::parse();
    let device = pick_device();
    info!(?device, "device");

    // ---- Domain (cargo verifier).
    let mut rcd = RustCodeDomain::new(&args.scratch_dir);
    rcd.run_program = true;
    rcd.ensure_scratch_project()?;
    let domain = Arc::new(rcd);

    // ---- Build pretrain corpus + char tokenizer.
    let pretrain_text = synth_pretrain_corpus(args.pretrain_examples, 7);
    let mut seed_chars = String::from(domain.charset());
    seed_chars.push_str(&pretrain_text);
    let tk = Arc::new(Tokenizer::char_from_text(&seed_chars));
    let vocab = tk.vocab_size();
    info!(vocab, "tokenizer ready");

    // ---- K9 base config (LoRA r=32 α=64 — Phase 6's best generator).
    let gpt_cfg = GPTConfig {
        vocab_size: vocab,
        block_size: 80,
        n_layer: 4,
        n_head: 4,
        n_embd: 128,
        dropout: 0.0,
        bias: false,
        ffn_mult: 4,
        use_rope: true,
        rope_base: 10_000.0,
        n_kv_head: 4,
        n_experts: 1,
        moe_top_k: 0,
        moe_aux_weight: 0.0,
        activation: nanogpt_rs::config::ActivationKind::SwiGlu,
        weight_tying: false,
        norm_kind: nanogpt_rs::config::NormKind::RmsNorm,
        norm_position: nanogpt_rs::config::NormPosition::Pre,
        lora_rank: 32,
        lora_alpha: 64.0,
    };

    // ---- Pretrain.
    info!("pretraining generator (will also be used as critic)...");
    let ids = tk.encode(&pretrain_text)?;
    let ds = TokenDataset::new(ids, gpt_cfg.block_size);
    let mut tcfg = TrainConfig::smoke();
    tcfg.max_steps = args.pretrain_steps;
    tcfg.batch_size = 64;
    tcfg.eval_interval = args.pretrain_steps;
    tcfg.lr = 3e-3;
    tcfg.min_lr = 3e-4;
    tcfg.warmup_steps = 50;
    let outcome = train_from(
        &gpt_cfg,
        &ds,
        None,
        &tcfg,
        &device,
        Some(&args.seed_ckpt),
        None,
    )?;
    info!(
        train_loss = outcome.last_train_loss,
        steps = outcome.final_step,
        "pretrain done"
    );

    // ---- Build a fresh GPT for generation (loads from the checkpoint).
    // The VarMap must be `mut` so `.load()` can populate the named tensors;
    // we keep it alive for the lifetime of `gen_model`.
    let mut gen_varmap = VarMap::new();
    let gen_vb = VarBuilder::from_varmap(&gen_varmap, DType::F32, &device);
    let gen_model = GPT::new(gpt_cfg.clone(), gen_vb)?;
    gen_varmap.load(&args.seed_ckpt)?;

    // ---- Build the LogitCritic from the same checkpoint (independent VarMap).
    let critic =
        LogitCritic::from_checkpoint(gpt_cfg.clone(), tk.clone(), device.clone(), &args.seed_ckpt)?;
    let random_critic = RandomCritic::new(0xC417);
    let always_critic = AlwaysCorrectCritic;

    // ---- Harvest labeled (prompt, completion, verdict, score) tuples.
    let prompts: Vec<&str> = CHALLENGES.iter().map(|(p, _)| *p).collect();
    let mut samples: Vec<(String, String, bool, f32, f32)> = Vec::new();
    info!(
        n_prompts = prompts.len(),
        samples_per_prompt = args.samples_per_prompt,
        "harvesting candidates"
    );
    let mut total_correct = 0usize;
    let mut total_attempts = 0usize;
    for (pi, prompt) in prompts.iter().enumerate() {
        let prompt_ids = tk.encode(prompt)?;
        for j in 0..args.samples_per_prompt {
            let cfg = GenerateConfig {
                max_new_tokens: args.max_new_tokens,
                temperature: args.temperature,
                top_k: Some(args.top_k),
                top_p: None,
                seed: Some(0xCAFE + (pi as u64) * 1_000 + j as u64),
            };
            let out_ids = generate(&gen_model, &prompt_ids, &cfg, &device)?;
            // out_ids is the full sequence (prompt + completion). Slice out
            // the completion.
            let completion_ids = &out_ids[prompt_ids.len()..];
            let completion = tk.decode(completion_ids)?;
            // Trim at the first newline (matches GeneratorActor's stop_char).
            let completion_trim = match completion.find('\n') {
                Some(p) => &completion[..p],
                None => &completion[..],
            };
            // Verify with cargo. Verdict is binary for our purposes.
            let verdict = domain.verify(prompt, completion_trim);
            let label = matches!(verdict, llm_actors::Verdict::Correct);
            // Score with each critic. The critic's input is the COMPLETION
            // up to the stop character (no trailing \n).
            let score_logit = critic.score(prompt, completion_trim);
            let score_random = random_critic.score(prompt, completion_trim);
            samples.push((
                (*prompt).to_string(),
                completion_trim.to_string(),
                label,
                score_logit,
                score_random,
            ));
            total_attempts += 1;
            if label {
                total_correct += 1;
            }
        }
    }
    info!(
        total_attempts,
        total_correct,
        pass_rate = total_correct as f32 / total_attempts as f32,
        "harvest done"
    );

    // ---- Compute AUC for each critic.
    let pairs_logit: Vec<(f32, bool)> = samples.iter().map(|(_, _, l, sl, _)| (*sl, *l)).collect();
    let pairs_random: Vec<(f32, bool)> = samples.iter().map(|(_, _, l, _, sr)| (*sr, *l)).collect();
    // AlwaysCorrectCritic gives the same score to everything → all ties.
    let pairs_always: Vec<(f32, bool)> = samples
        .iter()
        .map(|(_, _, l, _, _)| (always_critic.score("", ""), *l))
        .collect();

    let auc_logit = roc_auc(&pairs_logit);
    let auc_random = roc_auc(&pairs_random);
    let auc_always = roc_auc(&pairs_always);

    println!("\n=== Phase 6 Shape C — LogitCritic AUC ===");
    println!(
        "samples:        {total_attempts} ({total_correct} correct, {} incorrect)",
        total_attempts - total_correct
    );
    println!("LogitCritic:    {auc_logit:.3}  (mean log-prob over completion tokens)");
    println!("RandomCritic:   {auc_random:.3}  (deterministic random scoring; expect ~0.5)");
    println!("AlwaysCritic:   {auc_always:.3}  (constant score → all ties → 0.5)");
    println!();
    println!("Acceptance criterion (docs/phase6-shape-c.md): AUC ≥ 0.6");
    if auc_logit.is_nan() {
        println!("=> result: undefined (one of the classes is empty in the harvest)");
    } else if auc_logit >= 0.6 {
        println!("=> PASS — LogitCritic carries enough signal to pre-filter cargo");
        println!("        Session 3 (--critic-threshold integration) is worth running.");
    } else {
        println!("=> FAIL — LogitCritic doesn't beat the {auc_logit:.3} ceiling");
        println!("        Either Session 4 (dedicated critic head) or pivot.");
    }

    // ---- Print a few examples for a sanity check.
    let mut by_score = samples.clone();
    by_score.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
    println!("\n=== Top 5 by LogitCritic score ===");
    for (p, c, lab, sl, _) in by_score.iter().take(5) {
        println!(
            "  [{}] score={sl:.4}  {p:?} → {c:?}",
            if *lab { "ok" } else { "no" }
        );
    }
    println!("\n=== Bottom 5 by LogitCritic score ===");
    for (p, c, lab, sl, _) in by_score.iter().rev().take(5) {
        println!(
            "  [{}] score={sl:.4}  {p:?} → {c:?}",
            if *lab { "ok" } else { "no" }
        );
    }

    Ok(())
}
