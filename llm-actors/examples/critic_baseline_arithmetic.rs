//! Phase 7 Session 1: Shape C transfer test on `ArithmeticDomain`.
//!
//! Phase 6 Shape C demonstrated `LogitCritic` AUC 0.727 on
//! `RustCodeDomain` (cargo-verified Rust slot-fill). This example
//! repeats the **same protocol** on `ArithmeticDomain` (single-digit
//! addition, parse-then-compare verifier). If AUC ≥ 0.6 here too,
//! Shape C is a general pattern, not a Rust-specific quirk.
//!
//! Differences from `critic_baseline.rs`:
//!   - Verifier is a microsecond `parse + compare` (vs ~100ms cargo).
//!   - Char tokenizer over the small `0..9 + = \n` charset
//!     (vocab ~13) vs the K9 vocab ~97.
//!   - Block size 32 (one example per window) vs 80.
//!   - Pretrain converges fast (~500 steps) on the small task.
//!
//! Same metric, same acceptance gate (AUC ≥ 0.6 from
//! `docs/phase6-shape-c.md`).
//!
//! Run:
//!   cargo run -p llm-actors --example critic_baseline_arithmetic --features cuda --release

use std::path::PathBuf;
use std::sync::Arc;

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use clap::Parser;
use llm_actors::{
    domain::{
        arithmetic::{ArithmeticDomain, SeedMode},
        Domain,
    },
    roc_auc, AlwaysCorrectCritic, Critic, LogitCritic, RandomCritic,
};
use nanogpt_rs::{
    config::{ActivationKind, GPTConfig, NormKind, NormPosition},
    data::TokenDataset,
    generate::{generate, GenerateConfig},
    model::GPT,
    tokenizer::Tokenizer,
    train::{train_from, TrainConfig},
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use tracing::info;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_t = 800)]
    pretrain_steps: usize,
    #[arg(long, default_value_t = 2000)]
    pretrain_examples: usize,
    /// Number of distinct prompts to harvest candidates for.
    #[arg(long, default_value_t = 30)]
    n_prompts: usize,
    /// Stochastic generations per prompt.
    #[arg(long, default_value_t = 30)]
    samples_per_prompt: usize,
    #[arg(long, default_value_t = 1.0)]
    temperature: f64,
    #[arg(long, default_value_t = 8)]
    top_k: usize,
    /// 4 chars handles the largest correct sum (9+9=18, 2 digits + \n).
    #[arg(long, default_value_t = 4)]
    max_new_tokens: usize,
    #[arg(long, default_value = "checkpoints/critic_baseline_arith.safetensors")]
    seed_ckpt: PathBuf,
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

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    let args = Args::parse();
    let device = pick_device();
    info!(?device, "device");

    let domain = Arc::new(ArithmeticDomain::default());

    // ---- Pretrain corpus + char tokenizer (covers all chars the
    // domain ever emits; vocab is tiny).
    let pretrain_text = domain.synth_corpus(args.pretrain_examples, 7);
    // Add seed pairs (NoCarry) so the model's pretrain set is balanced
    // — without this, max_operand=9 has the carry half (a+b > 9) under-
    // represented, and the AUC measurement becomes about base rate
    // rather than ranking quality.
    let mut full_corpus = pretrain_text.clone();
    for (a, b) in domain.enumerate_seed_pairs(SeedMode::Full) {
        full_corpus.push_str(&domain.render_example(a, b));
    }
    let mut seed_chars = String::from(domain.charset());
    seed_chars.push_str(&full_corpus);
    let tk = Arc::new(Tokenizer::char_from_text(&seed_chars));
    let vocab = tk.vocab_size();
    info!(
        vocab,
        corpus_chars = full_corpus.len(),
        "tokenizer + corpus"
    );

    // ---- Small char-level model. The arithmetic task is trivial
    // compared to K9 — a 6L/192-dim is overkill but fast on GPU.
    let gpt_cfg = GPTConfig {
        vocab_size: vocab,
        block_size: 32,
        n_layer: 6,
        n_head: 6,
        n_embd: 192,
        dropout: 0.0,
        bias: false,
        ffn_mult: 4,
        use_rope: true,
        rope_base: 10_000.0,
        n_kv_head: 6,
        n_experts: 1,
        moe_top_k: 0,
        moe_aux_weight: 0.0,
        activation: ActivationKind::SwiGlu,
        weight_tying: false,
        norm_kind: NormKind::RmsNorm,
        norm_position: NormPosition::Pre,
        // No LoRA — full FT for the critic-baseline check, matches
        // the original Phase 6 Shape C pretrain regime.
        lora_rank: 0,
        lora_alpha: 16.0,
    };
    info!(params = gpt_cfg.num_params_estimate(), "model config");

    // ---- Pretrain.
    info!("pretraining...");
    let ids = tk.encode(&full_corpus)?;
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

    // ---- Reload generator (independent VarMap from critic).
    let mut gen_varmap = VarMap::new();
    let gen_vb = VarBuilder::from_varmap(&gen_varmap, DType::F32, &device);
    let gen_model = GPT::new(gpt_cfg.clone(), gen_vb)?;
    gen_varmap.load(&args.seed_ckpt)?;

    // ---- Two LogitCritic variants: mean (default, length-normalized)
    // and sum (raw log P). Mean rewards short completions when the
    // domain has variable-length answers — we measure both to isolate
    // the normalization effect.
    let critic_mean =
        LogitCritic::from_checkpoint(gpt_cfg.clone(), tk.clone(), device.clone(), &args.seed_ckpt)?;
    let mut critic_sum =
        LogitCritic::from_checkpoint(gpt_cfg.clone(), tk.clone(), device.clone(), &args.seed_ckpt)?;
    critic_sum.normalize_by_length = false;
    let random_critic = RandomCritic::new(0xC417);
    let always_critic = AlwaysCorrectCritic;

    // ---- Sample prompts via the domain (so distribution matches
    // what self_improve_round.rs would see). The same prompt may be
    // drawn multiple times; that's fine — gen samples are independent.
    let mut rng = StdRng::seed_from_u64(0xE0FF);
    let prompts: Vec<String> = (0..args.n_prompts)
        .map(|_| domain.sample_prompt(&mut rng))
        .collect();

    info!(
        n_prompts = args.n_prompts,
        samples_per_prompt = args.samples_per_prompt,
        "harvesting candidates"
    );
    // (prompt, completion, label, score_mean, score_sum, score_random)
    let mut samples: Vec<(String, String, bool, f32, f32, f32)> = Vec::new();
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
            let completion_ids = &out_ids[prompt_ids.len()..];
            let completion = tk.decode(completion_ids)?;
            let completion_trim = match completion.find('\n') {
                Some(p) => &completion[..p],
                None => &completion[..],
            };
            let verdict = domain.verify(prompt, completion_trim);
            let label = matches!(verdict, llm_actors::Verdict::Correct);
            let score_mean = critic_mean.score(prompt, completion_trim);
            let score_sum = critic_sum.score(prompt, completion_trim);
            let score_random = random_critic.score(prompt, completion_trim);
            samples.push((
                prompt.clone(),
                completion_trim.to_string(),
                label,
                score_mean,
                score_sum,
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

    let pairs_mean: Vec<(f32, bool)> = samples
        .iter()
        .map(|(_, _, l, sm, _, _)| (*sm, *l))
        .collect();
    let pairs_sum: Vec<(f32, bool)> = samples
        .iter()
        .map(|(_, _, l, _, ss, _)| (*ss, *l))
        .collect();
    let pairs_random: Vec<(f32, bool)> = samples
        .iter()
        .map(|(_, _, l, _, _, sr)| (*sr, *l))
        .collect();
    let pairs_always: Vec<(f32, bool)> = samples
        .iter()
        .map(|(_, _, l, _, _, _)| (always_critic.score("", ""), *l))
        .collect();

    let auc_mean = roc_auc(&pairs_mean);
    let auc_sum = roc_auc(&pairs_sum);
    let auc_random = roc_auc(&pairs_random);
    let auc_always = roc_auc(&pairs_always);

    println!("\n=== Phase 7 S1 — Shape C transfer to ArithmeticDomain ===");
    println!(
        "samples:        {total_attempts} ({total_correct} correct, {} incorrect)",
        total_attempts - total_correct
    );
    println!("LogitCritic mean:  {auc_mean:.3}   (mean log-prob — K9 default)");
    println!("LogitCritic sum:   {auc_sum:.3}   (sum log-prob — length-aware variant)");
    println!("RandomCritic:      {auc_random:.3}   (sampling variance, ~0.5)");
    println!("AlwaysCritic:      {auc_always:.3}   (constant → all ties → 0.5)");
    println!();
    println!("Acceptance criterion (docs/phase6-shape-c.md): AUC ≥ 0.6");
    let best_auc = auc_mean.max(auc_sum);
    if best_auc.is_nan() {
        println!("=> undefined (one class missing in harvest)");
    } else if best_auc >= 0.6 {
        let which = if auc_mean >= auc_sum { "mean" } else { "sum" };
        println!("=> PASS via {which}-variant — Shape C transfers with the right scoring.");
    } else {
        println!("=> FAIL on BOTH variants — Shape C does NOT cleanly transfer.");
    }

    let mut by_mean = samples.clone();
    by_mean.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
    println!("\n=== Top 5 by mean log-prob ===");
    for (p, c, lab, sm, _, _) in by_mean.iter().take(5) {
        println!(
            "  [{}] {sm:.4}  {p:?} → {c:?}",
            if *lab { "ok" } else { "no" }
        );
    }
    let mut by_sum = samples.clone();
    by_sum.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
    println!("\n=== Top 5 by sum log-prob ===");
    for (p, c, lab, _, ss, _) in by_sum.iter().take(5) {
        println!(
            "  [{}] {ss:.4}  {p:?} → {c:?}",
            if *lab { "ok" } else { "no" }
        );
    }
    println!("\n=== Bottom 5 by sum log-prob ===");
    for (p, c, lab, _, ss, _) in by_sum.iter().rev().take(5) {
        println!(
            "  [{}] {ss:.4}  {p:?} → {c:?}",
            if *lab { "ok" } else { "no" }
        );
    }

    // Selection sweep — same as critic_baseline.rs Session 3.
    println!("\n=== Selection sweep (random vs critic at oversample F) ===");
    println!("(per-prompt: draw F random samples, compare random-pick-1 vs critic-pick-top-1)\n");
    println!("  F  random_pass  critic_pass    Δ      lift");
    println!("  -  -----------  -----------  ------   -----");
    let prompts_unique: Vec<String> = {
        use std::collections::HashSet;
        let mut seen: HashSet<String> = HashSet::new();
        samples
            .iter()
            .filter_map(|(p, _, _, _, _, _)| {
                if seen.insert(p.clone()) {
                    Some(p.clone())
                } else {
                    None
                }
            })
            .collect()
    };
    let n_trials = 1000;
    println!("\n--- mean log-prob critic ---");
    println!("  F  random_pass  critic_pass    Δ      lift");
    println!("  -  -----------  -----------  ------   -----");
    for &factor in &[1, 2, 4, 8, 16] {
        if factor > args.samples_per_prompt {
            continue;
        }
        let (random_pass, critic_pass) = bake_off(
            &samples,
            &prompts_unique,
            factor,
            n_trials,
            0xB00B + factor as u64,
            |s| s.3, // mean log-prob
        );
        let delta = critic_pass - random_pass;
        let lift = if random_pass > 1e-6 {
            critic_pass / random_pass
        } else {
            f32::INFINITY
        };
        println!("  {factor:>1}    {random_pass:>9.3}    {critic_pass:>9.3}   {delta:+.3}   {lift:>4.2}×");
    }
    println!("\n--- sum log-prob critic ---");
    println!("  F  random_pass  critic_pass    Δ      lift");
    println!("  -  -----------  -----------  ------   -----");
    for &factor in &[1, 2, 4, 8, 16] {
        if factor > args.samples_per_prompt {
            continue;
        }
        let (random_pass, critic_pass) = bake_off(
            &samples,
            &prompts_unique,
            factor,
            n_trials,
            0xB00B + factor as u64,
            |s| s.4, // sum log-prob
        );
        let delta = critic_pass - random_pass;
        let lift = if random_pass > 1e-6 {
            critic_pass / random_pass
        } else {
            f32::INFINITY
        };
        println!("  {factor:>1}    {random_pass:>9.3}    {critic_pass:>9.3}   {delta:+.3}   {lift:>4.2}×");
    }
    println!();
    println!("Compare to RustCodeDomain (Phase 6 Shape C S3): F=4 lift 1.22×, F=16 0.41×.");

    Ok(())
}

/// `score_of`: closure that maps a sample to its scalar critic score.
/// Lets us run the same bake-off for mean vs sum scoring.
fn bake_off<F>(
    samples: &[(String, String, bool, f32, f32, f32)],
    prompts: &[String],
    factor: usize,
    n_trials: usize,
    seed: u64,
    score_of: F,
) -> (f32, f32)
where
    F: Fn(&(String, String, bool, f32, f32, f32)) -> f32,
{
    use rand::seq::SliceRandom;
    let cohorts: Vec<Vec<usize>> = prompts
        .iter()
        .map(|p| {
            samples
                .iter()
                .enumerate()
                .filter(|(_, s)| &s.0 == p)
                .map(|(i, _)| i)
                .collect()
        })
        .collect();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut random_correct = 0u64;
    let mut critic_correct = 0u64;
    let mut total = 0u64;
    for _ in 0..n_trials {
        for cohort in &cohorts {
            if cohort.len() < factor {
                continue;
            }
            let picks: Vec<usize> = cohort.choose_multiple(&mut rng, factor).copied().collect();
            if samples[picks[0]].2 {
                random_correct += 1;
            }
            let best = picks
                .iter()
                .max_by(|&&a, &&b| {
                    score_of(&samples[a])
                        .partial_cmp(&score_of(&samples[b]))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .copied()
                .expect("non-empty picks");
            if samples[best].2 {
                critic_correct += 1;
            }
            total += 1;
        }
    }
    if total == 0 {
        return (f32::NAN, f32::NAN);
    }
    (
        random_correct as f32 / total as f32,
        critic_correct as f32 / total as f32,
    )
}
