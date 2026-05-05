//! Phase 8 Session 1: Apply the Phase 7 Shape C playbook to
//! `KoreanCompletionDomain`.
//!
//! Phase 7 design doc decision tree:
//!   1. Length-varying domain? Korean sentences are 8-400 chars → YES.
//!      Use sum-scoring (`LogitCritic::sum_scoring_*`).
//!   2. Held-out sum-AUC ≥ 0.6 → apply Shape C; else don't.
//!   3. Pass rate is informative but not deciding (Phase 7 S2).
//!
//! Differences from arithmetic baseline:
//!   - Real-world checkpoint: K8's 30K-step KoWiki 50M Llama-recipe
//!     model (val_loss 7.43). Loaded from disk, no pretraining here.
//!   - BPE tokenizer (16K vocab, KoWiki-trained) instead of char.
//!   - Heuristic verifier (Hangul + sentence-ending + length window)
//!     instead of execution / arithmetic equality.
//!   - Default to sum-scoring per Phase 7 design doc.
//!
//! This is the cleanest application of the Phase 7 playbook:
//! identify the gate, measure, decide. No code changes to the
//! domain — just a new measurement script.
//!
//! Run:
//!   cargo run -p llm-actors --example critic_baseline_korean --features cuda --release

use std::path::PathBuf;
use std::sync::Arc;

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use clap::Parser;
use llm_actors::{
    domain::{korean_completion::KoreanCompletionDomain, Domain},
    roc_auc, AlwaysCorrectCritic, Critic, LogitCritic, RandomCritic,
};
use nanogpt_rs::{
    config::GPTConfig,
    generate::{generate, GenerateConfig},
    model::GPT,
    tokenizer::Tokenizer,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use tracing::info;

#[derive(Parser, Debug)]
struct Args {
    /// K8's 30K-step KoWiki checkpoint (from `train_kowiki`).
    #[arg(long, default_value = "checkpoints/kowiki_50m_30k.safetensors")]
    init: PathBuf,
    /// 16K BPE tokenizer (sibling to the corpus).
    #[arg(long, default_value = "data/kowiki/kowiki_bpe.json")]
    tokenizer: PathBuf,
    /// Number of distinct prompts to harvest candidates for.
    /// KoreanCompletionDomain has a fixed seed list of ~10 prompts;
    /// we sample with replacement from it.
    #[arg(long, default_value_t = 30)]
    n_prompts: usize,
    /// Stochastic generations per prompt.
    #[arg(long, default_value_t = 20)]
    samples_per_prompt: usize,
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,
    #[arg(long, default_value_t = 40)]
    top_k: usize,
    /// Generation length cap. Real Korean sentences are 8–400 chars
    /// in tokens — 80 BPE tokens is roughly 100–200 Korean chars.
    #[arg(long, default_value_t = 80)]
    max_new_tokens: usize,
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

fn load_cfg(path: &std::path::Path) -> anyhow::Result<GPTConfig> {
    let cfg_path = path.with_extension("cfg.json");
    let s = std::fs::read_to_string(&cfg_path)
        .map_err(|e| anyhow::anyhow!("read {:?}: {e}", cfg_path))?;
    Ok(serde_json::from_str(&s)?)
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    let args = Args::parse();
    let device = pick_device();
    info!(?device, "device");

    // ---- Load checkpoint config (sibling .cfg.json).
    let gpt_cfg = load_cfg(&args.init)?;
    info!(
        params = gpt_cfg.num_params_estimate(),
        n_layer = gpt_cfg.n_layer,
        n_embd = gpt_cfg.n_embd,
        block_size = gpt_cfg.block_size,
        vocab = gpt_cfg.vocab_size,
        "loaded model config from K8 30K checkpoint"
    );

    // ---- Tokenizer (BPE).
    let tk = Arc::new(Tokenizer::from_hf_file(&args.tokenizer)?);
    if tk.vocab_size() != gpt_cfg.vocab_size {
        anyhow::bail!(
            "tokenizer vocab {} != model vocab {} — mismatched checkpoint/tokenizer pair",
            tk.vocab_size(),
            gpt_cfg.vocab_size
        );
    }
    info!(vocab = tk.vocab_size(), "tokenizer loaded");

    // ---- Domain (heuristic verifier; no training data needed).
    let domain = Arc::new(KoreanCompletionDomain::default());

    // ---- Reload generator + LogitCritic from the checkpoint.
    let mut gen_varmap = VarMap::new();
    let gen_vb = VarBuilder::from_varmap(&gen_varmap, DType::F32, &device);
    let gen_model = GPT::new(gpt_cfg.clone(), gen_vb)?;
    gen_varmap.load(&args.init)?;

    // Phase 7 design doc tier 2: Korean is length-varying → sum-scoring.
    // Also instantiate mean-scoring for comparison so we can confirm
    // the doc's prediction (mean fails on length-varying domains).
    let critic_mean =
        LogitCritic::from_checkpoint(gpt_cfg.clone(), tk.clone(), device.clone(), &args.init)?;
    let critic_sum = LogitCritic::sum_scoring_from_checkpoint(
        gpt_cfg.clone(),
        tk.clone(),
        device.clone(),
        &args.init,
    )?;
    let random_critic = RandomCritic::new(0xC417);
    let always_critic = AlwaysCorrectCritic;

    // ---- Sample prompts via the domain's seed list.
    let mut rng = StdRng::seed_from_u64(0xE0FF);
    let prompts: Vec<String> = (0..args.n_prompts)
        .map(|_| domain.sample_prompt(&mut rng))
        .collect();

    info!(
        n_prompts = args.n_prompts,
        samples_per_prompt = args.samples_per_prompt,
        "harvesting candidates from K8 model"
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
            // Heuristic verifier looks at the whole completion, no \n
            // truncation (KoreanCompletionDomain doesn't define one).
            let verdict = domain.verify(prompt, &completion);
            let label = matches!(verdict, llm_actors::Verdict::Correct);
            let score_mean = critic_mean.score(prompt, &completion);
            let score_sum = critic_sum.score(prompt, &completion);
            let score_random = random_critic.score(prompt, &completion);
            samples.push((
                prompt.clone(),
                completion.clone(),
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

    println!("\n=== Phase 8 S1 — Shape C on KoreanCompletionDomain (K8 30K) ===");
    println!(
        "samples:           {total_attempts} ({total_correct} correct, {} incorrect)",
        total_attempts - total_correct
    );
    println!("LogitCritic mean:  {auc_mean:.3}   (Phase 7 predicts: FAIL on length-varying)");
    println!("LogitCritic sum:   {auc_sum:.3}   ★ PRIMARY GATE (Phase 7 design)");
    println!("RandomCritic:      {auc_random:.3}   (sampling variance, ~0.5)");
    println!("AlwaysCritic:      {auc_always:.3}   (constant → all ties)");
    println!();
    println!("Phase 7 acceptance gate: sum-AUC ≥ 0.6");
    if auc_sum.is_nan() {
        println!("=> undefined (one class missing from harvest)");
    } else if auc_sum >= 0.6 {
        println!("=> PASS — Shape C applies cleanly to Korean self-improve loop.");
        println!("        Recommended deployment: F=4 critic-rerank in self_improve_korean.");
    } else {
        println!("=> FAIL — Shape C gate not met.");
        println!("        Either (a) train KoWiki LM longer for better calibration");
        println!("                    (Phase 7 S2 showed sum-AUC climbs with pretrain),");
        println!("        or (b) skip Shape C for Korean and use Shape B at 50M scale.");
    }

    let mut by_sum = samples.clone();
    by_sum.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
    println!("\n=== Top 5 by sum log-prob ===");
    for (p, c, lab, _, ss, _) in by_sum.iter().take(5) {
        let c_short = if c.chars().count() > 60 {
            let s: String = c.chars().take(60).collect();
            format!("{s}...")
        } else {
            c.clone()
        };
        println!(
            "  [{}] {ss:.2}  {p:?} → {c_short:?}",
            if *lab { "ok" } else { "no" }
        );
    }
    println!("\n=== Bottom 5 by sum log-prob ===");
    for (p, c, lab, _, ss, _) in by_sum.iter().rev().take(5) {
        let c_short = if c.chars().count() > 60 {
            let s: String = c.chars().take(60).collect();
            format!("{s}...")
        } else {
            c.clone()
        };
        println!(
            "  [{}] {ss:.2}  {p:?} → {c_short:?}",
            if *lab { "ok" } else { "no" }
        );
    }

    // Selection sweep.
    println!("\n=== Selection sweep (sum log-prob critic) ===");
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
            |s| s.4,
        );
        let delta = critic_pass - random_pass;
        let lift = if random_pass > 1e-6 {
            critic_pass / random_pass
        } else {
            f32::INFINITY
        };
        println!(
            "  {factor:>1}    {random_pass:>9.3}    {critic_pass:>9.3}   {delta:+.3}   {lift:>4.2}×"
        );
    }
    println!();
    println!(
        "Phase 6 RustCode reference: F=4 lift 1.22×, F=16 lift 0.41× (top-tail poison at high F)."
    );

    Ok(())
}

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
