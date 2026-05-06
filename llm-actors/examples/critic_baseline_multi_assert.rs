//! Phase 9 Session 3: harder real-world variant — multi-assert
//! challenges where the model must learn a *function* (not memorize a
//! single input-output pair) for the slot to compile under all
//! cargo asserts simultaneously.
//!
//! Each challenge defines a function with 2-3 asserts:
//!   add(a, b) → a + b   verified by f(2,3)=5, f(10,20)=30, f(0,0)=0
//!   double(x) → x * 2   verified by g(5)=10, g(7)=14, g(0)=0
//!   decrement(n) → n - 1 verified by h(5)=4, h(0)=-1, h(100)=99
//!
//! Why this matters for the Phase 7/8 matrix: previous K9 Rust
//! challenges had ONE assert per program. The model could memorize
//! "for prompt X, slot S works" without learning any abstraction.
//! Multi-assert forces generalization (you can't memorize three
//! input-output pairs in five chars; you must compute).
//!
//! Predicted outcome:
//!   - Pass rate likely lower than K9 (harder task)
//!   - If still in 15-25% sweet spot → Shape C should still work
//!   - If pass rate drops below 10% → calibration may be borderline
//!     (Phase 7 S2 territory)
//!
//! Mirror of `critic_baseline.rs` plumbing; just swaps in
//! MULTI_ASSERT_CHALLENGES via RustCodeDomain's pub `challenges` field.
//!
//! Run:
//!   cargo run -p llm-actors --example critic_baseline_multi_assert --features cuda --release

use std::path::PathBuf;
use std::sync::Arc;

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use clap::Parser;
use llm_actors::{
    domain::{
        rust_code::{RustChallenge, RustCodeDomain},
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
    #[arg(long, default_value_t = 1500)]
    pretrain_steps: usize,
    #[arg(long, default_value_t = 900)]
    pretrain_examples: usize,
    #[arg(long, default_value_t = 30)]
    samples_per_prompt: usize,
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,
    #[arg(long, default_value_t = 10)]
    top_k: usize,
    /// Multi-assert programs need ~10-20 char slots (function body).
    #[arg(long, default_value_t = 20)]
    max_new_tokens: usize,
    #[arg(long, default_value = "checkpoints/critic_baseline_multi.safetensors")]
    seed_ckpt: PathBuf,
    #[arg(long, default_value = "/tmp/workllm-rust-multi-scratch")]
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

/// Each challenge has DISTINCT prompt prefix (Phase 6 lesson) and
/// 2-3 asserts in the suffix that force the slot to be a *function*
/// rather than a memorized constant.
static MULTI_ASSERT_CHALLENGES: &[RustChallenge] = &[
    RustChallenge {
        name: "add",
        prompt: "fn main() { fn f(a: i32, b: i32) -> i32 { ",
        suffix: " } assert_eq!(f(2, 3), 5); assert_eq!(f(10, 20), 30); assert_eq!(f(0, 0), 0); }\n",
    },
    RustChallenge {
        name: "double",
        prompt: "fn main() { fn g(x: i32) -> i32 { ",
        suffix: " } assert_eq!(g(5), 10); assert_eq!(g(7), 14); assert_eq!(g(0), 0); }\n",
    },
    RustChallenge {
        name: "decrement",
        prompt: "fn main() { fn h(n: i32) -> i32 { ",
        suffix: " } assert_eq!(h(5), 4); assert_eq!(h(0), -1); assert_eq!(h(100), 99); }\n",
    },
];

/// Slot completions that satisfy ALL asserts for each challenge. A
/// model that memorizes one input-output won't generalize; it needs
/// to learn the function.
const CORRECT_SLOTS: &[(&str, &[&str])] = &[
    (
        "fn main() { fn f(a: i32, b: i32) -> i32 { ",
        &["a + b", "b + a", "a+b", "(a + b)"],
    ),
    (
        "fn main() { fn g(x: i32) -> i32 { ",
        &["x * 2", "2 * x", "x + x", "x*2"],
    ),
    (
        "fn main() { fn h(n: i32) -> i32 { ",
        &["n - 1", "n-1", "(n - 1)"],
    ),
];

fn synth_pretrain_corpus(n: usize, seed: u64) -> String {
    use rand::seq::SliceRandom;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = String::with_capacity(n * 64);
    for _ in 0..n {
        let (prompt, slots) = CORRECT_SLOTS.choose(&mut rng).expect("non-empty");
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

    let mut rcd = RustCodeDomain::new(&args.scratch_dir);
    rcd.run_program = true;
    rcd.challenges = MULTI_ASSERT_CHALLENGES; // override default single-assert
    rcd.ensure_scratch_project()?;
    let domain = Arc::new(rcd);

    let pretrain_text = synth_pretrain_corpus(args.pretrain_examples, 7);
    let mut seed_chars = String::from(domain.charset());
    seed_chars.push_str(&pretrain_text);
    let tk = Arc::new(Tokenizer::char_from_text(&seed_chars));
    let vocab = tk.vocab_size();
    info!(
        vocab,
        corpus_chars = pretrain_text.len(),
        "tokenizer + corpus"
    );

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
        activation: ActivationKind::SwiGlu,
        weight_tying: false,
        norm_kind: NormKind::RmsNorm,
        norm_position: NormPosition::Pre,
        lora_rank: 32,
        lora_alpha: 64.0,
    };
    info!(params = gpt_cfg.num_params_estimate(), "model config");

    info!("pretraining...");
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

    let mut gen_varmap = VarMap::new();
    let gen_vb = VarBuilder::from_varmap(&gen_varmap, DType::F32, &device);
    let gen_model = GPT::new(gpt_cfg.clone(), gen_vb)?;
    gen_varmap.load(&args.seed_ckpt)?;

    let critic_mean =
        LogitCritic::from_checkpoint(gpt_cfg.clone(), tk.clone(), device.clone(), &args.seed_ckpt)?;
    let critic_sum = LogitCritic::sum_scoring_from_checkpoint(
        gpt_cfg.clone(),
        tk.clone(),
        device.clone(),
        &args.seed_ckpt,
    )?;
    let random_critic = RandomCritic::new(0xC417);
    let always_critic = AlwaysCorrectCritic;

    let prompts: Vec<&str> = MULTI_ASSERT_CHALLENGES.iter().map(|c| c.prompt).collect();
    info!(
        n_prompts = prompts.len(),
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
                (*prompt).to_string(),
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

    println!("\n=== Phase 9 S3 — Multi-assert challenges (harder Rust) ===");
    println!(
        "samples:           {total_attempts} ({total_correct} correct, {} incorrect)",
        total_attempts - total_correct
    );
    println!("LogitCritic mean:  {auc_mean:.3}");
    println!("LogitCritic sum:   {auc_sum:.3}   ★ PRIMARY GATE");
    println!("RandomCritic:      {auc_random:.3}");
    println!("AlwaysCritic:      {auc_always:.3}");
    println!();
    let best = auc_mean.max(auc_sum);
    if best.is_nan() {
        println!("=> undefined");
    } else if best >= 0.6 {
        println!("=> PASS — Shape C survives the multi-assert difficulty bump.");
    } else {
        println!("=> FAIL — multi-assert breaks Shape C at this scale.");
    }

    let mut by_sum = samples.clone();
    by_sum.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
    println!("\n=== Top 5 by sum log-prob ===");
    for (p, c, lab, _, ss, _) in by_sum.iter().take(5) {
        let p_short = if p.chars().count() > 30 {
            format!("{}...", p.chars().take(30).collect::<String>())
        } else {
            p.clone()
        };
        println!(
            "  [{}] {ss:.3}  {p_short:?} → {c:?}",
            if *lab { "ok" } else { "no" }
        );
    }
    println!("\n=== Bottom 5 by sum log-prob ===");
    for (p, c, lab, _, ss, _) in by_sum.iter().rev().take(5) {
        let p_short = if p.chars().count() > 30 {
            format!("{}...", p.chars().take(30).collect::<String>())
        } else {
            p.clone()
        };
        println!(
            "  [{}] {ss:.3}  {p_short:?} → {c:?}",
            if *lab { "ok" } else { "no" }
        );
    }

    println!("\n=== Selection sweep (sum log-prob) ===");
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
    println!("References (Phase 6/7/8):");
    println!("  K9 Rust single-assert: pass 19%, AUC 0.727, F=4 1.22×");
    println!("  Python:                pass 36%, AUC 0.848, F=4 1.00×");
    println!("  Arithmetic:            pass 10%, AUC 0.632, F=4 mild");
    println!("  Korean K8 30K:         pass  2%, AUC 0.363, F=4 0.83× (anti-cal)");
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
