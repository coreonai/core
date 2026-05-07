//! Phase 8 Session 2: Apply the Phase 7 Shape C playbook to
//! `PythonCodeDomain`.
//!
//! Mirror of `critic_baseline.rs` (which targets RustCodeDomain) but
//! uses Python source completion with a `python -c` verifier. This
//! tests whether Phase 6's Shape C result generalizes across
//! different verifier mechanisms (cargo build/run vs Python
//! interpreter) on what is otherwise the same kind of task
//! (slot-fill into a tiny program).
//!
//! Predicted outcome (from Phase 7 design doc):
//!   - Slot completions are roughly length-uniform (3–10 chars), so
//!     mean and sum should give similar AUC.
//!   - The pretrain regime mirrors K9, so we expect harvest pass
//!     rate ≥ 2× chance and AUC ≥ 0.6 — i.e. a clean PASS.
//!
//! If both PASS, Shape C is genuinely cross-language; if Python
//! fails while Rust passes, K9's success was language- or
//! tokenizer-specific.
//!
//! Run:
//!   cargo run -p llm-actors --example critic_baseline_python --features cuda --release

use std::path::PathBuf;
use std::sync::Arc;

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use clap::Parser;
use llm_actors::{
    domain::{python_code::PythonCodeDomain, Domain},
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
    #[arg(long, default_value_t = 16)]
    max_new_tokens: usize,
    #[arg(long, default_value = "checkpoints/critic_baseline_python.safetensors")]
    seed_ckpt: PathBuf,
    /// Phase 10 S2: JEPA aux loss weight on the pretraining stage.
    /// `0.0` = standard CE only (matches Phase 8 S2 setup).
    #[arg(long, default_value_t = 0.0)]
    jepa_lambda: f32,
    /// JEPA future-token offset.
    #[arg(long, default_value_t = 4)]
    jepa_offset: usize,
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
    // equals_5: `def f(): return <slot>` then `assert f() == 5`
    (
        "def f(): return ",
        &[
            "2 + 3",
            "1 + 4",
            "5 + 0",
            "0 + 5",
            "10 - 5",
            "5 * 1",
            "1 * 5",
            "100 // 20",
        ],
    ),
    // equals_14_via_doubling: 2 * (slot) == 14 → slot == 7
    (
        "def f(): return 2 * (",
        &["7", "3 + 4", "4 + 3", "1 + 6", "6 + 1", "10 - 3", "14 // 2"],
    ),
    // len_5_string: s = <slot>; len(s) == 5
    (
        "s = ",
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
    use rand::seq::SliceRandom;
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

    let domain = Arc::new(PythonCodeDomain::new());

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

    // Same arch as critic_baseline.rs (Rust) for direct comparability.
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
    // Phase 10 S2: optional JEPA aux loss during pretraining.
    tcfg.jepa_lambda = args.jepa_lambda;
    tcfg.jepa_offset = args.jepa_offset;
    if args.jepa_lambda > 0.0 {
        info!(
            jepa_lambda = args.jepa_lambda,
            jepa_offset = args.jepa_offset,
            "JEPA aux active during pretraining"
        );
    }
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

    // Phase 7 design doc tier 2: Python slot completions are
    // length-uniform-ish (3–10 chars), so mean ≈ sum is expected.
    // We still measure both for the comparison table.
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

    let prompts: Vec<&str> = CHALLENGES.iter().map(|(p, _)| *p).collect();
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

    println!("\n=== Phase 8 S2 — Shape C on PythonCodeDomain ===");
    println!(
        "samples:           {total_attempts} ({total_correct} correct, {} incorrect)",
        total_attempts - total_correct
    );
    println!("LogitCritic mean:  {auc_mean:.3}");
    println!("LogitCritic sum:   {auc_sum:.3}   ★ PRIMARY GATE (Phase 7)");
    println!("RandomCritic:      {auc_random:.3}");
    println!("AlwaysCritic:      {auc_always:.3}");
    println!();
    println!("Phase 7 acceptance gate: sum-AUC ≥ 0.6");
    let best = auc_mean.max(auc_sum);
    if best.is_nan() {
        println!("=> undefined");
    } else if best >= 0.6 {
        println!("=> PASS — Shape C generalizes from Rust to Python.");
        println!("        Cross-verifier-mechanism transfer confirmed.");
    } else {
        println!("=> FAIL — Shape C did NOT transfer.");
    }

    let mut by_sum = samples.clone();
    by_sum.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
    println!("\n=== Top 5 by sum log-prob ===");
    for (p, c, lab, _, ss, _) in by_sum.iter().take(5) {
        println!(
            "  [{}] {ss:.3}  {p:?} → {c:?}",
            if *lab { "ok" } else { "no" }
        );
    }
    println!("\n=== Bottom 5 by sum log-prob ===");
    for (p, c, lab, _, ss, _) in by_sum.iter().rev().take(5) {
        println!(
            "  [{}] {ss:.3}  {p:?} → {c:?}",
            if *lab { "ok" } else { "no" }
        );
    }

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
    println!("Reference: K9 RustCode F=4 lift 1.22×, F=16 lift 0.41×.");
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
