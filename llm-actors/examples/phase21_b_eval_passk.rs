//! Phase 21 Stage B — clean pass@k comparison on a FIXED checkpoint.
//!
//! Loads a `self_improve_rust`-trained checkpoint and runs the standard
//! RustCodeDomain eval with multiple `passk` values, reporting all on
//! the SAME model. Avoids the independent-pretrain contamination that
//! comparing two separate `self_improve_rust` runs would introduce.
//!
//! Run:
//!   cargo run -p llm-actors --example phase21_b_eval_passk --features cuda --release -- \
//!       --ckpt checkpoints/rust_round_b_passk1.r1.safetensors --eval-n 24
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use candle_core::Device;
use clap::Parser;
use llm_actors::{
    domain::{rust_code::RustCodeDomain, Domain},
    EvaluatorActor, EvaluatorMessage, ModelActor,
};
use nanogpt_rs::{config::GPTConfig, generate::GenerateConfig, tokenizer::Tokenizer};
use pekko_actor::ActorSystem;
use tokio::sync::oneshot;

#[derive(Parser, Debug)]
struct Args {
    /// Path to the trained safetensors checkpoint. Must match the model
    /// config below (defaults reproduce `phase21_b/run_smoke.sh` scale-up).
    #[arg(long)]
    ckpt: PathBuf,
    #[arg(long, default_value_t = 24)]
    eval_n: usize,
    #[arg(long, default_value_t = 6)]
    n_layer: usize,
    #[arg(long, default_value_t = 8)]
    n_head: usize,
    #[arg(long, default_value_t = 4)]
    n_kv_head: usize,
    #[arg(long, default_value_t = 512)]
    n_embd: usize,
    #[arg(long, default_value_t = 16)]
    max_new_tokens: usize,
    /// pass@k values to compare. Each is a separate eval run on the
    /// same model. Eval uses temp=0 (greedy) when passk=1, temp=0.8 +
    /// top-k=10 when passk>1.
    #[arg(long, num_args=1.., default_values_t = vec![1usize, 3, 5, 10])]
    passks: Vec<usize>,
    #[arg(long, default_value = "/tmp/workllm-rust-scratch-phase21b-eval")]
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

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    let args = Args::parse();
    let device = pick_device();
    println!("[Phase21B] device = {:?}", device);
    println!("[Phase21B] ckpt = {}", args.ckpt.display());

    // RustCodeDomain mirrors the self_improve_rust setup.
    // ensure_scratch_project lays down Cargo.toml + src/main.rs skeleton —
    // without it `cargo run` would fail at verify-time and every verdict
    // would be Inconclusive (== Incorrect for pass-rate purposes).
    let domain_concrete = RustCodeDomain::new(&args.scratch_dir);
    domain_concrete.ensure_scratch_project()?;
    let domain: Arc<dyn Domain> = Arc::new(domain_concrete);

    // Build the char tokenizer the same way self_improve_rust does — by
    // dumping the DEFAULT_CHALLENGES prompts + suffixes + a small synth
    // pretrain corpus so the char vocab matches.
    let pretrain_text = synth_pretrain_corpus(600, 7);
    let mut seed_chars = String::from(domain.charset());
    seed_chars.push_str(&pretrain_text);
    let tk = Arc::new(Tokenizer::char_from_text(&seed_chars));
    let vocab = tk.vocab_size();

    // Must mirror `self_improve_rust.rs::gpt_cfg` exactly so the
    // safetensors load and forward pass produce the same numerics
    // the training run did. block_size=80 is load-bearing — char-level
    // prompts can hit ~31 chars + ~16 generated, so 32 truncates context.
    let gpt_cfg = GPTConfig {
        vocab_size: vocab,
        block_size: 80,
        n_layer: args.n_layer,
        n_head: args.n_head,
        n_embd: args.n_embd,
        dropout: 0.0,
        bias: false,
        ffn_mult: 4,
        use_rope: true,
        rope_base: 10_000.0,
        n_kv_head: args.n_kv_head,
        n_experts: 1,
        moe_top_k: 0,
        moe_aux_weight: 0.0,
        activation: nanogpt_rs::config::ActivationKind::SwiGlu,
        weight_tying: false,
        norm_kind: nanogpt_rs::config::NormKind::RmsNorm,
        norm_position: nanogpt_rs::config::NormPosition::Pre,
        lora_rank: 0,
        lora_alpha: 16.0,
    };
    let model_actor =
        ModelActor::from_checkpoint(gpt_cfg.clone(), device.clone(), tk.clone(), &args.ckpt)?;
    let system = ActorSystem::new("phase21-b-eval");
    let model_ref = system.spawn(model_actor, "model").await?;

    let evaluator = EvaluatorActor::new(model_ref.clone(), tk.clone(), domain.clone(), Some('\n'));
    let evaluator_ref = system.spawn(evaluator, "evaluator").await?;

    println!("\n[Phase21B] eval_n = {}", args.eval_n);
    println!("[Phase21B] pass@k results (same checkpoint, same seed_set):");
    println!("  passk |  pass-rate  | correct/total | eval_sampling");
    let mut summary: Vec<(usize, f32)> = Vec::new();
    for &passk in &args.passks {
        let (temp, topk) = if passk > 1 { (0.8f64, 10) } else { (0.0f64, 1) };
        let sampling = GenerateConfig {
            max_new_tokens: args.max_new_tokens,
            temperature: temp,
            top_k: Some(topk),
            top_p: None,
            seed: Some(0xE5A2),
        };
        let (tx, rx) = oneshot::channel();
        evaluator_ref
            .tell(EvaluatorMessage::Eval {
                n: args.eval_n,
                seed: 0xE5A2,
                sampling,
                passk,
                reply: tx,
            })
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let report = rx.await??;
        let rate = report.pass_rate();
        summary.push((passk, rate));
        println!(
            "  {:>5} |   {:.4}   |    {}/{}     | temp={} topk={}",
            passk, rate, report.correct, report.total, temp, topk
        );
        for (i, s) in report.samples.iter().take(3).enumerate() {
            println!(
                "    sample {i} prompt={:?} completion={:?}",
                s.prompt, s.completion
            );
        }
    }

    println!("\n[Phase21B] pass@k lift table (vs pass@1 baseline):");
    let baseline = summary
        .iter()
        .find(|(k, _)| *k == 1)
        .map(|(_, r)| *r)
        .unwrap_or(0.0);
    for (k, r) in &summary {
        let delta = r - baseline;
        let ratio = if baseline > 0.0 { r / baseline } else { 0.0 };
        println!(
            "  pass@{:<3} = {:.4}   Δ vs pass@1 = {:+.4}   ratio = {:.2}×",
            k, r, delta, ratio
        );
    }
    println!("\nphase21_b_eval_passk: PASS");
    Ok(())
}

/// Mirror of `self_improve_rust.rs::CHALLENGES` — kept in sync by hand
/// so this binary builds the same char vocab as the trained model used.
/// If the upstream list changes, update here too (vocab mismatch will
/// surface as a load-time shape error).
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
    (
        "fn main() { let x: i32 = ",
        &["10", "5 + 5", "2 * 5", "20 / 2", "12 - 2", "1 + 9", "3 + 7"],
    ),
    (
        "fn main() { let z: i32 = ",
        &["0", "1 - 1", "5 - 5", "10 - 10", "2 * 0", "0 * 7"],
    ),
    (
        "fn main() { let b: bool = ",
        &["true", "1 == 1", "2 > 1", "!false", "1 != 2"],
    ),
    (
        "fn main() { let f: bool = ",
        &["false", "1 == 2", "2 < 1", "!true", "1 != 1"],
    ),
    (
        "fn main() { let t: &str = ",
        &[r#""abc""#, r#""123""#, r#""hey""#, r#""xyz""#, r#""foo""#],
    ),
    (
        "fn main() { let xs: [i32; 3] = ",
        &[
            "[1, 2, 3]",
            "[2, 2, 2]",
            "[0, 3, 3]",
            "[3, 0, 3]",
            "[1, 1, 4]",
        ],
    ),
    (
        "fn main() { let o: Option<i32> = ",
        &["Some(5)", "Some(2 + 3)", "Some(10 - 5)"],
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
