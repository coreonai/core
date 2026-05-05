//! Phase 3 end-to-end smoke: evolutionary architecture search on the
//! arithmetic domain.
//!
//! - Population of N random `GPTConfig`s.
//! - Each variant trained from scratch on a fixed corpus + greedy-eval'd.
//! - Top-2 elites carry over; rest filled by mutation / crossover.
//! - `--n-gpus` > 1 dispatches variants round-robin across CUDA devices in
//!   parallel via `spawn_blocking`.
//!
//! Run:
//!   cargo run -p llm-actors --example evolve_arithmetic --release --features cuda \
//!       -- --population 6 --generations 3 --train-steps 1500 --n-gpus 5

use std::sync::Arc;

use clap::Parser;
use llm_actors::{
    domain::{
        arithmetic::{ArithmeticDomain, SeedMode},
        Domain,
    },
    evolution::{EvolutionConfig, EvolutionRunner, FitnessInputs, SearchSpace},
};
use nanogpt_rs::tokenizer::Tokenizer;
use tracing::info;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_t = 6)]
    population: usize,
    #[arg(long, default_value_t = 3)]
    generations: usize,
    #[arg(long, default_value_t = 2)]
    elite_keep: usize,
    #[arg(long, default_value_t = 1500)]
    train_steps: usize,
    #[arg(long, default_value_t = 128)]
    batch_size: usize,
    #[arg(long, default_value_t = 100)]
    eval_n: usize,
    #[arg(long, default_value_t = 1)]
    n_gpus: usize,
    #[arg(long, default_value_t = 4000)]
    pretrain_examples: usize,
    #[arg(long, default_value_t = 0xE0)]
    seed: u64,
    /// full | nocarry | none
    #[arg(long, default_value = "full")]
    seed_mode: String,
}

fn parse_seed_mode(s: &str) -> anyhow::Result<SeedMode> {
    match s {
        "full" => Ok(SeedMode::Full),
        "nocarry" => Ok(SeedMode::NoCarry),
        "none" => Ok(SeedMode::None),
        other => anyhow::bail!("invalid --seed-mode {other:?}"),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    let args = Args::parse();

    let domain = Arc::new(ArithmeticDomain::default());
    let seed_mode = parse_seed_mode(&args.seed_mode)?;

    // Corpus: pretraining synthetic + (optional) seed pairs concatenated.
    let mut corpus = domain.synth_corpus(args.pretrain_examples, args.seed);
    for (a, b) in domain.enumerate_seed_pairs(seed_mode) {
        corpus.push_str(&domain.render_example(a, b));
    }

    // Tokenizer: deterministic char vocab over the domain charset (so all
    // variants share the same vocab_size and prompt encodings).
    let mut seed_chars = String::from(domain.charset());
    seed_chars.push_str(&corpus);
    let tk = Arc::new(Tokenizer::char_from_text(&seed_chars));
    let vocab = tk.vocab_size();
    info!(
        vocab,
        corpus_chars = corpus.len(),
        "tokenizer + corpus ready"
    );

    let space = SearchSpace::small_char(vocab);

    let inputs = FitnessInputs {
        tokenizer: tk,
        domain: domain.clone() as Arc<dyn Domain>,
        corpus: Arc::new(corpus),
        eval_n: args.eval_n,
        eval_seed: 0xE7A1,
        train_steps: args.train_steps,
        batch_size: args.batch_size,
        stop_char: Some('\n'),
        max_new_tokens: 4,
        min_corpus_chars: 8000,
    };

    let cfg = EvolutionConfig {
        population_size: args.population,
        generations: args.generations,
        elite_keep: args.elite_keep,
        train_steps: args.train_steps,
        batch_size: args.batch_size,
        eval_n: args.eval_n,
        eval_seed: 0xE7A1,
        min_corpus_chars: 8000,
        n_gpus: args.n_gpus,
    };

    let mut runner = EvolutionRunner::new(space, cfg, inputs, args.seed);
    let history = runner.run().await?;

    println!("\n=== generation history ===");
    for r in &history {
        println!(
            "gen {}: best id={:?} fitness={:?}",
            r.generation, r.best_id, r.best_fitness
        );
        for (i, v) in r.variants.iter().enumerate().take(3) {
            println!(
                "  #{}: id={} fit={:?} cfg=L{}H{}/Kv{}E{}B{}F{}x{}exp rope={} act={} tie={} {:?}-{:?} origin={:?}",
                i + 1,
                v.id,
                v.fitness,
                v.config.n_layer,
                v.config.n_head,
                v.config.n_kv_head,
                v.config.n_embd,
                v.config.block_size,
                v.config.ffn_mult,
                v.config.n_experts,
                v.config.use_rope,
                format_args!("{:?}", v.config.activation),
                v.config.weight_tying,
                v.config.norm_kind,
                v.config.norm_position,
                v.origin,
            );
        }
    }

    if let Some(last) = history.last() {
        if let Some(best) = last.variants.first() {
            println!(
                "\nbest overall: id={} fitness={:?}  cfg=L{}H{}/Kv{}E{}B{}F{}x{}exp rope={} act={} tie={} {:?}-{:?}  params={}",
                best.id,
                best.fitness,
                best.config.n_layer,
                best.config.n_head,
                best.config.n_kv_head,
                best.config.n_embd,
                best.config.block_size,
                best.config.ffn_mult,
                best.config.n_experts,
                best.config.use_rope,
                format_args!("{:?}", best.config.activation),
                best.config.weight_tying,
                best.config.norm_kind,
                best.config.norm_position,
                best.config.num_params_estimate(),
            );
        }
    }

    Ok(())
}
