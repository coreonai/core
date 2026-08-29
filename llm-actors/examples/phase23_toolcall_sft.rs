//! Phase 23 — SFT the tool-call format into the 7B base, then re-gate.
//!
//! The gate (be3b9ac, 060f1c0, 1370192) found the base model produces the
//! call *structure* ~70% few-shot but the exact call 0% of the time, that
//! prompting does not close it, and that the model's own JSON format is two
//! orders of magnitude worse. So the format has to be trained in.
//!
//! ## What is trained, and why not `synth_corpus`
//!
//! `ToolUseArithmeticDomain::synth_corpus` concatenates *full* trajectories:
//!
//! ```text
//! Q: 3+4=
//! (arith add 3 4\u{2192}7)     <- RESOLVED; this is what the model READS BACK
//! A: 7                     after the executor splices a result in
//! ```
//!
//! Training on that teaches the model to emit `=7` itself, and
//! `parse_first_tool_call` skips bodies containing the resolved marker by
//! design, so the loop would never dispatch. The probe already made this
//! exact mistake once and measured 2.8% instead of 72%.
//!
//! So the pairs are built here, two per problem, covering both turns of the
//! agentic loop:
//!
//! ```text
//! turn 1   prompt "Q: a+b=\n"                  -> "(arith add a b)\n"
//! turn 2   prompt "Q: a+b=\n(arith add a b\u{2192}r)\n" -> "A: r\n"
//! ```
//!
//! Completion-only loss (`TrainSftPairs`) — CE on the completion span only.
//! Without it the prompt dominates and the model over-trains on prompt
//! reproduction (CLAUDE.md gotcha #9, which cost Phase 22 four batches).
//!
//! ## Held-out
//!
//! Trains on the `i % 5 != 0` side of the 100-pair grid (80 pairs); the probe
//! evaluates the `i % 5 == 0` side (20 pairs). The rule is written out in
//! both files rather than shared, because it is the one thing a reader has to
//! check before trusting a held-out number.
//!
//! ## `--tool python`
//!
//! Same treatment for `PythonTool`. Few-shot prompting the base model to emit
//! a python call gives 0/12: it derails into filename completion
//! (`(python/sum_of.py`), a repo-context habit, even with control tokens
//! suppressed. So the answer is the same one `arith` needed — train the
//! format in. The tasks (sums of squares, multiples, prime counts) are ones
//! the model cannot answer by guessing, so a correct final answer is evidence
//! the tool ran.
//!
//! Held-out here is by problem size: `PYTHON_EVAL_N` is excluded from
//! training and is exactly what `phase23_python_tool_7b` evaluates.
//!
//! Run:
//!   cargo run -p llm-actors --example phase23_toolcall_sft --features cuda --release -- \
//!       --train-steps 200 --out scratch-7b-sft/p23_fmt_sft.safetensors

use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device};
use clap::Parser;
use llm_actors::{qwen2_lora::LoraConfig, QwenTrainerActor, QwenTrainerMessage};
use pekko_actor::ActorSystem;
use tokio::sync::oneshot;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    model_dir: Option<PathBuf>,
    #[arg(long, default_value = "Qwen2.5-Coder-7B")]
    model_id: String,
    /// Optimizer steps. The corpus is tiny (160 pairs), so this is epochs in
    /// disguise; too many and the model memorises the 80 training pairs.
    #[arg(long, default_value_t = 200)]
    train_steps: usize,
    #[arg(long, default_value_t = 2e-4)]
    lr: f64,
    #[arg(long, default_value_t = 16)]
    lora_rank: usize,
    #[arg(long, default_value_t = 32.0)]
    lora_alpha: f32,
    #[arg(long, default_value_t = 4)]
    batch_size: usize,
    /// Where to write the merged (base + LoRA) safetensors.
    #[arg(long)]
    out: PathBuf,
    /// Train the turn-2 pairs (resolved call -> answer) as well as turn-1.
    #[arg(long, default_value_t = true)]
    both_turns: bool,
    /// Which tool's call format to train.
    #[arg(long, default_value = "arith")]
    tool: String,
}

/// Problem sizes held out of `--tool python` training. Must match the values
/// `phase23_python_tool_7b` evaluates — this is the one thing a reader has to
/// check before trusting the held-out number, so it is written out in both
/// files rather than shared through a helper.
const PYTHON_EVAL_N: [u32; 12] = [37, 53, 71, 96, 23, 41, 67, 88, 17, 29, 43, 61];

/// The three task families, as (question, one-line snippet, answer).
///
/// The snippet must be a single line: the grammar closes a call at the first
/// `)` followed by a newline, so a multi-line body cannot be expressed (see
/// `tools::python_tool`).
fn python_task(family: usize, n: u32) -> (String, String, i64) {
    match family {
        0 => (
            format!("sum of squares from 1 to {n}?"),
            format!("print(sum(i*i for i in range(1,{n}+1)))"),
            (1..=n as i64).map(|i| i * i).sum(),
        ),
        1 => (
            format!("sum of numbers below {n} divisible by 3 or 5?"),
            format!("print(sum(i for i in range(1,{n}) if i%3==0 or i%5==0))"),
            (1..n as i64).filter(|i| i % 3 == 0 || i % 5 == 0).sum(),
        ),
        _ => (
            format!("how many primes below {n}?"),
            format!("print(sum(1 for d in range(2,{n}) if all(d%k for k in range(2,d))))"),
            (2..n as i64)
                .filter(|&d| (2..d).all(|k| d % k != 0))
                .count() as i64,
        ),
    }
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

fn resolve_snapshot(dir: Option<&std::path::Path>, id: &str) -> Result<PathBuf> {
    if let Some(d) = dir {
        return Ok(d.to_path_buf());
    }
    let home = std::env::var("HOME").context("HOME unset")?;
    let snaps = PathBuf::from(format!(
        "{home}/.cache/huggingface/hub/models--Qwen--{id}/snapshots"
    ));
    std::fs::read_dir(&snaps)
        .with_context(|| format!("read_dir {snaps:?}"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.is_dir() && p.join("config.json").exists())
        .ok_or_else(|| anyhow!("no snapshot under {snaps:?}"))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .init();
    let args = Args::parse();
    let device = pick_device();
    let on_cuda = device.is_cuda();
    println!("[Phase23SFT] device = {device:?}, on_cuda = {on_cuda}");
    if !on_cuda && std::env::var("PHASE22_ALLOW_CPU").is_err() {
        anyhow::bail!("Refusing to run on CPU. Rebuild with `--features cuda`.");
    }

    let snapshot = resolve_snapshot(args.model_dir.as_deref(), &args.model_id)?;

    let marker = llm_actors::tools::RESOLVED_MARKER;
    let mut pairs: Vec<(String, String)> = Vec::new();
    let n_problems;

    match args.tool.as_str() {
        "arith" => {
            // Training side of the split: i % 5 != 0 (80 of 100 pairs).
            let all: Vec<(u32, u32)> = (0..=9u32)
                .flat_map(|a| (0..=9u32).map(move |b| (a, b)))
                .collect();
            let train_pairs: Vec<(u32, u32)> = all
                .iter()
                .copied()
                .enumerate()
                .filter(|(i, _)| i % 5 != 0)
                .map(|(_, p)| p)
                .collect();
            n_problems = train_pairs.len();
            for &(a, b) in &train_pairs {
                let r = a + b;
                // turn 1 — the call the parser must dispatch (unresolved)
                pairs.push((format!("Q: {a}+{b}=\n"), format!("(arith add {a} {b})\n")));
                if args.both_turns {
                    // turn 2 — after the executor splices the result back in
                    pairs.push((
                        format!("Q: {a}+{b}=\n(arith add {a} {b}{marker}{r})\n"),
                        format!("A: {r}\n"),
                    ));
                }
            }
            println!(
                "[Phase23SFT] {} pairs from {n_problems} held-in problems (i%5!=0); probe uses the other 20",
                pairs.len()
            );
        }
        "python" => {
            let train_n: Vec<u32> = (10u32..=99)
                .filter(|n| !PYTHON_EVAL_N.contains(n))
                .collect();
            n_problems = train_n.len() * 3;
            for family in 0..3usize {
                for &n in &train_n {
                    let (q, code, r) = python_task(family, n);
                    pairs.push((format!("Q: {q}\n"), format!("(python {code})\n")));
                    if args.both_turns {
                        pairs.push((
                            format!("Q: {q}\n(python {code}{marker}{r})\n"),
                            format!("A: {r}\n"),
                        ));
                    }
                }
            }
            println!(
                "[Phase23SFT] {} pairs from {n_problems} held-in problems (n not in PYTHON_EVAL_N); probe uses the other {}",
                pairs.len(),
                PYTHON_EVAL_N.len()
            );
        }
        other => anyhow::bail!("unknown --tool {other:?} (expected arith or python)"),
    }
    println!("[Phase23SFT] sample: {:?} -> {:?}", pairs[0].0, pairs[0].1);

    let trainer = QwenTrainerActor::from_snapshot_dir(
        &snapshot,
        device.clone(),
        DType::BF16,
        LoraConfig {
            rank: args.lora_rank,
            alpha: args.lora_alpha,
        },
        args.lr,
    )?
    .with_sft_batch_size(args.batch_size)
    .with_fresh_optimizer(true);

    let system = ActorSystem::new("phase23-sft");
    let trainer_ref = system.spawn(trainer, "qwen-trainer").await?;

    println!("[Phase23SFT] training {} steps...", args.train_steps);
    let t0 = std::time::Instant::now();
    let (tx, rx) = oneshot::channel();
    trainer_ref
        .tell(QwenTrainerMessage::TrainSftPairs {
            pairs,
            train_steps: args.train_steps,
            reply: tx,
        })
        .map_err(|e| anyhow!("{e:?}"))?;
    let outcome = rx.await??;
    println!(
        "[Phase23SFT] loss {:.4} -> {:.4} in {:.1}s",
        outcome.initial_loss,
        outcome.final_loss,
        t0.elapsed().as_secs_f64()
    );

    let (tx, rx) = oneshot::channel();
    trainer_ref
        .tell(QwenTrainerMessage::SaveMergedCheckpoint {
            base_path: snapshot.clone(),
            out_path: args.out.clone(),
            reply: tx,
        })
        .map_err(|e| anyhow!("{e:?}"))?;
    rx.await??;
    println!("[Phase23SFT] merged checkpoint -> {}", args.out.display());
    println!("\nRe-gate with:");
    println!(
        "  phase23_toolcall_probe --checkpoint {} --n-problems 20 --passk 10 \\\n    --n-shot 4 --suppress-special",
        args.out.display()
    );
    println!("\nphase23_toolcall_sft: PASS");
    Ok(())
}
