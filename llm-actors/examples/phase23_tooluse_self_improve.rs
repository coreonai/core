//! Phase 23 — tool-use self-improve on the 7B, through the Pekko loop.
//!
//! Everything before this measured a fixed model. This is the loop the
//! project exists for, pointed at the gap Phase 23 identified:
//!
//!   - `phase23_toolcall_sft` taught the call format. The base already had
//!     the format few-shot; what SFT bought was **grounding** — using the
//!     tool's result instead of inventing an answer (3/12 → 12/12 under
//!     `--sabotage`).
//!   - `phase23_python_tool_7b --novel` showed the format generalises to
//!     unseen task families (12/12 calls emitted) while **task-solving does
//!     not** (4/12 correct, identical to the base).
//!
//! So the model can call the tool and will believe it. It cannot write
//! correct code for a problem it has not seen. That is a learning problem
//! with a free verifier, which is exactly what self-improve wants.
//!
//! ## The verifier is free, and that is the point
//!
//! Every previous domain paid for verification — `RustCodeDomain` shells out
//! to `cargo`, `HumanEvalDomain` runs a test harness, and the HBM-style
//! domains have no verifier at all. Here the answer is an integer computed in
//! Rust when the question was generated, and the candidate is executed by the
//! very tool the model is learning to call. Harvest cost is generation only.
//!
//! ## What is trained, and on what
//!
//! `ToolUsePythonDomain` covers eight families disjoint from both the SFT
//! families (already saturated) and the `--novel` transfer set (kept clean).
//! Harvest is **turn 1 only** — prompt → `(python code)` — because that is
//! where the signal is; turn 2 (state the tool's result) is already 12/12 and
//! synthesising it adds nothing. Completions are truncated at the call
//! boundary so harvest, training and eval all see the same string.
//!
//! ## Reward hacking
//!
//! A verifier that compares numbers invites `print(3025)`. The domain rejects
//! a bare `print(<literal>)` outright, and *reports* the weaker "snippet never
//! mentions n" signal rather than filtering on it — that heuristic rejects
//! genuinely correct solutions (`range(1, 38)` for n=37).
//!
//! ## Starting point
//!
//! Round 0 starts from the format-SFT'd checkpoint, not the base. A base
//! model prompted 0-shot emits nothing dispatchable, so the harvest would be
//! empty and the loop could never bootstrap — the cold-start failure this
//! codebase has hit before. `--init-dir` must therefore be a directory
//! holding that checkpoint as `model.safetensors` alongside the snapshot's
//! `config.json` and `tokenizer.json` (symlinks are fine).
//!
//! Run:
//!   CUDA_VISIBLE_DEVICES=0,1 ./phase23_tooluse_self_improve \
//!       --init-dir scratch-7b-sft/p23_py_sft_dir --trainer-gpu 1 --rounds 3

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device};
use clap::Parser;
use llm_actors::{
    curator_actor::SampleMode,
    domain::{
        tool_use_python::{task, ToolUsePythonDomain},
        Domain,
    },
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
    /// Directory holding the STARTING weights as `model.safetensors` plus
    /// `config.json` and `tokenizer.json`. See the module docs on why this is
    /// the format-SFT'd checkpoint and not the base snapshot.
    #[arg(long)]
    init_dir: PathBuf,
    #[arg(long, default_value_t = 3)]
    rounds: usize,
    /// Prompts harvested per round.
    #[arg(long, default_value_t = 48)]
    gen_n: usize,
    /// Samples per prompt. Phase 22 measured saturation at 16 for transfer
    /// and continued in-domain gains to 32, at 70+ min/step for 64 prompts.
    /// The default here is lower because a call line is ~25 tokens against
    /// HumanEval's ~192, and because this is the first run of the mechanism.
    #[arg(long, default_value_t = 8)]
    samples_per_prompt: usize,
    #[arg(long, default_value_t = 64)]
    eval_n: usize,
    #[arg(long, default_value_t = 1)]
    eval_passk: usize,
    #[arg(long, default_value_t = 64)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,
    #[arg(long, default_value_t = 40)]
    top_k: usize,
    #[arg(long, default_value_t = 60)]
    train_steps: usize,
    #[arg(long, default_value_t = 1e-4)]
    lr: f64,
    #[arg(long, default_value_t = 16)]
    lora_rank: usize,
    #[arg(long, default_value_t = 32.0)]
    lora_alpha: f32,
    #[arg(long, default_value_t = 4)]
    batch_size: usize,
    /// Families to include, comma-separated. The eight are not equally hard
    /// — see `ToolUsePythonDomain::with_families`. Default is all of them.
    #[arg(long, value_delimiter = ',')]
    families: Vec<usize>,
    /// Measure the per-family pass rate of the starting checkpoint and stop.
    /// One sample per prompt, at the loop's eval temperature. This is the
    /// measurement that says where the headroom is, and `gen_n` does not
    /// bound it: systematic harvest sweeps the whole pool, so pool size is
    /// set by `--families` and the `n` range, not by a sample count.
    #[arg(long)]
    baseline: bool,
    /// Samples per prompt in `--baseline` mode. `1` gives pass@1. Higher
    /// gives pass@k, which is the number that decides whether the loop can
    /// bootstrap at all: a family at pass@1 = 0 has nothing to harvest unless
    /// *some* sample succeeds.
    #[arg(long, default_value_t = 1)]
    baseline_k: usize,
    /// Harvest with self-repair: when a sampled call fails, hand the tool's
    /// error back and keep the retry if it verifies. Required here — the
    /// target families are 0/12 at pass@16, so a turn-1-only harvest is
    /// empty and the loop cannot start.
    #[arg(long)]
    harvest_repair: bool,
    /// After a failed call, splice the tool's error back in and give the
    /// model another turn — then report how often that second turn fixes it.
    ///
    /// This is the measurement that decides whether the loop can bootstrap
    /// without injected knowledge. Sampling cannot discover the import
    /// (0/576 snippets contained one), so the only source of the missing
    /// information is the tool's own error message. If self-repair works,
    /// the repaired trajectories are harvestable and the behaviour can be
    /// trained in; if it does not, nothing short of telling the model the
    /// contract will start the loop.
    #[arg(long)]
    repair: bool,
    /// Print this many failing (prompt, snippet, verdict) triples in
    /// `--baseline` mode. A family at exactly 0/32 fails systematically, and
    /// whether the loop can bootstrap on it depends on *how*: a near-miss
    /// that sampling will occasionally get right, or one wrong idea the model
    /// holds every time.
    #[arg(long, default_value_t = 0)]
    show_failures: usize,
    /// Smallest `n` in the task pool, and the largest.
    #[arg(long, default_value_t = 12)]
    n_lo: u32,
    #[arg(long, default_value_t = 60)]
    n_hi: u32,
    /// Put the trainer on its own GPU. Inference runs F32 (28 GB at 7B, see
    /// CLAUDE.md gotcha #11) so it will not share a 40 GB card with training.
    #[arg(long)]
    trainer_gpu: Option<usize>,
    #[arg(long, default_value = "scratch-7b-sft/p23_si")]
    out_dir: PathBuf,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value = "Qwen2.5-Coder-7B")]
    model_id: String,
}

/// Per-family pass rate of the starting checkpoint.
///
/// Deliberately not routed through `EvaluatorActor`: it reports one aggregate
/// number, and an aggregate is exactly what hides a mix of saturated and
/// unsolved families. Same generate → truncate → verify path as the loop, so
/// the numbers are comparable to the round-0 eval.
async fn run_baseline(
    domain: ToolUsePythonDomain,
    args: &Args,
    device: Device,
    dtype: DType,
    tk: Arc<NgptTokenizer>,
) -> Result<()> {
    use llm_actors::ModelMessage;
    use tokio::sync::oneshot;

    let model = QwenModelActor::from_snapshot_dir(&args.init_dir, device, dtype)?;
    let system = ActorSystem::new("phase23-si-baseline");
    let model_ref = system.spawn(model, "qwen-model").await?;

    let n = domain.n_tasks();
    println!(
        "[Phase23SI] baseline over {n} prompts x {} samples\n",
        args.baseline_k.max(1)
    );
    let mut per_family: std::collections::BTreeMap<usize, (usize, usize)> = Default::default();
    let mut no_call = 0usize;
    let mut shown = 0usize;
    let mut shown_ok = 0usize;
    // Whether the model ever writes the import its tool actually needs.
    let mut with_import = 0usize;
    let mut attempts = 0usize;
    let (mut repair_attempts, mut repair_ok, mut repair_import) = (0usize, 0usize, 0usize);
    let mut answer_before_call = 0usize;
    let mut shown_repair = 0usize;
    let mut looks_hardcoded = 0usize;

    let k = args.baseline_k.max(1);
    for i in 0..n {
        let t = domain.task_at(i).expect("index in range").clone();
        let prompt = domain.nth_prompt(i).expect("index in range");
        let ids = tk.encode(&prompt)?;
        let mut any_correct = false;
        for s in 0..k {
            let (tx, rx) = oneshot::channel();
            model_ref
                .tell(ModelMessage::GenerateTokens {
                    prompt_ids: ids.clone(),
                    cfg: GenerateConfig {
                        max_new_tokens: args.max_new_tokens,
                        temperature: 0.8,
                        top_k: Some(40),
                        top_p: Some(0.95),
                        // Distinct per (prompt, sample) or every draw of a
                        // prompt repeats the same completion and pass@k
                        // collapses to pass@1.
                        seed: Some(
                            args.seed
                                .wrapping_add((i as u64) << 8)
                                .wrapping_add(s as u64),
                        ),
                    },
                    reply: tx,
                })
                .map_err(|e| anyhow!("{e:?}"))?;
            let tokens = tokio::time::timeout(std::time::Duration::from_secs(180), rx).await???;
            let full = tk.decode(&tokens)?;
            // Measure contamination on the RAW text. `truncate_completion`
            // now strips anything before the call, so checking the truncated
            // string would report 0 by construction and blind the detector to
            // the very artifact it was added to catch.
            let raw = full[prompt.len().min(full.len())..].to_string();
            let completion = domain.truncate_completion(&raw);
            let verdict = domain.verify(&prompt, &completion);
            let code = ToolUsePythonDomain::snippet_of(&completion);
            match &code {
                None => no_call += 1,
                Some(c) => {
                    if c.contains("import ") {
                        with_import += 1;
                    }
                    if verdict.is_correct() && ToolUsePythonDomain::looks_hardcoded(c, t.n) {
                        looks_hardcoded += 1;
                    }
                }
            }
            attempts += 1;
            // Text before the call. The harvested repair completions carried
            // an `A: <guess>` line ahead of the fixed call — it made sense in
            // the two-turn context it came from, but paired with the original
            // prompt it trains the model to state an answer BEFORE computing
            // one. The snippet still verifies, so nothing else here catches
            // it.
            if !raw.trim_start().starts_with('(') && ToolUsePythonDomain::snippet_of(&raw).is_some()
            {
                answer_before_call += 1;
            }
            // Second turn: the model sees its own call with the error where
            // the result would be. Same splice the agentic loop performs.
            if !verdict.is_correct() && args.repair {
                if let Some(bad) = &code {
                    let err = match domain.verify(&prompt, &completion) {
                        llm_actors::types::Verdict::Incorrect { reason } => reason,
                        _ => String::new(),
                    };
                    let marker = llm_actors::tools::RESOLVED_MARKER;
                    let turn2 = format!("{prompt}(python {bad}{marker}ERR:{err})\n");
                    let ids2 = tk.encode(&turn2)?;
                    let (tx2, rx2) = oneshot::channel();
                    model_ref
                        .tell(ModelMessage::GenerateTokens {
                            prompt_ids: ids2,
                            cfg: GenerateConfig {
                                max_new_tokens: args.max_new_tokens,
                                temperature: 0.8,
                                top_k: Some(40),
                                top_p: Some(0.95),
                                seed: Some(args.seed.wrapping_add(0x5eed).wrapping_add(s as u64)),
                            },
                            reply: tx2,
                        })
                        .map_err(|e| anyhow!("{e:?}"))?;
                    let t2 =
                        tokio::time::timeout(std::time::Duration::from_secs(180), rx2).await???;
                    let full2 = tk.decode(&t2)?;
                    let comp2 = domain.truncate_completion(&full2[turn2.len().min(full2.len())..]);
                    repair_attempts += 1;
                    if let Some(c2) = ToolUsePythonDomain::snippet_of(&comp2) {
                        if c2.contains("import ") {
                            repair_import += 1;
                        }
                        // Verify the REPAIRED call against the original
                        // prompt: the second turn's job is to produce a
                        // snippet that answers the question.
                        let fixed = format!("(python {c2})\n");
                        if domain.verify(&prompt, &fixed).is_correct() {
                            repair_ok += 1;
                            if shown_repair < args.show_failures {
                                println!("  [REPAIRED f{} n={}] {}", t.family, t.n, c2);
                                shown_repair += 1;
                            }
                        } else if shown_repair < args.show_failures {
                            println!("  [repair failed f{} n={}] {}", t.family, t.n, c2);
                            shown_repair += 1;
                        }
                    }
                }
            }
            if verdict.is_correct() {
                any_correct = true;
                if shown_ok < args.show_failures {
                    println!(
                        "  [SOLVED f{} n={} sample {s}] {}",
                        t.family,
                        t.n,
                        completion.trim()
                    );
                    shown_ok += 1;
                }
            } else if shown < args.show_failures {
                println!(
                    "  [f{} n={}] want {}\n      {}\n      {:?}",
                    t.family,
                    t.n,
                    t.answer,
                    completion.trim(),
                    verdict
                );
                shown += 1;
            }
        }
        let e = per_family.entry(t.family).or_insert((0, 0));
        e.1 += 1;
        if any_correct {
            e.0 += 1;
        }
    }

    println!("\n[Phase23SI] === per-family baseline (pass@{k}) ===");
    let (mut tc, mut tt) = (0usize, 0usize);
    for (f, (c, t)) in &per_family {
        println!(
            "  family {f}  {c:3}/{t:3}  {:.3}   {}",
            *c as f32 / *t as f32,
            task(*f, 42).question
        );
        tc += c;
        tt += t;
    }
    println!("  ---");
    println!("  overall   {tc:3}/{tt:3}  {:.3}", tc as f32 / tt as f32);
    println!("  attempts: {attempts}, no dispatchable call: {no_call}");
    println!("  snippets containing an import: {with_import}/{attempts}");
    println!("  text before the call (answer stated first): {answer_before_call}/{attempts}");
    if args.repair {
        println!(
            "  self-repair after the error: {repair_ok}/{repair_attempts} fixed, \
             {repair_import}/{repair_attempts} wrote an import"
        );
    }
    println!("  verified but never mentions n (weak hardcode signal): {looks_hardcoded}/{tc}");
    Ok(())
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

fn resolve_snapshot(id: &str) -> Result<PathBuf> {
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
        .with_max_level(tracing::Level::INFO)
        .init();
    let args = Args::parse();
    let device = pick_device();
    let on_cuda = device.is_cuda();
    println!("[Phase23SI] device = {device:?}, rounds = {}", args.rounds);
    if !on_cuda && std::env::var("PHASE22_ALLOW_CPU").is_err() {
        anyhow::bail!(
            "Refusing to run on CPU. Rebuild with `--features cuda` \
             (see CLAUDE.md gotcha #8). PHASE22_ALLOW_CPU=1 overrides."
        );
    }
    std::fs::create_dir_all(&args.out_dir)?;

    // The init dir must be self-contained: both the inference actor and the
    // trainer load from it, and SaveMergedCheckpoint folds the LoRA delta
    // back onto it. A missing tokenizer here surfaces much later as a
    // confusing decode error, so check up front.
    for f in ["config.json", "tokenizer.json"] {
        if !args.init_dir.join(f).exists() {
            anyhow::bail!(
                "--init-dir {:?} has no {f}. It must hold the starting weights \
                 as model.safetensors plus the snapshot's config.json and \
                 tokenizer.json (symlinks are fine).",
                args.init_dir
            );
        }
    }
    let snapshot = resolve_snapshot(&args.model_id)?;
    println!("[Phase23SI] init  = {}", args.init_dir.display());
    println!("[Phase23SI] snap  = {}", snapshot.display());

    // F32 inference: this domain generates code, and F16 corrupts dense
    // snippets into `(p):` even at training loss 1e-4 (CLAUDE.md gotcha #11).
    // BF16 training — the 7B recipe from Phase 22 C3.
    let (inference_dtype, train_dtype) = if on_cuda {
        (DType::F32, DType::BF16)
    } else {
        (DType::F32, DType::F32)
    };

    let tk = Arc::new(NgptTokenizer::from_hf_file(
        args.init_dir.join("tokenizer.json"),
    )?);

    let families: Vec<usize> = if args.families.is_empty() {
        (0..llm_actors::domain::tool_use_python::N_FAMILIES).collect()
    } else {
        args.families.clone()
    };
    let tud = ToolUsePythonDomain::with_families(args.n_lo, args.n_hi, &families);
    println!(
        "[Phase23SI] ToolUsePythonDomain: {} tasks (8 families x n={}..={})",
        tud.n_tasks(),
        args.n_lo,
        args.n_hi
    );
    if args.baseline {
        return run_baseline(tud, &args, device, inference_dtype, tk).await;
    }
    let domain: Arc<dyn Domain> = Arc::new(tud);

    let trainer_device = match args.trainer_gpu {
        Some(idx) if on_cuda => {
            let d = Device::new_cuda(idx)
                .with_context(|| format!("--trainer-gpu {idx}: CUDA device unavailable"))?;
            println!("[Phase23SI] trainer on cuda:{idx}, inference on cuda:0");
            d
        }
        _ => device.clone(),
    };

    let qwen_model =
        QwenModelActor::from_snapshot_dir(&args.init_dir, device.clone(), inference_dtype)?;
    let qwen_trainer = QwenTrainerActor::from_snapshot_dir(
        &args.init_dir,
        trainer_device,
        train_dtype,
        LoraConfig {
            rank: args.lora_rank,
            alpha: args.lora_alpha,
        },
        args.lr,
    )?
    .with_sft_batch_size(args.batch_size)
    .with_fresh_optimizer(true);

    let system = ActorSystem::new("phase23-si");
    let model_ref = system.spawn(qwen_model, "qwen-model").await?;
    let trainer_ref = system.spawn(qwen_trainer, "qwen-trainer").await?;
    let generator_ref = system
        .spawn(
            GeneratorActor::<QwenModelActor>::new(
                model_ref.clone(),
                tk.clone(),
                domain.clone(),
                None,
                "qwen".to_string(),
            )
            .with_repair_failures(args.harvest_repair),
            "generator",
        )
        .await?;
    let verifier_ref = system
        .spawn(VerifierActor::new(domain.clone()), "verifier")
        .await?;
    let curator_ref = system.spawn(CuratorActor::new(2048), "curator").await?;
    let evaluator_ref = system
        .spawn(
            EvaluatorActor::<QwenModelActor>::new(
                model_ref.clone(),
                tk.clone(),
                domain.clone(),
                None,
            ),
            "evaluator",
        )
        .await?;
    // The merge base is the init dir, not the pristine snapshot: the LoRA
    // delta sits on top of the format-SFT'd weights, and merging it onto the
    // base instead would silently discard that SFT.
    let trainer_handle = Arc::new(QwenTrainerActorHandle::new(
        trainer_ref,
        args.train_steps,
        args.init_dir.clone(),
    )) as Arc<dyn TrainerHandle>;

    let actors = RoundActors::<QwenModelActor> {
        model: model_ref,
        generator: generator_ref,
        verifier: verifier_ref,
        curator: curator_ref,
        trainer: trainer_handle,
        evaluator: evaluator_ref,
    };
    println!("[Phase23SI] 6 actors spawned\n");

    let mut train_cfg = TrainConfig::smoke();
    train_cfg.max_steps = args.train_steps;
    train_cfg.optimizer = OptimizerKind::Adam;

    let gen_seed = args.seed;
    let eval_seed = args.seed.wrapping_sub(35);
    let base = RoundConfig {
        round: 0,
        gen_n: args.gen_n,
        gen_seed,
        gen_sampling: GenerateConfig {
            max_new_tokens: args.max_new_tokens,
            temperature: args.temperature,
            top_k: (args.top_k > 0).then_some(args.top_k),
            top_p: Some(0.95),
            seed: Some(gen_seed),
        },
        eval_n: args.eval_n,
        eval_seed,
        eval_sampling: GenerateConfig {
            max_new_tokens: args.max_new_tokens,
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
        corpus_seed: Some(args.seed.wrapping_sub(42)),
        anchor: None,
        freeze_base: false,
        gen_oversample: 1,
        dpo_beta: None,
        dpo_reference_path: None,
        dpo_max_pairs_per_prompt: 0,
        dpo_sft_anchor_weight: 0.0,
        eval_passk: args.eval_passk,
        sft_mask_prompt: true,
        samples_per_prompt: Some(args.samples_per_prompt),
    };

    // `run_multi_round` derives per-round save paths from the r0 template
    // and auto-chains `init_from` to the previous round's checkpoint.
    let reports = run_multi_round(
        &actors,
        MultiRoundConfig::new(args.rounds, base),
        |r, rep| {
            // `None` is "eval-after never ran" (empty-corpus early return),
            // which is not the same as a measured zero — conflating them
            // prints a collapse that did not happen.
            let fmt = |c: Option<usize>| match c {
                Some(n) => format!("{:.3}", n as f32 / rep.eval_total.max(1) as f32),
                None => "skipped".to_string(),
            };
            println!(
                "[Phase23SI] round {r}: harvested {}/{} | pass {} -> {}",
                rep.correct,
                rep.generated,
                fmt(rep.eval_correct_before),
                fmt(rep.eval_correct_after),
            );
        },
    )
    .await?;

    println!("\n[Phase23SI] === tool-use self-improve ===");
    for (r, rep) in reports.iter().enumerate() {
        let total = rep.eval_total.max(1) as f32;
        let pct = |c: Option<usize>| match c {
            Some(n) => format!("{:.3}", n as f32 / total),
            None => "skipped".to_string(),
        };
        println!(
            "  round {r}: harvest {:4}/{:4}  pass@{} {} -> {}",
            rep.correct,
            rep.generated,
            args.eval_passk,
            pct(rep.eval_correct_before),
            pct(rep.eval_correct_after),
        );
    }
    println!("\nphase23_tooluse_self_improve: PASS");
    Ok(())
}
