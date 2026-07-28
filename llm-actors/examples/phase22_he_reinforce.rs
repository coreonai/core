//! Phase 22 Stage E — REINFORCE on HumanEval through Pekko.
//!
//! Port of Phase 21 Stage G (`phase21_g_smoke`) from the trivial
//! `PythonReturnDomain` to the real `HumanEvalDomain`. The structure
//! is identical: sample k completions per prompt at temp=0.8, verify
//! via Domain, compute RLOO baseline-subtracted rewards, send a
//! `TrainPolicyGradient { samples }` to `QwenTrainerActor`.
//!
//! What's different from Stage D's `phase22_he_mr_sft`:
//!   - **Loss**: REINFORCE policy gradient (reward × log-prob) instead
//!     of cross-entropy SFT on chosen completions.
//!   - **Reward source**: same `HumanEvalDomain::verify` (1.0 if test
//!     passes, 0.0 otherwise) — *verifier-as-reward*, the cleanest
//!     extrinsic signal a coding model can train against.
//!   - **No multi-round supervisor**. Stage G's RL loop runs directly:
//!     each RL step is its own `(gen × k → verify → reward → policy
//!     gradient step)` cycle, no separate train/eval phases.
//!
//! Memory: k_per_prompt=2 + max_new=16 fit Qwen2.5-Coder-0.5B F32
//! gradient graph in 40 GB. Larger k or longer max_new OOM in Phase
//! 21 G's exploration; keep this conservative.
//!
//! Smoke recipe (~6 min on A100):
//!   cargo run -p llm-actors --example phase22_he_reinforce \
//!       --features cuda --release -- --rl-steps 3 --n-prompts 6 \
//!       --k-per-prompt 2 --max-new-tokens 16
//!
//! Larger smoke (15 min, real-ish max_new):
//!   cargo run -p llm-actors --example phase22_he_reinforce \
//!       --features cuda --release -- --rl-steps 5 --n-prompts 8 \
//!       --k-per-prompt 2 --max-new-tokens 80
//!
//! Signal-bearing run (max_new=64, k=4, micro-batch to avoid OOM):
//!   cargo run -p llm-actors --example phase22_he_reinforce \
//!       --features cuda --release -- --rl-steps 20 --n-prompts 16 \
//!       --k-per-prompt 4 --max-new-tokens 64 --pg-micro-batch-size 4

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device};
use clap::Parser;
use llm_actors::{
    domain::{human_eval::HumanEvalDomain, Domain},
    qwen2_lora::LoraConfig,
    ModelMessage, QwenModelActor, QwenTrainerActor, QwenTrainerMessage,
};
use nanogpt_rs::{generate::GenerateConfig, Tokenizer as NgptTokenizer};
use pekko_actor::ActorSystem;
use tokio::sync::oneshot;

#[derive(Parser, Debug)]
struct Args {
    /// Qwen model snapshot directory (config.json + tokenizer.json +
    /// model.safetensors or its shard index). Overrides --model-id.
    /// Point at a 7B snapshot to run 7B.
    #[arg(long)]
    model_dir: Option<PathBuf>,
    /// HF repo suffix under `models--Qwen--<id>` when --model-dir absent.
    #[arg(long, default_value = "Qwen2.5-Coder-0.5B")]
    model_id: String,
    /// Run BF16 throughout (inference + policy-gradient training) instead
    /// of the 0.5B default (F16 inference / F32 train). Required to fit 7B
    /// on 40GB. Ignored on CPU.
    #[arg(long, default_value_t = false)]
    train_bf16: bool,
    /// Put the QwenTrainerActor on a SEPARATE CUDA device (index among
    /// visible GPUs) from the sampling model. For 7B both are ~15GB; the
    /// policy-gradient graph needs a full card's headroom. Run e.g.
    /// `CUDA_VISIBLE_DEVICES=0,1 ... --trainer-gpu 1`.
    #[arg(long)]
    trainer_gpu: Option<usize>,
    /// Start index into the HumanEval prompt series. Prompts used are
    /// `nth_prompt(prompt_offset .. prompt_offset+n_prompts)`. For the hard
    /// tail: `--prompt-offset 100 --n-prompts 64` (idx 100..164). Default 0.
    #[arg(long, default_value_t = 0)]
    prompt_offset: usize,
    /// Number of RL outer steps (each = sample + verify + policy gradient).
    #[arg(long, default_value_t = 3)]
    rl_steps: usize,
    /// Number of HumanEval problems sampled per RL step.
    #[arg(long, default_value_t = 6)]
    n_prompts: usize,
    /// Samples per prompt (used for RLOO baseline subtraction).
    #[arg(long, default_value_t = 2)]
    k_per_prompt: usize,
    /// max tokens per completion. Stage G's OOM ceiling at k=2 on
    /// 0.5B F32: ~24 before the gradient graph blows up. We default
    /// conservatively to 16; bump if you've shrunk k or moved to FP16
    /// gradient accumulation.
    #[arg(long, default_value_t = 16)]
    max_new_tokens: usize,
    /// HumanEval JSONL path.
    #[arg(long)]
    jsonl: Option<PathBuf>,
    /// Generation sampling temperature.
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,
    /// AdamW learning rate.
    #[arg(long, default_value_t = 2e-4)]
    lr: f64,
    /// LoRA rank.
    #[arg(long, default_value_t = 16)]
    lora_rank: usize,
    /// LoRA alpha.
    #[arg(long, default_value_t = 32.0)]
    lora_alpha: f32,
    /// Base seed for prompt-pick + per-sample seed derivation.
    #[arg(long, default_value_t = 7)]
    seed: u64,
    /// Phase 22 follow-up C2 — adapter-sync cadence. Every `N` RL
    /// steps, the trainer's accumulated LoRA delta is baked into base
    /// (`SaveMergedCheckpoint`) and the sampling model reloads
    /// (`ModelMessage::ReloadCheckpoint`). This closes the off-policy
    /// gap: without sync, `QwenModelActor` keeps sampling from the
    /// frozen base while `QwenTrainerActor`'s LoRA drifts, so the
    /// policy gradient updates a distribution that's never used for
    /// sampling. `0` disables sync (the historical Stage G behavior);
    /// `1` syncs every step (most on-policy); `N>1` trades sync
    /// frequency against the ~30s save+reload cost per sync.
    #[arg(long, default_value_t = 0)]
    sync_every: usize,
    /// Directory for the intermediate merged-checkpoint file used by
    /// adapter sync. Default writes to `/dev/shm/...` (tmpfs) to cut
    /// the 988-MB-per-sync disk I/O.
    #[arg(long, default_value = "/dev/shm/phase22_he_reinforce_sync.safetensors")]
    sync_path: PathBuf,
    /// Base safetensors path for the merged-checkpoint save. Defaults
    /// to the HF snapshot's `model.safetensors`.
    #[arg(long)]
    base_safetensors: Option<PathBuf>,
    /// If set, save a merged (base + LoRA baked in) safetensors file
    /// here after the final RL step. Enables post-run aggregate eval
    /// via `phase22_humaneval_baseline --checkpoint <path> --sequential
    /// --aggregate`. Skipped if not provided.
    #[arg(long)]
    final_checkpoint: Option<PathBuf>,
    /// Policy-gradient micro-batch size. `0` (default) collects all
    /// k×n_prompts samples into one backward pass — the original Stage G
    /// behaviour, fits when max_new ≤ 16 + k ≤ 2. `> 0` chunks samples
    /// and calls `backward_step` once per chunk, bounding peak GPU memory
    /// so that max_new ≥ 64 or k > 2 become viable.
    #[arg(long, default_value_t = 0)]
    pg_micro_batch_size: usize,
    /// Phase 22 follow-up C3 — restore the Stage E policy-gradient step
    /// semantics: one AdamW update **per micro-batch** instead of one per
    /// RL step. At `--n-prompts 64 --k-per-prompt 4 --pg-micro-batch-size 1`
    /// that is 256 optimizer updates per RL step (~1024 before the first
    /// `--sync-every 4` sync), which is what collapsed the 7B hard-tail
    /// policy. Only pass this to reproduce the collapse.
    #[arg(long, default_value_t = false)]
    pg_legacy_updates: bool,
    /// Phase 22 follow-up C3 — keep RLOO zero-advantage samples (prompts
    /// where all k completions share a verdict). They contribute no
    /// gradient; on the hard tail they are ~94% of the batch, so keeping
    /// them costs a forward+backward each. Stage E kept them.
    #[arg(long, default_value_t = false)]
    pg_keep_zero_advantage: bool,
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

fn resolve_snapshot(model_dir: Option<&std::path::Path>, model_id: &str) -> Result<PathBuf> {
    if let Some(d) = model_dir {
        if !d.join("config.json").exists() {
            anyhow::bail!("--model-dir {d:?} has no config.json");
        }
        return Ok(d.to_path_buf());
    }
    let home = std::env::var("HOME").context("HOME unset")?;
    let snapshots_dir = PathBuf::from(format!(
        "{home}/.cache/huggingface/hub/models--Qwen--{model_id}/snapshots"
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
        .with_max_level(tracing::Level::WARN)
        .init();

    let args = Args::parse();
    let device = pick_device();
    let on_cuda = device.is_cuda();
    println!(
        "[Phase22E] device = {device:?}, on_cuda = {on_cuda}, rl_steps = {}",
        args.rl_steps
    );
    if !on_cuda && std::env::var("PHASE22_ALLOW_CPU").is_err() {
        anyhow::bail!(
            "Refusing to run on CPU. Rebuild with `--features cuda`. \
             Set PHASE22_ALLOW_CPU=1 to override."
        );
    }

    let snapshot = resolve_snapshot(args.model_dir.as_deref(), &args.model_id)?;
    println!("[Phase22E] snapshot = {}", snapshot.display());
    // dtype: default (0.5B) = F16 inference / F32 train; --train-bf16 = BF16
    // throughout (7B on 40GB). CPU = F32.
    let (inference_dtype, train_dtype) = if !on_cuda {
        (DType::F32, DType::F32)
    } else if args.train_bf16 {
        (DType::BF16, DType::BF16)
    } else {
        (DType::F16, DType::F32)
    };

    let tk = Arc::new(NgptTokenizer::from_hf_file(
        snapshot.join("tokenizer.json"),
    )?);

    let jsonl = args
        .jsonl
        .unwrap_or_else(|| PathBuf::from("data/humaneval/HumanEval.jsonl"));
    let scratch = std::env::temp_dir().join("workllm-phase22e-humaneval");
    let humaneval = HumanEvalDomain::from_jsonl(&jsonl, &scratch)
        .with_context(|| format!("loading HumanEval from {}", jsonl.display()))?;
    println!(
        "[Phase22E] HumanEvalDomain loaded {} problems",
        humaneval.n_problems()
    );

    // Pick `n_prompts` HumanEval problems deterministically (the first
    // `n_prompts` via `nth_prompt`). Phase 22 D's smoke also uses a
    // determinis subset so RL and SFT can be cross-compared on the
    // same problem distribution if desired.
    let prompts: Vec<String> = (args.prompt_offset..args.prompt_offset + args.n_prompts)
        .filter_map(|i| humaneval.nth_prompt(i))
        .collect();
    println!(
        "[Phase22E] using {} HumanEval prompts (task_id {}..{})",
        prompts.len(),
        args.prompt_offset,
        args.prompt_offset + prompts.len()
    );

    // Optionally place the trainer on its own GPU (see --trainer-gpu): for 7B
    // the sampling model + policy-gradient trainer are ~15GB each, and the PG
    // graph over generated tokens needs a full card's headroom.
    let trainer_device = match args.trainer_gpu {
        Some(idx) if on_cuda => {
            let d = Device::new_cuda(idx)
                .with_context(|| format!("--trainer-gpu {idx}: CUDA device unavailable"))?;
            println!("[Phase22E] trainer on separate device cuda:{idx} (model on cuda:0)");
            d
        }
        _ => device.clone(),
    };
    let qwen_model = QwenModelActor::from_snapshot_dir(&snapshot, device.clone(), inference_dtype)?;
    let qwen_trainer = QwenTrainerActor::from_snapshot_dir(
        &snapshot,
        trainer_device,
        train_dtype,
        LoraConfig {
            rank: args.lora_rank,
            alpha: args.lora_alpha,
        },
        args.lr,
    )?
    .with_pg_micro_batch_size(args.pg_micro_batch_size)
    .with_pg_step_semantics(!args.pg_legacy_updates, !args.pg_keep_zero_advantage);

    let system = ActorSystem::new("phase22-e");
    let model_ref = system.spawn(qwen_model, "qwen-model").await?;
    let trainer_ref = system.spawn(qwen_trainer, "qwen-trainer").await?;
    println!("[Phase22E] 2 actors spawned (model + trainer)\n");

    let mut losses: Vec<f32> = Vec::with_capacity(args.rl_steps);
    let mut step_passes: Vec<usize> = Vec::with_capacity(args.rl_steps);
    let mut step_totals: Vec<usize> = Vec::with_capacity(args.rl_steps);

    let t_run = std::time::Instant::now();
    for rl_step in 0..args.rl_steps {
        let mut samples: Vec<(Vec<u32>, Vec<u32>, f32)> = Vec::new();
        let mut prompt_passes: Vec<usize> = Vec::with_capacity(prompts.len());
        // Phase 22 follow-up C3 — mean completion length is the collapse
        // tell-tale: a mode-collapsed policy emits EOS immediately, so the
        // length (and the step wallclock) falls off a cliff while `pass`
        // goes to 0.
        let mut comp_len_sum = 0usize;
        let mut comp_len_n = 0usize;

        let t_step = std::time::Instant::now();
        for (p_idx, prompt) in prompts.iter().enumerate() {
            let prompt_ids = tk.encode(prompt)?;
            let mut prompt_samples: Vec<(Vec<u32>, f32)> = Vec::with_capacity(args.k_per_prompt);

            for k in 0..args.k_per_prompt {
                let cfg = GenerateConfig {
                    max_new_tokens: args.max_new_tokens,
                    temperature: args.temperature,
                    top_k: Some(40),
                    top_p: Some(0.95),
                    seed: Some(args.seed.wrapping_add(
                        ((rl_step as u64) << 24) ^ ((p_idx as u64) << 8) ^ (k as u64),
                    )),
                };
                let (tx, rx) = oneshot::channel();
                model_ref
                    .tell(ModelMessage::GenerateTokens {
                        prompt_ids: prompt_ids.clone(),
                        cfg,
                        reply: tx,
                    })
                    .map_err(|e| anyhow!("{e:?}"))?;
                let full = rx.await??;
                let comp_ids: Vec<u32> = if full.len() > prompt_ids.len() {
                    full[prompt_ids.len()..].to_vec()
                } else {
                    vec![]
                };
                comp_len_sum += comp_ids.len();
                comp_len_n += 1;
                let comp_text = tk.decode(&comp_ids)?;
                let verdict = humaneval.verify(prompt, &comp_text);
                let v_value: f32 = if verdict.is_correct() { 1.0 } else { 0.0 };
                prompt_samples.push((comp_ids, v_value));
            }

            // RLOO baseline — center rewards on the prompt's mean
            // verdict so high-variance prompts don't dominate.
            let baseline: f32 =
                prompt_samples.iter().map(|(_, v)| *v).sum::<f32>() / prompt_samples.len() as f32;
            let mut passes_this_prompt = 0usize;
            for (comp_ids, v) in prompt_samples {
                if v > 0.5 {
                    passes_this_prompt += 1;
                }
                let reward = v - baseline;
                if !comp_ids.is_empty() {
                    samples.push((prompt_ids.clone(), comp_ids, reward));
                }
            }
            prompt_passes.push(passes_this_prompt);
        }

        let total_pass: usize = prompt_passes.iter().sum();
        let total_samples = prompts.len() * args.k_per_prompt;

        // Send TrainPolicyGradient to the trainer.
        let (tx, rx) = oneshot::channel();
        trainer_ref
            .tell(QwenTrainerMessage::TrainPolicyGradient { samples, reply: tx })
            .map_err(|e| anyhow!("{e:?}"))?;
        let stats = rx.await??;
        let loss = stats.loss;
        losses.push(loss);
        step_passes.push(total_pass);
        step_totals.push(total_samples);

        // C2 — adapter sync. Every `sync_every` RL steps, push the
        // accumulated LoRA delta from the trainer into the inference
        // model so subsequent samples are drawn from the updated
        // policy (closer to on-policy).
        let synced = if args.sync_every > 0 && (rl_step + 1) % args.sync_every == 0 {
            let base = args
                .base_safetensors
                .clone()
                .unwrap_or_else(|| snapshot.clone());
            let (sx, sr) = oneshot::channel();
            trainer_ref
                .tell(QwenTrainerMessage::SaveMergedCheckpoint {
                    base_path: base,
                    out_path: args.sync_path.clone(),
                    reply: sx,
                })
                .map_err(|e| anyhow!("{e:?}"))?;
            sr.await??;
            let (rx2, rr2) = oneshot::channel();
            model_ref
                .tell(ModelMessage::ReloadCheckpoint {
                    path: args.sync_path.clone(),
                    reply: rx2,
                })
                .map_err(|e| anyhow!("{e:?}"))?;
            rr2.await??;
            true
        } else {
            false
        };

        let mean_comp_len = if comp_len_n == 0 {
            0.0
        } else {
            comp_len_sum as f64 / comp_len_n as f64
        };
        println!(
            "[Phase22E] rl_step {rl_step}  loss = {loss:+.4}  pass = {}/{}  \
             upd = {}  used/skip = {}/{}  comp_len = {:.1}  \
             elapsed_step = {:.1}s  synced = {}",
            total_pass,
            total_samples,
            stats.n_updates,
            stats.n_used,
            stats.n_skipped,
            mean_comp_len,
            t_step.elapsed().as_secs_f64(),
            synced
        );
    }

    println!("\n[Phase22E] === RL summary ===");
    println!(
        "[Phase22E] {} steps × {} prompts × k={} = {} samples total, elapsed = {:.1}s",
        args.rl_steps,
        prompts.len(),
        args.k_per_prompt,
        args.rl_steps * prompts.len() * args.k_per_prompt,
        t_run.elapsed().as_secs_f64()
    );
    println!("[Phase22E] losses across RL steps: {:?}", losses);
    println!(
        "[Phase22E] passes/total per step:  {:?}/{:?}",
        step_passes, step_totals
    );

    // The deliverable: RL loop closes against a real benchmark verifier.
    // Reward signal is sparse on HumanEval at this scale (most prompts
    // yield 0 passes); RLOO collapses to (v - 0)/k = 0 for those
    // prompts → no gradient contribution. The non-zero-reward prompts
    // (any prompt where >=1 of k samples passes) drive learning.
    let nonzero_steps: usize = step_passes.iter().filter(|&&p| p > 0).count();
    println!(
        "\n[Phase22E] {}/{} RL steps had >0 prompt passes (gradient signal)",
        nonzero_steps, args.rl_steps
    );
    // Save final merged checkpoint if requested.
    if let Some(ref out_path) = args.final_checkpoint {
        let base = args
            .base_safetensors
            .clone()
            .unwrap_or_else(|| snapshot.clone());
        let (tx, rx) = oneshot::channel();
        trainer_ref
            .tell(QwenTrainerMessage::SaveMergedCheckpoint {
                base_path: base,
                out_path: out_path.clone(),
                reply: tx,
            })
            .map_err(|e| anyhow!("{e:?}"))?;
        rx.await??;
        println!("[Phase22E] final checkpoint saved → {}", out_path.display());
    }

    println!("\nphase22_he_reinforce: PASS");
    Ok(())
}
