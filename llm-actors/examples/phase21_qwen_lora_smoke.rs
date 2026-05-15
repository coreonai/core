//! Phase 21 Stage F — Qwen2 LoRA training smoke (Candle-native).
//!
//! Loads the same Qwen2.5-Coder-0.5B checkpoint Phase 14-20 used,
//! wraps it with LoRA adapters on `q_proj` + `v_proj` (the Phase
//! 14-20 PEFT recipe), and runs N training steps over a fixed
//! `(input, target)` next-token-prediction pair.
//!
//! Acceptance: loss strictly decreases across the steps, demonstrating
//! that gradients flow through LoRA Vars and AdamW updates them.
//! This is the **inference + training bridge** — pair with
//! Stage D's `QwenModelActor` for the inference side and the SAME
//! model now has a Rust-side training stack too.
//!
//! Not in this smoke:
//! - Real corpus (uses a hand-crafted prompt → completion)
//! - Adapter save/load (VarMap stays in process memory)
//! - Actor integration (deferred to a future stage)
//!
//! Run:
//!   cargo run -p llm-actors --example phase21_qwen_lora_smoke \
//!       --features cuda --release -- --steps 6
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, ParamsAdamW, VarBuilder, VarMap};
use candle_transformers::models::qwen2::Config as Qwen2Config;
use clap::Parser;
use llm_actors::qwen2_lora::{lora_grad_norms, train_qwen_lora_step, LoraConfig, ModelForCausalLM};
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    model_dir: Option<PathBuf>,
    #[arg(long, default_value_t = 6)]
    steps: usize,
    #[arg(long, default_value_t = 2e-4)]
    lr: f64,
    #[arg(long, default_value_t = 16)]
    lora_rank: usize,
    #[arg(long, default_value_t = 32.0)]
    lora_alpha: f32,
    #[arg(
        long,
        default_value = "def fibonacci(n):\n    if n < 2:\n        return n"
    )]
    text: String,
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

fn resolve_default_snapshot() -> Result<PathBuf> {
    let home = std::env::var("HOME").context("HOME unset")?;
    let snapshots_dir = PathBuf::from(format!(
        "{home}/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B/snapshots"
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

fn main() -> Result<()> {
    let args = Args::parse();
    let device = pick_device();
    let on_cuda = device.is_cuda();
    println!("[Phase21F] device = {device:?}, on_cuda = {on_cuda}");

    let dir = args
        .model_dir
        .map(Ok)
        .unwrap_or_else(resolve_default_snapshot)?;
    println!("[Phase21F] model_dir = {}", dir.display());

    // -- Tokenizer
    let tokenizer =
        Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("tokenizer: {e}"))?;
    println!(
        "[Phase21F] tokenizer loaded (vocab={})",
        tokenizer.get_vocab_size(true)
    );

    // -- Config
    let cfg_text = std::fs::read_to_string(dir.join("config.json"))?;
    let cfg: Qwen2Config = serde_json::from_str(&cfg_text)?;

    // -- Build model: frozen base from safetensors + trainable LoRA VarMap.
    //
    // Training is done in F32 throughout — gradients through F16 LoRA Vars
    // are numerically too coarse at rank=16 and lr=2e-4.
    let dtype = DType::F32;
    let safetensors = dir.join("model.safetensors");
    let base_vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&safetensors], dtype, &device)? };

    let lora_map = VarMap::new();
    let lora_vb = VarBuilder::from_varmap(&lora_map, dtype, &device);
    let lora_cfg = LoraConfig {
        rank: args.lora_rank,
        alpha: args.lora_alpha,
    };

    let mut model = ModelForCausalLM::new(&cfg, base_vb, Some(lora_vb), lora_cfg)?;
    println!(
        "[Phase21F] model built  weight_tied={}  lora(r={}, α={})",
        model.weight_tied(),
        lora_cfg.rank,
        lora_cfg.alpha,
    );

    // Sanity: count LoRA parameters.
    let lora_vars = lora_map.all_vars();
    let lora_param_count: usize = lora_vars
        .iter()
        .map(|v| v.dims().iter().product::<usize>())
        .sum();
    println!(
        "[Phase21F] LoRA Vars: {} tensors / {} params",
        lora_vars.len(),
        lora_param_count
    );
    if lora_vars.is_empty() {
        return Err(anyhow!(
            "no LoRA Vars registered — adapter injection failed"
        ));
    }
    // Dump unique LoRA Var shapes — should include both (rank, in_dim)
    // and (out_dim, rank) families per (q_proj, v_proj).
    let mut shape_counts: std::collections::HashMap<Vec<usize>, usize> =
        std::collections::HashMap::new();
    for v in &lora_vars {
        *shape_counts.entry(v.dims().to_vec()).or_insert(0) += 1;
    }
    let mut pairs: Vec<_> = shape_counts.into_iter().collect();
    pairs.sort();
    println!("[Phase21F] LoRA Var shape inventory:");
    for (s, c) in pairs {
        println!("  shape {s:?} × {c}");
    }

    // -- AdamW over LoRA Vars only.
    let params = ParamsAdamW {
        lr: args.lr,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0, // LoRA traditionally has 0 weight decay
    };
    use candle_nn::Optimizer;
    let mut optimizer = AdamW::new(lora_vars, params)?;

    // -- Build the (input, target) pair: shifted next-token prediction.
    let encoded = tokenizer
        .encode(args.text.as_str(), true)
        .map_err(|e| anyhow!("encode: {e}"))?;
    let ids: Vec<u32> = encoded.get_ids().to_vec();
    if ids.len() < 2 {
        return Err(anyhow!(
            "training text tokenizes to < 2 tokens — need at least 2 for next-token loss"
        ));
    }
    let input_ids = Tensor::from_slice(&ids[..ids.len() - 1], (1, ids.len() - 1), &device)?;
    let target_ids = Tensor::from_slice(&ids[1..], (1, ids.len() - 1), &device)?;
    println!(
        "[Phase21F] training text: {:?} → {} tokens, (input, target) shapes ({:?}, {:?})",
        args.text.lines().next().unwrap_or(&args.text),
        ids.len(),
        input_ids.shape(),
        target_ids.shape(),
    );

    // -- Diagnostic: gradient norms over ALL LoRA Vars.
    let all_vars = lora_map.all_vars();
    let norms = lora_grad_norms(&mut model, &input_ids, &target_ids, &all_vars)?;
    let with_grad = norms
        .iter()
        .filter(|(_, n)| !n.is_nan() && *n != 0.0)
        .count();
    let nan_count = norms.iter().filter(|(_, n)| n.is_nan()).count();
    let zero_count = norms.iter().filter(|(_, n)| *n == 0.0).count();
    println!(
        "[Phase21F] LoRA grad-norms across {} Vars: with_grad={with_grad}  zero={zero_count}  nan={nan_count}",
        norms.len()
    );
    // Print a few examples from each category.
    for (label, want_nan, want_zero) in [
        ("with_grad", false, false),
        ("zero", false, true),
        ("nan", true, false),
    ] {
        let matches: Vec<&(String, f32)> = norms
            .iter()
            .filter(|(_, n)| n.is_nan() == want_nan && (*n == 0.0) == want_zero)
            .take(3)
            .collect();
        if !matches.is_empty() {
            println!("  {label} examples:");
            for (name, n) in matches {
                println!("    {name} = {n:.6}");
            }
        }
    }
    if with_grad == 0 {
        return Err(anyhow!(
            "ZERO LoRA Vars receive gradients — gradients are not flowing into ANY LoRA Var. \
             Aborting training (Stage F bug)."
        ));
    }

    // -- Train loop
    let mut losses = Vec::with_capacity(args.steps);
    for step in 0..args.steps {
        let loss = train_qwen_lora_step(&mut model, &mut optimizer, &input_ids, &target_ids)?;
        losses.push(loss);
        println!("[Phase21F]  step {step}  loss = {loss:.4}");
    }

    let initial = losses.first().copied().unwrap_or(f32::NAN);
    let final_loss = losses.last().copied().unwrap_or(f32::NAN);
    println!(
        "\n[Phase21F] loss: {initial:.4} → {final_loss:.4}  Δ = {:+.4}",
        final_loss - initial
    );

    // Acceptance: loss must strictly decrease over the run.
    if final_loss >= initial {
        return Err(anyhow!(
            "loss did NOT decrease ({initial:.4} → {final_loss:.4}) — \
             LoRA gradients may be broken"
        ));
    }
    println!("phase21_qwen_lora_smoke: PASS");
    Ok(())
}
