//! Phase 5 scaffolding: N-actor ensemble plumbing.
//!
//! Spawns N `ModelActor`s (each owns its own `VarMap`), validates that
//! they share a tokenizer (i.e. same `vocab_size`), and exposes
//! `ensemble_generate` to run all of them on the same prompt set.
//!
//! This is the **Session 1** deliverable from `docs/phase5-design.md` —
//! plumbing only. Consensus filtering / curator scoring / ensemble
//! training round live in Sessions 2+.
//!
//! Models can be heterogeneous (different `GPTConfig`s) but must share
//! `vocab_size` so a single tokenizer covers all of them. Each model
//! optionally loads from its own checkpoint path; pass `None` to spawn
//! with random weights (handy for smoke tests and seed-diversity
//! ensembles where the random init *is* the source of independence).

use std::path::PathBuf;
use std::sync::Arc;

use candle_core::Device;
use nanogpt_rs::{config::GPTConfig, generate::GenerateConfig, tokenizer::Tokenizer};
use pekko_actor::{ActorRef, ActorSystem};
use tokio::sync::oneshot;
use tracing::info;

use crate::model_actor::{ModelActor, ModelMessage};
use crate::types::Trajectory;

/// Architecture + checkpoint plan for an N-actor ensemble.
pub struct EnsembleConfig {
    pub models: Vec<GPTConfig>,
    /// Optional starting checkpoint per model. `None` = random init.
    pub init_paths: Vec<Option<PathBuf>>,
    pub device: Device,
}

impl EnsembleConfig {
    pub fn n(&self) -> usize {
        self.models.len()
    }

    /// All models must agree on vocab_size (single shared tokenizer)
    /// and the lengths of `models` and `init_paths` must match.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.models.is_empty() {
            anyhow::bail!("EnsembleConfig.models is empty");
        }
        if self.models.len() != self.init_paths.len() {
            anyhow::bail!(
                "EnsembleConfig: models.len() = {} but init_paths.len() = {}",
                self.models.len(),
                self.init_paths.len(),
            );
        }
        let v0 = self.models[0].vocab_size;
        for (i, c) in self.models.iter().enumerate() {
            if c.vocab_size != v0 {
                anyhow::bail!(
                    "ensemble vocab_size mismatch: model 0 = {v0}, model {i} = {} \
                     — all members must share the tokenizer",
                    c.vocab_size,
                );
            }
        }
        Ok(())
    }
}

/// Live actor refs for the spawned ensemble.
pub struct EnsembleActors {
    pub models: Vec<ActorRef<ModelActor>>,
    pub tokenizer: Arc<Tokenizer>,
    pub model_names: Vec<String>,
}

impl EnsembleActors {
    pub async fn spawn(
        cfg: &EnsembleConfig,
        tokenizer: Arc<Tokenizer>,
        system: &ActorSystem,
    ) -> anyhow::Result<Self> {
        cfg.validate()?;
        let mut models = Vec::with_capacity(cfg.n());
        let mut model_names = Vec::with_capacity(cfg.n());
        for (i, (gpt_cfg, init_path)) in cfg.models.iter().zip(cfg.init_paths.iter()).enumerate() {
            let name = format!("ensemble-model-{i}");
            let actor = match init_path {
                Some(path) => ModelActor::from_checkpoint(
                    gpt_cfg.clone(),
                    cfg.device.clone(),
                    tokenizer.clone(),
                    path,
                )?,
                None => ModelActor::new(gpt_cfg.clone(), cfg.device.clone(), tokenizer.clone())?,
            };
            let actor_ref = system.spawn(actor, &name).await?;
            models.push(actor_ref);
            model_names.push(name);
        }
        info!(n = models.len(), "ensemble spawned");
        Ok(Self {
            models,
            tokenizer,
            model_names,
        })
    }

    pub fn n(&self) -> usize {
        self.models.len()
    }
}

/// Run each model on each prompt, drawing `samples_per_model` samples
/// per (model, prompt) pair. Returns `Vec<Vec<Trajectory>>` indexed by
/// `[model_id]`. Each inner vec has `prompts.len() * samples_per_model`
/// entries in row-major order: (prompt 0, sample 0), (prompt 0, sample
/// 1), ..., (prompt 1, sample 0), ...
///
/// Seeds are deterministic: model `i`'s sample `j` on prompt `k` uses
/// `seed = seed_base + i * 1_000 + k * 100 + j`. This means re-running
/// with the same `seed_base` yields identical trajectories — useful
/// for testing the consensus filter without compounding randomness.
pub async fn ensemble_generate(
    actors: &EnsembleActors,
    prompts: &[String],
    samples_per_model: usize,
    sampling: &GenerateConfig,
    seed_base: u64,
) -> anyhow::Result<Vec<Vec<Trajectory>>> {
    let mut out: Vec<Vec<Trajectory>> = (0..actors.n()).map(|_| Vec::new()).collect();
    for (i, model) in actors.models.iter().enumerate() {
        for (k, prompt) in prompts.iter().enumerate() {
            for j in 0..samples_per_model {
                let mut s = sampling.clone();
                s.seed = Some(seed_base + (i as u64) * 1_000 + (k as u64) * 100 + j as u64);
                let (tx, rx) = oneshot::channel();
                model
                    .tell(ModelMessage::Generate {
                        prompt: prompt.clone(),
                        cfg: s,
                        reply: tx,
                    })
                    .map_err(|e| anyhow::anyhow!("{e:?}"))?;
                let reply = rx.await??;
                out[i].push(Trajectory {
                    prompt: prompt.clone(),
                    completion: reply.text,
                    source: format!("ensemble-model-{i}"),
                });
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nanogpt_rs::config::{ActivationKind, GPTConfig, NormKind, NormPosition};

    fn tiny_cfg(n_layer: usize, n_embd: usize) -> GPTConfig {
        GPTConfig {
            vocab_size: 16,
            block_size: 8,
            n_layer,
            n_head: 2,
            n_embd,
            dropout: 0.0,
            bias: false,
            ffn_mult: 2,
            use_rope: false,
            rope_base: 10_000.0,
            n_kv_head: 2,
            n_experts: 1,
            moe_top_k: 0,
            moe_aux_weight: 0.0,
            activation: ActivationKind::Gelu,
            weight_tying: true,
            norm_kind: NormKind::LayerNorm,
            norm_position: NormPosition::Pre,
            lora_rank: 0,
            lora_alpha: 16.0,
        }
    }

    #[test]
    fn validate_rejects_vocab_mismatch() {
        let mut a = tiny_cfg(2, 16);
        let mut b = tiny_cfg(2, 16);
        a.vocab_size = 16;
        b.vocab_size = 32;
        let cfg = EnsembleConfig {
            models: vec![a, b],
            init_paths: vec![None, None],
            device: Device::Cpu,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_length_mismatch() {
        let cfg = EnsembleConfig {
            models: vec![tiny_cfg(2, 16)],
            init_paths: vec![None, None],
            device: Device::Cpu,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_empty() {
        let cfg = EnsembleConfig {
            models: vec![],
            init_paths: vec![],
            device: Device::Cpu,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_accepts_heterogeneous_archs_with_shared_vocab() {
        let cfg = EnsembleConfig {
            // Different n_layer / n_embd, same vocab — a small + a smaller model.
            models: vec![tiny_cfg(2, 16), tiny_cfg(4, 32)],
            init_paths: vec![None, None],
            device: Device::Cpu,
        };
        cfg.validate().expect("heterogeneous archs allowed");
        assert_eq!(cfg.n(), 2);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn smoke_two_model_ensemble_generates_trajectories() {
        // Two tiny models, random init (seed-diversity), tiny char vocab.
        // Verifies the full path: spawn → ensemble_generate →
        // [model][prompt × sample] layout.
        let cfg = EnsembleConfig {
            models: vec![tiny_cfg(2, 16), tiny_cfg(2, 16)],
            init_paths: vec![None, None],
            device: Device::Cpu,
        };
        let tk = Arc::new(Tokenizer::char_from_text("abcdefghijklmnop"));
        // tiny_cfg specifies vocab=16; tokenizer must match.
        assert_eq!(tk.vocab_size(), 16);

        let system = ActorSystem::new("ensemble-smoke");
        let ens = EnsembleActors::spawn(&cfg, tk, &system).await.unwrap();
        assert_eq!(ens.n(), 2);

        let prompts = vec!["abc".to_string(), "def".to_string()];
        let sampling = GenerateConfig {
            max_new_tokens: 4,
            temperature: 1.0,
            top_k: Some(4),
            top_p: None,
            seed: Some(0),
        };
        let trajectories = ensemble_generate(&ens, &prompts, 3, &sampling, 0xC0DE)
            .await
            .unwrap();

        // Layout invariant: outer = N=2 models; inner = prompts × samples.
        assert_eq!(trajectories.len(), 2);
        for per_model in &trajectories {
            assert_eq!(per_model.len(), prompts.len() * 3);
            for traj in per_model {
                assert!(prompts.contains(&traj.prompt));
                assert!(traj.source.starts_with("ensemble-model-"));
            }
        }

        // Random-init divergence: at least ONE (model 0, sample) must differ
        // from the corresponding (model 1, sample) at the same index. If
        // every entry matched, the two random inits would be acting
        // identically — that defeats the ensemble's whole purpose.
        let differ = trajectories[0]
            .iter()
            .zip(trajectories[1].iter())
            .any(|(a, b)| a.completion != b.completion);
        assert!(
            differ,
            "expected at least one disagreement between two random-init models"
        );
    }
}
