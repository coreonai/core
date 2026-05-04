//! Architectural evolution / NAS over `GPTConfig`.
//!
//! Population-based search:
//! - **SearchSpace** lists candidate values per architectural field.
//! - **Variant** = a `GPTConfig` + fitness/lineage metadata.
//! - **Operators** = `random_init`, `mutate`, `crossover`.
//! - **EvolutionRunner** orchestrates generations: train each variant from
//!   scratch on a fixed corpus, eval pass-rate, select top-k, fill the rest
//!   via mutation / crossover, repeat.
//!
//! Multi-GPU is opt-in via `n_gpus` in [`EvolutionConfig`]: variants are
//! dispatched round-robin across `Cuda(0)..Cuda(n_gpus)` using
//! `spawn_blocking`. With `n_gpus == 1` (or no CUDA) execution is serial.
//!
//! Fitness here is task-specific (pass-rate on a held-out prompt set);
//! callers supply the `Domain` and the corpus so the same runner can drive
//! arithmetic, RustCode, or future domains.

use std::sync::Arc;
use std::time::Instant;

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use nanogpt_rs::{
    config::{ActivationKind, GPTConfig, NormKind, NormPosition},
    data::TokenDataset,
    generate::{generate, GenerateConfig},
    model::GPT,
    train::{train_from, TrainConfig},
    Tokenizer,
};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use tokio::task::JoinSet;
use tracing::{info, warn};

use crate::domain::Domain;

/// Discrete search space — one slot per `GPTConfig` field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchSpace {
    pub n_layer: Vec<usize>,
    pub n_head: Vec<usize>,
    pub n_embd: Vec<usize>,
    pub block_size: Vec<usize>,
    pub ffn_mult: Vec<usize>,
    /// Whether RoPE is enabled. Search across {true, false} typically.
    pub use_rope: Vec<bool>,
    /// Group sizes for GQA: `n_kv_head = n_q_head / kv_group`. Each picked
    /// kv_group must divide the chosen `n_head`. Use {1} for MHA-only.
    pub kv_group: Vec<usize>,
    /// Number of MoE experts per block. `1` = dense (no MoE).
    pub n_experts: Vec<usize>,
    /// MLP activation kind to consider.
    pub activation: Vec<ActivationKind>,
    /// Whether the LM head shares weights with `wte`.
    pub weight_tying: Vec<bool>,
    pub norm_kind: Vec<NormKind>,
    pub norm_position: Vec<NormPosition>,
    pub vocab_size: usize,
    pub bias: bool,
}

impl SearchSpace {
    /// Sensible default for a small char-level domain (arithmetic). Tuned so
    /// that even the smallest sampled variant has enough capacity to learn
    /// the task within ~3000 training steps — earlier wider spaces wasted
    /// budget on configs that never escaped uniform prediction.
    pub fn small_char(vocab_size: usize) -> Self {
        Self {
            n_layer: vec![4, 6, 8],
            n_head: vec![4, 6, 8],
            n_embd: vec![192, 256, 384],
            block_size: vec![16, 32, 64],
            ffn_mult: vec![2, 4, 6],
            use_rope: vec![false, true],
            kv_group: vec![1, 2, 4],
            // MoE expensive: cap at 4 experts so worst-case compute stays
            // within ~3 minutes per variant on A100.
            n_experts: vec![1, 2, 4],
            activation: vec![ActivationKind::Gelu, ActivationKind::SwiGlu, ActivationKind::GeGlu],
            weight_tying: vec![true, false],
            norm_kind: vec![NormKind::LayerNorm, NormKind::RmsNorm],
            norm_position: vec![NormPosition::Pre, NormPosition::Post],
            vocab_size,
            bias: false,
        }
    }

    /// Pick a `kv_group` from `self.kv_group` that divides `n_head`. Falls
    /// back to 1 (MHA) if nothing matches — guarantees a valid pair.
    fn sample_kv_group(&self, n_head: usize, rng: &mut StdRng) -> usize {
        let valid: Vec<usize> = self
            .kv_group
            .iter()
            .copied()
            .filter(|g| *g >= 1 && n_head % g == 0)
            .collect();
        if valid.is_empty() {
            1
        } else {
            *valid.choose(rng).unwrap()
        }
    }

    /// Pick a uniform-random valid (n_embd, n_head) pair such that
    /// `n_embd % n_head == 0`.
    fn sample_compatible_embd_head(&self, rng: &mut StdRng) -> (usize, usize) {
        for _ in 0..32 {
            let e = *self.n_embd.choose(rng).unwrap();
            let h = *self.n_head.choose(rng).unwrap();
            if e % h == 0 {
                return (e, h);
            }
        }
        // Fallback: pair up the smallest-compatible.
        for &e in &self.n_embd {
            for &h in &self.n_head {
                if e % h == 0 {
                    return (e, h);
                }
            }
        }
        panic!("SearchSpace has no n_embd/n_head pair where n_embd % n_head == 0");
    }

    pub fn sample(&self, rng: &mut StdRng) -> GPTConfig {
        let (n_embd, n_head) = self.sample_compatible_embd_head(rng);
        let n_layer = *self.n_layer.choose(rng).unwrap();
        let block_size = *self.block_size.choose(rng).unwrap();
        let ffn_mult = *self.ffn_mult.choose(rng).unwrap();
        let use_rope = *self.use_rope.choose(rng).unwrap();
        let kv_group = self.sample_kv_group(n_head, rng);
        let n_kv_head = n_head / kv_group;
        let n_experts = *self.n_experts.choose(rng).unwrap();
        // Default top-k policy: when MoE active, use top-2 (Mixtral-style).
        // Single-expert dense gets top_k = 0.
        let moe_top_k = if n_experts >= 2 { 2.min(n_experts) } else { 0 };
        let activation = *self.activation.choose(rng).unwrap();
        let weight_tying = *self.weight_tying.choose(rng).unwrap();
        let norm_kind = *self.norm_kind.choose(rng).unwrap();
        let norm_position = *self.norm_position.choose(rng).unwrap();
        GPTConfig {
            vocab_size: self.vocab_size,
            block_size,
            n_layer,
            n_head,
            n_embd,
            dropout: 0.0,
            bias: self.bias,
            ffn_mult,
            use_rope,
            rope_base: 10_000.0,
            n_kv_head,
            n_experts,
            moe_top_k,
            // Load-balance aux loss is a regularizer that helps at scale to
            // prevent expert collapse, but on this 3000-step toy task it
            // mostly adds noise. Keep at 0 here; turn on for real training.
            moe_aux_weight: 0.0,
            activation,
            weight_tying,
            norm_kind,
            norm_position,
            lora_rank: 0,
            lora_alpha: 16.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variant {
    pub id: usize,
    pub generation: usize,
    pub config: GPTConfig,
    pub fitness: Option<f32>,
    pub parents: Vec<usize>,
    pub origin: VariantOrigin,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VariantOrigin {
    Random,
    Mutated { from: usize, fields: Vec<String> },
    Crossover { a: usize, b: usize },
    Elite,
}

impl Variant {
    pub fn new(id: usize, generation: usize, config: GPTConfig, origin: VariantOrigin, parents: Vec<usize>) -> Self {
        Self { id, generation, config, fitness: None, parents, origin }
    }
}

#[derive(Debug, Clone)]
pub struct EvolutionConfig {
    pub population_size: usize,
    pub generations: usize,
    pub elite_keep: usize,
    pub train_steps: usize,
    pub batch_size: usize,
    pub eval_n: usize,
    pub eval_seed: u64,
    pub min_corpus_chars: usize,
    pub n_gpus: usize,
}

impl EvolutionConfig {
    pub fn small_default() -> Self {
        Self {
            population_size: 6,
            generations: 3,
            elite_keep: 2,
            train_steps: 1500,
            batch_size: 128,
            eval_n: 100,
            eval_seed: 0xE7A1,
            min_corpus_chars: 8000,
            n_gpus: 1,
        }
    }
}

/// Mutation: pick 1-2 fields and swap them out for a random alternative
/// from the search space. Compatibility (n_embd % n_head == 0) is preserved
/// by re-pairing if either of those fields is touched.
pub fn mutate(parent: &GPTConfig, space: &SearchSpace, rng: &mut StdRng) -> (GPTConfig, Vec<String>) {
    let mut cfg = parent.clone();
    let mut touched: Vec<String> = Vec::new();
    let n_changes = rng.gen_range(1..=2);
    let fields: [&str; 12] = [
        "n_layer", "n_head", "n_embd", "block_size", "ffn_mult",
        "use_rope", "kv_group", "n_experts", "activation",
        "weight_tying", "norm_kind", "norm_position",
    ];
    let chosen: Vec<&&str> = fields.choose_multiple(rng, n_changes).collect();
    for f in chosen {
        match *f {
            "n_layer" => {
                cfg.n_layer = *space.n_layer.choose(rng).unwrap();
                touched.push("n_layer".into());
            }
            "n_head" => {
                cfg.n_head = *space.n_head.choose(rng).unwrap();
                touched.push("n_head".into());
            }
            "n_embd" => {
                cfg.n_embd = *space.n_embd.choose(rng).unwrap();
                touched.push("n_embd".into());
            }
            "block_size" => {
                cfg.block_size = *space.block_size.choose(rng).unwrap();
                touched.push("block_size".into());
            }
            "ffn_mult" => {
                cfg.ffn_mult = *space.ffn_mult.choose(rng).unwrap();
                touched.push("ffn_mult".into());
            }
            "use_rope" => {
                cfg.use_rope = *space.use_rope.choose(rng).unwrap();
                touched.push("use_rope".into());
            }
            "kv_group" => {
                let g = space.sample_kv_group(cfg.n_head, rng);
                cfg.n_kv_head = cfg.n_head / g;
                touched.push("kv_group".into());
            }
            "n_experts" => {
                cfg.n_experts = *space.n_experts.choose(rng).unwrap();
                touched.push("n_experts".into());
            }
            "activation" => {
                cfg.activation = *space.activation.choose(rng).unwrap();
                touched.push("activation".into());
            }
            "weight_tying" => {
                cfg.weight_tying = *space.weight_tying.choose(rng).unwrap();
                touched.push("weight_tying".into());
            }
            "norm_kind" => {
                cfg.norm_kind = *space.norm_kind.choose(rng).unwrap();
                touched.push("norm_kind".into());
            }
            "norm_position" => {
                cfg.norm_position = *space.norm_position.choose(rng).unwrap();
                touched.push("norm_position".into());
            }
            _ => unreachable!(),
        }
    }
    if cfg.n_embd % cfg.n_head != 0 {
        let (e, h) = space.sample_compatible_embd_head(rng);
        cfg.n_embd = e;
        cfg.n_head = h;
        if !touched.iter().any(|s| s == "n_embd") {
            touched.push("n_embd".into());
        }
        if !touched.iter().any(|s| s == "n_head") {
            touched.push("n_head".into());
        }
    }
    // After any change to n_head, re-pick kv_group so divisibility holds.
    if cfg.n_kv_head == 0 || cfg.n_head % cfg.n_kv_head != 0 || cfg.n_kv_head > cfg.n_head {
        let g = space.sample_kv_group(cfg.n_head, rng);
        cfg.n_kv_head = cfg.n_head / g;
    }
    (cfg, touched)
}

/// Crossover: each field independently picked from one parent, with
/// post-hoc compatibility check.
pub fn crossover(a: &GPTConfig, b: &GPTConfig, space: &SearchSpace, rng: &mut StdRng) -> GPTConfig {
    let mut cfg = a.clone();
    cfg.n_layer = if rng.gen_bool(0.5) { a.n_layer } else { b.n_layer };
    cfg.n_head = if rng.gen_bool(0.5) { a.n_head } else { b.n_head };
    cfg.n_embd = if rng.gen_bool(0.5) { a.n_embd } else { b.n_embd };
    cfg.block_size = if rng.gen_bool(0.5) { a.block_size } else { b.block_size };
    cfg.ffn_mult = if rng.gen_bool(0.5) { a.ffn_mult } else { b.ffn_mult };
    cfg.use_rope = if rng.gen_bool(0.5) { a.use_rope } else { b.use_rope };
    cfg.n_experts = if rng.gen_bool(0.5) { a.n_experts } else { b.n_experts };
    cfg.activation = if rng.gen_bool(0.5) { a.activation } else { b.activation };
    cfg.weight_tying = if rng.gen_bool(0.5) { a.weight_tying } else { b.weight_tying };
    cfg.norm_kind = if rng.gen_bool(0.5) { a.norm_kind } else { b.norm_kind };
    cfg.norm_position = if rng.gen_bool(0.5) { a.norm_position } else { b.norm_position };
    if cfg.n_embd % cfg.n_head != 0 {
        let (e, h) = space.sample_compatible_embd_head(rng);
        cfg.n_embd = e;
        cfg.n_head = h;
    }
    // kv_group inherited proportionally where possible, else resampled.
    let inherited_kv = if rng.gen_bool(0.5) { a.n_kv_head } else { b.n_kv_head };
    if inherited_kv > 0 && cfg.n_head % inherited_kv == 0 {
        cfg.n_kv_head = inherited_kv;
    } else {
        let g = space.sample_kv_group(cfg.n_head, rng);
        cfg.n_kv_head = cfg.n_head / g;
    }
    cfg
}

/// All inputs the per-variant fitness evaluator needs. Cloned once per call
/// (cheap — Arc shares).
#[derive(Clone)]
pub struct FitnessInputs {
    pub tokenizer: Arc<Tokenizer>,
    pub domain: Arc<dyn Domain>,
    pub corpus: Arc<String>,
    pub eval_n: usize,
    pub eval_seed: u64,
    pub train_steps: usize,
    pub batch_size: usize,
    pub stop_char: Option<char>,
    pub max_new_tokens: usize,
    pub min_corpus_chars: usize,
}

#[derive(Debug, Clone)]
pub struct VariantOutcome {
    pub id: usize,
    pub fitness: f32,
    pub last_train_loss: f32,
    pub eval_correct: usize,
    pub eval_total: usize,
    pub elapsed_ms: u128,
}

/// Train a fresh model with the given config, evaluate, return outcome.
/// Runs entirely on `device` (no actor crossings).
pub fn evaluate_variant(
    id: usize,
    config: &GPTConfig,
    inputs: &FitnessInputs,
    device: &Device,
) -> anyhow::Result<VariantOutcome> {
    let t0 = Instant::now();

    // Pad / repeat corpus if too short.
    let corpus_str = if inputs.corpus.len() < inputs.min_corpus_chars && !inputs.corpus.is_empty() {
        let factor = (inputs.min_corpus_chars + inputs.corpus.len() - 1) / inputs.corpus.len();
        inputs.corpus.repeat(factor)
    } else {
        (*inputs.corpus).clone()
    };
    let ids = inputs.tokenizer.encode(&corpus_str)?;
    if ids.len() < config.block_size + 2 {
        anyhow::bail!("corpus too short ({} ids) for block_size {}", ids.len(), config.block_size);
    }
    let ds = TokenDataset::new(ids, config.block_size);

    let mut tcfg = TrainConfig::smoke();
    tcfg.max_steps = inputs.train_steps;
    tcfg.batch_size = inputs.batch_size;
    tcfg.eval_interval = inputs.train_steps; // skip mid-train eval
    tcfg.lr = 1e-3;
    tcfg.min_lr = 1e-4;
    tcfg.warmup_steps = 50;

    // Save to a per-variant temp checkpoint so we can reload as a fresh GPT.
    let ckpt = std::env::temp_dir().join(format!("evolve_var_{id}.safetensors"));
    let outcome = train_from(config, &ds, None, &tcfg, device, Some(&ckpt), None)?;

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = GPT::new(config.clone(), vb)?;
    varmap.load(&ckpt)?;

    // Greedy eval on a fixed prompt set (for comparability across variants).
    let mut rng = StdRng::seed_from_u64(inputs.eval_seed);
    let mut correct = 0usize;
    let cfg = GenerateConfig {
        max_new_tokens: inputs.max_new_tokens,
        temperature: 0.0,
        top_k: Some(1),
        top_p: None,
        seed: Some(inputs.eval_seed),
    };
    for _ in 0..inputs.eval_n {
        let prompt = inputs.domain.sample_prompt(&mut rng);
        let prompt_ids = inputs.tokenizer.encode(&prompt)?;
        let tokens = generate(&model, &prompt_ids, &cfg, device)?;
        let comp_ids = if tokens.len() > prompt_ids.len() {
            &tokens[prompt_ids.len()..]
        } else {
            &[][..]
        };
        let mut completion = inputs.tokenizer.decode(comp_ids)?;
        if let Some(stop) = inputs.stop_char {
            if let Some(idx) = completion.find(stop) {
                completion.truncate(idx + stop.len_utf8());
            }
        }
        if inputs.domain.verify(&prompt, &completion).is_correct() {
            correct += 1;
        }
    }

    let _ = std::fs::remove_file(&ckpt);

    let fitness = correct as f32 / inputs.eval_n.max(1) as f32;
    Ok(VariantOutcome {
        id,
        fitness,
        last_train_loss: outcome.last_train_loss,
        eval_correct: correct,
        eval_total: inputs.eval_n,
        elapsed_ms: t0.elapsed().as_millis(),
    })
}

/// Round-robin device picker.
fn pick_device(idx: usize, n_gpus: usize) -> anyhow::Result<Device> {
    if n_gpus == 0 {
        return Ok(Device::Cpu);
    }
    let g = idx % n_gpus;
    Device::new_cuda(g).map_err(|e| anyhow::anyhow!("cuda {g}: {e}"))
}

pub struct EvolutionRunner {
    pub space: SearchSpace,
    pub cfg: EvolutionConfig,
    pub inputs: FitnessInputs,
    pub seed: u64,
    next_id: usize,
}

#[derive(Debug, Clone)]
pub struct GenerationReport {
    pub generation: usize,
    pub variants: Vec<Variant>,
    pub best_id: Option<usize>,
    pub best_fitness: Option<f32>,
}

impl EvolutionRunner {
    pub fn new(space: SearchSpace, cfg: EvolutionConfig, inputs: FitnessInputs, seed: u64) -> Self {
        Self { space, cfg, inputs, seed, next_id: 0 }
    }

    fn alloc_id(&mut self) -> usize {
        let i = self.next_id;
        self.next_id += 1;
        i
    }

    fn random_population(&mut self, rng: &mut StdRng, generation: usize) -> Vec<Variant> {
        (0..self.cfg.population_size)
            .map(|_| {
                let id = self.alloc_id();
                let config = self.space.sample(rng);
                Variant::new(id, generation, config, VariantOrigin::Random, vec![])
            })
            .collect()
    }

    /// Async because we use `spawn_blocking` for parallel GPU eval.
    pub async fn run(&mut self) -> anyhow::Result<Vec<GenerationReport>> {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut population = self.random_population(&mut rng, 0);
        let mut history: Vec<GenerationReport> = Vec::with_capacity(self.cfg.generations);

        for gen in 0..self.cfg.generations {
            info!(generation = gen, n = population.len(), "starting generation");

            // Evaluate all unfit variants in parallel.
            let mut joinset = JoinSet::new();
            for (i, v) in population.iter().enumerate() {
                if v.fitness.is_some() {
                    continue;
                }
                let inputs = self.inputs.clone();
                let id = v.id;
                let config = v.config.clone();
                let device = pick_device(i, self.cfg.n_gpus)?;
                joinset.spawn_blocking(move || evaluate_variant(id, &config, &inputs, &device));
            }

            while let Some(joined) = joinset.join_next().await {
                match joined {
                    Ok(Ok(out)) => {
                        if let Some(v) = population.iter_mut().find(|v| v.id == out.id) {
                            v.fitness = Some(out.fitness);
                        }
                        info!(
                            id = out.id,
                            fitness = out.fitness,
                            correct = out.eval_correct,
                            train_loss = out.last_train_loss,
                            elapsed_ms = out.elapsed_ms,
                            "variant evaluated"
                        );
                    }
                    Ok(Err(e)) => warn!(error = %e, "variant eval failed"),
                    Err(e) => warn!(error = %e, "variant join failed"),
                }
            }

            // Sort descending by fitness (None last).
            population.sort_by(|a, b| {
                b.fitness
                    .partial_cmp(&a.fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let best = population.first().cloned();
            let report = GenerationReport {
                generation: gen,
                variants: population.clone(),
                best_id: best.as_ref().map(|v| v.id),
                best_fitness: best.as_ref().and_then(|v| v.fitness),
            };
            info!(
                generation = gen,
                best_id = report.best_id,
                best_fitness = report.best_fitness,
                "generation complete"
            );
            history.push(report);

            if gen + 1 == self.cfg.generations {
                break;
            }

            // Build next generation: keep elites, fill with mutate/crossover.
            let elite_keep = self.cfg.elite_keep.min(population.len());
            let elites: Vec<Variant> = population.iter().take(elite_keep).cloned().collect();
            let mut next: Vec<Variant> = elites
                .iter()
                .map(|v| Variant {
                    id: self.alloc_id(),
                    generation: gen + 1,
                    config: v.config.clone(),
                    fitness: v.fitness,
                    parents: vec![v.id],
                    origin: VariantOrigin::Elite,
                })
                .collect();
            while next.len() < self.cfg.population_size {
                let id = self.alloc_id();
                let v = if rng.gen_bool(0.5) && elites.len() >= 2 {
                    let parent_a = elites.choose(&mut rng).unwrap();
                    let parent_b = loop {
                        let cand = elites.choose(&mut rng).unwrap();
                        if cand.id != parent_a.id {
                            break cand;
                        }
                    };
                    let cfg = crossover(&parent_a.config, &parent_b.config, &self.space, &mut rng);
                    Variant::new(
                        id,
                        gen + 1,
                        cfg,
                        VariantOrigin::Crossover { a: parent_a.id, b: parent_b.id },
                        vec![parent_a.id, parent_b.id],
                    )
                } else {
                    let parent = elites.choose(&mut rng).unwrap();
                    let (cfg, fields) = mutate(&parent.config, &self.space, &mut rng);
                    Variant::new(
                        id,
                        gen + 1,
                        cfg,
                        VariantOrigin::Mutated { from: parent.id, fields },
                        vec![parent.id],
                    )
                };
                next.push(v);
            }
            population = next;
        }

        Ok(history)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn space() -> SearchSpace {
        SearchSpace::small_char(13)
    }

    #[test]
    fn search_space_samples_compatible_pairs() {
        let mut rng = StdRng::seed_from_u64(7);
        let s = space();
        for _ in 0..50 {
            let cfg = s.sample(&mut rng);
            assert!(cfg.n_embd % cfg.n_head == 0, "{} % {} != 0", cfg.n_embd, cfg.n_head);
        }
    }

    #[test]
    fn mutate_preserves_compatibility() {
        let mut rng = StdRng::seed_from_u64(11);
        let s = space();
        let parent = s.sample(&mut rng);
        for _ in 0..50 {
            let (child, _) = mutate(&parent, &s, &mut rng);
            assert!(child.n_embd % child.n_head == 0);
        }
    }

    #[test]
    fn crossover_preserves_compatibility() {
        let mut rng = StdRng::seed_from_u64(13);
        let s = space();
        let a = s.sample(&mut rng);
        let b = s.sample(&mut rng);
        for _ in 0..50 {
            let c = crossover(&a, &b, &s, &mut rng);
            assert!(c.n_embd % c.n_head == 0);
        }
    }

    #[test]
    fn ffn_mult_is_sampled_from_space() {
        let mut rng = StdRng::seed_from_u64(17);
        let s = space();
        // Every sampled config has ffn_mult drawn from the space.
        for _ in 0..50 {
            let cfg = s.sample(&mut rng);
            assert!(s.ffn_mult.contains(&cfg.ffn_mult), "ffn_mult={} not in space {:?}", cfg.ffn_mult, s.ffn_mult);
        }
    }

    #[test]
    fn sampled_configs_are_kv_compatible() {
        let mut rng = StdRng::seed_from_u64(101);
        let s = space();
        for _ in 0..100 {
            let cfg = s.sample(&mut rng);
            assert!(cfg.n_kv_head > 0, "n_kv_head must be positive");
            assert!(cfg.n_head % cfg.n_kv_head == 0, "GQA divisibility violated");
        }
    }

    #[test]
    fn mutation_keeps_gqa_valid() {
        let mut rng = StdRng::seed_from_u64(103);
        let s = space();
        let parent = s.sample(&mut rng);
        for _ in 0..100 {
            let (child, _) = mutate(&parent, &s, &mut rng);
            assert!(child.n_kv_head > 0);
            assert!(child.n_head % child.n_kv_head == 0);
        }
    }

    #[test]
    fn rope_appears_in_population() {
        let mut rng = StdRng::seed_from_u64(105);
        let s = space();
        let mut saw_rope = false;
        let mut saw_no_rope = false;
        for _ in 0..50 {
            let cfg = s.sample(&mut rng);
            if cfg.use_rope { saw_rope = true; } else { saw_no_rope = true; }
        }
        assert!(saw_rope && saw_no_rope, "expected both RoPE on/off in 50 samples");
    }

    #[test]
    fn norm_axes_appear_in_population() {
        let mut rng = StdRng::seed_from_u64(141);
        let s = space();
        let mut saw_ln = false;
        let mut saw_rms = false;
        let mut saw_pre = false;
        let mut saw_post = false;
        for _ in 0..80 {
            let cfg = s.sample(&mut rng);
            match cfg.norm_kind {
                NormKind::LayerNorm => saw_ln = true,
                NormKind::RmsNorm => saw_rms = true,
            }
            match cfg.norm_position {
                NormPosition::Pre => saw_pre = true,
                NormPosition::Post => saw_post = true,
            }
        }
        assert!(saw_ln && saw_rms, "expected both LN/RMS in 80 samples");
        assert!(saw_pre && saw_post, "expected both pre/post in 80 samples");
    }

    #[test]
    fn weight_tying_appears_in_population() {
        let mut rng = StdRng::seed_from_u64(131);
        let s = space();
        let mut saw_tied = false;
        let mut saw_untied = false;
        for _ in 0..50 {
            let cfg = s.sample(&mut rng);
            if cfg.weight_tying { saw_tied = true; } else { saw_untied = true; }
        }
        assert!(saw_tied && saw_untied, "expected both tied/untied in 50 samples");
    }

    #[test]
    fn activation_kinds_appear_in_population() {
        let mut rng = StdRng::seed_from_u64(123);
        let s = space();
        let mut saw_gelu = false;
        let mut saw_swi = false;
        let mut saw_ge = false;
        for _ in 0..80 {
            let cfg = s.sample(&mut rng);
            match cfg.activation {
                ActivationKind::Gelu => saw_gelu = true,
                ActivationKind::SwiGlu => saw_swi = true,
                ActivationKind::GeGlu => saw_ge = true,
            }
        }
        assert!(saw_gelu && saw_swi && saw_ge, "all 3 activations must appear in 80 samples");
    }

    #[test]
    fn moe_appears_in_population() {
        let mut rng = StdRng::seed_from_u64(107);
        let s = space();
        let mut saw_dense = false;
        let mut saw_moe = false;
        for _ in 0..50 {
            let cfg = s.sample(&mut rng);
            if cfg.n_experts <= 1 { saw_dense = true; } else { saw_moe = true; }
        }
        assert!(saw_dense && saw_moe, "expected both dense and MoE in 50 samples");
    }

    #[test]
    fn n_experts_always_in_space() {
        let mut rng = StdRng::seed_from_u64(109);
        let s = space();
        for _ in 0..50 {
            let cfg = s.sample(&mut rng);
            assert!(s.n_experts.contains(&cfg.n_experts));
            let (child, _) = mutate(&cfg, &s, &mut rng);
            assert!(s.n_experts.contains(&child.n_experts));
        }
    }

    #[test]
    fn mutate_reaches_ffn_mult() {
        // With 50 mutations of a parent fixed at ffn_mult=4, the touched
        // field set should include "ffn_mult" at least once (1/5 chance per
        // mutation × ~1.5 fields per mutation × 50 ≈ 15 expected hits).
        let mut rng = StdRng::seed_from_u64(19);
        let s = space();
        let mut parent = s.sample(&mut rng);
        parent.ffn_mult = 4;
        let mut saw = false;
        for _ in 0..50 {
            let (_, fields) = mutate(&parent, &s, &mut rng);
            if fields.iter().any(|f| f == "ffn_mult") {
                saw = true;
                break;
            }
        }
        assert!(saw, "mutation never touched ffn_mult in 50 tries");
    }
}
