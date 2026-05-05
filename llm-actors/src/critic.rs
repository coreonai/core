//! Phase 6 Shape C scaffolding: a `Critic` is a cheap proxy that
//! predicts whether a candidate `(prompt, completion)` will pass the
//! domain's expensive verifier (cargo, in K9's case).
//!
//! Used as a pre-filter: rank-and-keep top-K candidates per prompt
//! before sending them to cargo. The expensive verifier remains the
//! ultimate label source for the curator; the critic only decides
//! who gets to ASK cargo, not who gets accepted.
//!
//! See `docs/phase6-shape-c.md` for the design rationale, the
//! sessions plan, and the acceptance criteria for promoting Shape C
//! out of "scaffolding only" status.

use std::path::Path;
use std::sync::{Arc, Mutex};

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use nanogpt_rs::{config::GPTConfig, model::GPT, tokenizer::Tokenizer};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

/// A `Critic` produces a real-valued score for `(prompt, completion)`
/// pairs; higher = more likely to pass the actual verifier.
///
/// Scores are NOT required to be probabilities — they're for ranking.
/// Convention: score in `[0.0, 1.0]` when an implementation can
/// produce one cheaply (so callers can apply a threshold), but
/// implementations are free to use any monotone scale.
pub trait Critic: Send + Sync {
    fn score(&self, prompt: &str, completion: &str) -> f32;
}

/// Trivial baseline: every candidate gets the same score, so all
/// pass the threshold. Equivalent to "no filter" and reproduces
/// the current self-improve loop's behavior. Useful as a
/// sanity-check baseline against learned critics.
pub struct AlwaysCorrectCritic;

impl Critic for AlwaysCorrectCritic {
    fn score(&self, _prompt: &str, _completion: &str) -> f32 {
        1.0
    }
}

/// Random scoring (deterministic per `(prompt, completion)` for a
/// given seed via interior mutability of an RNG). Useful as a
/// **negative baseline** — a critic should beat random.
pub struct RandomCritic {
    rng: Mutex<StdRng>,
}

impl RandomCritic {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
        }
    }
}

impl Critic for RandomCritic {
    fn score(&self, _prompt: &str, _completion: &str) -> f32 {
        let mut g = self.rng.lock().expect("RandomCritic mutex poisoned");
        g.gen_range(0.0..1.0)
    }
}

/// Phase 6 Shape C MVP: use the existing generator's own per-token
/// log-probability of `(prompt, completion)` as the critic score.
/// No separate model is trained — the LM's `lm_head` already encodes
/// "how confident am I in this completion?", and if that confidence
/// correlates with the verifier's verdict we get the critic for free.
///
/// Two scoring regimes — pick by the domain's completion-length
/// distribution:
///
/// - **mean** (default, `normalize_by_length = true`): score is
///   `sum_t log P(c_t | ...) / len(c)`. Length-invariant, works on
///   length-uniform domains (Phase 6 K9 RustCode AUC 0.727). Has a
///   **fatal short-bias on length-varying domains** — empty / 1-token
///   completions get artificially high mean log-prob because there's
///   no opportunity for low-confidence tokens to drag the average down.
///   Phase 7 S1 measured this: arithmetic mean-AUC 0.445, F=16 lift
///   0.04× (catastrophic).
///
/// - **sum** (`normalize_by_length = false`): score is the raw
///   `sum_t log P(c_t | ...)`. Penalizes longer completions by
///   accumulating more negative log-probs. Robust on length-varying
///   domains *iff the model has sufficient confidence calibration*.
///   Phase 7 S2 measured this: arithmetic sum-AUC crossed 0.6 at 5000
///   pretrain steps even while pass rate stayed at chance baseline.
///
/// **The right gate is held-out sum-AUC ≥ 0.6**, not pass rate. See
/// `docs/phase7-design.md` for the full decision tree. Use
/// [`LogitCritic::sum_scoring`] / [`LogitCritic::sum_scoring_from_checkpoint`]
/// for the sum variant; defaults are mean for backward compatibility
/// with the Phase 6 K9 measurement.
///
/// `LogitCritic` owns a `VarMap` + `GPT` so it can be used standalone
/// for measurement (e.g. `examples/critic_baseline.rs`). For a
/// production self-improve loop you'd typically wrap this around the
/// same `ModelActor` the generator uses, sharing the VarMap to avoid
/// double the GPU memory.
pub struct LogitCritic {
    /// Kept alive so the model's tensors stay valid.
    _varmap: VarMap,
    model: GPT,
    tokenizer: Arc<Tokenizer>,
    device: Device,
    /// `true` (default) = mean log-prob per token; `false` = raw sum.
    /// **Use `false` (sum) on length-varying domains.** See type-level
    /// docs and `docs/phase7-design.md`.
    pub normalize_by_length: bool,
}

impl LogitCritic {
    /// Load a model from disk and wrap as a `Critic`.
    pub fn from_checkpoint(
        cfg: GPTConfig,
        tokenizer: Arc<Tokenizer>,
        device: Device,
        path: &Path,
    ) -> anyhow::Result<Self> {
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = GPT::new(cfg, vb)?;
        varmap.load(path)?;
        Ok(Self {
            _varmap: varmap,
            model,
            tokenizer,
            device,
            normalize_by_length: true,
        })
    }

    /// Phase 7 ergonomic helper: load with `normalize_by_length = false`
    /// (sum log-prob) for length-varying domains. Equivalent to calling
    /// `from_checkpoint` and then setting `normalize_by_length = false`,
    /// but the constructor name documents intent at the call site.
    pub fn sum_scoring_from_checkpoint(
        cfg: GPTConfig,
        tokenizer: Arc<Tokenizer>,
        device: Device,
        path: &Path,
    ) -> anyhow::Result<Self> {
        let mut me = Self::from_checkpoint(cfg, tokenizer, device, path)?;
        me.normalize_by_length = false;
        Ok(me)
    }

    /// Sum-variant of [`LogitCritic::from_model`].
    pub fn sum_scoring_from_model(
        varmap: VarMap,
        model: GPT,
        tokenizer: Arc<Tokenizer>,
        device: Device,
    ) -> Self {
        let mut me = Self::from_model(varmap, model, tokenizer, device);
        me.normalize_by_length = false;
        me
    }

    /// Wrap an already-built model. Caller is responsible for keeping
    /// the VarMap alive (we take ownership) and ensuring the tokenizer
    /// matches the model's vocab.
    pub fn from_model(
        varmap: VarMap,
        model: GPT,
        tokenizer: Arc<Tokenizer>,
        device: Device,
    ) -> Self {
        Self {
            _varmap: varmap,
            model,
            tokenizer,
            device,
            normalize_by_length: true,
        }
    }
}

impl Critic for LogitCritic {
    fn score(&self, prompt: &str, completion: &str) -> f32 {
        let prompt_ids = match self.tokenizer.encode(prompt) {
            Ok(ids) if !ids.is_empty() => ids,
            // Empty prompt or tokenization failure → treat as "unknown".
            // Returning -inf pushes this candidate to the bottom of any
            // ranking, which is the safe default.
            _ => return f32::NEG_INFINITY,
        };
        let completion_ids = match self.tokenizer.encode(completion) {
            Ok(ids) => ids,
            Err(_) => return f32::NEG_INFINITY,
        };
        if completion_ids.is_empty() {
            return 0.0;
        }
        // Truncate prompt's tail-end to fit if the combined sequence
        // exceeds block_size — mirrors the generator's behavior.
        let block = self.model.block_size();
        let total = prompt_ids.len() + completion_ids.len();
        let (p, c): (&[u32], &[u32]) = if total > block {
            let keep_prompt = block.saturating_sub(completion_ids.len()).max(1);
            let p_start = prompt_ids.len().saturating_sub(keep_prompt);
            (&prompt_ids[p_start..], &completion_ids[..])
        } else {
            (&prompt_ids[..], &completion_ids[..])
        };
        match self.model.sequence_log_prob(p, c, &self.device) {
            Ok(sum_lp) => {
                if self.normalize_by_length {
                    sum_lp / c.len() as f32
                } else {
                    sum_lp
                }
            }
            Err(_) => f32::NEG_INFINITY,
        }
    }
}

/// Compute the area under the ROC curve for binary `(score, label)`
/// pairs. `label = true` for cargo-correct, `false` for cargo-rejected.
/// Returns the probability that a random positive scores higher than a
/// random negative; `0.5` = no signal, `1.0` = perfect ranker.
///
/// Implementation is the Mann-Whitney U variant: sort by score and
/// count concordant pairs. Ties contribute 0.5. O(n log n) time,
/// O(n) space.
pub fn roc_auc(scores: &[(f32, bool)]) -> f32 {
    let n_pos = scores.iter().filter(|(_, l)| *l).count();
    let n_neg = scores.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        // AUC undefined when one class is missing.
        return f32::NAN;
    }
    let mut indexed: Vec<(usize, f32, bool)> = scores
        .iter()
        .enumerate()
        .map(|(i, (s, l))| (i, *s, *l))
        .collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    // Compute mid-ranks (1-indexed). Tied scores share the average rank.
    let mut ranks = vec![0.0f32; scores.len()];
    let n = indexed.len();
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && indexed[j + 1].1 == indexed[i].1 {
            j += 1;
        }
        let avg_rank = ((i + 1) as f32 + (j + 1) as f32) / 2.0;
        for k in i..=j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j + 1;
    }
    let sum_ranks_pos: f32 = scores
        .iter()
        .enumerate()
        .filter(|(_, (_, l))| *l)
        .map(|(idx, _)| ranks[idx])
        .sum();
    let u = sum_ranks_pos - (n_pos as f32 * (n_pos as f32 + 1.0)) / 2.0;
    u / (n_pos as f32 * n_neg as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn always_correct_returns_one() {
        let c = AlwaysCorrectCritic;
        assert_eq!(c.score("any", "thing"), 1.0);
        assert_eq!(c.score("", ""), 1.0);
    }

    #[test]
    fn random_critic_is_deterministic_per_seed() {
        let c1 = RandomCritic::new(42);
        let c2 = RandomCritic::new(42);
        // Same seed, called in the same order → same scores.
        assert_eq!(c1.score("p", "x"), c2.score("p", "x"));
        assert_eq!(c1.score("p", "y"), c2.score("p", "y"));
    }

    #[test]
    fn random_critic_scores_in_unit_interval() {
        let c = RandomCritic::new(0);
        for i in 0..32 {
            let s = c.score(&format!("prompt-{i}"), "completion");
            assert!(
                (0.0..1.0).contains(&s),
                "score {s} out of [0, 1) at iter {i}"
            );
        }
    }

    /// Trait can be passed as `&dyn Critic`. Guards against the trait
    /// accidentally requiring `Sized` etc.
    #[test]
    fn dyn_dispatch_compiles() {
        fn take_dyn(c: &dyn Critic) -> f32 {
            c.score("", "")
        }
        assert_eq!(take_dyn(&AlwaysCorrectCritic), 1.0);
        let r = RandomCritic::new(7);
        let s = take_dyn(&r);
        assert!((0.0..1.0).contains(&s));
    }

    #[test]
    fn auc_perfect_ranker_is_one() {
        // All positives strictly greater than all negatives.
        let pairs: Vec<(f32, bool)> = vec![
            (0.1, false),
            (0.2, false),
            (0.3, false),
            (0.7, true),
            (0.8, true),
            (0.9, true),
        ];
        assert!((roc_auc(&pairs) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn auc_inverted_is_zero() {
        let pairs: Vec<(f32, bool)> = vec![
            (0.9, false),
            (0.8, false),
            (0.7, false),
            (0.3, true),
            (0.2, true),
            (0.1, true),
        ];
        assert!(roc_auc(&pairs).abs() < 1e-6);
    }

    #[test]
    fn auc_ties_contribute_half() {
        // All scores identical — pure tie. AUC = 0.5.
        let pairs: Vec<(f32, bool)> = vec![(0.5, true), (0.5, false), (0.5, true), (0.5, false)];
        let auc = roc_auc(&pairs);
        assert!((auc - 0.5).abs() < 1e-6, "got {auc}");
    }

    #[test]
    fn auc_degenerate_one_class_is_nan() {
        let pairs: Vec<(f32, bool)> = vec![(0.1, true), (0.5, true)];
        assert!(roc_auc(&pairs).is_nan());
    }

    /// Phase 7 ergonomic API: the sum-scoring constructor produces a
    /// LogitCritic with `normalize_by_length = false`. We test the
    /// invariant via from_model (which doesn't require a real
    /// checkpoint on disk) — the from_checkpoint variant is the same
    /// pattern, validated indirectly when examples run.
    #[test]
    fn sum_scoring_constructor_disables_length_normalization() {
        use candle_core::{DType, Device};
        use candle_nn::{VarBuilder, VarMap};
        use nanogpt_rs::config::{ActivationKind, GPTConfig, NormKind, NormPosition};

        let cfg = GPTConfig {
            vocab_size: 8,
            block_size: 4,
            n_layer: 2,
            n_head: 2,
            n_embd: 16,
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
        };
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = nanogpt_rs::model::GPT::new(cfg, vb).unwrap();
        let tk = Arc::new(Tokenizer::char_from_text("abcdefgh"));

        let mean_critic = LogitCritic::from_model(VarMap::new(), model, tk.clone(), device.clone());
        assert!(
            mean_critic.normalize_by_length,
            "default constructor must produce mean variant"
        );

        // sum_scoring_from_model needs its own VarMap+model, since
        // mean_critic took ownership of the previous one.
        let varmap2 = VarMap::new();
        let vb2 = VarBuilder::from_varmap(&varmap2, DType::F32, &device);
        let model2 = nanogpt_rs::model::GPT::new(
            GPTConfig {
                vocab_size: 8,
                block_size: 4,
                n_layer: 2,
                n_head: 2,
                n_embd: 16,
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
            },
            vb2,
        )
        .unwrap();
        let sum_critic = LogitCritic::sum_scoring_from_model(varmap2, model2, tk, device);
        assert!(
            !sum_critic.normalize_by_length,
            "sum_scoring constructor must disable length normalization"
        );
    }
}
