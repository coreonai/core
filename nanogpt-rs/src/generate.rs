//! Sampling: temperature, top-k, top-p (nucleus).

use candle_core::{DType, Device, Tensor};
use candle_nn::ops;
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::error::Result;
use crate::model::GPT;

#[derive(Debug, Clone)]
pub struct GenerateConfig {
    pub max_new_tokens: usize,
    pub temperature: f64,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub seed: Option<u64>,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            temperature: 1.0,
            top_k: Some(50),
            top_p: None,
            seed: None,
        }
    }
}

/// Autoregressive sampler.
pub fn generate(
    model: &GPT,
    prompt_ids: &[u32],
    cfg: &GenerateConfig,
    device: &Device,
) -> Result<Vec<u32>> {
    let mut rng: StdRng = match cfg.seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let block_size = model.block_size();
    let mut tokens: Vec<u32> = prompt_ids.to_vec();

    for _ in 0..cfg.max_new_tokens {
        let context: &[u32] = if tokens.len() <= block_size {
            &tokens[..]
        } else {
            &tokens[tokens.len() - block_size..]
        };
        let input = Tensor::from_vec(context.to_vec(), (1, context.len()), device)?;
        let logits = model.forward_last(&input)?; // (1, vocab)
        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
        let next = sample_logits(&logits, cfg, &mut rng)?;
        tokens.push(next);
    }
    Ok(tokens)
}

fn sample_logits(logits: &Tensor, cfg: &GenerateConfig, rng: &mut StdRng) -> Result<u32> {
    // temperature == 0 → greedy. Avoid divide-by-zero (which produces ±inf
    // logits and silently collapses sampling to whichever token has the
    // largest *positive* raw logit — usually a frequent structural token).
    if cfg.temperature <= 0.0 {
        let v = logits.to_vec1::<f32>()?;
        let argmax = v
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        return Ok(argmax);
    }

    let logits = if cfg.temperature != 1.0 {
        (logits / cfg.temperature)?
    } else {
        logits.clone()
    };

    // top-k filtering
    let logits = if let Some(k) = cfg.top_k {
        let v = logits.to_vec1::<f32>()?;
        let k = k.min(v.len());
        if k > 0 && k < v.len() {
            let mut sorted = v.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let kth = sorted[k - 1];
            let masked: Vec<f32> = v
                .into_iter()
                .map(|x| if x < kth { f32::NEG_INFINITY } else { x })
                .collect();
            Tensor::from_vec(masked, logits.shape(), logits.device())?
        } else {
            logits
        }
    } else {
        logits
    };

    let probs = ops::softmax_last_dim(&logits)?;
    let mut probs_v: Vec<f32> = probs.to_vec1()?;

    // top-p (nucleus) filtering
    if let Some(p) = cfg.top_p {
        let p = p as f32;
        let mut idx: Vec<usize> = (0..probs_v.len()).collect();
        idx.sort_by(|a, b| {
            probs_v[*b]
                .partial_cmp(&probs_v[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut cumsum = 0.0;
        let mut keep = vec![false; probs_v.len()];
        for i in &idx {
            keep[*i] = true;
            cumsum += probs_v[*i];
            if cumsum >= p {
                break;
            }
        }
        for (i, k) in keep.iter().enumerate() {
            if !k {
                probs_v[i] = 0.0;
            }
        }
        let s: f32 = probs_v.iter().sum();
        if s > 0.0 {
            for v in probs_v.iter_mut() {
                *v /= s;
            }
        }
    }

    // Guard: any non-finite or all-zero -> fallback to argmax
    if !probs_v.iter().all(|x| x.is_finite()) || probs_v.iter().all(|x| *x <= 0.0) {
        let argmax = probs_v
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        return Ok(argmax);
    }

    let dist =
        WeightedIndex::new(&probs_v).map_err(|e| crate::error::Error::Data(e.to_string()))?;
    Ok(dist.sample(rng) as u32)
}
