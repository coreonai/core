//! Token-stream dataset.
//!
//! Holds a flat `Vec<u32>` of token ids and produces random
//! (idx, target) batches of shape `(B, T)`.

use candle_core::{Device, Tensor};
use rand::Rng;

use crate::error::Result;

pub struct TokenDataset {
    pub tokens: Vec<u32>,
    pub block_size: usize,
}

impl TokenDataset {
    pub fn new(tokens: Vec<u32>, block_size: usize) -> Self {
        Self { tokens, block_size }
    }

    pub fn len(&self) -> usize {
        self.tokens.len().saturating_sub(self.block_size + 1)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Sample one batch (B, T) of inputs and targets.
    pub fn random_batch(&self, batch_size: usize, device: &Device) -> Result<(Tensor, Tensor)> {
        let n = self.len();
        assert!(
            n > 0,
            "dataset is too small for block_size {}",
            self.block_size
        );

        let mut rng = rand::thread_rng();
        let t = self.block_size;
        let mut x = Vec::with_capacity(batch_size * t);
        let mut y = Vec::with_capacity(batch_size * t);
        for _ in 0..batch_size {
            let start = rng.gen_range(0..n);
            x.extend_from_slice(&self.tokens[start..start + t]);
            y.extend_from_slice(&self.tokens[start + 1..start + 1 + t]);
        }
        let x_t = Tensor::from_vec(x, (batch_size, t), device)?;
        let y_t = Tensor::from_vec(y, (batch_size, t), device)?;
        Ok((x_t, y_t))
    }

    /// Split off the last `frac` fraction as validation.
    pub fn split_train_val(self, frac: f32) -> (Self, Self) {
        let n = self.tokens.len();
        let split = ((1.0 - frac as f64) * n as f64) as usize;
        let train = self.tokens[..split].to_vec();
        let val = self.tokens[split..].to_vec();
        (
            Self::new(train, self.block_size),
            Self::new(val, self.block_size),
        )
    }
}
