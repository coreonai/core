//! nanogpt-rs: Rust-native nanoGPT on Candle.
//!
//! Module layout:
//! - [`config`]    GPTConfig
//! - [`model`]     GPT, Block, CausalSelfAttention, MLP
//! - [`tokenizer`] HuggingFace BPE wrapper + char-level fallback
//! - [`data`]      Token streaming dataset
//! - [`train`]     AdamW + cosine LR + grad-accum loop
//! - [`generate`]  top-k / top-p / temperature sampling
//! - [`error`]     crate-wide Result/Error

pub mod config;
pub mod data;
pub mod error;
pub mod ewc;
pub mod generate;
pub mod model;
pub mod tokenizer;
pub mod train;

pub use config::GPTConfig;
pub use error::{Error, Result};
pub use generate::{GenerateConfig, generate};
pub use model::GPT;
pub use tokenizer::Tokenizer;
