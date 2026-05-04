use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("candle: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("tokenizer: {0}")]
    Tokenizer(String),

    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("serde: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("config: {0}")]
    Config(String),

    #[error("data: {0}")]
    Data(String),
}

pub type Result<T> = std::result::Result<T, Error>;
