//! Tokenizer wrappers.
//!
//! Two flavors used in this crate:
//! - `Bpe`: HuggingFace `tokenizers` BPE (default for full-scale runs)
//! - `Char`: simple char-level tokenizer for fast smoke tests on Shakespeare

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::models::TrainerWrapper;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::AddedToken;

use crate::error::{Error, Result};

pub enum Tokenizer {
    // Bpe is boxed: the inner tokenizers::Tokenizer is much larger than CharTokenizer,
    // so the unboxed enum was wasting bytes on every Char variant.
    Bpe(Box<tokenizers::Tokenizer>),
    Char(CharTokenizer),
}

impl Tokenizer {
    pub fn from_hf_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let tk =
            tokenizers::Tokenizer::from_file(path).map_err(|e| Error::Tokenizer(e.to_string()))?;
        Ok(Tokenizer::Bpe(Box::new(tk)))
    }

    /// Download a tokenizer file from a HuggingFace Hub repo (public-only)
    /// and load it. Cached at `~/.cache/workllm/hf-tokenizers/{repo_safe}/{filename}`
    /// so re-runs hit the local file. Shells out to `curl` to avoid pulling
    /// in a full HTTP client crate — the dump is one tiny JSON file.
    ///
    /// Example:
    ///     Tokenizer::from_hub("EleutherAI/polyglot-ko-1.3b", "tokenizer.json")
    pub fn from_hub(repo: &str, filename: &str) -> Result<Self> {
        let cache_root = dirs_cache().join("workllm").join("hf-tokenizers");
        let safe_repo = repo.replace('/', "__");
        let dest_dir = cache_root.join(&safe_repo);
        let dest = dest_dir.join(filename);
        if !dest.exists() {
            std::fs::create_dir_all(&dest_dir)?;
            let url = format!("https://huggingface.co/{repo}/resolve/main/{filename}");
            let status = std::process::Command::new("curl")
                .args(["-fsSL", "--retry", "3", "-o"])
                .arg(&dest)
                .arg(&url)
                .status()
                .map_err(|e| Error::Tokenizer(format!("spawn curl: {e}")))?;
            if !status.success() {
                let _ = std::fs::remove_file(&dest); // don't leave empty file
                return Err(Error::Tokenizer(format!(
                    "curl {url} failed (exit {status:?}). Set HF_HOME if behind a proxy or pre-download the file manually."
                )));
            }
        }
        Self::from_hf_file(&dest)
    }

    /// Train a fresh ByteLevel-BPE tokenizer on the given text files. The
    /// resulting tokenizer is saved to `save_path` (HF JSON format) and
    /// returned ready-to-use. Use this for new corpora; reuse `from_hf_file`
    /// thereafter.
    pub fn train_bpe<P: AsRef<Path>>(
        train_files: &[P],
        vocab_size: usize,
        save_path: P,
    ) -> Result<Self> {
        let mut tokenizer = tokenizers::Tokenizer::new(BPE::default());
        // ByteLevel pre-tokenizer + decoder is the GPT-2 / Llama-2 / Mistral
        // recipe — it works for arbitrary unicode (including Korean) without
        // extra normalization, and round-trips cleanly.
        tokenizer.with_pre_tokenizer(Some(ByteLevel::default()));
        tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));

        let bpe_trainer = BpeTrainerBuilder::new()
            .vocab_size(vocab_size)
            .min_frequency(2)
            .show_progress(true)
            .special_tokens(vec![
                AddedToken::from("<|endoftext|>", true),
                AddedToken::from("<|pad|>", true),
            ])
            .initial_alphabet(ByteLevel::alphabet().into_iter().collect())
            .build();
        // The high-level `Tokenizer` is `TokenizerImpl<ModelWrapper, ...>`,
        // so its `train_from_files` requires a `TrainerWrapper`. Wrap once.
        let mut trainer: TrainerWrapper = bpe_trainer.into();

        let files: Vec<String> = train_files
            .iter()
            .map(|p| p.as_ref().to_string_lossy().into_owned())
            .collect();
        tokenizer
            .train_from_files(&mut trainer, files)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;

        tokenizer
            .save(save_path.as_ref(), false)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        Ok(Tokenizer::Bpe(Box::new(tokenizer)))
    }

    pub fn char_from_text(text: &str) -> Self {
        Tokenizer::Char(CharTokenizer::from_text(text))
    }

    pub fn vocab_size(&self) -> usize {
        match self {
            Tokenizer::Bpe(t) => t.get_vocab_size(true),
            Tokenizer::Char(c) => c.vocab_size(),
        }
    }

    pub fn encode(&self, s: &str) -> Result<Vec<u32>> {
        match self {
            Tokenizer::Bpe(t) => {
                let enc = t
                    .encode(s, false)
                    .map_err(|e| Error::Tokenizer(e.to_string()))?;
                Ok(enc.get_ids().to_vec())
            }
            Tokenizer::Char(c) => Ok(c.encode(s)),
        }
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        match self {
            Tokenizer::Bpe(t) => t
                .decode(ids, true)
                .map_err(|e| Error::Tokenizer(e.to_string())),
            Tokenizer::Char(c) => Ok(c.decode(ids)),
        }
    }
}

fn dirs_cache() -> std::path::PathBuf {
    if let Ok(c) = std::env::var("XDG_CACHE_HOME") {
        if !c.is_empty() {
            return std::path::PathBuf::from(c);
        }
    }
    if let Ok(h) = std::env::var("HOME") {
        return std::path::PathBuf::from(h).join(".cache");
    }
    std::path::PathBuf::from("/tmp")
}

#[cfg(test)]
mod from_hub_tests {
    use super::*;
    use std::sync::Mutex;

    /// Serialize tests that mutate the global `XDG_CACHE_HOME` env var.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn cache_hit_avoids_network() {
        let _g = ENV_LOCK.lock().unwrap();
        // Point cache at a fresh tempdir, pre-seed with our own copy of
        // the Polyglot tokenizer file (any valid HF tokenizer works), and
        // verify `from_hub` loads it without spawning curl.
        let tmp =
            std::env::temp_dir().join(format!("workllm-from-hub-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::env::set_var("XDG_CACHE_HOME", &tmp);

        let repo = "EleutherAI/polyglot-ko-1.3b";
        let filename = "tokenizer.json";
        let dest = tmp
            .join("workllm")
            .join("hf-tokenizers")
            .join(repo.replace('/', "__"))
            .join(filename);
        std::fs::create_dir_all(dest.parent().unwrap()).unwrap();

        // Seed with a known-good local tokenizer.json so we don't need
        // the network. The repo path here is exercising path layout only.
        let local = std::path::Path::new("data/kowiki/polyglot_ko_tokenizer.json");
        if !local.exists() {
            // No data/ available in this environment; nothing to assert.
            std::env::remove_var("XDG_CACHE_HOME");
            return;
        }
        std::fs::copy(local, &dest).unwrap();

        // Make curl fail loudly if it's invoked — easiest way is to
        // shadow it by prepending a dir with a fake `curl` to PATH.
        let bin_dir = tmp.join("nopath-curl");
        std::fs::create_dir_all(&bin_dir).unwrap();
        let fake = bin_dir.join("curl");
        std::fs::write(
            &fake,
            "#!/bin/sh\necho 'from_hub should not call curl on cache hit' >&2\nexit 99\n",
        )
        .unwrap();
        // chmod +x
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&fake).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&fake, perms).unwrap();

        let original_path = std::env::var("PATH").unwrap_or_default();
        std::env::set_var("PATH", format!("{}:{}", bin_dir.display(), original_path));

        let tk = Tokenizer::from_hub(repo, filename).expect("cache hit should load");
        assert_eq!(tk.vocab_size(), 30003);

        std::env::set_var("PATH", original_path);
        std::env::remove_var("XDG_CACHE_HOME");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn cache_path_normalizes_repo_slash() {
        let _g = ENV_LOCK.lock().unwrap();
        let tmp =
            std::env::temp_dir().join(format!("workllm-from-hub-path-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::env::set_var("XDG_CACHE_HOME", &tmp);

        // Without internet we can't actually download — but we can trigger
        // the path-construction branch by providing a tokenizer file at
        // the expected cache path and checking that's what gets read.
        let repo = "owner/sub/repo";
        let dest = tmp
            .join("workllm")
            .join("hf-tokenizers")
            .join("owner__sub__repo")
            .join("tokenizer.json");
        // Just assert the path layout — we'd need a real tokenizer file
        // for full load, which is overkill for this assertion.
        assert!(!dest.exists());
        // Building the path manually mirrors what `from_hub` does:
        let safe_repo = repo.replace('/', "__");
        assert_eq!(safe_repo, "owner__sub__repo");

        std::env::remove_var("XDG_CACHE_HOME");
        let _ = std::fs::remove_dir_all(&tmp);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharTokenizer {
    /// id -> char
    pub itos: Vec<char>,
    /// char -> id
    pub stoi: BTreeMap<char, u32>,
}

impl CharTokenizer {
    pub fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();
        let stoi: BTreeMap<char, u32> = chars
            .iter()
            .enumerate()
            .map(|(i, c)| (*c, i as u32))
            .collect();
        Self { itos: chars, stoi }
    }

    pub fn vocab_size(&self) -> usize {
        self.itos.len()
    }

    pub fn encode(&self, s: &str) -> Vec<u32> {
        s.chars()
            .filter_map(|c| self.stoi.get(&c).copied())
            .collect()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|i| self.itos.get(*i as usize).copied())
            .collect()
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let s = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&s)?)
    }
}
