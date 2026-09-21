//! Phase 21 Stage D — actor wrapper around `candle_transformers::models::qwen2`.
//!
//! Bridges Phase 14-20's production model (Qwen2.5-Coder-0.5B served
//! through HuggingFace transformers in Python) into the Rust `llm-actors`
//! framework. The smoke binary `phase21_qwen_candle_smoke` proves Candle
//! can load + serve the model natively; this actor wraps that into the
//! same `ModelMessage` enum the rest of the actor pipeline already uses.
//!
//! ## Supported `ModelMessage` variants
//! - `Ping` — health check (trivial)
//! - `Generate { prompt, cfg, reply }` — HF-tokenize → generate → decode
//! - `GenerateTokens { prompt_ids, cfg, reply }` — raw HF token IDs in/out
//! - `ReloadCheckpoint { path, reply }` — re-load `model.safetensors`
//!
//! ## Implemented (Phase 22 follow-ups)
//! - `ScoreLogProb` — uses the same KV-cache + last-position forward
//!   trick as `generate_autoregressive`: feed prompt → last logits =
//!   P(next | prompt) → look up `completion[0]` log-prob → feed
//!   `completion[0]` → last logits = P(next | prompt + comp[..=0]) →
//!   ... etc. Returns the **mean** log-prob per completion token
//!   matching `ModelActor`'s semantics. Unlocks Phase 6 Shape C
//!   best-of-K filter (`--gen-oversample > 1`) on QwenModelActor.
//! - `LossOn` — slow per-position cross-entropy using the same
//!   KV-cache pattern. For (B, T) input/target tensors, runs T
//!   forward steps; at each step computes -log_softmax(logits)[y[:, t]]
//!   and averages over (B × T). Wallclock: ~T × 50ms (~12 s at T=256).
//!   Acceptable for evaluation (PPL); not designed for hot training
//!   paths. Unlocks OPD / multi-teacher distillation against
//!   `QwenModelActor`.
//!
//! ## NOT covered by this actor
//! - Training (LoRA / SFT). The Trainer actor's path is heavily tied
//!   to `nanogpt_rs::GPT` + Candle VarMap. A Qwen-side training stack
//!   is its own multi-day project (deferred to Phase 21 Stage E+).
//!
//! ## Type compatibility
//! `QwenModelActor::Message == ModelMessage` so the enum is shared with
//! `ModelActor`. But `ActorRef<QwenModelActor>` and `ActorRef<ModelActor>`
//! are different types — callers like `EvaluatorActor` that hold
//! `ActorRef<ModelActor>` cannot directly accept a Qwen actor. Plumbing
//! the eval/gen actors to be generic over the model type is Stage E.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
// The vendored model, not candle's: upstream casts rotary position indices
// to the model dtype, which silently destroys positions past 256 in BF16.
// `Config` is candle's, re-exported, so every call site that deserializes
// config.json is unaffected. See `crate::qwen2_f32rope`.
use crate::qwen2_f32rope::ModelForCausalLM;
use candle_transformers::models::qwen2::Config as Qwen2Config;
use nanogpt_rs::generate::GenerateConfig;
use pekko_actor::{Actor, ActorContext};
use rand::rngs::StdRng;
use rand::SeedableRng;
use tokenizers::Tokenizer as HfTokenizer;
use tokio::sync::oneshot;
use tracing::{error, info, warn};

use crate::model_actor::{GenerateReply, ModelMessage};
use crate::qwen2_lora::resolve_safetensors;

pub struct QwenModelActor {
    /// `Option` so `ReloadCheckpoint` can drop the old model (freeing its
    /// GPU memory) BEFORE building the replacement. For 7B the base is ~15GB
    /// and a co-resident trainer holds another ~15GB, so loading the new
    /// model while the old is still resident would need 3×15=45GB and OOM a
    /// 40GB card. Always `Some` outside of a reload.
    pub model: Option<ModelForCausalLM>,
    pub tokenizer: Arc<HfTokenizer>,
    pub config: Qwen2Config,
    pub device: Device,
    pub dtype: DType,
    /// Token ids masked to -inf before sampling. Empty by default, so every
    /// existing caller is unaffected.
    ///
    /// Why this exists: on the Phase 23 tool-call probe the base 7B lands
    /// `(arith` and then samples a *special* token (`<|fim_prefix|>`) where
    /// ` add` belongs, wrecking the call 83% of the time. The tokenizer
    /// round-trips the target string fine and ` add` is a single token, so
    /// this is real sampling, not a decode artifact — Qwen-Coder base
    /// reaching for its FIM machinery on an unfamiliar format. Masking the
    /// special ids at sample time is the cheapest way to find out whether the
    /// exact-call rate is recoverable by decoding alone, before concluding
    /// the format needs SFT.
    ///
    /// Lives on the actor rather than `GenerateConfig` deliberately: that
    /// struct is built by 47 literal sites with no `..Default::default()`,
    /// so adding a field there is pure churn for an experiment.
    pub suppress_tokens: Vec<u32>,
    /// Stop decoding as soon as one of these appears in the generated text.
    ///
    /// Not cosmetic — it is the single largest waste on the serving path.
    /// Decoding costs ~26 ms/token for the 7B in F32, and the agentic loop
    /// cuts each chunk at a stop sequence *after* generation returns. So a
    /// 20-token tool call under `max_new_tokens = 64` paid for 44 tokens that
    /// were thrown away: ~1.1 s per loop step, twice or three times per
    /// answer. Checking here ends the forward passes instead of the string.
    ///
    /// Empty by default, so nothing that does not set it changes behaviour.
    pub stop_sequences: Vec<String>,
    /// Path of the most recently loaded safetensors checkpoint. Used by
    /// `ReloadCheckpoint` to re-initialize the model.
    pub model_path: std::path::PathBuf,
}

impl QwenModelActor {
    /// Build a fresh actor by mmap-loading the safetensors at `model_path`
    /// into a Qwen2 `ModelForCausalLM`. `model_path` may be a single
    /// `.safetensors` file (a merged checkpoint) OR an HF snapshot
    /// directory whose weights are sharded across
    /// `model-0000N-of-0000M.safetensors` (resolved via
    /// `resolve_safetensors`). The tokenizer and config are expected to
    /// live next to the safetensors but the caller passes them in
    /// explicitly so this constructor is testable.
    pub fn new(
        model_path: std::path::PathBuf,
        tokenizer: Arc<HfTokenizer>,
        config: Qwen2Config,
        device: Device,
        dtype: DType,
    ) -> anyhow::Result<Self> {
        let model = load_qwen_model(&resolve_safetensors(&model_path)?, &config, dtype, &device)?;
        Ok(Self {
            model: Some(model),
            tokenizer,
            config,
            device,
            dtype,
            model_path,
            suppress_tokens: Vec::new(),
            stop_sequences: Vec::new(),
        })
    }

    /// Convenience loader: read `config.json` + `tokenizer.json` +
    /// `model.safetensors` from a single HF snapshot directory.
    /// Mask these token ids out of every sample. Builder-style.
    /// Stop decoding at any of these. See [`Self::stop_sequences`].
    pub fn with_stop_sequences<I, S>(mut self, seqs: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.stop_sequences = seqs
            .into_iter()
            .map(Into::into)
            .filter(|s: &String| !s.is_empty())
            .collect();
        self
    }

    pub fn with_suppressed_tokens(mut self, ids: Vec<u32>) -> Self {
        self.suppress_tokens = ids;
        self
    }

    pub fn from_snapshot_dir(
        snapshot_dir: &Path,
        device: Device,
        dtype: DType,
    ) -> anyhow::Result<Self> {
        let cfg_text = std::fs::read_to_string(snapshot_dir.join("config.json"))?;
        let config: Qwen2Config = serde_json::from_str(&cfg_text)?;
        let tokenizer = HfTokenizer::from_file(snapshot_dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        // Pass the snapshot directory itself; `new` → `resolve_safetensors`
        // picks the single `model.safetensors` (0.5B/1.5B) or the shard set
        // listed in `model.safetensors.index.json` (7B).
        Self::new(
            snapshot_dir.to_path_buf(),
            Arc::new(tokenizer),
            config,
            device,
            dtype,
        )
    }
}

fn load_qwen_model(
    model_paths: &[PathBuf],
    config: &Qwen2Config,
    dtype: DType,
    device: &Device,
) -> anyhow::Result<ModelForCausalLM> {
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(model_paths, dtype, device)? };
    let model = ModelForCausalLM::new(config, vb)?;
    Ok(model)
}

impl Actor for QwenModelActor {
    type Message = ModelMessage;

    fn receive(
        &mut self,
        msg: Self::Message,
        _ctx: &mut ActorContext<Self>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + '_>> {
        Box::pin(async move {
            match msg {
                ModelMessage::Ping { reply } => {
                    let _ = reply.send(());
                }
                ModelMessage::Generate { prompt, cfg, reply } => {
                    let result = self.handle_generate(&prompt, &cfg);
                    log_send(reply, result, "qwen_generate");
                }
                ModelMessage::GenerateTokens {
                    prompt_ids,
                    cfg,
                    reply,
                } => {
                    let result = self.handle_generate_tokens(&prompt_ids, &cfg);
                    log_send(reply, result, "qwen_generate_tokens");
                }
                ModelMessage::ReloadCheckpoint { path, reply } => {
                    let result = self.handle_reload(&path);
                    log_send(reply, result, "qwen_reload_checkpoint");
                }
                ModelMessage::LossOn { x, y, reply } => {
                    let result = self.handle_loss_on(&x, &y);
                    log_send(reply, result, "qwen_loss_on");
                }
                ModelMessage::GenerateBatch {
                    prompt_ids,
                    n_samples,
                    cfg,
                    reply,
                } => {
                    let result = self.generate_batch(&prompt_ids, n_samples.max(1), &cfg);
                    log_send(reply, result, "qwen_generate_batch");
                }
                ModelMessage::GenerateBatchPrompts {
                    prompts,
                    seeds,
                    cfg,
                    reply,
                } => {
                    let result = self.generate_batch_prompts(&prompts, &seeds, &cfg);
                    log_send(reply, result, "qwen_generate_batch_prompts");
                }
                ModelMessage::ScoreLogProb {
                    prompt_ids,
                    completion_ids,
                    reply,
                } => {
                    let result = self.handle_score_log_prob(&prompt_ids, &completion_ids);
                    log_send(reply, result, "qwen_score_log_prob");
                }
            }
        })
    }

    fn pre_start(
        &mut self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + '_>> {
        Box::pin(async {
            info!("QwenModelActor started");
        })
    }
}

impl QwenModelActor {
    fn handle_generate(
        &mut self,
        prompt: &str,
        cfg: &GenerateConfig,
    ) -> anyhow::Result<GenerateReply> {
        let encoded = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        let prompt_ids: Vec<u32> = encoded.get_ids().to_vec();
        let tokens = self.generate_autoregressive(&prompt_ids, cfg)?;
        let comp_ids = if tokens.len() > prompt_ids.len() {
            &tokens[prompt_ids.len()..]
        } else {
            &[][..]
        };
        let text = self
            .tokenizer
            .decode(comp_ids, true)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
        Ok(GenerateReply { text, tokens })
    }

    fn handle_generate_tokens(
        &mut self,
        prompt_ids: &[u32],
        cfg: &GenerateConfig,
    ) -> anyhow::Result<Vec<u32>> {
        self.generate_autoregressive(prompt_ids, cfg)
    }

    /// Mutable access to the loaded model, erroring if a reload left it empty
    /// (should never happen outside `handle_reload`).
    fn model_mut(&mut self) -> anyhow::Result<&mut ModelForCausalLM> {
        self.model
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("QwenModelActor model not loaded"))
    }

    fn handle_reload(&mut self, path: &Path) -> anyhow::Result<()> {
        let paths = resolve_safetensors(path)?;
        // Free the old model's GPU memory BEFORE building the new one, so the
        // reload reuses the freed allocations (peak ~15GB) instead of holding
        // old+new simultaneously (~30GB) on top of a co-resident trainer.
        self.model = None;
        let new_model = load_qwen_model(&paths, &self.config, self.dtype, &self.device)?;
        self.model = Some(new_model);
        self.model_path = path.to_path_buf();
        info!(?path, "QwenModelActor checkpoint reloaded");
        Ok(())
    }

    /// Per-position cross-entropy. Input `x` and target `y` are
    /// `(B, T)` u32 token tensors. For each position `t ∈ 0..T`, we
    /// forward `x[:, t]` (shape `(B, 1)`) at `seqlen_offset = t` to
    /// get logits `(B, 1, vocab)`, narrowed by qwen2's forward to
    /// the last position. We log_softmax along vocab, gather
    /// `-log P(y[:, t])`, and accumulate. Final return is the mean
    /// over `B × T`, matching `ModelActor::LossOn` semantics.
    ///
    /// Wallclock: T sequential forwards on the GPU. At T=256 and
    /// ~50 ms per forward step, ~13 s per LossOn call. Acceptable for
    /// PPL evaluation; not designed for hot training paths (a true
    /// all-position forward + lm_head over `(B, T, hidden)` would
    /// require forking `candle_transformers::models::qwen2` like
    /// Stage F did for LoRA training).
    fn handle_loss_on(&mut self, x: &Tensor, y: &Tensor) -> anyhow::Result<f32> {
        let (b, t) = x.dims2()?;
        if t == 0 {
            return Ok(0.0);
        }
        let (yb, yt) = y.dims2()?;
        anyhow::ensure!(
            yb == b && yt == t,
            "LossOn shape mismatch: x={b}x{t}, y={yb}x{yt}"
        );
        self.model_mut()?.clear_kv_cache();
        let mut total_loss = 0.0f64;
        for (seqlen_offset, pos) in (0..t).enumerate() {
            // x[:, pos:pos+1] → (B, 1)
            let x_step = x.narrow(1, pos, 1)?;
            // forward returns (B, 1, vocab); cast to F32 for log_softmax.
            let logits = self
                .model_mut()?
                .forward(&x_step, seqlen_offset)?
                .squeeze(1)?
                .to_dtype(DType::F32)?;
            let log_probs = candle_nn::ops::log_softmax(&logits, 1)?;
            // y[:, pos:pos+1] → (B, 1). gather expects same rank as input.
            let y_step = y.narrow(1, pos, 1)?;
            let gathered = log_probs.gather(&y_step, 1)?; // (B, 1)
            let neg_log_prob_sum = gathered.sum_all()?.to_scalar::<f32>()?;
            total_loss += -neg_log_prob_sum as f64;
        }
        let count = (b * t) as f64;
        Ok((total_loss / count) as f32)
    }

    /// Compute the model's mean log-probability per completion token.
    /// Matches `ModelActor::ScoreLogProb` semantics (length-normalized,
    /// returns `mean` not `sum`) so callers like `GeneratorActor`'s
    /// oversample-and-rerank path work identically regardless of the
    /// underlying model.
    ///
    /// Algorithm: forward the prompt, take its last-position logits
    /// (= distribution over `completion[0]`), look up the log-prob of
    /// `completion[0]`, then advance one token at a time, each forward
    /// emitting the distribution over the next completion token.
    /// Total: `completion.len()` forward steps using the existing
    /// KV-cache pattern from `generate_autoregressive`.
    fn handle_score_log_prob(
        &mut self,
        prompt_ids: &[u32],
        completion_ids: &[u32],
    ) -> anyhow::Result<f32> {
        if completion_ids.is_empty() {
            return Ok(0.0);
        }
        if prompt_ids.is_empty() {
            anyhow::bail!("ScoreLogProb requires a non-empty prompt");
        }
        self.model_mut()?.clear_kv_cache();
        // Prime with the full prompt → last-position logits is the
        // distribution over `completion[0]`.
        let mut logits = self.forward_chunk(prompt_ids, 0)?;
        let mut seqlen_offset = prompt_ids.len();
        let vocab = logits.dims1()?;
        let mut total_log_prob = 0.0f64;
        for (i, &target) in completion_ids.iter().enumerate() {
            if (target as usize) >= vocab {
                anyhow::bail!("completion token id {target} out of vocab range (vocab={vocab})");
            }
            // log_softmax along the last dim. logits is (vocab,).
            let log_probs = candle_nn::ops::log_softmax(&logits, 0)?;
            let tgt_log_prob = log_probs.narrow(0, target as usize, 1)?.to_vec1::<f32>()?[0];
            if !tgt_log_prob.is_finite() {
                anyhow::bail!(
                    "non-finite log-prob {tgt_log_prob} at completion index {i} token {target}"
                );
            }
            total_log_prob += tgt_log_prob as f64;
            // Advance KV cache by one token unless this was the last one
            // (no need to forward past it; we already scored it).
            if i + 1 < completion_ids.len() {
                logits = self.forward_chunk(&[target], seqlen_offset)?;
                seqlen_offset += 1;
            }
        }
        let mean = (total_log_prob / completion_ids.len() as f64) as f32;
        Ok(mean)
    }

    /// Does the decoded completion contain a stop sequence?
    ///
    /// Decodes the whole completion each step rather than accumulating
    /// per-token strings: byte-level BPE splits multi-byte characters across
    /// tokens, so appending `decode(&[tok])` would produce replacement
    /// characters and could both miss a real stop and invent one. A full
    /// decode of <200 tokens is microseconds against a 26 ms forward pass.
    fn hits_stop(&self, completion_ids: &[u32]) -> bool {
        let Ok(text) = self.tokenizer.decode(completion_ids, true) else {
            return false;
        };
        self.stop_sequences
            .iter()
            .any(|s| text.contains(s.as_str()))
    }

    fn generate_autoregressive(
        &mut self,
        prompt_ids: &[u32],
        cfg: &GenerateConfig,
    ) -> anyhow::Result<Vec<u32>> {
        // Each call is a fresh sequence — clear the KV cache so previous
        // priming doesn't leak in.
        self.model_mut()?.clear_kv_cache();
        let mut rng: StdRng = match cfg.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let mut tokens: Vec<u32> = prompt_ids.to_vec();
        if tokens.is_empty() {
            return Ok(tokens);
        }

        // Prime with the full prompt at seqlen_offset = 0.
        let mut logits = self.forward_chunk(&tokens, 0)?;
        let mut seqlen_offset = tokens.len();

        let prompt_len = tokens.len();
        for _ in 0..cfg.max_new_tokens {
            let next = sample_logits(&logits, cfg, &self.suppress_tokens, &mut rng)?;
            // Qwen2 uses `eos_token_id` = 151643. Stop on EOS if encountered.
            if next == 151_643 {
                break;
            }
            tokens.push(next);
            // Check before the next forward pass, so hitting a stop costs
            // nothing extra. The token that completed the stop is kept: the
            // callers that cut text at a stop sequence expect it present.
            if !self.stop_sequences.is_empty() && self.hits_stop(&tokens[prompt_len..]) {
                break;
            }
            logits = self.forward_chunk(&[next], seqlen_offset)?;
            seqlen_offset += 1;
        }
        Ok(tokens)
    }

    /// Decode `n` completions of one prompt in a single batched pass.
    ///
    /// Every sequence shares the prompt, so all `n` are the same length at
    /// every step: no padding, no attention mask beyond the causal one the
    /// model already builds per batch size. That is what makes this tractable
    /// where general cross-request batching is not.
    ///
    /// Sequences that finish early (EOS, or a stop sequence) keep being fed
    /// their own last token and their extra output is trimmed at the end.
    /// Compacting the batch mid-flight would mean rebuilding the KV cache,
    /// which costs more than the wasted tokens.
    fn generate_batch(
        &mut self,
        prompt_ids: &[u32],
        n: usize,
        cfg: &GenerateConfig,
    ) -> anyhow::Result<Vec<Vec<u32>>> {
        if prompt_ids.is_empty() {
            return Ok(vec![Vec::new(); n]);
        }
        self.model_mut()?.clear_kv_cache();
        // One RNG per sample, offset from the base seed, so the draws differ.
        // Matches how `GeneratorActor` seeds its K samples and how the
        // ModelActor fallback seeds its loop.
        let mut rngs: Vec<StdRng> = (0..n)
            .map(|i| match cfg.seed {
                Some(s) => StdRng::seed_from_u64(s.wrapping_add(i as u64)),
                None => StdRng::from_entropy(),
            })
            .collect();

        let t = prompt_ids.len();
        let mut flat: Vec<u32> = Vec::with_capacity(n * t);
        for _ in 0..n {
            flat.extend_from_slice(prompt_ids);
        }
        let input = Tensor::from_vec(flat, (n, t), &self.device)?;
        let mut logits = self.forward_batch(&input, 0)?; // (n, vocab)
        let mut seqlen_offset = t;

        let mut seqs: Vec<Vec<u32>> = vec![prompt_ids.to_vec(); n];
        let mut done = vec![false; n];
        for _ in 0..cfg.max_new_tokens {
            let mut next = Vec::with_capacity(n);
            for (i, rng) in rngs.iter_mut().enumerate() {
                let row = logits.narrow(0, i, 1)?.squeeze(0)?;
                let tok = sample_logits(&row, cfg, &self.suppress_tokens, rng)?;
                if !done[i] {
                    if tok == 151_643 {
                        done[i] = true;
                    } else {
                        seqs[i].push(tok);
                        if !self.stop_sequences.is_empty() && self.hits_stop(&seqs[i][t..]) {
                            done[i] = true;
                        }
                    }
                }
                // A finished sequence still needs a token to feed forward, or
                // the batch dimension changes and the cache no longer lines up.
                next.push(*seqs[i].last().unwrap_or(&tok));
            }
            if done.iter().all(|d| *d) {
                break;
            }
            let step = Tensor::from_vec(next, (n, 1), &self.device)?;
            logits = self.forward_batch(&step, seqlen_offset)?;
            seqlen_offset += 1;
        }
        Ok(seqs)
    }

    /// Decode one completion for each of `prompts`, which may differ in
    /// length, in a single batched pass.
    ///
    /// Left-padded to the longest prompt. Left rather than right because RoPE
    /// encodes *relative* position: shifting a whole sequence by its pad count
    /// leaves every real-token distance unchanged, and all sequences then
    /// share one `seqlen_offset` at every step. Right padding would put the
    /// generated tokens at different offsets per row, which the shared KV
    /// cache cannot express.
    ///
    /// Pad columns are masked to `-inf` at every step, so no real token ever
    /// attends to one. The prefill mask is causal AND padding together —
    /// supplying only the padding part would silently drop causality, since
    /// the model's own `prepare_attention_mask` expands a keep/drop mask and
    /// never ANDs the causal term in.
    ///
    /// This is what batching the repair turns needs: each failed sample has
    /// its own repair prompt, containing its own failed call, so the prompts
    /// genuinely differ.
    fn generate_batch_prompts(
        &mut self,
        prompts: &[Vec<u32>],
        seeds: &[u64],
        cfg: &GenerateConfig,
    ) -> anyhow::Result<Vec<Vec<u32>>> {
        let n = prompts.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if prompts.iter().any(|p| p.is_empty()) {
            anyhow::bail!("generate_batch_prompts: empty prompt");
        }
        self.model_mut()?.clear_kv_cache();
        let mut rngs: Vec<StdRng> = (0..n)
            .map(|i| match seeds.get(i) {
                Some(s) => StdRng::seed_from_u64(*s),
                None => StdRng::from_entropy(),
            })
            .collect();

        let max_len = prompts.iter().map(|p| p.len()).max().unwrap_or(0);
        let pads: Vec<usize> = prompts.iter().map(|p| max_len - p.len()).collect();
        // Pad id is never attended to; EOS is used so a bug that leaks it into
        // the output is visible rather than looking like ordinary text.
        const PAD: u32 = 151_643;
        let mut flat = Vec::with_capacity(n * max_len);
        for (p, &pad) in prompts.iter().zip(pads.iter()) {
            flat.extend(std::iter::repeat(PAD).take(pad));
            flat.extend_from_slice(p);
        }
        let input = Tensor::from_vec(flat, (n, max_len), &self.device)?;

        // Prefill mask: (n, 1, T, T), -inf where j > i (causal) or column j is
        // a pad for that row.
        let mut m = vec![0f32; n * max_len * max_len];
        for (b, &pad) in pads.iter().enumerate() {
            for i in 0..max_len {
                for j in 0..max_len {
                    // Causal always. The pad columns are blocked only for
                    // REAL rows: masking a pad row's entire line leaves a
                    // softmax over all -inf, which is NaN, and a real query
                    // then multiplies that NaN value by a zero weight — 0 *
                    // NaN is NaN, so one padded row poisons the whole batch.
                    // The first version did exactly that and produced empty
                    // output for every padded sequence while the one
                    // unpadded sequence was correct.
                    //
                    // Pad rows are free to attend to earlier pads; their
                    // outputs are never read, and as keys they are masked out
                    // of every real query.
                    let blocked = j > i || (i >= pad && j < pad);
                    if blocked {
                        m[(b * max_len + i) * max_len + j] = f32::NEG_INFINITY;
                    }
                }
            }
        }
        let mask = Tensor::from_vec(m, (n, 1, max_len, max_len), &self.device)?;
        let mut logits = {
            let l = self
                .model_mut()?
                .forward_with_mask(&input, 0, Some(&mask))?;
            l.squeeze(1)?.to_dtype(DType::F32)?
        };
        let mut seqlen_offset = max_len;

        let mut seqs: Vec<Vec<u32>> = prompts.to_vec();
        let mut done = vec![false; n];
        for _ in 0..cfg.max_new_tokens {
            let mut next = Vec::with_capacity(n);
            for (i, rng) in rngs.iter_mut().enumerate() {
                let row = logits.narrow(0, i, 1)?.squeeze(0)?;
                let tok = sample_logits(&row, cfg, &self.suppress_tokens, rng)?;
                if !done[i] {
                    if tok == PAD {
                        done[i] = true;
                    } else {
                        seqs[i].push(tok);
                        if !self.stop_sequences.is_empty()
                            && self.hits_stop(&seqs[i][prompts[i].len()..])
                        {
                            done[i] = true;
                        }
                    }
                }
                next.push(*seqs[i].last().unwrap_or(&tok));
            }
            if done.iter().all(|d| *d) {
                break;
            }
            // Decode mask: (n, 1, 1, kv_len), blocking only the pads.
            let kv = seqlen_offset + 1;
            let mut dm = vec![0f32; n * kv];
            for (b, &pad) in pads.iter().enumerate() {
                for j in 0..pad {
                    dm[b * kv + j] = f32::NEG_INFINITY;
                }
            }
            let dmask = Tensor::from_vec(dm, (n, 1, 1, kv), &self.device)?;
            let step = Tensor::from_vec(next, (n, 1), &self.device)?;
            logits = self
                .model_mut()?
                .forward_with_mask(&step, seqlen_offset, Some(&dmask))?
                .squeeze(1)?
                .to_dtype(DType::F32)?;
            seqlen_offset += 1;
        }
        Ok(seqs)
    }

    /// Batched forward returning last-position logits as `(batch, vocab)`.
    fn forward_batch(&mut self, input: &Tensor, seqlen_offset: usize) -> anyhow::Result<Tensor> {
        let logits = self.model_mut()?.forward(input, seqlen_offset)?; // (b, 1, vocab)
        Ok(logits.squeeze(1)?.to_dtype(DType::F32)?)
    }

    fn forward_chunk(&mut self, chunk: &[u32], seqlen_offset: usize) -> anyhow::Result<Tensor> {
        let input = Tensor::from_slice(chunk, (1, chunk.len()), &self.device)?;
        let logits = self.model_mut()?.forward(&input, seqlen_offset)?; // (1, 1, vocab)
        let logits = logits.squeeze(0)?.squeeze(0)?;
        Ok(logits.to_dtype(DType::F32)?)
    }
}

fn sample_logits(
    logits: &Tensor,
    cfg: &GenerateConfig,
    suppress: &[u32],
    rng: &mut StdRng,
) -> anyhow::Result<u32> {
    use rand::distributions::{Distribution, WeightedIndex};

    // Applied before temperature/top-k/top-p and before the greedy branch, so
    // a suppressed id can never be selected by any path.
    let logits = if suppress.is_empty() {
        logits.clone()
    } else {
        let mut v = logits.to_vec1::<f32>()?;
        for &id in suppress {
            if let Some(x) = v.get_mut(id as usize) {
                *x = f32::NEG_INFINITY;
            }
        }
        Tensor::from_vec(v, logits.shape(), logits.device())?
    };
    let logits = &logits;

    // temperature == 0 → greedy
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

    let probs = candle_nn::ops::softmax_last_dim(&logits)?;
    let mut probs_v: Vec<f32> = probs.to_vec1()?;

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

    if !probs_v.iter().all(|x| x.is_finite()) || probs_v.iter().all(|x| *x <= 0.0) {
        let argmax = probs_v
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        return Ok(argmax);
    }

    let dist = WeightedIndex::new(&probs_v).map_err(|e| anyhow::anyhow!("weighted index: {e}"))?;
    Ok(dist.sample(rng) as u32)
}

fn log_send<T>(reply: oneshot::Sender<anyhow::Result<T>>, r: anyhow::Result<T>, op: &str) {
    if let Err(e) = &r {
        error!(op, error = %e, "actor op failed");
    }
    if reply.send(r).is_err() {
        warn!(op, "reply channel dropped before response");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_model_actor_has_modelmessage() {
        // Compile-time assertion that QwenModelActor uses the same
        // ModelMessage enum as ModelActor, so the messages are
        // interchangeable at the API level (only ActorRef<...> types differ).
        fn assert_uses<A>()
        where
            A: Actor<Message = ModelMessage>,
        {
        }
        assert_uses::<QwenModelActor>();
    }
}
