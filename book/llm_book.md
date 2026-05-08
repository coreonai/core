---
title: "LLM, 이렇게 만든다"
subtitle: "Rust × Apache Pekko 위에 자기-진화하는 에이전틱 파운데이션 모델 짓기"
author: "paul.yu@unomic.com (with Claude)"
date: "2026-05-08"
documentclass: book
lang: ko
fontsize: 11pt
toc: true
toc-depth: 2
numbersections: true
---

# 머리말

이 책은 `workLLM` 프로젝트가 한 일의 기록이다. 12개 phase 동안
약 6개월에 걸쳐 GPT 모델을 처음부터 Rust로 다시 쓰고, 그 위에
self-improvement 루프를 올리고, 그 루프를 자기-진화 시스템으로
확장한 과정을 남긴다. 모든 코드는 `github.com/coreonai/core`에
공개되어 있고, 이 책은 그 코드의 *왜* 와 *어떻게* 를 한 흐름으로
엮은 것이다.

## 이 책이 누구를 위한 것인가

- LLM을 한 줄씩 이해하면서 직접 짓고 싶은 시스템 엔지니어
- "fine-tune 한다"가 아니라 "scratch부터 만든다"가 무슨 뜻인지
  궁금한 ML 엔지니어
- Apache Pekko / actor 모델을 ML 인프라에 적용해본 적 없는
  Scala/Rust 백엔드 개발자
- 자기 framing을 falsifier test로 뒤집는 작업 패턴을 갖고 싶은
  연구 엔지니어

## 이 책이 *아닌* 것

- "어떻게 SOTA에 도달하는가"를 다루는 책이 아니다. 이 프로젝트의
  toy task ceiling은 데이터/스케일에 묶여 있고, 의도적으로 그렇게
  남겨뒀다. 인프라가 본체이고, 성능 숫자는 그 인프라가 제대로
  도는지 보여주는 신호일 뿐이다.
- "이 paper를 그대로 따라하면 된다"는 책이 아니다. 12개 paper의
  핵심 아이디어를 어디에 어떻게 끼워넣었는지, 그리고 어디에서
  *안* 통하는지를 honest negative와 함께 적었다.
- "프레임워크 카탈로그"가 아니다. PyTorch도 transformers 라이브러리도
  쓰지 않는다. Candle 위에서 모든 것을 한 번씩 다시 짓는다.

## 한 줄 요약

LLM을 짓는 일에 마법은 없다. 거의 모든 단계에서 명시적인 측정,
explicit한 기준, 그리고 자기 가설을 cheap하게 falsify하는 워크플로가
있을 뿐이다. 이 책은 그 워크플로를 어떻게 코드와 메모리와 git history에
새겼는지를 보여준다.

\newpage

# 1부: 비전과 배경 {.unnumbered}

# 왜 자기-진화하는 LLM인가

대부분의 "LLM 만들기" 자료는 한 모델을 한 번 학습시키는
방법까지만 다룬다. 학습 후의 inference, fine-tuning,
deployment는 별도의 책이다. 그러나 실제 production LLM 시스템의
가치는 *학습이 끝난 뒤에 무엇을 하는가* 에서 나온다.

이 프로젝트의 출발점은 다음 질문이었다.

> "한 번 학습된 모델 위에서, 인간이 매번 데이터를 라벨링하지
> 않고도, 모델 자신이 자기 출력을 검증하고, 검증된 출력으로
> 다시 학습하고, 그 결과를 평가하는 *루프*를 어떻게 짓는가?"

이 질문은 세 가지 부품을 요구한다:

1. **Generator** — 모델이 후보 출력을 만든다.
2. **Verifier** — 그 후보가 옳은지 결정하는 기준이 있다.
   토이 단계에서는 cargo build, python -c, heuristic regex 같은
   결정론적 기준이고, 큰 단계에서는 unit test, type check,
   LLM-judge가 된다.
3. **Trainer** — verified 출력을 다음 round의 학습 데이터로
   되먹인다.

이 세 부품이 actor pattern으로 묶이면 self-improvement 루프가
된다. 그 루프 위에서 evolution, tool use, agentic loop,
ensemble, critic-rerank를 차례로 쌓아 올리는 것이 이 책의
12개 phase 줄거리다.

## 왜 "scratch부터" 인가

이미 transformers, vLLM, accelerate가 있다. 왜 다시 짓는가?
세 가지 이유가 있다.

**첫째, 인프라를 모르면 인프라를 못 늘린다.** Phase 6의 핵심
발견 — Shape C, 즉 모델의 자기 log-prob을 critic으로 재활용
— 은 `GPT::sequence_log_prob` 메서드 한 줄에서 시작한다. 이
메서드는 모델 내부 forward pass에 손을 댈 수 있을 때만 짤 수
있다. 라이브러리가 그 hook을 제공하지 않으면 그 패턴 자체를
시도할 수 없다. 자기 코드면 hook을 자기가 판다.

**둘째, Rust + actor가 ML production에 대한 답에 가까워서다.**
Python ML 스택은 학습에는 좋지만 actor concurrency, type-safe
config, 결정론적 deployment에는 약하다. Apache Pekko의 Rust
포팅 (`pekko-rust`) 위에 ML 워크로드를 올리면, supervision
tree로 trainer/generator/verifier를 묶고, message-passing으로
재현 가능한 round를 굴릴 수 있다.

**셋째, "한 번씩 다 짜본다"가 멘탈 모델이 된다.** GPTConfig의
12개 axis 중 무엇이 무엇과 상호작용하는지, RoPE가 GQA와
함께 쓰일 때 attention 모듈이 어떻게 변하는지, distillation의
T²·KL이 왜 batch dimension으로 나뉘어야 하는지 — 이 모든 것은
"model.rs를 600줄 보고 매번 만져봤다"는 경험에서만 나온다.

\newpage

# Rust + Apache Pekko 선택의 이유

## Rust

Rust는 ML 자료에서 거의 등장하지 않는다. 대부분의 paper repo는
Python이고, 학습 루프는 PyTorch를 쓴다. 그럼에도 이 프로젝트는
Rust를 골랐다. 이유:

- **Type-safe 12-axis config**. `GPTConfig`는 12개 enum/scalar
  필드의 곱집합이다. `ActivationKind::Gelu | SwiGlu | GeGlu`,
  `NormKind::LayerNorm | RmsNorm`, `n_kv_head: NonZeroUsize`. 이
  모든 필드가 컴파일 타임에 검증되고, JSON으로 round-trip 되고,
  evolution의 mutation 연산자가 type-safe하게 변형한다. Python
  dataclass + dict로는 가능하지만 결정론과 strictness에서 차이가
  난다.
- **Candle**: HuggingFace가 만든 Rust 네이티브 ML 프레임워크.
  PyTorch 의존성 없이 CUDA 커널을 직접 호출하고, safetensors로
  체크포인트를 round-trip한다. 모든 학습 / 추론 / safetensors
  serialization이 한 process binary 안에서 끝난다.
- **결정론적 빌드**. `cargo build --workspace --release` 한 줄로
  모든 example이 빌드되고, CI 4-gate (build / test / fmt / clippy
  -D warnings)가 strict하게 통과한다. Python의 `requirements.txt`
  + venv + CUDA mismatch 지옥을 한 번도 만나지 않는다.

## Apache Pekko

Apache Pekko는 Akka(JVM actor framework)의 Apache 포크다. 사용자가
직접 Rust로 포팅한 `pekko-rust` (path dependency, `../AgenticAI/rust_pekko/`)를
쓴다. 사용 이유:

- **Actor = 자연스러운 경계**. ML 시스템에서 generator, verifier,
  trainer는 서로 다른 lifecycle을 갖는다. Generator는 stateless,
  trainer는 mutable VarMap을 보유하고, verifier는 외부 process
  (cargo, python -c)를 spawn한다. 각각을 actor로 분리하면 한
  actor의 panic이 supervision tree로 격리된다.
- **Hot reload**. `ModelActor`는 학습이 끝난 새 checkpoint를
  `ReloadCheckpoint` 메시지로 받아 in-place로 weights를 swap
  한다. 이 패턴은 Pekko message-passing이 없으면 mutex 지옥이
  된다.
- **Determinism**. Round-by-round 재현을 위해서는 message
  ordering이 보장돼야 한다. Pekko가 그 보장을 한다.

자세한 actor 설명은 8장에서 다룬다.

\newpage

# 프로젝트 구조 개관

```
workLLM/
├── nanogpt-rs/         # Phase 1 — GPT model + tokenizer + training
├── llm-actors/         # Phase 2+ — actor wrappers + domains + critics
├── docs/               # 12 risk register, design docs, postmortems
├── scripts/            # phase별 한 회차 실험 스크립트 (P9 S4/S5, P10 S2)
├── checkpoints/        # gitignored: safetensors
├── data/               # gitignored: corpora, tokenizers
└── README.md
```

12개 phase는 다음 순서로 쌓인다:

| Phase | 한 줄 요약 |
|------:|---|
| 1 | nanoGPT를 Candle/Rust로 |
| 1 epi | KoWiki BPE + 50M Llama recipe |
| 2 | 6 actor self-improvement 루프 |
| 2.5 | priority replay + RustCodeDomain |
| 3 | NAS / evolution (12-axis × 7 turns) |
| 4 | tool use + agentic loop + EWC + LoRA |
| 5 | consensus ensemble (honest negative) |
| 6 S1 | specialist routing (honest negative) |
| 6 SC | LogitCritic Shape C — 4.8× lift ★ |
| 7 | Shape C 일반화 검증 (decision tree) |
| 8 | Korean / Python / multi-assert 매트릭스 |
| 9 | 외부 1B HF 모델 + 실세계 self-improve |
| 10 | LLM-JEPA aux loss (S1 negative → S2 sweep 정정) |

각 phase는 다음 phase의 가설 공간을 좁힌다. 그 좁히기는 두
가지 방식으로 일어난다: positive 결과는 다음 phase가 그 위에
쌓이고, honest negative는 다음 phase가 *그 길을 가지 않도록*
한다.

\newpage

# 2부: 모델 (Phase 1) {.unnumbered}

# nanoGPT를 Candle/Rust로 옮기기

`nanogpt-rs/src/model.rs`는 1000줄의 GPT 구현이다. PyTorch
nanoGPT의 모듈 구조를 그대로 따라가지만 Candle의 Tensor /
Module trait 위에서 다시 짠다. 핵심 모듈:

- `CausalSelfAttention` — multi-head + RoPE + GQA + KV-cache 가능
- `MLP` / `FeedForward` — Dense / SwiGLU / GeGLU + MoE top-k
- `Block` — Pre/Post Norm + Attention + FeedForward
- `Norm` — LayerNorm / RmsNorm 래퍼
- `LoraAdapter` — rank/alpha 분리 가능
- `GPT` — 위 모듈들을 묶고 `forward`, `loss`, `forward_with_aux`,
  `forward_with_hidden`, `sequence_log_prob` 메서드 제공

## Candle vs PyTorch

Candle이 PyTorch와 다른 점:

1. **Tensor는 immutable이다.** 모든 연산은 새 Tensor를 만든다.
   in-place 연산이 없다. 처음에는 어색하지만 결과적으로 모든
   메모리 흐름이 명시적이 된다.
2. **Var는 별도 타입이다.** 학습 가능한 파라미터는 `Var`이고,
   `VarMap`이 이름→Var를 holding한다. `varmap.all_vars()`가
   AdamW의 trainable list가 된다.
3. **`forward(x)`는 `Result<Tensor>`를 반환한다.** 모든 에러가
   `Result`로 흐르므로 `?`로 propagate. Panic 없는 학습 루프.

```rust
pub fn forward(&self, idx: &Tensor) -> CResult<Tensor> {
    let (_b, t) = idx.dims2()?;
    if t > self.cfg.block_size {
        candle_core::bail!("seq len {t} > block_size {}", self.cfg.block_size);
    }
    let tok_emb = self.wte.forward(idx)?;
    let mut x = if let Some(wpe) = &self.wpe {
        let pos = Tensor::arange(0u32, t as u32, &self.device)?;
        let pos_emb = wpe.forward(&pos)?.unsqueeze(0)?;
        tok_emb.broadcast_add(&pos_emb)?
    } else {
        tok_emb
    };
    for blk in &self.blocks { x = blk.forward(&x)?; }
    let x = self.ln_f.forward(&x)?;
    self.head_logits(&x)
}
```

`use_rope`이 true면 `wpe`(positional embedding)을 만들지 않는다.
Position 정보는 `CausalSelfAttention` 내부에서 RoPE로 들어간다.
이 분기 하나가 GPTConfig의 `use_rope` 필드의 효과다.

## 12-axis GPTConfig

```rust
pub struct GPTConfig {
    pub vocab_size: usize,
    pub block_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    pub dropout: f64,
    pub bias: bool,
    pub ffn_mult: usize,
    pub use_rope: bool,
    pub rope_base: f64,
    pub n_kv_head: usize,
    pub n_experts: usize,
    pub moe_top_k: usize,
    pub moe_aux_weight: f64,
    pub activation: ActivationKind,
    pub weight_tying: bool,
    pub norm_kind: NormKind,
    pub norm_position: NormPosition,
    pub lora_rank: usize,
    pub lora_alpha: f32,
}
```

12개 핵심 axis (vocab/block/layer/head/embd/ffn_mult/use_rope/
n_kv_head/n_experts/activation/weight_tying/norm_kind/norm_position)는
Phase 3 NAS의 search space가 된다. 13장에서 evolution이 어떻게
이 12-axis 공간을 탐험해서 RoPE + GQA + SwiGLU + RmsNorm-Pre +
untied head — Llama-2 recipe — 를 *독립적으로 재발견*하는지를
보인다.

\newpage

# Tokenizer와 데이터 파이프라인

`nanogpt-rs/src/tokenizer.rs`는 두 가지 토크나이저를 지원한다:

1. **CharTokenizer** — 글자 단위. Toy task (arithmetic, Korean
   completion, Python slot-fill)에 충분.
2. **HuggingFace BPE** — `tokenizers` crate를 통해 BPE 학습.
   KoWiki, Shakespeare 같은 실제 자연어에 사용.

BPE 학습은 한 줄이다:

```rust
let tk = Tokenizer::train_bpe(
    &[corpus_path],
    16_000,                       // vocab size
    "data/kowiki/kowiki_bpe.json".into(),
)?;
```

내부적으로 `tokenizers::Tokenizer::train`이 ByteLevel pre-tokenizer
+ BPE 알고리즘으로 16K vocab을 만든다. Polyglot-Ko의 30K BPE와
비교하면 우리 16K가 in-domain Korean에 더 강하고, 코드/영어 혼재
입력에는 30K가 강하다 (`compare_tokenizers` 실험).

## TokenDataset

`data.rs::TokenDataset`은 `Vec<u32>` (corpus 전체 토큰)을 들고
`random_batch(batch_size, device) -> (Tensor, Tensor)`을 제공한다.
한 step마다 `block_size+1` 길이 sliding window를 batch_size개
뽑아서 `(input, target)` pair를 만든다.

```rust
pub fn random_batch(&self, batch_size: usize, device: &Device)
    -> CResult<(Tensor, Tensor)>
{
    let mut rng = thread_rng();
    let max_start = self.ids.len().saturating_sub(self.block_size + 1);
    let mut x_data = Vec::with_capacity(batch_size * self.block_size);
    let mut y_data = Vec::with_capacity(batch_size * self.block_size);
    for _ in 0..batch_size {
        let s = rng.gen_range(0..max_start);
        x_data.extend_from_slice(&self.ids[s..s + self.block_size]);
        y_data.extend_from_slice(&self.ids[s + 1..s + 1 + self.block_size]);
    }
    let x = Tensor::from_vec(x_data, (batch_size, self.block_size), device)?;
    let y = Tensor::from_vec(y_data, (batch_size, self.block_size), device)?;
    Ok((x, y))
}
```

명시적이라는 게 핵심이다. PyTorch DataLoader의 worker pool,
collate_fn, pin_memory를 다 안 쓴다. 한 process, 한 thread,
한 dataset, 한 random_batch.

\newpage

# AdamW + Cosine LR + EWC + LoRA + Distillation

`nanogpt-rs/src/train.rs`는 580줄의 학습 루프다. 핵심 entry
point는 `train_from_full`이고, 다음 옵션이 모두 한 함수에
통합되어 있다:

- AdamW + cosine LR schedule + warmup
- EWC penalty (`anchor: Option<&WeightAnchor>`)
- LoRA-only fine-tune (`freeze_base: bool`, "lora" 이름이
  포함된 Var만 trainable)
- JEPA aux loss + EMA target encoder (Phase 10)
- Distillation (별도 함수 `train_with_teacher`)

전체 호출 그래프:

```
train()
 └─ train_from()
     └─ train_from_with_anchor()
         └─ train_from_full()  ← 모든 entry point의 정점
```

## Cosine LR

```rust
fn cosine_lr(step: usize, cfg: &TrainConfig) -> f64 {
    if step < cfg.warmup_steps {
        return cfg.lr * (step + 1) as f64 / cfg.warmup_steps.max(1) as f64;
    }
    let progress = (step - cfg.warmup_steps) as f64
        / (cfg.max_steps.saturating_sub(cfg.warmup_steps).max(1)) as f64;
    let progress = progress.clamp(0.0, 1.0);
    let coeff = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
    cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)
}
```

linear warmup → cosine decay → min_lr에서 평탄.

## 학습 step 본체

```rust
for step in 0..cfg.max_steps {
    let lr = cosine_lr(step, cfg);
    opt.set_learning_rate(lr);

    let (x, y) = train_ds.random_batch(cfg.batch_size, device)?;
    let task_loss = if let Some(predictor) = &jepa_predictor {
        // Phase 10 JEPA branch (자세한 건 11부)
        let (logits, hidden_main) = model.forward_with_hidden(&x)?;
        let ce = cross_entropy(&logits, &y)?;
        let jl = jepa_loss(predictor, &hidden_main, cfg.jepa_offset)?;
        ce + (jl * cfg.jepa_lambda)?
    } else {
        model.loss(&x, &y)?
    };
    let total = if let Some(a) = anchor {
        (&task_loss + a.penalty(&varmap)?)?
    } else {
        task_loss.clone()
    };
    opt.backward_step(&total)?;
}
```

세 갈래의 분기 — JEPA 활성, EWC anchor 활성, 둘 다 비활성 —
가 한 step에 모인다. 이 통합이 evolution / multi-actor /
self-improve 루프를 짤 때 단일 entry point만 호출하면 되도록
만들어 준다.

## Critical bug 회복: temp=0 무한대 logits

Phase 4 작업 중 `generate.rs::sample_logits`에서 다음 패턴을
발견했다:

```rust
let logits = (logits / cfg.temperature)?;  // temp=0이면 ÷0!
```

`temperature == 0.0`일 때 logits가 ±∞가 되어 `softmax`가
NaN을 토하거나, 첫 non-negative logit으로 silent collapse하는
버그였다. 수정:

```rust
if cfg.temperature == 0.0 {
    // greedy path — argmax 직접
    return logits.argmax(D::Minus1);
}
let logits = (logits / cfg.temperature)?;
// ... softmax + sampling
```

이 버그 한 줄이 Phase 2의 demo dynamics 결과를 망가뜨리고
있었다. 발견 후 즉시 단위 테스트를 추가했고 (`greedy_returns_argmax`),
이후로 `cargo test --workspace`가 정확히 이 contract를 보호한다.

\newpage

# 3부: 자기-개선 인프라 (Phase 2 / 2.5) {.unnumbered}

# 6개 actor의 안무

Phase 2의 핵심은 다음 6개 actor가 메시지로 협조해서
self-improvement 한 round를 만드는 것이다.

```
GeneratorActor  → 후보 trajectory 생성
VerifierActor   → 도메인 verifier로 정/오 판정
CuratorActor    → priority replay buffer 관리
TrainerActor    → spawn_blocking으로 fine-tune
ModelActor      → VarMap + GPT 보유, hot reload
EvaluatorActor  → 전/후 metric 측정
SupervisorActor → 위 6개를 1 round로 안무
```

한 round의 메시지 흐름:

```
Supervisor → Evaluator: Eval(before)
Supervisor → Generator: Generate(prompts)
Generator  → Supervisor: trajectories
Supervisor → Verifier: VerifyBatch
Verifier   → Supervisor: verdicts
Supervisor → Curator: Add(verified)
Curator    → Supervisor: training_corpus
Supervisor → Trainer: Train(corpus, hyperparams)
Trainer    → Supervisor: new_checkpoint_path
Supervisor → ModelActor: ReloadCheckpoint(path)
Supervisor → Evaluator: Eval(after)
Supervisor → Caller: RoundResult{ before, after, gen_pass }
```

이 안무가 `supervisor.rs::RoundConfig`로 파라미터화된다.
`gen_oversample`(Phase 6 Shape C에서 critic-rerank용),
`anchor`(Phase 4 EWC), `freeze_base`(Phase 4 LoRA), `min_agreement`
(Phase 5 ensemble 합의 임계) 같은 옵션이 모두 한 struct에 모인다.

## Domain trait — task의 추상화

```rust
pub trait Domain: Send + Sync {
    fn charset(&self) -> &str;
    fn build_prompts(&self) -> Vec<String>;
    fn verify(&self, prompt: &str, completion: &str) -> Verdict;
    fn render_training_example(&self, prompt: &str, completion: &str) -> String;
}

pub enum Verdict { Correct, Incorrect, Skip }
```

세 가지 구현이 있다:

1. **ArithmeticDomain** — 단순 산수 (`"3+4="` → `"7"`)
2. **ToolUseArithmeticDomain** — tool-call grammar 학습용
3. **RustCodeDomain** — cargo build/run으로 검증
4. **PythonCodeDomain** — `python -c`로 검증
5. **KoreanCompletionDomain** — heuristic regex로 한국어 fluency

Domain trait가 있어서 supervisor 코드는 task에 무지하다. Trait
하나 구현 추가하면 새 domain이 self-improvement 루프에 들어간다.

## RustCodeDomain — cargo as ground truth

가장 흥미로운 domain이다. Verifier가 cargo다:

```rust
fn verify(&self, prompt: &str, completion: &str) -> Verdict {
    let program = format!("{prompt}{completion}\n{}", self.suffix);
    let dir = tempfile::tempdir().expect("tempdir");
    let main_path = dir.path().join("src/main.rs");
    std::fs::create_dir_all(dir.path().join("src")).expect("mkdir");
    std::fs::write(&main_path, &program).expect("write");
    std::fs::write(dir.path().join("Cargo.toml"), CARGO_TOML).expect("toml");

    let out = Command::new("cargo")
        .args(["run", "--release", "--quiet"])
        .current_dir(dir.path())
        .output();
    match out {
        Ok(o) if o.status.success() => Verdict::Correct,
        _ => Verdict::Incorrect,
    }
}
```

Cargo가 ground truth라서 모델이 *어떻게* 정답에 도달했는지에 관계없이
program이 동작하면 통과다. 이 domain이 Phase 6 Shape C의 4.8×
lift를 가능하게 했다.

\newpage

# Priority Replay Buffer

`CuratorActor`는 `PriorityBuffer<TrainingExample>`을 들고 있다.
`SampleMode::Priority { recency_decay }` 모드에서는:

- 새로 들어온 verified example의 priority = 1.0
- 매 round 끝에 모든 priority가 `recency_decay`로 곱해짐
- 학습 corpus 샘플링 확률 = priority / sum(priorities)

이 구조가 Phase 4에서 catastrophic forgetting 비교의 baseline이
된다. ER (Experience Replay)이라고 불리는 단순 mixing 전략과
같은 mechanism이다.

```rust
pub enum SampleMode {
    Uniform,
    Priority { recency_decay: f32 },
}

impl PriorityBuffer<TrainingExample> {
    pub fn sample(&self, n: usize, mode: SampleMode) -> Vec<TrainingExample> {
        match mode {
            SampleMode::Uniform => self.sample_uniform(n),
            SampleMode::Priority { recency_decay: _ } => self.sample_weighted(n),
        }
    }
}
```

Phase 4에서 plain fine-tune (forgetting massive) → ER (forgetting
완화) → EWC (안정성 더 강함) → LoRA (frozen base, 0 forgetting,
하지만 capacity 한계) 순서로 ablation했다. 그 ablation 매트릭스가
17장에서 다뤄진다.

\newpage

# 4부: NAS / Evolution (Phase 3) {.unnumbered}

# 진화가 Llama recipe를 자동 발견하다

Phase 3의 목표: 12-axis GPTConfig 공간에서 어떤 조합이 가장 좋은
fitness를 내는지를 evolution으로 탐색한다. Genetic algorithm:

```rust
pub struct EvolutionRunner {
    search_space: SearchSpace,
    population_size: usize,
    generations: usize,
    elite_fraction: f32,
    mutation_prob: f32,
    crossover_prob: f32,
}

pub struct Variant {
    pub cfg: GPTConfig,
    pub origin: VariantOrigin,
    pub fitness: Option<f32>,
}

pub enum VariantOrigin {
    Random,
    Mutated(VariantId),
    Crossover(VariantId, VariantId),
    Elite(VariantId),
}
```

## Multi-GPU dispatch

`tokio::task::JoinSet`이 각 variant의 학습을 `spawn_blocking`으로
보낸다. GPU 라운드-로빈:

```rust
for (i, variant) in variants.into_iter().enumerate() {
    let device = Device::new_cuda(i % n_gpus)?;
    join_set.spawn_blocking(move || {
        let outcome = train_from(...);
        let fitness = evaluate(&model, &eval_set);
        (variant, fitness)
    });
}
while let Some(result) = join_set.join_next().await {
    population.push(result?);
}
```

5장의 A100에서 generation당 5×4 = 20 variants를 병렬로 돈다.

## 7-turn 진화

12-axis 공간을 한 번에 다 풀지 않았다. 한 axis씩 추가하면서
generation을 돌렸다:

| Turn | 추가된 axis | Best fitness | 발견된 패턴 |
|--:|---|--:|---|
| 1 | n_layer / n_head / n_embd | 0.01 | baseline |
| 2 | + ffn_mult | 0.05 | larger FFN 선호 |
| 3 | + use_rope, n_kv_head (GQA) | 0.07 | RoPE + GQA |
| 4 | + n_experts (MoE) | 0.08 | 2-expert MoE |
| 5 | + activation (Gelu/SwiGLU/GeGLU) | 0.10 | SwiGLU |
| 6 | + weight_tying (untied) | 0.12 | untied head |
| 7 | + norm_kind, norm_position | **0.49** | RmsNorm + Pre |

마지막 turn에서 fitness 0.49는 6 turns 전 baseline의 **49배**다.
그리고 그 winning recipe — RoPE + GQA + SwiGLU + RmsNorm-Pre +
untied head — 가 정확히 Llama-2의 recipe였다. **사람이 그 recipe를
주입하지 않았다.** Evolution이 12-axis 공간에서 독립적으로 찾았다.

이 결과가 Phase 3 마지막 commit message의 한 줄로 남아있다:
"Evolution rediscovered the Llama-2 recipe."

## 함의

Llama-2 paper가 RoPE + GQA + SwiGLU + RmsNorm-Pre를 고른 데에는
empirical 정당화가 있고, 이 결과가 그것과 일치한다. 더 흥미로운
함의는: **이 12-axis 공간 위에서 evolution은 상대적으로 cheap한
경로다.** 12-axis 손으로 sweep하려면 $3^{12}$ ≈ 50만 조합이지만,
evolution은 100여 variants × 20 generations로 같은 답에 도달했다.

\newpage

# 5부: Tool Use, 에이전트, EWC, LoRA (Phase 4) {.unnumbered}

# Tool, 그리고 에이전트의 두 번째 본능

Phase 4의 첫 chunk: 모델이 tool을 호출하게 만든다.

```rust
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn call(&self, args: &str) -> Result<String, ToolError>;
}

pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

pub struct ArithmeticTool;

impl Tool for ArithmeticTool {
    fn name(&self) -> &str { "arithmetic" }
    fn call(&self, args: &str) -> Result<String, ToolError> {
        // "3+4" → "7"
        let result = parse_and_eval(args)?;
        Ok(result.to_string())
    }
}
```

모델은 다음과 같은 grammar로 tool을 호출하도록 학습된다:

```
Q: 17 + 25 = ?
A: <tool_call>arithmetic: 17+25</tool_call>
```

`AgenticGeneratorActor`가 multi-turn dispatch 루프를 돌린다:

```
1. generate(prompt) → output까지
2. parse_first_tool_call(output) → tool_name, tool_args
3. dispatch(tool_name, tool_args) → tool_result
4. splice_result(output, tool_result)  // "<tool_call>...</tool_call>=42"
5. generate(spliced) → 더 진행
6. parse_first_tool_call(spliced) → 새 호출이 있으면 dispatch
7. 새 호출이 없으면 종료
```

## Critical invariant: `=` skip

`parse_first_tool_call`에는 한 가지 특수 처리가 있다.
`splice_result`가 결과를 다음과 같이 적는다:

```
<tool_call>arithmetic: 17+25</tool_call>=42
```

이 splicing 후에 또 한 번 `parse_first_tool_call`을 부르면
"방금 resolve된" 호출이 또 매칭돼서 무한 루프가 된다. 그래서
parser는 *body 안에 `=`가 들어있는 호출은 skip*한다:

```rust
fn parse_first_tool_call(text: &str) -> Option<(String, String)> {
    for cap in TOOL_CALL_RE.captures_iter(text) {
        let body = cap.get(2).unwrap().as_str();
        if body.contains('=') { continue; }  // 이미 해결됨
        return Some((cap.get(1).unwrap().as_str().into(), body.into()));
    }
    None
}
```

이 한 줄이 multi-turn agentic loop의 무한 루프를 막는다. Test에
`agentic_loop_skips_resolved_calls`가 이 contract를 lock한다.

\newpage

# Catastrophic Forgetting의 5가지 처방

같은 모델을 여러 round에 걸쳐 fine-tune하면 이전 round의 학습이
새 round의 학습에 의해 잊힌다 (catastrophic forgetting).
Phase 4에서 5가지 처방을 ablation했다.

| 처방 | 안정성 | Capacity | Best Δ | Mechanism |
|---|:-:|:-:|--:|---|
| Plain fine-tune | ✗ | 100% | +12 (불안정) | 매 round 새 데이터로만 |
| Replay (ER) | △ | 100% | +7 sustain | 누적 corpus 일부 mix |
| Uniform Fisher EWC | ◯ | 100% | +12 best | 모든 weight L2 regular |
| Real Fisher EWC | ◯ | 100% | +7,+3 sustain | grad²로 importance |
| LoRA (per-Linear, freeze_base) | ◎ | ~30% | +0 sustain | base frozen, 1% params |
| **Full LoRA r=32** | ◎ | ~60% | **+15** | r=32 α=64 sweet spot |

표는 trade-off 매트릭스다. 안정성과 capacity는 trade-off다.
LoRA로 frozen base를 만들면 0 forgetting이지만 적은 trainable params는
학습 ceiling을 낮춘다. r=32 (충분히 높은 rank) + α=64 (높은 scale)
조합이 stability와 capacity의 sweet spot으로 발견됐다.

이 매트릭스가 Phase 5/6/7로 넘어갈 때 어떤 조합을 default로 쓸지를
결정한다 (대부분 LoRA r=32 α=64 freeze_base).

\newpage

# 6부: 실제 Korean 데이터 (Phase 1 epilogue) {.unnumbered}

# KoWiki를 1.2GB → 깨끗한 92MB로

`extract_kowiki.rs`는 `bzcat | extract_kowiki > clean.txt` 한
줄 파이프로 KoWiki 덤프를 plaintext로 푼다. 1.2 GB bz2 → 92 MB
text. 핵심 단계:

1. `quick-xml` SAX-style stream parser로 `<page><text>...</text></page>`만
   추출. DOM 안 만든다 (메모리 폭발).
2. `regex` 기반 줄 단위 cleanup: `[[파일:...]]`, `{{...}}`,
   `<ref>...</ref>`, `​` 같은 unicode noise 제거.
3. 너무 짧은 줄 (< 50 chars) drop.
4. LaTeX-only 줄 (`\begin{...}` ~ `\end{...}`) drop.

## 진짜 한국어 학습

50M Llama-recipe 모델 (`nano_50m()`)로 30K steps 학습. A100에서
~30분.

```
val_loss curve (50 batches × 16, held-out 5% tail):
  step 5K   → 7.46  (perp 1746)
  step 30K  → 7.43  (perp 1680)
  step 100K → 7.50  (perp 1808)  — multi-epoch overfit
```

`ln(16K) = 9.68` 와 비교하면 7.43은 의미있는 학습이다 (random
guessing 대비 -2.25 nats). 그러나 fluent Korean을 내기엔 멀었다
(ChatGPT급 perplexity는 ~2-3 nats 영역). 이 ceiling이 Phase 5
이후 self-improve 루프의 metric collapse 원인이 된다 (15장에서
다룸).

## 30K vs 100K 의 anti-calibration

100K로 늘리면 train_loss는 떨어지지만 val_loss는 *더 나빠진다*.
한국어 corpus에서 가장 흔한 토큰 (`\n`, 공백 변형, 자주 쓰이는
조사)에 mass가 몰린다. 이 mode collapse가 Phase 9 S2에서
*재발견*된다 — sum-AUC 측정에서 K8 100K가 30K보다 anti-calibrated.

\newpage

# 7부: 다중 Actor 실험 (Phase 5) {.unnumbered}

# 세 가지 후보 형태

Phase 5 design doc(`docs/phase5-design.md`)은 다중 actor
구성에서 가능한 세 가지 형태를 정의했다:

- **Shape A — Consensus**: N개 모델이 같은 prompt에 generate.
  N 중 ≥ k개가 같은 답을 내면 그 답을 verified로 취급. ML에서
  흔한 self-consistency 패턴.
- **Shape B — Specialist**: N개 모델이 각자 다른 challenge에만
  학습. Eval에서는 challenge → specialist routing.
- **Shape C — Adversarial**: 한 모델이 generate, 다른 actor가
  critic. Critic이 score를 매겨 best-of-N filter.

Phase 5 Session 1-3은 Shape A의 인프라(`ensemble.rs`,
`AddEnsemble` curator API)를 짰다. Session 4가 핵심 측정.

## Phase 5 S4 — Compute-matched honest negative

비교 setup:
- **Single 1M**: 1 model, 1× compute per round (1200 trainer steps)
- **Ensemble N=3**: 3 models, 1/3× each (3 × 400 = 1200 total)

총 compute가 같다.

| | Round 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| Single 1M | 8→15, gen 0 | 15→15, gen 0 | 15→14, gen 9/24 | 14→8, **gen 13/24 (54%)** |
| Ensemble N=3 | max 15 | max 15 | max **21/21 (100%)** | max 8 |

Ensemble peak이 round 2에 100%지만 그 round 끝에서 (다음 round 시작에)
8로 추락. 그 이유: lucky member가 한 번 좋았을 뿐 ensemble
mechanism이 아님. 다른 round에서 stochastic gen 0/72 (consensus
filter `kept=0` 매번).

Single은 stochastic gen 54%까지 도달. Curator turnover가
ensemble의 fragmented data보다 강함.

**Lesson**: Toy K9 scale에서는 task distribution이 N-way split의
이득을 못 준다. Multi-actor는 task distribution이 *실제로* split을
보상할 때만 유효.

이 negative는 Phase 5 design doc의 contingency를 발동시켜 Phase 6을
Shape B/C 방향으로 보낸다.

\newpage

# 8부: Shape C — 발견 (Phase 6) {.unnumbered}

# LogitCritic — 모델 자기 logits을 critic으로

Phase 6 Shape C의 핵심 발견은 한 줄이다:

> **모델의 sequence log-probability 자체가 cargo verdict와
> 의미있게 correlation한다.**

Mechanism: 한 모델이 generator + critic 양쪽 역할을 한다. 별도
critic을 학습할 필요 없이 *같은 모델*이 자기 confidence를 자기
filter로 재활용한다.

## `sequence_log_prob` 메서드

```rust
pub fn sequence_log_prob(
    &self,
    prompt_ids: &[u32],
    completion_ids: &[u32],
    device: &Device,
) -> CResult<f32> {
    let full = [prompt_ids, completion_ids].concat();
    let input = Tensor::from_vec(
        full[..full.len() - 1].to_vec(),
        (1, full.len() - 1),
        device,
    )?;
    let logits = self.forward(&input)?;
    // logits: [1, T-1, V]
    // we want log P(completion[t] | prompt + completion[:t])
    let log_probs = ops::log_softmax(&logits, candle_core::D::Minus1)?;
    let prompt_len = prompt_ids.len();
    let mut total = 0.0f32;
    for (i, &tok_id) in completion_ids.iter().enumerate() {
        let pos = prompt_len + i - 1;
        let lp = log_probs.i((0, pos, tok_id as usize))?.to_scalar::<f32>()?;
        total += lp;
    }
    Ok(total)
}
```

`completion`의 모든 토큰에 대해 conditional log-probability를
누적. 이게 sequence log-prob, *Shape C critic의 score*다.

## AUC 0.727 — Phase 6 Session 2

K9 RustCode에서 90개 후보 (3 challenges × 30 stochastic samples)를
harvest해서 cargo verdict로 라벨, LogitCritic으로 score:

| Critic | AUC |
|---|--:|
| **LogitCritic** | **0.727** ★ |
| RandomCritic (negative baseline) | 0.377 |
| AlwaysCorrectCritic | 0.500 (모두 ties) |

Acceptance gate (≥ 0.6) 통과. **모델 자체 logits이 cargo verdict와
의미있게 correlation.**

## Selection sweep — F=4 sweet spot

같은 90개 pool에서 F-of-many random subset, random pick vs
critic top-1 비교:

| F | Random | Critic | Lift |
|--:|--:|--:|---|
| 1 | 0.199 | 0.199 | 1.00× |
| 2 | 0.193 | 0.207 | 1.07× |
| **4** | 0.181 | **0.221** | **1.22×** ★ |
| 8 | 0.181 | 0.207 | 1.14× |
| 16 | 0.192 | 0.079 | **0.41× (inverts!)** |

F=4가 sweet spot. F=16에서 inversion — critic의 가장 confident한
완성이 cargo verdict와 *anti-correlate*. 이것이 risk #4 (top-tail
outlier poisoning at high F)의 시작이다.

## 4.8× compounding lift

Session 4에서 Shape C를 self-improve 루프 안에 통합:

```rust
ModelMessage::ScoreLogProb
GeneratorMessage::GenerateBatch.oversample
RoundConfig.gen_oversample
self_improve_rust --critic-oversample F
```

K9 v5 r=32 α=64 base에서 4 round 비교:

| Metric | F=1 baseline | F=4 critic |
|---|--:|--:|
| Round 0 gen | 0/24 (0%) | **10/24 (41.7%)** |
| Round 1 gen | 0/24 | **13/24 (54.2%)** |
| Round 2 gen | 9/24 (37.5%) | 10/24 (41.7%) |
| Round 3 gen | 0/24 | **10/24 (41.7%)** |
| **Mean gen-pass** | **9.4%** | **44.8%** |
| Wall-clock/round | 13s | 16s (+25%) |

**4.8× mean gen-pass at +25% wall-clock.** S3의 1.22× single-pool
lift가 self-improve 루프에서 4.8× round-over-round로 *증폭*된다.

## Mechanism: critic은 ranker가 아니라 amplifier

Round 0의 critic-rerank가 *더 나은 candidates*를 curator에 줌
→ round 1의 학습 corpus가 더 좋음 → round 1의 모델이 더 강함 →
round 1의 critic-rerank가 *더 효과적*. 이 cumulative loop가
1.22× single-pool lift를 4.8× multi-round lift로 만든다.

**Critic은 단순 ranker가 아니라 self-improve multiplier 역할.**

\newpage

# 9부: Shape C 일반화 검증 (Phase 7-8) {.unnumbered}

# Falsifier test로 자기 framing 뒤집기

Phase 7의 핵심 작업은 **Phase 6의 결과가 *general*인지 검증**이다.
Cargo로 검증되는 Rust 도메인에서 4.8× lift가 났다고 해서 다른
도메인에서도 같은가?

## Phase 7 S1 — Arithmetic 전이 시도

Same Shape C, ArithmeticDomain. 결과:

| Domain | LogitCritic AUC | F=4 lift | 결과 |
|---|--:|--:|---|
| RustCode (Phase 6) | 0.727 | 1.22× | PASS |
| Arithmetic (mean) | **0.447** | 0.75× | **FAIL** |
| Arithmetic (sum) | <0.6 | 0.93× | FAIL |

Mean과 sum 두 length-normalization variant 모두 acceptance gate를
통과하지 못함.

## 처음에 잘못 짠 framing

S1의 첫 분석은 "harvest pass rate가 낮아서다 — 모델이 답을 모르면
critic이 noise"라고 결론지었다. 그래서 acceptance criterion으로
"pass_rate ≥ 2 × chance_baseline"을 제안했다.

## Phase 7 S2 — Falsifier test로 위 framing을 falsify

Pretrain budget을 sweep해서 (800 → 10000 steps) AUC가 어떻게
변하는지 측정:

| Pretrain | Pass rate | Mean AUC | Sum AUC | Verdict |
|--:|--:|--:|--:|---|
| 800 | 7.6% | 0.445 | 0.545 | FAIL both |
| 2000 | 8.6% | 0.509 | 0.581 | FAIL both |
| **5000** | 9.8% | 0.564 | **0.632** | **PASS sum** ★ |
| 10000 | 9.9% | 0.569 | 0.658 | PASS sum |

**S1 framing이 틀렸다.** Pass rate가 chance baseline (~9%)에
머물러도 sum-AUC는 5K steps에서 0.632로 충분. **Calibration ≠
accuracy.**

## 정정된 lesson — 3 tier 가이던스

1. **Mean log-prob critic**: length-uniform 도메인 (K9 slot)에서만
   안전. Length-varying 도메인 (arithmetic, korean)에서는 short-bias로
   영원히 broken.
2. **Sum log-prob critic**: model이 confidence calibration을 갖추면
   작동. Pass rate가 chance여도 됨.
3. **진짜 gate**: held-out sum-AUC ≥ 0.6 측정. Pass rate는
   informative하지만 deciding은 아니다.

이 가이던스가 `docs/phase7-design.md`에 12개 risk와 함께 lock된다.

\newpage

# Risk Register — 12개 위험

`docs/phase7-design.md`의 risk #1-#12는 매번 한 phase가 발견한 한 가지
실패 모드를 정량화해 등록한 결과다. 짧게:

1. **Cargo syntactic checks**: critic이 cargo의 빌드 검증만 흉내내고 의미는 못 잡을 위험. 실측: 그 이상으로, critic은 model의 training distribution과 verifier의 verdict가 정렬될 때 작동.
2. **Critic over-fits to training set**: holdout AUC로 covered.
3. **Wall-clock penalty**: F=4 +25% bounded.
4. **Top-tail outlier poisoning at high F**: F=16에서 inversion (Phase 6 S3).
5. **Length-varying domains** (Phase 7 S1+S2): mean critic의 short-bias로 broken. Sum + AUC gate 필요.
6. **Calibration gate, not accuracy gate** (Phase 7 S2): pass rate 아니라 sum-AUC가 deploy gate.
7. **Anti-calibration on undertrained models** (Phase 8 S1, KoreanCompletionDomain). Sum-AUC < 0.4면 모델이 mode collapse, 재학습 필요.
8. **High AUC ≠ high selection lift** (Phase 8 S2, Python). Pass rate ≥ 30%면 random baseline이 강해서 critic-rerank lift가 압축.
9. **Pretrain can WORSEN calibration** (Phase 9 S2). K8 100K가 30K보다 anti-calibrated. Multi-epoch + 큰 corpus = mode collapse.
10. **External-scale validation** (Phase 9 S4). 1.5B-Coder가 0.5B-Coder보다 worse. 더 큰 모델이 더 좋다는 보장 없음.
11. **Cold-start dominates self-improve fate** (Phase 9 S5). round 0에 verifier-passed 0개면 LoRA-FT으로 영원히 0.
12. **JEPA-style aux losses interact non-trivially with calibration** (Phase 10 S1+S2). HP-, 도메인-민감. Single point에서 일반화 금지.

12개 모두 *측정으로 발견됐고*, *측정으로 정량화됐다*. Paper의
"future work"가 아니라 commit message + memory entry + design doc
risk가 됐다.

\newpage

# 5-Domain Matrix

Phase 8까지 5개 도메인 × 3 critic을 다 측정한 결과:

| Domain | Pass | Mean AUC | Sum AUC | F=4 lift | Verdict |
|---|--:|--:|--:|--:|---|
| K9 RustCode (1M GPT) | 19% | 0.564 | **0.727** | 1.22× | ★ PASS |
| Arithmetic (1M, 800 steps) | 7.6% | 0.445 | 0.545 | 0.75× | FAIL |
| Arithmetic (1M, 5000 steps) | 9.8% | 0.564 | **0.632** | 0.93× | PASS sum |
| KoreanCompletion (50M, 30K) | 2.2% | 0.418 | 0.421 | 0.54× | FAIL anti-cal |
| Python (1M, 1500 steps) | 35.6% | 0.787 | **0.859** | 1.01× | PASS but no lift |
| MultiAssert (1M, 1500 steps) | 7.8% | 0.564 | **0.789** | **1.32×** | PASS ★ |

Sweet spot for selection lift: pass rate **5–15%**. 너무 낮으면
critic이 calibrate 안 됨, 너무 높으면 random baseline 강해서 lift
압축.

\newpage

# 10부: 외부 모델 + 실세계 (Phase 9) {.unnumbered}

# 외부 1B 모델로 Decision Tree 검증

Phase 7 design tree는 in-house 1M GPT 위에서만 측정됐다. 1B-스케일
HF 모델로 가져갔을 때 그대로 작동하는지가 Phase 9 S4의 질문.

## Setup

- 6 Python challenges (3 from PythonCodeDomain + 3 simpler arithmetic)
- 32 stochastic samples per challenge
- Line-truncated to mirror in-house Generator
- `python -c` verify

## 결과

| Model | Params | Pass | Mean AUC | Sum AUC | F=8 lift | Verdict |
|---|--:|--:|--:|--:|--:|---|
| Qwen2.5-Coder-0.5B | 494M | 9.9% | 0.502 | **0.702** | **1.95×** | PASS |
| Qwen2.5-Coder-1.5B | 1.54B | 6.8% | 0.232 | 0.474 | 0.19× | NO SIGNAL |

**0.5B-Coder F=8 lift 1.95×** — matrix의 모든 행보다 강함.
Phase 7 design tree가 외부 모델로 transfer.

**1.5B의 회귀 mechanism**: 더 강한 priors가 common patterns
(`s = 0`, `return 1`)에 집중 → 드문 verifier-aligned completions
(`"hello"`, `5`)의 log-prob을 깔아뭉갬. Anti-calibration이 0.5에서
*위로* 접근.

## Phase 9 S5 — 외부 self-improve 루프 통합

S4는 measurement-only. S5는 in-house `self_improve_rust` 루프를
외부 모델에 그대로 포팅 + 실세계 task 추가.

```python
# scripts/phase9_s5/self_improve.py
for round in range(num_rounds):
    candidates = generate_K_per_challenge(model, challenges, K=8)
    scored = critic_rerank(candidates, sum_logp)
    verified = [c for c in candidates if python_c_verify(c)]
    train_set = [(prompt, completion) for c in verified]
    apply_lora_finetune(model, train_set, steps=60)
```

11 challenges = 6 S4 slot-fill + 5 HumanEval-style (`is_even`,
`list_sum`, `is_positive`, `count_chars`, `double_it`).

| Round | Pass | F=4 lift |
|--:|--:|--:|
| 0 | 39.8% | 0.81× |
| 1 | **72.7%** | 1.00× |
| 2 | 72.7% | 1.00× |
| 3 | 72.7% | 1.00× |

**+33 pp in 1 LoRA round**. 11 challenges 중 8개가 round 1에
100% 도달.

## 그러나 cold-start는 못 고친다

3 challenges (`equals_14_via_doubling`, `len_5_string`,
`ten_minus_to_3`)는 영원히 0/8. Round 0에서 verifier-passed가
0개라서 LoRA training set에 시드가 없음. 다른 8개의 LoRA pair로는
전이 안 됨.

**Risk #11 추가**: Self-improve는 cold-start를 못 고친다.
Curriculum / few-shot / hand-injection이 필요한 영역.

\newpage

# 11부: LLM-JEPA (Phase 10) {.unnumbered}

# Latent Prediction을 추가하면 좋아질까?

LLM-JEPA(Joint-Embedding Predictive Architecture)는 next-token CE
대신 *latent space에서 미래 hidden state를 예측*하는 보조
objective다. 이 프로젝트의 적용 동기:

> Phase 9 S2가 발견한 K8 mode collapse — 100K로 길게 학습하면
> `\n` 같은 고빈도 토큰에 mass가 몰리면서 sum-AUC가 *떨어진다*.
> JEPA의 latent objective는 "토큰 빈도가 아니라 representation
> distinctiveness"를 보상하므로, 이 mode collapse를 풀어줄 수
> 있을까?

## 단순 implementation

`nanogpt-rs/src/jepa.rs`:

```rust
pub struct JepaPredictor { fc1: Linear, fc2: Linear }

impl JepaPredictor {
    pub fn new(n_embd: usize, vb: VarBuilder) -> CResult<Self> {
        let hidden = 2 * n_embd;
        Ok(Self {
            fc1: linear(n_embd, hidden, vb.pp("fc1"))?,
            fc2: linear(hidden, n_embd, vb.pp("fc2"))?,
        })
    }
    pub fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let h = self.fc1.forward(x)?.gelu()?;
        self.fc2.forward(&h)
    }
}

pub fn jepa_loss(
    predictor: &JepaPredictor,
    hidden: &Tensor,
    offset: usize,
) -> CResult<Tensor> {
    let (_b, t, _d) = hidden.dims3()?;
    let context = hidden.narrow(1, 0, t - offset)?;
    let target = hidden.narrow(1, offset, t - offset)?.detach();  // stop-grad
    let predicted = predictor.forward(&context)?;
    (predicted - target)?.sqr()?.mean_all()
}
```

Single-encoder + stop-gradient (BYOL/SimSiam style). target이
detach되어 grad가 context branch로만 흐른다.

`TrainConfig.jepa_lambda` + `jepa_offset`로 한 step에 CE +
λ·JEPA가 합쳐진다.

## Phase 10 S1 — Honest Negative

K8 5K steps × (λ=0.1, k=8) baseline와 비교:

| metric | baseline | JEPA λ=0.1 k=8 |
|---|--:|--:|
| **top-1 mass** | 0.146 | **0.097 (−33%)** ✓ |
| pass rate | 2.2% | **3.3% (+50%)** ✓ |
| **sum-AUC** | 0.421 | **0.238 (anti-cal)** ✗ |
| F=4 lift | 0.54× | 0.21× ✗ |
| F=16 lift | 0.14× | 0.00× ✗ |

JEPA가 *자기 가설*은 달성 (mode collapse 약화) — 그러나 Shape C
critic을 *깨뜨린다*. F=16에서 critic_pass = 0.00× (matrix 최저).

Risk #12 초기 framing: "diversity ≠ calibration. Anti-mode-collapse
aux loss가 Shape-C를 악화시킬 수 있음."

## Phase 10 S2 — Falsifier sweep

S1 결과는 hyperparameter 공간의 *한 점* — 그것을 일반화해도 되는지
sweep으로 검증.

4 axes: λ ∈ {0.01, 0.03, 0.1, 0.3}, k ∈ {2, 4, 8}, EMA decay 0.99,
Python domain.

| variant | λ | k | EMA | top1 | sum-AUC | Δ |
|---|--:|--:|:-:|--:|--:|--:|
| baseline | 0.0 | — | no | 0.146 | 0.421 | — |
| λ=0.01 | 0.01 | 8 | no | 0.091 | 0.342 | −0.079 |
| λ=0.03 | 0.03 | 8 | no | 0.066 | 0.291 | −0.130 |
| **λ=0.1 k=8 (S1)** | 0.1 | 8 | no | 0.097 | **0.238** | **−0.183** worst |
| **λ=0.3 k=8** | 0.3 | 8 | no | 0.069 | **0.433** | **+0.012** recovered |
| **λ=0.1 k=2** | 0.1 | 2 | no | 0.073 | **0.432** | **+0.011** recovered |
| λ=0.1 k=4 | 0.1 | 4 | no | 0.050 | 0.396 | −0.025 |
| EMA | 0.1 | 8 | 0.99 | **0.049** | 0.292 | −0.129 |

**sum-AUC가 λ에서 U-shaped.** λ=0.1 k=8이 U의 *바닥*. 양 끝
(λ=0.3 또는 k=2)에서 calibration이 *완전히 회복*. EMA target은
mode collapse를 가장 잘 잡지만 calibration은 회복 못 한다.

Python domain (`λ ∈ {0, 0.03, 0.1}`, k=4): 모든 variant sum-AUC
≈ 0.86 (PASS). λ=0.1 k=4가 F=4 lift 1.05×로 매트릭스 최고.

**S1의 단일 점 negative가 일반화되지 않는다.** Risk #12 reframe:
"JEPA-style aux loss는 calibration과 비단조 상호작용. HP 민감 +
도메인 민감. Single point에서 결론 내지 말 것."

## Practical recipe

새 도메인에 JEPA를 deploy 전:

1. baseline (no JEPA) sum-AUC + top-1 mass 측정
2. 최소 3 점: (λ=0.05, k=2), (λ=0.1, k=2), (λ=0.2, k=4) — S2의 안전한 점들
3. λ vs sum-AUC plot. U-shape이면 양 끝 선호.
4. EMA target은 calibration 자유 업그레이드 *아님* — top1 mass가 진짜 metric이 아니면 안 씀.
5. "JEPA가 도움/해" 결론을 single point에서 내리지 말 것.

\newpage

# 12부: 메타 — 일하는 방식 {.unnumbered}

# Falsifier-test Workflow

이 프로젝트가 Phase 5 이후 가장 자주 쓴 패턴: 자기 framing을
cheap experiment로 falsify. 8개 사례:

| Phase | 초기 framing | falsifier 결과 | 정정된 claim |
|---|---|---|---|
| 5 | Consensus ensemble이 더 좋음 | 0/72 throughout | Toy scale에선 task split tax |
| 6 S1 | Specialist routing이 더 좋음 | compute-matched 8/21 | Per-challenge data ≥ 1/N compute 필요 |
| 7 S1→S2 | "≥ 2× chance" gate | 5K steps에서 sum-AUC 0.632 PASS | Calibration ≠ accuracy. sum-AUC가 진짜 gate |
| 8 S2 | Shape C가 Python에도 작동 | F=4 lift 1.00× (no-op) | High pass rate에서 lift 압축 |
| 9 S2 | 더 많은 pretrain ⇒ 더 좋은 calibration | K8 100K가 30K보다 worse | data uniqueness × epochs interaction |
| 9 S4 | 더 큰 모델 ⇒ 더 좋은 Shape-C | 1.5B가 0.5B보다 worse | Priors over-fit to common patterns |
| 9 S5 | Self-improve가 cold-start를 푼다 | 3 challenges 영원히 0/8 | Round-0 seed 0이면 못 고침 |
| 10 S1→S2 | JEPA가 Shape-C 망친다 | λ=0.3 또는 k=2에서 calibration 회복 | HP- + 도메인-민감, sweep 전 일반화 금지 |

8개 honest negative + 정정. 각 phase가 다음 phase의 가설 공간을
좁힌다.

## 워크플로의 codified 형태

```
1. 다음 phase의 가설 H를 한 줄로 적어라
2. H가 틀린다면 어떤 cheap experiment가 그것을 보여줄지 적어라
3. cheap experiment가 빠르게 끝나는가? (≤ 30분 권장)
4. 끝나면 H가 (a) 통과하는지, (b) 정확히 어디에서 깨지는지를 측정
5. (a)면 phase 진행. (b)면 H를 좁혀서 다시 1.
```

이 workflow가 8개 honest negative를 만들었다. 그 negative들은
*감춰진 게 아니라 git history와 risk register와 메모리 엔트리에
명시적으로 기록*됐다.

\newpage

# 가드레일 — CI Gates, 메모리, Pre-commit

## CI 4-gate strict

`.github/workflows/ci.yml`:

```yaml
- cargo build --workspace --release
- cargo test --workspace --release
- cargo build --workspace --examples
- cargo fmt --check
- cargo clippy --workspace --all-targets -- -D warnings
```

5-gate 모두 strict (`continue-on-error: false`). 한 gate라도 깨지면
PR merge 차단. 이 strict policy 덕분에 프로젝트는 **104 unit tests
+ 0 clippy warnings + 0 fmt drift**를 6개월간 유지했다.

## 메모리 시스템

`~/.claude/projects/-raid-users-paul-workLLM/memory/`에 phase별
상태 entry. 30+ files. `MEMORY.md`가 인덱스. 각 entry는:

- 무엇을 측정했는가
- 어떤 결론에 도달했는가
- *왜* 그 결론을 내렸는가 (mechanism)
- 다음에 무엇을 시도해야 하는가 (또는 무엇을 *시도하지 않아야*
  하는가)

이 메모리가 있어서 한 phase 끝나고 다음 phase로 넘어갈 때
"왜 이 선택을 했지?"를 git log만 봐서는 알 수 없는 *맥락*까지
보존된다. Phase 9 S5의 cold-start finding은 메모리 한 줄로
다음 phase를 starvation 검증으로 보낸다.

## Pre-commit checklist (CLAUDE.md)

매 commit 전:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Optional but useful:

```bash
cargo build --workspace --examples --release
```

CI가 잡기 전에 로컬에서 잡는다. 이 강제가 있어서 PR이 main을
한 번도 break 시키지 않았다.

\newpage

# 부록 A — 12-axis GPTConfig 풀 스펙

| Axis | Type | Default (`nano_50m`) | Range |
|---|---|---|---|
| vocab_size | usize | 32000 | dataset에 종속 |
| block_size | usize | 1024 | 64 ~ 4096 |
| n_layer | usize | 8 | 2 ~ 24 |
| n_head | usize | 8 | 1 ~ 16 |
| n_embd | usize | 512 | 64 ~ 2048 |
| dropout | f64 | 0.0 | 0.0 ~ 0.5 |
| bias | bool | false | — |
| ffn_mult | usize | 4 | 1 ~ 8 |
| use_rope | bool | true | — |
| rope_base | f64 | 10000.0 | 100 ~ 1e6 |
| n_kv_head | usize | 4 | 1 ~ n_head |
| n_experts | usize | 1 | 1 ~ 16 (1=Dense) |
| moe_top_k | usize | 0 | 0 ~ n_experts |
| moe_aux_weight | f64 | 0.0 | 0.0 ~ 0.1 |
| activation | enum | SwiGLU | Gelu / SwiGLU / GeGLU |
| weight_tying | bool | false | — |
| norm_kind | enum | RmsNorm | LayerNorm / RmsNorm |
| norm_position | enum | Pre | Pre / Post |
| lora_rank | usize | 0 | 0 ~ 64 |
| lora_alpha | f32 | 0.0 | 0.0 ~ 128.0 |

\newpage

# 부록 B — 12 Risk Register

`docs/phase7-design.md` 전체. 각 risk는 (1) Phase + Session,
(2) 측정 결과, (3) operational implication 세 부분으로 구성.

(본문에 9장의 표가 있음.)

\newpage

# 부록 C — 5-Domain Shape-C Matrix

(본문 9부 끝의 표.)

\newpage

# 부록 D — 명령어 레시피

## CUDA 환경

```bash
export CUDA_HOME=/usr/local/cuda-12.5
export PATH=/usr/local/cuda-12.5/bin:$PATH
```

driver 555 → cuda-12.5 toolkit 필수. cuda-12.9는 PTX
incompatibility로 runtime panic.

## 빌드

```bash
# CPU
cargo build --workspace --release

# CUDA
CUDA_HOME=/usr/local/cuda-12.5 PATH=/usr/local/cuda-12.5/bin:$PATH \
  cargo build -p llm-actors --example evolve_arithmetic --features cuda --release
```

## Tests

```bash
cargo test --workspace            # 104 tests, all CPU
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
```

## 핵심 example (run order)

```bash
# Phase 1
cargo run -p nanogpt-rs --example train_shakespeare --release
cargo run -p nanogpt-rs --example train_kowiki --features cuda --release

# Phase 2.5
cargo run -p llm-actors --example self_improve_round --release

# Phase 3
cargo run -p llm-actors --example evolve_arithmetic --features cuda --release

# Phase 4
cargo run -p llm-actors --example self_improve_tool_use --release

# Phase 6 Shape C
cargo run -p llm-actors --example self_improve_rust --features cuda --release \
  -- --critic-oversample 4

# Phase 9 S5
CUDA_VISIBLE_DEVICES=0 /tmp/s4_env/bin/python \
  scripts/phase9_s5/self_improve.py --rounds 3 --samples 8 --train-steps 60

# Phase 10 S2
cargo run -p nanogpt-rs --example train_kowiki_jepa --features cuda --release \
  -- --jepa-lambda 0.1 --jepa-offset 2
```

\newpage

# 맺음말

이 책은 12개 phase의 코드, 30+ memory entry, 12개 risk, 5-domain
matrix, 그리고 8개 honest negative를 한 흐름으로 엮은 결과다.

이 프로젝트가 SOTA 모델을 만들지는 않았다. Toy task ceiling은
arithmetic ~50%, K9 RustCode 75%, Korean fluency 미달이다. 그러나
그 ceiling은 데이터/스케일에 묶인 것이지, 인프라에 묶인 게
아니다. 50M → 7B로 가면 그대로 작동할 인프라가 남아있다.

**가져갈 한 줄**: LLM 만드는 일에 마법은 없다. 명시적인 측정,
explicit한 기준, 자기 가설을 cheap하게 falsify하는 워크플로가
있을 뿐이다. 이 워크플로가 곧 *인프라*다.

이 책의 모든 코드, 모든 risk, 모든 honest negative는
[github.com/coreonai/core](https://github.com/coreonai/core)에서
바로 cargo로 빌드 가능하다.

— 2026-05-08, paul.yu@unomic.com
