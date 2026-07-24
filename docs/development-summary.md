---
title: "workLLM — Rust × Pekko Self-Evolving Agentic Foundation Model"
subtitle: "개발 내용 종합 정리 (Phase 1 – 22)"
date: "2026-07-24"
---

# 1. 프로젝트 개요

**workLLM** 은 순수 Rust 로 구현한 nanoGPT 스타일 트랜스포머 위에, 사용자가
직접 포팅한 Apache Pekko 의 Rust 포트(`pekko-rust`)를 얹어 **self-evolving
agentic foundation model** 인프라를 구축한 프로젝트입니다.

- **비전**: "Rust nanoGPT × Pekko-Rust self-evolving Agentic Foundation Model."
  각 Phase 는 다음 Phase 가 조합해 쓰는 인프라를 ship 합니다.
- **핵심 원칙**: state-of-the-art 모델이 아니라 **인프라**를 만드는 것.
  toy task 의 천장(arithmetic ~50%, Korean loss ~7.0)은 dataset/scale 한계이지
  버그가 아니므로 억지로 끌어올리지 않는다.
- **규모**: 163 commits, 2026-05-05 ~ 2026-07-24, **176 unit tests**,
  37 worked examples, 11 인프라 Phase + 측정/알고리즘 Phase(5–22).
- **하드웨어**: 로컬 A100 40GB × 5+ 장 (host `dgxa100`).

프로젝트는 크게 **5개의 시대(era)** 로 나뉩니다.

| Era | Phase | 주제 | 한 줄 요약 |
|-----|-------|------|-----------|
| 1. 기반 | 1–4 | 모델·루프·NAS·툴 | GPT + self-improve loop + 진화적 NAS + agentic tool-use |
| 2. 측정 규율 | 5–16 | multi-actor·calibration·분산·paper-port | 정직한 부정 결과 + 분산 하한 확립 |
| 3. 알고리즘 승리 | 17–20 | multi-round SFT·pass@k | 6 Phase 만의 첫 robust positive, saturation curve |
| 4. Pekko 브릿지 | 21 | Candle-native Qwen2 | 추론·학습 양쪽 모두 Rust-native 화 |
| 5. 실제 벤치마크 | 22 | HumanEval·MBPP on Pekko | Python 레퍼런스를 Rust/Pekko 로 수치 재현 |

---

# 2. 기술 스택 & 저장소 구조

- **언어/런타임**: Rust, Candle (텐서/자동미분), `candle_transformers`.
- **액터 프레임워크**: `pekko-rust` (사용자의 Apache Pekko Rust 포트).
- **모델**: 자체 nanoGPT(`nanogpt-rs`) + Candle-native Qwen2.5-Coder-0.5B.
- **빌드 주의**: CUDA 12.5 툴체인 고정 필수(드라이버 555 가 cuda-12.9 PTX 거부).

```
workLLM/
├── nanogpt-rs/     # GPT 모델 + tokenizer + train + EWC + sampling + DPO + Muon + JEPA
├── llm-actors/     # Pekko 액터 래퍼 (Model/Trainer/Curator/Generator/Verifier/
│                   #   Evaluator/Supervisor) + evolution + tools + Qwen2 LoRA 브릿지
│   └── src/domain/ # arithmetic / tool_use / rust_code / humaneval / mbpp
├── scripts/        # phase14-22 측정 스크립트 (Python 레퍼런스 + Rust 런처)
├── data/           # (gitignored) 코퍼스·토크나이저·HumanEval/MBPP jsonl
└── checkpoints/    # (gitignored) safetensors
```

**GPTConfig 12축**: `n_layer, n_head, n_embd, block_size, ffn_mult, use_rope,
kv_group, n_experts, activation(Gelu/SwiGlu/GeGlu), weight_tying,
norm_kind(LayerNorm/RmsNorm), norm_position(Pre/Post)`.

---

# 3. Era 1 — 기반 인프라 (Phase 1–4)

### Phase 1 — 모델
- `GPT` 를 12축으로 configurable 하게 구현. 기본 `nano_50m()` 은 Phase 3 가
  발견한 Llama recipe(RoPE + 4× GQA + SwiGLU + RmsNorm-Pre + untied head, ~46.8M).
- **KoWiki epilogue**: 실제 한국어 위키 21.5M 토큰으로 50M Llama 학습, loss 8→7.2,
  한국어 토큰 emerge. `ln(16K)=9.68` 대비 예상된 plateau.
- **핵심 버그 수정**: `generate.rs::sample_logits` 가 `temperature==0` 에서도
  `logits/temp` 를 수행해 ±∞ 로 붕괴하던 것을 greedy=argmax 분기로 수정.

### Phase 2 / 2.5 — self-improvement 루프
- 6개 액터(`Generator, Verifier, Curator, Trainer, Evaluator, Supervisor`)가
  한 라운드 `Gen → Verify → Curate → Train → Reload → Eval` 로 조합.
- `Domain` trait 추상화(arithmetic / tool-use / rust-code). priority replay.
- arithmetic eval 4→13/100 로 self-improve 신호 실증.

### Phase 3 — 진화적 NAS (7 turns)
- `EvolutionRunner` 가 `population × generations` eval 을 각 variant scratch 학습.
- 12축 search space 를 한 축씩 추가 → **evolution 이 사람 개입 없이 Llama-2
  recipe(RoPE+GQA+SwiGLU+RmsNorm-Pre+untied)를 독립적으로 재발견**. fitness 0.49.
- MoE(top-k + Switch aux loss), gated activation 등도 이 과정에서 발견.

### Phase 4 — tool use / agent loop / distillation / EWC / LoRA (11 turns)
- `Tool` trait + registry, `AgenticGeneratorActor` 의 multi-turn 툴 디스패치 루프.
  (불변식: `=` 포함 툴콜은 skip — splice_result 마커라 무한루프 방지.)
- **Catastrophic forgetting 완전 ablation**: plain FT / replay(ER) /
  uniform-Fisher EWC / real diagonal-Fisher EWC / LoRA(c_attn-only, per-Linear).
- Full LoRA(r=32, 모든 linear+lm_head) 가 **31% pass rate (당시 최고)**.

---

# 4. Era 2 — 측정 규율의 확립 (Phase 5–16)

이 시대의 주제는 **"정직한 부정 결과(honest negative)"** 와 **분산 하한**.
멋진 single-run 결과를 falsifier test 로 뒤집는 규율이 자리잡습니다.

### Phase 5–7 — multi-actor & calibration
- **Phase 5 (앙상블)**: 3-member ensemble 은 compute-matched 로는 single 에 패배.
  Honest negative — 다중 액터는 task 분포가 실제로 split benefit 을 줄 때만 도움.
- **Phase 6 Shape C**: **LogitCritic AUC 0.727 PASS** — 모델 자체 log-prob 이
  cargo 검증과 유의미하게 correlate. 별도 critic 학습 불필요(1 model = gen + critic).
- **Phase 7**: 길이-가변 domain 에서는 `sum` log-prob 이 필수, `mean` 은
  short-bias 로 broken. 5000 steps 에서 sum-AUC 0.632 PASS. 게이트는 pass-rate 가
  아니라 **confidence calibration**.

### Phase 9 — 외부 모델 검증
- **S4**: Qwen2.5-Coder-0.5B sum-AUC **0.702**, F=8 lift 1.95×. 1.5B-Coder 는
  오히려 열화(0.474) — 더 큰 모델이 더 나은 calibration 을 보장하지 않는다.
- **S5**: Qwen0.5B + LoRA 1 round 로 **39.8% → 72.7% (+33pp)**, 11개 중 8개 saturate.
  cold-start 3개는 영영 0 — 실세계 HumanEval-style 과 synthetic 이 같은 dynamics.

### Phase 10–11 — JEPA / DPO
- **JEPA**: diversity ↑ 이지만 calibration 과 antagonistic. 5K sweep 회복은
  transient, 30K 에서 baseline 아래. **교훈: 5K sweep 으로 결정하지 말고 ≥50%
  budget 까지 측정.**
- **DPO**: r0 41.7% (SFT 0% 대비 최강 단일 신호) 이지만 r1 catastrophic collapse
  (mode collapse onto `-`). β sweep·rolling ref·hybrid 모두 시도 → **pure DPO 는
  1M K9 scale 에서 viable 하지 않음**. Hybrid α≥0.3 만 collapse 방지.

### Phase 12–13 — Muon & 분산 붕괴 (중요 전환점)
- **Phase 12**: Muon(DeepSeek V4, Newton-Schulz 직교화 SGD) 이 gen +78% 로 보였으나…
- **Phase 13**: 5-seed 로 **그 +78% 는 seed-0 outlier** 로 판명. 진짜 평균은 noise
  내. **K9 1M substrate 의 σ 가 fine-grained 알고리즘 비교엔 너무 큼** → K9 1M 을
  smoke-test 로 강등, 알고리즘 비교는 Qwen+HumanEval 로 이관. (Risk #14, #15)

### Phase 14–16 — quiet substrate 에서의 paper-port 검증
- **Phase 14 S1**: Qwen0.5B+LoRA+25문제 substrate 가 K9 대비 **13–27× tighter**
  (0.851 ± 0.011). "조용한(quiet) substrate" 확립 → 부정 결과가 actionable 해짐.
- **연쇄 retraction**: Muon LoRA(Δ=−0.092), DPO variants(σ 10.6× blow-up),
  OPD multi-teacher(Δ=−0.088), reverse-KL OPD(Δ=−0.159), hybrid OPD+SFT(Δ=−0.114).
- **분산 분해**: 양쪽 substrate 모두 harvest-dominated(93/7, 97/3). multi-init
  averaging 은 무의미, **samples-per-prompt 가 진짜 noise 감소 레버**(CLT 검증).
- **누적 결론**: **DeepSeek V4 3대 기법(Muon, DPO, OPD)이 LoRA self-improve
  scale 에서 3/3 모두 실패** — 각기 다른 메커니즘(잘못된 inductive bias / pair
  scarcity / KL 방향).

---

# 5. Era 3 — 첫 알고리즘 승리 (Phase 17–20)

6개 Phase 만에 처음으로 **robust positive** 가 나온 시대.

### Phase 17 — 3개의 robust positive
- **S1 multi-round SFT (rounds=2)**: 0.405 ± 0.013 vs 0.230, **Δ=+0.174** (2.8×
  threshold), 게다가 분산도 감소(σ ratio 0.43×).
- **S6 inference-time pass@k**: base Qwen pass@10=0.524 vs pass@1=0.216,
  **Δ=+0.308** (5×). MBPP 로 cross-substrate 확인(S9 Δ=+0.270).
- **SA 메커니즘**: multi-round SFT 는 pass@1·pass@10 **둘 다** base 초과 —
  매 라운드 새 chosen-pair 방향으로 분포를 넓힘(single-round 는 pass@k 붕괴).
- **retraction**: label smoothing α=0.1 은 Δ=−0.049 (wrong-direction remedy).

### Phase 18–20 — saturation curve
- **rounds 확장**: r1=0.230 → r2=0.404 → r3=0.475 → r4=0.519 → r5=0.556 →
  r6=0.581 (**첫 plateau 신호 Δ<σ**). seed 1 record **0.645** (base 0.216 의 3.0×).
- **Muon / OPD 는 multi-round 로도 구제 안 됨** (Risk #20 falsified) — 각각 4개
  config 에서 LOSS 로 정착.
- **cross-substrate 수렴**: HE r=5=0.556, MBPP r=5=0.541 — recipe 는 substrate-
  agnostic.
- **배포 Pareto front** (`docs/phase20-deployment-recipe.md`): r=5 SFT single /
  r=3 SFT + pass@5 / r=2 SFT + pass@10 / base + pass@10. 학습축 × 추론축은
  additive with diminishing returns.

---

# 6. Era 4 — Pekko-native 브릿지 (Phase 21, 10 stages)

Python 사이드카 없이 **추론·학습 양쪽을 모두 Rust + Pekko-native** 로 만든 시대.
`candle_transformers::models::qwen2` 를 포크해 실제 Qwen2.5-Coder-0.5B HF
safetensors 를 그대로 로드.

| Stage | 내용 |
|-------|------|
| A | Phase 17 S6 pass@k 를 `EvaluatorActor` 에 이식 (per-(prompt,k) seed override) |
| C | `supervisor::run_multi_round` 헬퍼 (init_from 자동 체이닝) |
| D | **`QwenModelActor`** — Candle-native Qwen2 추론 액터 (no Python) |
| F | **Qwen2 LoRA 학습** (`qwen2_lora.rs`, q/v_proj r=16). Candle 0.10 gotcha 발견: `rope`/`softmax`/`rms_norm` 이 forward-only op 라 gradient 가 조용히 죽음 → `_slow` 경로로 dispatch. loss −57% 검증 |
| B | substrate scale-up (~24M) 에서 pass@k lift 재현 |
| E | Evaluator/Generator/RoundActors 를 `M: Actor<Message=ModelMessage>` 로 generic 화 |
| E.next | **`QwenTrainerActor`** — 학습 액터화, LoRA adapter(4.3MB) 저장 |
| E.next.next | `save_merged_lora` — LoRA 를 base 에 병합해 upstream 로더 호환 checkpoint 저장 |
| H | **`TrainerHandle` trait** — supervisor 가 Qwen 학습을 구동. README 의 "self-evolving agentic foundation model on Pekko" 실현 |
| G | **REINFORCE** (`train_qwen_lora_pg_step`) — verifier-as-reward policy gradient |

결과: `run_multi_round` 가 **Gen→Verify→Curate→Train→Reload→Eval 전체를 실제
Qwen2.5-Coder-0.5B 에 대해 end-to-end 로 Pekko 를 통해 구동**.

---

# 7. Era 5 — 실제 벤치마크 on Pekko (Phase 22)

Phase 17–20 의 모든 발견에 **Rust-native 실행 경로 + 수치 재현**을 부여.

### Stage A–C — 실제 벤치마크를 Rust Domain 으로
- **`HumanEvalDomain`** (164문제, python3 subprocess 검증), **`MbppDomain`**
  (MBPP-100). Stage B 에서 "pass@1" 의 정의 오해를 교정 — Phase 17 의 pass@1 은
  greedy 가 아니라 **temp=0.8/k=10 aggregate per-attempt**. `EvalSequential
  { aggregate }` 모드로 **aggregate pass@1 = 0.222 (ref 0.216 within 1σ)** 재현.

### Stage D — −0.20 regression 과 그 근본 원인 (G1–G9)
Stage D 의 multi-round SFT 는 처음 **base 의 절반**으로 붕괴. 원인은 알고리즘이
아니라 Phase 17 Python `self_improve.py` 와의 **recipe 세부 divergence 스택**이었고,
순서대로 해소:

1. `bc90db5` — **completion-only CE mask** (`labels[:prompt_len] = -100`). 우리는
   모든 위치에서 CE 계산 → prompt 가 loss 의 ~80% 지배 → catastrophic over-train.
2. `c7a7aed` — cosine LR + 10% warmup.
3. `59aab8d` — padded `batch_size=4` SFT (batch=1 collapse 수정, σ 0.063→0.009).
4. `e69de7e` — round 마다 fresh AdamW + non-cumulative harvest.
5. `aaf0594` — **completion truncation (결정타)**. `build_program` 이 RAW 완성문
   (trailing test scaffolding, 누출된 `<|fim_middle|>`, 잘린 tail)을 그대로 먹여
   verifier 실패(low harvest) + eval 실패. Phase 17 의 `truncate_completion` 이식.

**결과 (full recipe + truncation, 5-seed aggregate):**

| Substrate | base | r=2 (Rust/Pekko) | Phase 17 ref |
|-----------|------|------------------|--------------|
| HumanEval | 0.218 | **0.436** | 0.404 |
| MBPP | 0.201 | **0.447** | 0.453 (SB) |

saturation curve 도 diminishing→plateau SHAPE 재현(HE ~0.51@r≈4-5). 잔여 high-round
gap(~0.05)은 5개 paired ablation 으로 조사 — `top_k=0` 만 도움(+0.026, 채택),
cumulative buffer/harvest temp/weight_decay 는 기각, fp16 은 Candle 에 autocast/
GradScaler 가 없어 NaN(untestable). 잔차는 dominant divergence 가 아닌 plateau 변동.

### Stage E — REINFORCE 와 held-out generalization (최신, 2026-06-17)
- Phase 21 Stage G REINFORCE 를 실제 HumanEval 로 이식(verifier-as-reward, RLOO baseline).
- adapter-sync 변형이 "WINS 3/3" 로 보였으나, **이는 학습 프롬프트 위에서만 측정**된 것.
- `EvalSequential.offset` (`--offset`) 을 추가해 **held-out tail (task 64..164,
  학습에 안 쓰인 100문제)** 에서 clean 하게 재측정:

| Config | held-out pass@10 | aggregate pass@1 |
|--------|------------------|------------------|
| Base (untrained) | **0.28** | 0.066 |
| n16-sync × 3 seeds | 0.22–0.25 (전부 regress) | 0.061–0.083 |
| n64-sync seed42 | 0.26 | 0.113 |
| n64-sync seed100 / 200 | **0.00 (학습 중 붕괴)** | 0.00 |

- **결론**: n=16 "3/3 win" 은 on-distribution 암기였고 일반화 이득 0. "프롬프트를
  16→64 로 늘리면 일반화가 넓어진다"는 가설은 **falsify** — 3개 중 2개가 학습 도중
  mode-collapse. **RL 은 LoRA scale 에서 잘해야 on-distribution 암기이며 MR-SFT 를
  이기지 못한다.** (교훈: RL self-improve 는 반드시 held-out split 으로 평가할 것.)
  최신 commit `fcc39de`.

---

# 8. 관통하는 핵심 교훈

1. **Falsifier-test 규율**: 자신의 framing 을 자주 뒤집어 본다. 멋진 single-run 은
   5-seed 로 재현되기 전엔 믿지 않는다 (Phase 12→13 의 +78% outlier 가 상징).
2. **Quiet substrate 의 복리 효과**: 분산이 작은 substrate(Qwen+LoRA) 에서만
   부정 결과가 actionable 해진다. K9 1M 은 noise floor 가 너무 높아 강등.
3. **DeepSeek V4 3대 기법(Muon/DPO/OPD)은 LoRA self-improve scale 에서 3/3 실패** —
   각기 다른 메커니즘. 반대로 **단순한 multi-round SFT + inference-time pass@k 가
   robust win**.
4. **Recipe 세부가 전부다**: Pekko 구동이 Python 레퍼런스와 갈라지면 액터 배선보다
   **inner training loop 를 byte-compare** 하라 (completion-mask, truncation 이
   −0.20 gap 의 진짜 원인이었다).
5. **정직한 측정**: 학습 코퍼스 분포에서 뽑은 per-round eval 은 암기를 재고,
   benchmark-aligned aggregate eval + held-out split 만이 정직한 신호.

---

# 9. 현재 상태 (2026-07-24)

- **176 unit tests** green, **zero clippy warnings** (`-D warnings`), zero fmt drift.
- CI 4-gate strict (build / test / examples / fmt+clippy).
- Phase 17–20 의 모든 발견이 **Rust-native 실행 경로 + 수치 재현** 보유:
  pass@k inference scaling, MR-SFT, REINFORCE, HumanEval, MBPP 전부 Pekko 를
  통해 실제 Qwen2.5-Coder-0.5B 에 대해 동작.
- 진입점 문서: `docs/phase21-overview.md`, `docs/phase22-overview.md`,
  `docs/phase20-deployment-recipe.md`.
- 최신 작업: Phase 22 Stage E REINFORCE held-out generalization (honest negative),
  commit `fcc39de`, origin/master 에 push 완료.

## 주요 엔지니어링 gotcha
- **CUDA 12.5 고정 필수** (드라이버 555 가 cuda-12.9 PTX 거부; 컴파일은 성공,
  첫 커널에서 panic). Phase 22 GPU 바이너리는 `PHASE22_ALLOW_CPU` fail-fast 가드 보유.
- `--features cuda` 없이 `cargo build` 하면 example 바이너리가 CPU 전용으로 조용히
  덮여씀 → GPU 런이 timeout. 라이브러리 변경 후 반드시 example 재빌드.
- Candle 0.10: `rope`/`softmax`/`rms_norm` 은 forward-only op 라 학습 경로에서
  gradient 가 조용히 사라짐 → `_slow` 변형으로 dispatch.
