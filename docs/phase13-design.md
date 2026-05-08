# Phase 13 design — Scale-up plan

Phase 11/12 결과들의 신뢰도가 모두 같은 한계에 묶여 있음:

- **K9 21 prompts** — fine-grained 신호 추출에 부족 (Phase 11 결론)
- **1M-50M model scale** — Phase 12 S1 Muon +78% gen이 단일 run, variance bound 없음
- **Single-seed pretrain** — fresh seed별로 final eval이 11 vs 0 진폭 (Phase 11 baseline vs Phase 12 S1)
- **DeepSeek 기술은 1.6T에서 검증됨** — 1M에 transfer 안 될 수도 (Phase 9 S4의 1.5B 회귀 패턴)

이 한계들을 단계별로 풀어 나가는 계획. 각 stage는 (a) cost, (b)
무엇을 답하는지, (c) 의존성, (d) decision gate.

## 4-stage scale-up 로드맵

| stage | scope | cost | answers |
|--:|---|---:|---|
| A | 변량 측정 + 도메인 확장 (50M 유지) | 1 day | 단일 run 결과의 robustness |
| B | 200M 모델 + bf16 | 3-5 days | Muon/DPO/OPD가 4× scale에서 유지되나 |
| C | 외부 Qwen 1.5B-3B + PEFT | 2-3 days | 1B-scale 외부 모델 transfer |
| D | 500M+ in-house (deferred) | weeks | 자체 1B production track |

A → B는 **순차 (B가 A의 인프라 위에)**. C는 A와 **병렬 가능** (외부 모델은 별도 환경). D는 A/B/C 결과 후 결정.

---

## Stage A — Variance bound + 도메인 확장 (50M scale 유지)

### A1. K9 prompts 확장: 21 → 50

**목표**: Phase 11 결론의 caveat ("21 prompts 부족") 직격.

**구현**:
- `llm-actors/src/domain/rust_code.rs`의 `DEFAULT_CHALLENGES` 21 → 50으로 확장
- 패턴 보존: `assert_eq!(<slot>, expected)` 계열
- 새 challenge 종류 추가:
  - String length: `s.len()` for varied targets
  - Vec sum: `[1,2,3].iter().sum::<i32>()`
  - Boolean: `(a > b)`, `(a == b)`
  - Function composition: `add_one(2)` defined inline
- distinct prompt 보장 (`all_default_prompts_are_unique` 테스트 확장)
- 각 challenge slot 후보가 K9 v3처럼 reachable해야

**비용**: ~150 LOC + 새 unit tests. 1-2 hours.

**Falsifier 측정**: 기존 Phase 11 SFT/DPO matrix 일부 재실행 (50-prompt set 기준). Phase 11 결론이 그대로 유지되나? 18/24 (75%) → 38/50 (76%) 같은 일관성?

**Decision gate**: 50-prompt에서 같은 trend → Phase 11 결론 강화. 다르면 Phase 11 결론 K9-21-specific으로 재 framing.

### A2. Multi-seed variance measurement

**목표**: Phase 12 S1 Muon +78% / Phase 11 SFT 11/24의 variance 정량화.

**구현**:
- pretrain seed를 5개 fresh로 (`rust_seed_{0..4}`)
- 각 seed × {AdamW, Muon, hybrid α=0.3, round-0-only DPO} 4-round 측정
- 5 seeds × 4 variants = 20 runs × 5 min = ~100 min × 5 GPU 병렬 = 20 min wallclock
- Mean + std dev 보고

**비용**: 20 runs (자동화 스크립트, GPU 병렬 1.5h)

**Falsifier 측정**:
- Muon vs AdamW gen-pass: mean delta ± std → 95% CI 양수면 win
- mean(Muon) − mean(AdamW) > 2·std → 신뢰성 있는 win
- 그 외엔 noise → "결정적 시그널 없음" 결론

**Decision gate**:
- Muon 95% CI가 0 위 → Phase 12 S1 결론 robust. NAS axis 추가 confirmed.
- 95% CI가 0 포함 → Muon-vs-AdamW K9에선 차이 noise. Stage B에서 200M으로 다시 측정.

### A3. 5-seed Phase 11 DPO matrix 재현

Phase 11 S5의 11-variant matrix가 single seed였음. 가장 의미 있는 5
variants (SFT, pure DPO β=0.1, hybrid α=0.3, round-0-only, β=0.05)
× 5 seeds = 25 runs.

**가장 흥미로운 질문**: hybrid α=0.3의 r1 18/24 (75%) record가 전형
seed가 아니라 outlier인지? Mean이 14/24면 여전히 win, 12/24면 SFT 동등.

**비용**: 25 runs × 5 min ÷ 5 GPU = 25 min wallclock.

### A4. Stage A 종합

A1+A2+A3 모두 완료 후:
- Phase 11 + Phase 12 S1 결론을 variance bound 포함해 다시 commit
- 어느 결과가 robust, 어느 게 single-run noise였는지 분리
- Stage B 진입 결정 (어느 기술이 큰 scale에서 다시 검증할 가치 있는지)

---

## Stage B — 200M 모델 (4× current)

### B1. `nano_200m()` preset 추가

**구현**:
- `nanogpt-rs/src/config.rs`에 `nano_200m()` 함수
- Llama recipe (Phase 3에서 이미 검증됨) 그대로:
  - n_layer = 16 (was 8)
  - n_embd = 1024 (was 512)
  - n_head = 16, n_kv_head = 4 (GQA-4)
  - ffn_mult = 4
  - SwiGLU + RmsNorm-Pre + untied head + RoPE
  - block_size = 256 (유지) 또는 512
- ~200M params estimate

**메모리 budget** (A100 40GB, batch 16, block 256, fp32):
- params: 200M × 4 = **800 MB**
- activations: ~4 GB
- AdamW state: 2× params = **1.6 GB**
- gradients: 800 MB
- 총합 ~7 GB → fp32에서도 충분

**비용**: ~50 LOC + 1 unit test (`num_params_estimate ≈ 200M`).

### B2. bf16 학습 (선택 사항)

500M+로 갈 때 필요. 200M은 fp32로도 됨. bf16 도입 비용:
- `TrainConfig.dtype` 이미 존재 (DType::F32 default)
- DType::BF16으로 변경 시 candle bf16 ops 검증 필요
- 보통 memory −50%, throughput 1.5-2× 빨라짐

**Stage B에서 결정**: 200M fp32로 baseline, 그 다음 bf16 비교 측정.

### B3. K8 KoWiki 200M 재학습

- 기존 50M 30K → 200M 10K (or 같은 wallclock budget)
- val_loss 추적 (50M 30K가 7.43; 200M 10K가 어떤가?)
- top-1 mass + sum-AUC on KoreanCompletionDomain
- Phase 9 S2의 100K mode collapse 패턴이 200M에서도 보이나?

**비용**: A100 1대 ~1-2 hours (10K steps × 50% throughput at 4× params).

### B4. K9 200M 재현 — Phase 11/12 핵심 결과

A1의 50-prompt K9 도메인 + 200M 모델 + 5 seeds:
- SFT vs Muon vs hybrid α=0.3 vs round-0-only DPO
- 5 seeds × 4 variants = 20 runs
- 200M run wallclock ~15-20 min/run × 20 / 5 GPU = ~80 min

**가장 중요한 질문**: Phase 11 결과 (DPO multi-round collapse, hybrid α=0.3 75% record)가 200M에서 유지되나? 아니면 50M-specific?

**Decision gate**:
- Phase 11/12 trends가 200M에서 보존 → Phase 13 핵심 win, infrastructure validation
- 사라짐 → 1M-50M 결과는 toy artifacts. 더 큰 스케일이 *진짜* 결과 — Phase 13+에서 200M default

### B5. Stage B 종합

B1+B2+B3+B4 완료 후:
- 200M scale에서 Phase 11/12 결론 재현 여부 commit
- 도큐먼트: "Phase 11/12 결과는 X scale에서 valid"라는 명시적 scope
- Stage C/D 결정

---

## Stage C — 외부 Qwen 1.5B-3B + PEFT (Stage A와 병렬)

Phase 9 S4가 0.5B vs 1.5B 측정함. Stage C는 그것을 self-improve loop
+ Phase 12 기술까지 확장.

### C1. Phase 9 S6 측정 — 외부 critic-rerank

`scripts/phase9_s5/self_improve.py`에 이미 `--critic-oversample` flag
있음 (Phase 11 prep commit). GPU 측정만 미시행.

- Qwen2.5-Coder-0.5B + LoRA + critic-rerank F=4
- HumanEval-style task (이미 정의됨)
- baseline (F=1) vs F=4 비교

**비용**: ~30 min (Python venv + 측정).

### C2. Qwen + Muon for LoRA adapters

LoRA adapter는 작은 2-D matrix. Muon이 잘 맞는 use case:
- LoRA rank 16-32 → 작은 matrix → NS iteration 빠름
- AdamW 대비 +78% gen 효과가 LoRA에도 transfer되나?

**구현**:
- Qwen LoRA adapter forward를 Rust로 wrapping (또는 Python에서 직접)
- 가능하면 Python에서 muon 구현 빌리기 (`muon-optimizer` PyPI?)
- 1.5B-Coder + LoRA + Muon vs AdamW

**비용**: ~2-3 hours (Python 구현) + 30 min (GPU 측정).

### C3. Qwen variants OPD

- Specialists: Qwen2.5-Coder-0.5B (code), Qwen2.5-Math-0.5B (math),
  Qwen2.5-0.5B-Instruct (instruction)
- Student: 0.5B base, distill from 3 specialists
- HumanEval + GSM8K + AlpacaEval 3-domain eval

**비용**: ~1 day (multi-model orchestration + measurement).

### C4. Stage C 종합

외부 1B-scale에서 Phase 12 기술 검증. Phase 9 S4의 "0.5B-Coder
sum-AUC 0.702 PASS"와 함께 lift our claim from "1M K9 toy" to
"1B Qwen real-world".

---

## Stage D — 500M-1B in-house (deferred)

**진입 조건**: B 또는 C 결과가 명확한 win이라서 in-house production
track이 의미 있을 때만.

### D1. bf16 + gradient checkpointing
### D2. ZeRO-style optimizer state sharding (multi-GPU)
### D3. 1B Llama recipe 학습

**비용**: 수 주 — single phase 못 끝남. multi-month track.

**defer reason**: Stage A/B/C가 1B scale 결과 question을 어느 정도
풀 수 있음 (특히 C가 Qwen 3B로 가면 in-house 1B 필요성 약해짐).

---

## Phase 13 first session scope

**Phase 13 S1**: A1 + A2 — K9 50-prompt 확장 + 5-seed Muon variance.
- 가장 cheap, 가장 immediate
- Phase 12 S1 결론의 신뢰도 즉시 결정
- 한 세션 내 끝남 (1.5-2 hours coding + 30 min GPU)

**그 다음**:
- A2 결과가 Muon win robust → Phase 13 S2 = A3 (DPO matrix variance)
- A2 결과가 noise → Phase 13 S2 = B1+B4 (200M 직행)

---

## 우리 프로젝트의 falsifier-test workflow와 정합성

각 stage가 *정직한 falsifier*. 가능한 결과 분기:

- **Stage A에서 모든 게 noise**: Phase 11/12 결론 honest negative로
  강등. Phase 13의 핵심 win이 됨 ("우리는 이전 phase 결과를 falsify하는
  workflow가 정착됐다").
- **Stage A에서 robust**: Stage B로 진행. Phase 11/12가 50M scale에서
  검증된 결과 → Stage B가 200M scale에 transfer되나 묻기.
- **Stage B에서 transfer**: 큰 win. Project 결론 1M-50M에서 200M으로
  scope 확장.
- **Stage B에서 sun-setting**: 1M-50M 결과 toy-specific. Phase 13+가
  200M default로 재학습.
- **Stage C 외부 모델 결과**: 우리 phase별 결론이 1B Qwen에 transfer
  되는지 별도 데이터 포인트.

각 분기마다 commit + risk register update. 4-15개 commit 범위.

---

## Risk register 신규 후보

Stage 결과에 따라 risk #14-16 후보:

- **#14 (현 Phase 12 S1 후보)**: Optimizer choice는 metric별 trade-off
  (Muon = diversity↑ / sharpness↓). 일괄 교체 금지.
- **#15 (Stage A 결과 후 결정)**: Toy K9 21-prompts 결과는 50-prompts에서
  ?% 재현. 일반화 시 50+ prompt 측정 필수.
- **#16 (Stage B 결과 후 결정)**: 1M-50M 결과는 200M에서 ?% 재현. Scale
  jump가 결과를 ?

---

## 무엇을 NOT 하는가 (의도적 제외)

- **Architecture 큰 변화** — mHC (Phase 12 S4) / CSA-HCA (S5)는 Phase
  13 미포함. Stage B 결과 본 후 결정.
- **Multi-GPU 학습** — Single-GPU로 200M까지 충분. 1B+에서만 필요.
- **새 도메인 추가** — HumanEval/MBPP 도입은 Stage C 안에서만 (외부
  모델 측정용).
- **Phase 14+** — Stage A/B 결과를 본 후에야 Phase 14 정의 가능.

---

## See also

- `docs/phase12-design.md` — Phase 12 sequencing
- `docs/phase12-s1-muon.md` — Muon mixed result + caveats motivating
  Stage A2
- `docs/phase11-s5-hybrid-dpo.md` — 11-variant matrix, single-seed
  caveat motivating Stage A3
- `docs/phase7-design.md` — risk register (13 entries currently)
- DeepSeek V4 sources (Notion: workLLM Phase 10 S3 + 11 S1–S5 page)
