# Phase 11 S4 — DPO multi-round collapse는 HP 튜닝으로 안 풀림

S3가 발견한 round-1 catastrophic collapse를 두 가지 방향으로 수정 시도:
β sweep (frozen reference) + rolling reference (β=0.1). 결과: **둘 다
실패**. Pure DPO는 1M K9 scale에서 viable하지 않음. 다음 단계는
hybrid loss / round-0-only DPO.

## Setup

S3와 동일: 1M GPT, K9 RustCodeDomain, 4 rounds × 400 train_steps,
seed checkpoint 공유, 양 GPU 병렬.

5개 새 variant:
- DPO β=0.01 frozen reference (seed)
- DPO β=0.03 frozen reference
- DPO β=0.05 frozen reference
- DPO β=0.1 rolling reference (round n의 ref = round n-1's policy)
- (S3의 β=0.1 frozen + SFT control 포함)

## Result matrix

| variant | r0 gen / eval | r1 gen / eval | r2 / r3 | final eval | mean gen |
|---|---|---|---|---:|---:|
| **SFT** (control) | 0/24, 0→6 | 0/24, 7→7 | **9/24**, 7→11 / 0/24, 11→11 | **11/24 (45.8%)** ★ | 9.4% |
| DPO β=0.1 frozen (S3) | **10/24** (41.7%), 1→7 | 0/24, 7→**0** | 0, 0→0 / 0, 0→0 | **0/24 (0%)** | 10.4% |
| DPO β=0.1 rolling | 0/24, 0→7 | **7/24** (29.2%), 7→**0** | 0, 0→0 / 0, 0→0 | 0/24 (0%) | 7.3% |
| DPO β=0.01 frozen | 0/24, 0→6 | 0/24, 7→**0** | 0, 0→0 / 0, 0→**11** | **11/24 (45.8%)** | 0% |
| DPO β=0.03 frozen | 0/24, 0→7 | 0/24, 7→**0** | 0, 0→0 / 0, 0→0 | 0/24 (0%) | 0% |
| DPO β=0.05 frozen | 0/24, 0→7 | **8/24** (33.3%), 7→**0** | 0, 0→0 / 0, 0→7 | 7/24 (29.2%) | 8.3% |

## 핵심 관찰

### 1. Round 1 catastrophic collapse는 robust

모든 DPO variant가 round 1 eval-after = **0/24**. β를 10× 줄이거나 (0.01),
reference를 rolling으로 바꿔도 같은 패턴. **이건 hyperparameter 한계가
아니라 DPO 자체의 dynamics 문제**.

### 2. Rolling reference는 도움 안 됨

가설은 "(π − π_ref) drift가 ref 고정 시 무한히 커지므로 rolling이
풀어줄 것" — 데이터는 정반대. Rolling β=0.1도 r1 eval 7→0 collapse,
β=0.1 frozen과 같음. Reference 선택은 collapse 방지의 binding constraint
아님.

### 3. β=0.01만 r3에서 회복하지만 SFT 따라잡기에 불과

β=0.01: r1 collapse → r2 0 → r3 eval 0→11 (Δ +11, recover to baseline).
Final 11/24 = SFT final 11/24 — DPO 추가 신호의 net benefit은 **정확히 0**.

### 4. DPO round 0/1 gen-pass 스파이크는 진짜 신호

- β=0.1 frozen r0: 10/24 (41.7%)
- β=0.1 rolling r1: 7/24 (29.2%)
- β=0.05 r1: 8/24 (33.3%)

이 스파이크는 SFT에서 안 보임 (SFT r2 9/24가 가장 높음). DPO의
"chosen vs rejected" 신호가 *진짜* 정보를 추가하긴 함 — 단지
sustained가 안 됨. **One-shot signal, not a multi-round trainer**.

## 메커니즘 가설 (수정)

S3의 가설 "β 너무 큼 / reference 고정"은 falsified. 새 가설:

> **Rejected pile이 mode-collapse 유도 신호.** K9 RustCode에서 한
> round의 rejected는 24개 incorrect completions per challenge —
> 대부분 syntactic noise (`\n` repetition, 잘못된 토큰). DPO의
> negative gradient가 이 noise distribution에서 멀어지도록 push하면,
> 정작 *eval 분포에서도 멀어진다*. 즉 rejected가 "informative wrong"이
> 아니라 "noise"에 가까워서 informative한 push 신호를 못 만듦.

이 가설은 검증 가능 — Phase 11 S5에서 hybrid loss로 SFT가 anchor
역할을 하게 두면 DPO의 rejected push가 mode collapse 안 일으켜야 함.

## Phase 11 S5 후보

**가장 유망**: **Hybrid SFT+DPO loss within a round**

```
total_loss = (1 - α) * sft_ce(chosen) + α * dpo_loss(chosen, rejected)
```

α ∈ {0.1, 0.3, 0.5} sweep. SFT가 anchor → mode collapse 방지. DPO가
rejected에서 push.

**또 다른 옵션**: **Round-0-only DPO**

S3 frozen β=0.1 r0가 41.7% gen-pass — 강력. r1+에서는 SFT만 사용.
DPO를 "Round 0 boost"로만 활용. Phase 5/6의 Shape C와 유사한
패턴: 한 round의 강한 신호를 cumulative loop가 amplify.

## Risk #13 강화

S3의 #13 풀어쓰기:

> **DPO multi-round 동학은 single-round과 fundamental하게 다름**.
> Round 0/1 +30~+42pp gen-pass spike는 진짜 신호지만 sustainable
> 아님. β ∈ [0.01, 0.1] 모두에서 round 1 catastrophic collapse,
> rolling reference도 무효. **Pure DPO는 1M scale에서 SFT 대체재가
> 아님**. Hybrid SFT+DPO 또는 round-0-only deployment 형태로만 유효
> 가능성. 새 도메인 deploy 전 ≥ 4 rounds 측정 + hybrid 변종 검토 필수.

## Reproducing

```bash
# Build
CUDA_HOME=/usr/local/cuda-12.5 PATH=/usr/local/cuda-12.5/bin:$PATH \
  cargo build -p llm-actors --example self_improve_rust --features cuda --release

# Rolling reference (GPU 0)
bash /tmp/p11s4_rolling.sh

# β sweep — frozen reference (GPU 1)
bash /tmp/p11s4_beta_sweep.sh

# Aggregate
bash scripts/phase11_s4/aggregate.sh
```

## See also

- `docs/phase11-s3-dpo-vs-sft.md` — S3 honest negative
- `docs/phase7-design.md` risk #13 — strengthened with S4 data
- `nanogpt-rs/src/dpo.rs`, `train.rs::train_dpo` — DPO implementation
- `llm-actors/examples/self_improve_rust.rs` — `--dpo-rolling-reference`
  flag added in this phase
