# Phase 10 S3 — S2 winner recipes를 K8 long-run에 (honest negative, Phase 10 종료)

## 동기 — Phase 9 S2 재방문

Phase 9 S2: K8 100K가 K8 30K보다 sum-AUC가 *낮다* (0.307 < 0.363,
mode collapse onto `\n`).

S2: K8 5K-step에서 (λ=0.3, k=8)과 (λ=0.1, k=2) 둘 다 sum-AUC를
baseline 0.421 위로 회복.

S3 가설: **S2 winner recipes가 K8 30K에서도 baseline 0.363을 능가
(scale-stable)**. Falsifier: 두 recipe 모두 0.363 이하면 S2는 5K-step
한정 transient.

## Setup

- 50M Llama recipe, 16K BPE, K8 KoWiki, block 256, batch 16, lr 6e-4
- **30K steps** (S2의 5K → 6×)
- A100 fp32, 두 변종을 두 GPU 병렬 (~36분)
- 측정: `critic_baseline_korean` 30 prompts × 20 samples = 600

## Result — S2 recipe는 30K로 scale 안 됨

| variant | top1 mass | pass | sum-AUC | Δ vs base | F=4 lift |
|---|--:|--:|--:|--:|--:|
| **baseline 30K** | (n/a) | 2.17% | **0.363** | — | 0.84× |
| JEPA λ=0.3, k=8 | 0.073 | 2.50% | **0.289** | **−0.074** | 0.65× |
| JEPA λ=0.1, k=2 | 0.090 | 2.67% | **0.330** | **−0.033** | 0.57× |

두 S2 winner 모두 baseline 30K보다 sum-AUC가 *낮다*. Phase 10 S1과
같은 패턴: top-1 mass / pass rate는 약간 올라가지만 (diversity++)
sum-AUC는 떨어진다 (calibration--).

**S2의 5K 결과가 transient였다.** S2에서 λ=0.3 k=8이 baseline 0.421을
0.433으로 회복시킨 건 5K-step 한정 현상. 30K로 늘리면 JEPA의 latent
distinctiveness가 더 강하게 자리잡고, 그것이 verifier-aligned
calibration을 다시 깎아먹는다.

## 100K 확장 안 함 — 결정 근거

S3 30K 결과로부터:
- baseline 30K: sum-AUC 0.363
- baseline 100K (Phase 9 S2): sum-AUC 0.307
- JEPA λ=0.3 30K: sum-AUC 0.289 (baseline 30K보다 worse)
- JEPA λ=0.1 k=2 30K: sum-AUC 0.330 (baseline 30K보다 worse)

30K에서 이미 baseline에 못 미치는 recipe를 100K로 늘려서 baseline
100K(0.307)와 비교할 동기가 없다. 두 가지 시나리오 다 "JEPA가 도움
안 됨"으로 향한다:

1. JEPA 100K가 baseline 100K보다 좋다 → 하지만 *baseline 30K* 보다는 못함
2. JEPA 100K가 baseline 100K보다도 worse → 더 나쁜 결과

100K 학습 (~2시간 × 2 variants)을 절약하고 honest negative로 마무리.

## Phase 10 종합 결론 (S1 + S2 + S3)

| Phase | Setup | Sum-AUC 결과 | 해석 |
|---|---|---|---|
| S1 | K8 5K, λ=0.1 k=8 | 0.421 → **0.238** | JEPA가 Shape-C 망친다 |
| S2 | K8 5K λ/k sweep | λ=0.3 k=8: **0.433** (recovered), λ=0.1 k=2: **0.432** | S1은 single-point. HP에 따라 회복 가능 |
| S2 | Python 1500 step | 모든 variant ~0.86 PASS | K8 anti-cal pathology는 K8 특정 |
| **S3** | **K8 30K λ=0.3 k=8** | **0.289** (baseline 0.363보다 worse) | **S2 5K 회복은 transient** |
| **S3** | **K8 30K λ=0.1 k=2** | **0.330** (baseline 0.363보다 worse) | **S2 5K 회복은 transient** |

**Phase 10 합산**: JEPA aux loss는 K8 long-run mode-collapse 문제를
풀지 못한다. 5K에서 일시적으로 좋아 보이는 setting도 30K로 늘리면
무너진다. JEPA를 K8에서 deploy 가치 없음.

**다른 도메인 (Python)에서는** JEPA가 calibration을 망가뜨리지는
않지만 *눈에 띄게 도와주지도 않는다*. S2 Python: λ=0.1 k=4가 F=4
lift 1.05× (baseline 1.01×보다 약간 위) — 통계적으로 유의한지 확실치
않은 +4%.

## Risk #12 최종 framing (S1 + S2 + S3 통합)

**Updated**:
> JEPA-style auxiliary objectives는 Shape-C calibration과 비단조-
> 비scale-stable한 상호작용을 갖는다. 한 (λ, k) 점에서 측정한
> 결과는 학습 budget이 늘어나면 뒤집힐 수 있다. K8 5K에서 calibration을
> 회복시키는 것처럼 보이는 hyperparameter는 K8 30K에서 모두
> baseline 아래로 떨어진다. 새 도메인에 JEPA를 deploy하기 전에
> **목표 학습 budget의 ≥ 50% 지점까지 직접 측정**할 것. 5K-step
> sweep으로 결정 내리지 말 것.

## Operational implication

- K8 (Korean BPE pretrain, mode-collapse-prone)에서 **JEPA 사용 금지**
- 다른 도메인 (Python 같은 verifier-tight 슬롯-fill)에서는 calibration
  중립이지만 lift gain도 미미 → "JEPA는 default로 안 켬"이 안전
- 추후 JEPA 재시도 시 prerequisite: 더 큰 모델 (≥ 200M), 더 다양한
  데이터 (multi-source), EMA target encoder 다른 decay 또는 다른
  predictor 구조

## What "Phase 10 done" means

Phase 10 deliverables:
- ✓ JEPA infrastructure (`nanogpt-rs/src/jepa.rs`, EMA target encoder)
- ✓ S1 single-point honest negative (5K, λ=0.1 k=8)
- ✓ S2 sweep — single-point claim 정정 + Python 도메인 무영향 확인
- ✓ S3 30K — S2 5K 회복은 transient임을 확인
- ✓ Risk #12 최종 framing (HP-, scale-, domain-sensitive)

JEPA는 이 프로젝트에서 추가 작업 없이 **일반 default로 비활성**.
다른 paper / setting에서 다른 결과가 나올 수 있지만, *K8 + 1M-50M
스케일*에서는 도움 안 됨.

## Reproducing

```bash
CUDA_VISIBLE_DEVICES=0 ./target/release/examples/train_kowiki_jepa \
  --steps 30000 --jepa-lambda 0.3 --jepa-offset 8 \
  --save checkpoints/p10s3_30k_lam03.safetensors

CUDA_VISIBLE_DEVICES=1 ./target/release/examples/train_kowiki_jepa \
  --steps 30000 --jepa-lambda 0.1 --jepa-offset 2 \
  --save checkpoints/p10s3_30k_k2.safetensors

bash scripts/phase10_s3/measure_30k.sh
```

## See also

- `docs/phase10-s2-jepa.md` — S2 sweeps (5K) where these recipes won
- `docs/phase10-s1-jepa.md` — S1 single-point negative
- `docs/phase7-design.md` risk #12 — calibration-vs-diversity 통합 framing
- Phase 9 S2 memory entry — K8 100K mode collapse, Phase 10 originally
  meant to address
