# Phase 11 S5 — Hybrid SFT+DPO + Round-0-only DPO

S4 결정: pure DPO multi-round은 1M K9 scale에서 viable 아님 (β sweep
+ rolling reference 모두 r1 collapse). S5는 **structural** fix 두 가지
+ combined fix:

1. **Hybrid loss**: `total = (1-α)·DPO + α·SFT_chosen`. SFT가 anchor,
   DPO가 contrast. α ∈ {0.3, 0.5, 0.7}.
2. **Round-0-only DPO**: r0만 DPO (β=0.1), r1+ SFT. S3의 +41.7pp
   round-0 boost 살리고 multi-round collapse 회피.
3. **Combined**: β=0.05 (S4의 작은 β) + α=0.5 (S5의 SFT anchor).

## Implementation

### `train_dpo` 변경

새 파라미터 `sft_anchor_weight: f64` ∈ [0, 1]:

```rust
let loss = if sft_anchor_weight > 0.0 {
    let sft = -mean(per-pair sum_logp_chosen / n_chosen_tokens);
    (dpo * (1.0 - sft_anchor_weight)) + (sft * sft_anchor_weight)
} else {
    dpo
};
```

`0`이면 backward-compat (S2/S3/S4 동작 동일).

### 새 unit tests (2개)
- `train_dpo_hybrid_widens_gap_and_keeps_chosen_logp_high`: hybrid가 chosen-rejected gap을 넓히면서 chosen logp를 안 떨어뜨림
- `train_dpo_rejects_invalid_sft_anchor_weight`: α ∉ [0, 1] 거절

### `RoundConfig.dpo_sft_anchor_weight`, `--dpo-sft-anchor-weight`, `--dpo-round-zero-only`

기존 wiring에 새 필드 추가.

## Result — 11 variants matrix (S3 baseline + S4 + S5)

| variant | r0 (gen / eval) | r1 | r2 | r3 | final eval | best eval | mean gen |
|---|---|---|---|---|--:|--:|--:|
| **SFT** (control) | 0%, 0→6 | 0%, 7→7 | 9/24 (37.5%), 7→11 | 0%, 11→11 | **11/24** | 11 (r2) | 9.4% |
| pure DPO β=0.1 frozen (S3) | **41.7%**, 1→7 | 0%, 7→**0** | 0/0 | 0/0 | 0/24 | 7 (r0) | 10.4% |
| pure DPO β=0.1 rolling | 0%, 0→7 | 29.2%, 7→**0** | 0/0 | 0/0 | 0/24 | 7 (r0) | 7.3% |
| pure DPO β=0.01 frozen | 0%, 0→6 | 0%, 7→**0** | 0/0 | 0%, 0→11 | 11/24 | 11 (r3) | 0% |
| pure DPO β=0.03 frozen | 0%, 0→7 | 0%, 7→**0** | 0/0 | 0/0 | 0/24 | 7 (r0) | 0% |
| pure DPO β=0.05 frozen | 0%, 0→7 | 33.3%, 7→**0** | 0/0 | 0%, 0→7 | 7/24 | 7 (r0) | 8.3% |
| **hybrid α=0.3** | 0%, 0→0 | **41.7%**, 0→**18** ★ | 0%, 18→11 | 0%, 11→11 | 11/24 | **18 (r1)** ★ | 10.4% |
| hybrid α=0.5 | 0%, 0→0 | 45.8%, 0→7 | 0%, 7→11 | 0%, 11→11 | 11/24 | 11 (r2) | 11.5% |
| hybrid α=0.7 | 0%, 0→0 | 45.8%, 0→0 | 0%, 0→11 | 0%, 11→11 | 11/24 | 11 (r2) | 11.5% |
| **round-0-only DPO** | 0%, 0→**11** ★ | 0%, 7→7 | 0%, 7→11 | 0%, 11→11 | 11/24 | 11 (r0) | 0% |
| combined β=0.05 α=0.5 | 0%, 0→0 | 45.8%, 0→0 | 0%, 0→11 | 0%, 11→11 | 11/24 | 11 (r2) | 11.5% |

## 핵심 관찰

### 1. SFT의 final eval (11/24)이 robust ceiling

11개 variant 중 **모두** final eval ≤ 11/24. SFT를 능가하는 multi-round
DPO 구성은 **없음**.

### 2. Hybrid α=0.3 single-round 75% — 프로젝트 record

α=0.3 r1 eval **18/24 (75.0%)** — SFT의 best eval (11/24)을 +7 능가.
그러나 r2에 11로 떨어짐, sustain 안 됨. 다만 *intermediate
checkpoint*가 75% eval인 건 진짜 신호 — best-of-rounds로
체크포인트 선택하면 SFT를 능가할 수 있음.

### 3. Round-0-only DPO가 1 round 단축

r0 eval 0→11 (vs SFT 0→6). SFT 4 rounds → r2에 첫 11. Round-0-only
DPO는 r0에 11 도달. **Compute-efficiency win** (1 fewer round to
reach SFT max).

### 4. SFT anchor weight 단조 효과

```
α=0   (pure DPO):  collapse
α=0.3:             r1 +18 spike → r2 settle
α=0.5:             r1 +7,  r2 +4 → smooth climb
α=0.7:             r1 +0,  r2 +11 → SFT-like
α=1   (pure SFT):  baseline
```

α=0.3이 DPO 신호 + SFT 안전망 sweet spot.

### 5. Multi-round dynamics

- pure DPO β ≥ 0.03: 영원한 collapse (4 of 6 variants stuck at 0)
- pure DPO β=0.01: r3 recover by chance (matches SFT)
- hybrid α ≥ 0.3: 모두 final 11/24 도달, no permanent collapse
- round-0-only: 보호 ✓ + compute-efficient

## Phase 11 종합 결론 (S1 + S2 + S3 + S4 + S5)

| level | finding |
|---|---|
| S1 (DPO loss) | softplus-stable, 7 unit tests pass |
| S2 (loop wire) | actor wire-up, +6 preference_pair tests, 119 total |
| S3 (first GPU) | r0 +41.7pp single-round signal but r1 collapse |
| S4 (HP fix) | β + rolling 모두 collapse → fundamental mechanism |
| S5 (structural fix) | hybrid α=0.3 r1 75% peak, round-0-only 1-round-faster, **but no variant beats SFT final 11/24** |

### Risk #13 강화 (final form)

> **DPO multi-round은 1M K9 scale에서 SFT를 능가 못 함.** 11 variant
> matrix (β ∈ [0.01–0.1], rolling vs frozen ref, hybrid α ∈ [0–0.7],
> round-0-only, combined) 모두 final eval ≤ SFT의 11/24. Hybrid α=0.3은
> 단일 round eval 18/24 (75%, project record)을 만들지만 sustain 안 됨.
> **최선의 deployment 패턴**:
> 1. Round-0-only DPO — SFT보다 1 round 빨리 baseline 도달
> 2. Hybrid α=0.3 — best-of-rounds checkpoint selection 가능 시
> 3. Pure DPO — 권장 안 함

### 메커니즘 정리

S4가 의심한 "rejected pile noise"는 **부분적으로 맞음**: SFT anchor
가 들어가면 collapse 안 일어남. 하지만 SFT anchor 자체가 학습을
SFT 동학으로 수렴시킴 — DPO의 추가 신호가 결국 final eval에 흡수
안 됨.

K9 toy scale의 한계도 가능성: 21 distinct (prompt, slot) 페어가
fine-grained DPO 신호를 뽑기엔 너무 적음. 더 큰 도메인 (HumanEval,
실제 verifier-rich task)에서 다시 측정 가치 있음.

## Phase 11 deferred 후보 (Phase 12 영역)

1. **외부 모델 (Qwen 0.5B)에 DPO 적용** — 1M scale 한계 검증.
   Phase 9 S6 인프라 (--critic-oversample) + DPO 결합.
2. **Per-token DPO** — sequence sum 대신 per-token IPO/KTO 변종.
3. **Online DPO (round-K reference)** — reference를 round K snapshot로,
   매 round 후 reference 추가 학습.
4. **Reward-shaped DPO** — verifier verdict 외에 partial credit 지표
   사용 (cargo build success but assert fail = partial).

## Reproducing

```bash
# Hybrid α sweep — GPU 0 sequential (~15 min)
bash /tmp/p11s5_hybrid.sh

# Round-0-only DPO — GPU 1 (~5 min)
bash /tmp/p11s5_round_zero.sh

# Combined fix β=0.05 α=0.5 — GPU 1 (~5 min)
bash /tmp/p11s5_combined.sh

# Aggregate all S3 + S4 + S5 variants
bash scripts/phase11_s5/aggregate.sh
```

## See also

- `docs/phase11-s4-dpo-fixes.md` — S4 honest negative (β + rolling 둘 다 collapse)
- `docs/phase11-s3-dpo-vs-sft.md` — S3 first DPO measurement
- `docs/phase11-s2-dpo-loop-integration` (memory) — actor wire-up
- `nanogpt-rs/src/train.rs::train_dpo` — `sft_anchor_weight` parameter
- `nanogpt-rs/src/dpo.rs` — DPO loss
- `docs/phase7-design.md` risk #13 (final form, 5 sub-sessions)
