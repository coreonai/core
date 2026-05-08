# Phase 11 S3 — K9 DPO vs SFT 첫 측정 (honest negative)

Phase 11 S2가 DPO를 actor 기반 self-improve 루프에 wire-up. S3는
첫 직접 비교: K9 RustCodeDomain 4 rounds, 같은 seed에서 시작,
SFT vs DPO (β=0.1, reference = seed checkpoint).

## Setup

- 1M GPT (K9 default arch), char tokenizer, RustCodeDomain
- Pretrain seed: 1500 steps (둘 다 같은 `rust_seed.safetensors`)
- 4 rounds × (24 gen / 24 eval / 400 train steps)
- LR 5e-4 → 5e-5 cosine
- DPO β=0.1, reference = seed checkpoint, max_pairs_per_prompt=4
- A100 fp32, 두 GPU 병렬 (~5 min/run)

## Result

### SFT baseline (control)

| round | gen | eval before | eval after | Δ |
|--:|--:|--:|--:|--:|
| 0 | 0/24 (0%) | 0 | 6 | +6 |
| 1 | 0/24 | 7 | 7 | +0 |
| 2 | 9/24 (37.5%) | 7 | 11 | +4 |
| 3 | 0/24 | 11 | 11 | +0 |

Final eval **11/24 (45.8%)**, mean gen-pass **9/96 (9.4%)**. Phase 6
Shape C S4 baseline 재현 (그때 mean 9.4%).

### DPO β=0.1 (frozen reference at seed)

| round | gen | eval before | eval after | Δ |
|--:|--:|--:|--:|--:|
| 0 | **10/24 (41.7%)** | 1 | 7 | +6 |
| 1 | 0/24 | 7 | **0** | **−7** ⚠️ |
| 2 | 0/24 | 0 | 0 | +0 |
| 3 | 0/24 | 0 | 0 | +0 |

Final eval **0/24 (0.0%)**, mean gen-pass **10/96 (10.4%)**. **Round 0가
SFT 0/24 대비 41.7%로 압도적**, but round 1에 catastrophic collapse —
eval 7 → 0. Round 2, 3은 회복 불가.

DPO loss는 round 3 끝에 0.0158까지 떨어짐 — 매우 낮음. 모델이
degenerate state로 settle.

### Sample inspection (DPO round 3)

| prompt | completion |
|---|---|
| `fn main() { assert_eq!(` | `3 * - - - - - -` |
| `fn main() { assert_eq!(2 * (` | `3 - - - - - -` |
| `fn main() { assert_eq!(` | `3 * - - - - - -` |

Mode collapse onto repetitive `-` 토큰 — DPO가 *너무 강하게* rejected에서
멀어지려다가 의미있는 token distribution까지 깨버림.

## 메커니즘 분석

세 가지 가능성:

1. **β=0.1이 1M scale에서 너무 aggressive**. SFT가 r0→r1에서 부드럽게
   유지되는 데 비해 DPO는 한 round 만에 분포 깨뜨림.
2. **Reference 고정 문제**. Reference = seed (pretrain only). Round 1
   시작 시 policy = r0 (이미 DPO 한 번). `β · (π_r0 − π_seed)`의
   logp 차이가 커짐. r1 학습이 계속 seed로부터 멀어지려고 하면서
   r0의 좋은 학습이 wash out.
3. **400 train steps × β=0.1 = 너무 많은 negative gradient.**
   DPO에서 step 수 × β 조합이 effective drift budget인데, 둘 다
   conservative하게 줄여야 했을 수도.

## 정직한 해석

이 결과를 한 줄로 압축하면:

> DPO β=0.1이 round 0에서 SFT를 *압도*했지만, 그 lift는 sustainable
> 하지 않았다. 후속 round에서 mode collapse로 무너짐. DPO를 SFT의
> drop-in replacement로 쓰기엔 hyperparameter / reference 전략이
> 부족.

이건 Phase 5 (consensus)와 같은 패턴 — *처음에는 좋아 보이지만 multi-
round로 가면 무너지는* 것. round 0 시그널이 강하다는 게 더 의미있는
signal.

## Phase 11 S4 후보

Round 0의 +41.7pp lift를 sustain시키려면:

1. **β sweep**: 0.01, 0.03, 0.05 → 더 작은 step
2. **Rolling reference**: round n의 reference = round n-1's init (또는
   round n-1's policy). Reference가 policy를 따라 이동하면 logp
   차이가 무한정 커지지 않음.
3. **Train steps 줄이기**: 400 → 100 또는 200. Effective drift
   budget을 작게.
4. **Hybrid SFT+DPO**: round당 SFT 200 steps + DPO 100 steps.
   SFT가 pull-toward-correct, DPO가 push-away-from-rejected.
5. **β scheduling**: round 0에 β=0.1, 이후 β=0.03 (round 0의 강한
   신호만 활용하고 그 후로는 안정 유지).

## Risk register update — risk #13?

새 risk 후보를 docs/phase7-design.md에 추가 고려:

> **Risk #13: DPO multi-round은 single-round과 다른 dynamics.**
> 한 round의 DPO는 SFT보다 강한 신호 (Phase 11 S3 round 0:
> +41.7pp gen pass) but multi-round은 reference fixed면 catastrophic
> collapse. 새 도메인에 DPO deploy 전 *최소 4 rounds* 측정 + rolling
> reference 또는 β sweep 검토 필수.

## Stats

- Workspace: 119 unit tests, clippy + fmt clean (변동 없음 — 측정만)
- Commits: 측정 데이터 + 이 doc + memory entry

## Reproducing

```bash
# Build
CUDA_HOME=/usr/local/cuda-12.5 PATH=/usr/local/cuda-12.5/bin:$PATH \
  cargo build -p llm-actors --example self_improve_rust --features cuda --release

# SFT baseline (GPU 0)
CUDA_VISIBLE_DEVICES=0 ./target/release/examples/self_improve_rust \
  --rounds 4 --round-ckpt checkpoints/p11s3_sft \
  --seed-ckpt checkpoints/rust_seed.safetensors

# DPO β=0.1 (GPU 1, frozen reference at seed)
CUDA_VISIBLE_DEVICES=1 ./target/release/examples/self_improve_rust \
  --rounds 4 --round-ckpt checkpoints/p11s3_dpo \
  --seed-ckpt checkpoints/rust_seed.safetensors \
  --dpo-beta 0.1 --dpo-reference-from checkpoints/rust_seed.safetensors
```

## See also

- `docs/phase7-design.md` risk register (#13 후보)
- `nanogpt-rs/src/dpo.rs` — DPO loss
- `nanogpt-rs/src/train.rs::train_dpo` — DPO training entry point
- `llm-actors/src/curator_actor.rs::render_preference_pairs`
- Phase 11 S2 commit `a49a788` — wiring
