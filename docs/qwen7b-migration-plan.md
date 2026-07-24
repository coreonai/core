---
title: "Qwen2.5-Coder-0.5B → 7B 전환 계획 (eval + SFT LoRA, A100 40GB)"
date: "2026-07-25"
---

# 목표 & 범위

- **모델**: `Qwen2.5-Coder-0.5B` → **`Qwen2.5-Coder-7B`** (비양자화 bf16).
- **워크로드**: 추론/평가(HumanEval·MBPP) + **multi-round SFT LoRA**. REINFORCE 는
  이번 범위 제외(autograd 그래프가 생성 토큰까지 유지 → 40GB 단일 카드에서 OOM
  위험이 커서 별도 과제로 defer).
- **하드웨어**: A100 **40GB 단일 GPU**. 모델 1개는 반드시 1 GPU 에 적재
  (Candle `VarBuilder` 단일 device; 텐서 병렬 없음). bf16 7B ≈ 15GB 라 적재 OK.

# 전제(코드 변경 불필요) — 이미 동작

- **차원 자동 적응**: `QwenModelActor::from_snapshot_dir` 가 `config.json` 을
  `Qwen2Config` 로 파싱(`qwen_model_actor.rs:107`). 7B 의 layer/head/hidden/
  intermediate 가 그대로 반영됨.
- **tied/untied lm_head 자동 감지**: LoRA fork 가
  `base_vb.contains_tensor("lm_head.weight")` 로 분기(`qwen2_lora.rs:509`).
  0.5B(tied) → 7B(untied) 전환이 코드 변경 없이 처리됨. **단, 전 shard 로드 후에만
  정확** → 아래 2단계가 전제.

# 필수 코드 변경 3가지

### C1. Sharded safetensors 로더 (핵심)
7B 는 `model-0000N-of-0000M.safetensors` + `model.safetensors.index.json` 구조.
단일 `model.safetensors` 를 가정하는 3곳을 shard-aware 로:

| 위치 | 현재 | 변경 |
|------|------|------|
| `qwen_model_actor.rs:104` | `join("model.safetensors")` | 공용 헬퍼 사용 |
| `qwen_trainer_actor.rs:186` | 동일 | 공용 헬퍼 사용 |
| `qwen2_lora.rs:669` `save_merged_lora` | `safetensors::load(단일)` | 전 shard 로드 후 병합 |

- 공용 헬퍼 신설: `resolve_safetensors(dir) -> Vec<PathBuf>` — `index.json` 있으면
  `weight_map` 값의 유니크 shard 목록, 없으면 `[model.safetensors]`.
  `VarBuilder::from_mmaped_safetensors` 는 이미 슬라이스를 받으므로 로더 시그니처
  변경 최소.

### C2. 모델 경로 CLI 오버라이드
모든 example 이 `"…/Qwen2.5-Coder-0.5B/snapshots"` 를 포맷 문자열로 하드코딩.
`--model-dir <snapshot_dir>` 플래그 추가(기본값은 기존 0.5B 상수 유지). 대상:
`phase22_humaneval_baseline`, `phase22_mbpp_baseline`, `phase22_he_mr_sft`,
`phase22_mbpp_mr_sft`, 그리고 스모크(`phase21_qwen_candle_smoke`,
`phase21_qwen_lora_smoke`).

### C3. 학습 dtype F32 → BF16
`from_snapshot_dir(dtype)` 가 base·LoRA 를 같은 dtype 로 묶음. 학습 example 이
넘기던 F32 를 **BF16** 로.
- **bf16 ≠ fp16**: bf16 은 f32 와 동일 지수 범위 → GradScaler 불필요.
  메모리의 **fp16-NaN 이슈(Candle autocast/GradScaler 부재)와 무관**.
- 1차는 base+LoRA 전부 bf16. 발산 시에만 base bf16 + LoRA f32 마스터 분리
  (LoRA matmul 경계 캐스팅, 추가 소공수).
- `_slow` gradient 경로(`rope_slow`/`softmax`/`rms_norm_slow`) 유지 —
  attention materialize 지점이므로 seq len 을 짧게 유지해 메모리 관리.

# 메모리 예산 (A100 40GB, bf16)

- base 가중치: 7.6B × 2B ≈ **15.2 GB** (frozen, mmap).
- LoRA(q/v_proj r=16) 파라미터 + AdamW state: 수십 MB.
- 활성/autograd 그래프: batch·seq·`_slow` attention 에 지배됨.
- **평가/생성**: 여유(≤ 20GB). 문제 없음.
- **SFT LoRA**: `--sft-batch-size 1~2` + seq ≤ 256 로 시작 → 프로파일 후 상향.
  batch=2/seq=256 는 40GB 안쪽 가능성 높음. OOM 시 batch↓ / seq↓ / max_new↓.

# 실행 순서 (검증 게이트 포함)

1. **다운로드**: `huggingface-cli download Qwen/Qwen2.5-Coder-7B`
   (~15GB, 4-shard). ⚠ **AWQ 버전 금지**(양자화 → Candle 로더 비호환).
2. **C1+C2 구현** → 스모크 게이트:
   - `phase21_qwen_candle_smoke --model-dir <7B>` = 유효 Python 출력
     (config 파싱 + shard 로드 + tie 감지 + 생성 sanity 동시 검증).
3. **7B base 평가**: `phase22_humaneval_baseline --model-dir <7B> --sequential
   --aggregate` 로 pass@1/pass@k 측정. 7B 라 base 가 0.5B(0.22)보다 크게 높을 것 →
   headroom·saturation 형태가 다를 것으로 예상.
4. **C3 구현** → gradient 게이트:
   - `phase21_qwen_lora_smoke --model-dir <7B>` bf16 에서 loss 하락 확인
     (0.5B 는 −57% 였음).
   - merge 왕복 검증: `save_merged_lora` 후 재로드 logits 일치(G8 의심 지점).
5. **7B multi-round SFT**: `phase22_he_mr_sft --model-dir <7B> --rounds 2
   --sft-batch-size 2 --train-steps <작게>` → 메모리 프로파일 후 rounds/batch 상향.
   HumanEval saturation 재현, 필요 시 MBPP cross-substrate.

# 리스크 & 완화

| 리스크 | 완화 |
|--------|------|
| bf16 LoRA 학습 발산 | base bf16 + LoRA f32 마스터 분리(경계 캐스팅) |
| SFT OOM (batch/seq) | batch=1 / seq·max_new 축소 / 활성 최소화 |
| 7B config 필드가 fork 가정과 불일치(예: 명시적 `head_dim`) | 2단계 스모크에서 조기 발견, 필요 시 Config 필드 추가 |
| merged 체크포인트 15GB I/O | `/raid` 디스크 사용(shm 금지), 병합 파일 정리 |
| saturation curve wallclock 폭증(7B ≈ 14× FLOPs) | 먼저 base+r=1 단발로 신호 확인 후 curve 결정. seed 병렬은 GPU 당 1런 |

# 노력 추정

- **C1 shard 로더**: 반나절(대부분 여기).
- **C2 `--model-dir` + C3 bf16 스위치**: 각 1~2시간.
- **다운로드 + eval 스모크**: 수 시간(대역폭).
- **SFT 프로파일/튜닝**: 메모리 프로파일에 따라 가변.

# 향후(범위 밖)

- REINFORCE on 7B: micro-batch=1 + k=2 + 짧은 max_new + `/raid` adapter-sync 필요.
  40GB 에선 별도 실험으로 신중히.
- 멀티 GPU 텐서 병렬: 현재 아키텍처 밖(모델 1개 = 1 GPU).
