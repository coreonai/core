# Phase 12 design — DeepSeek V4 기술 4개를 workLLM에 가져오는 계획

DeepSeek V4 (2026-04-24)의 4개 핵심 혁신 — Muon optimizer, On-Policy
Distillation (OPD), mHC, CSA/HCA hybrid attention — 을 단계별로 우리
프로젝트에 반영하는 계획. 각 단계는 (a) 구현 범위, (b) falsifier
test, (c) risk, (d) decision gate 를 명시.

## Sequencing rationale

| # | Phase 12 sub | DeepSeek 기술 | 이유 (왜 이 순서) |
|--:|---|---|---|
| S1 | Muon optimizer | Muon | 가장 self-contained, 즉시 falsify 가능, 다른 기술의 기반 |
| S2 | OPD scaffolding | OPD | 가장 큰 leverage — 우리 Phase 11 RL 한계의 해답일 가능성 |
| S3 | OPD 첫 측정 | OPD | S2 인프라 위에서 K9 측정 |
| S4 | mHC | mHC | Architecture 변경 — Phase 10 JEPA처럼 risky, scale 검증 필요 |
| S5 (defer) | CSA/HCA | hybrid attn | block_size 1M 사용 사례 없음 — 큰 도메인 들어왔을 때 |

S1이 cheapest이자 다른 모든 phase에 영향 (학습 안정성). S2/S3가 most
impactful (Phase 11 RL collapse 해결책 후보). S4는 risky niche. S5는
사용 사례 부재로 보류.

---

## Phase 12 S1 — Muon optimizer

### Scope

`nanogpt-rs/src/optim/muon.rs` 새 모듈:

```rust
pub struct Muon {
    vars: Vec<Var>,
    lr: f64,
    momentum: f64,
    weight_decay: f64,
    /// 5 NS steps. Stage 1 (4 steps) coefficients
    /// (3.4445, −4.7750, 2.0315). Stage 2 (1 step) (2, −1.5, 0.5).
    ns_steps: usize,
    /// Per-Var momentum buffer.
    state: HashMap<usize, Tensor>,
}

impl Optimizer for Muon {
    fn step(&mut self, grads: &GradStore) -> CResult<()> {
        for var in &self.vars {
            let g = grads.get(var.as_tensor()).cloned();
            if g.is_none() { continue; }
            let g = g.unwrap();
            // Only orthogonalize 2D matrices (weights).
            // Bias/norm-scale: fall through to standard SGD-with-momentum.
            let update = if g.rank() == 2 {
                self.newton_schulz_orthogonalize(&g)?
            } else {
                g
            };
            // Momentum + weight decay update
            let m = self.state.entry(var.id()).or_insert_with(|| zeros_like(&update));
            *m = (m * self.momentum + &update)?;
            let new_v = (var.as_tensor() - &(m * self.lr)? - &(var.as_tensor() * (self.lr * self.weight_decay))?)?;
            var.set(&new_v)?;
        }
        Ok(())
    }
}

fn newton_schulz(x: &Tensor) -> CResult<Tensor> {
    // Orthogonalize via 5 Newton-Schulz iterations.
    // Normalize first to bound singular values.
    let norm = x.norm()?.max(1e-7);
    let mut x = (x / norm)?;
    let stage1_coeffs = [(3.4445_f64, -4.7750, 2.0315); 4];
    let stage2_coeffs = [(2.0_f64, -1.5, 0.5)];
    for (a, b, c) in stage1_coeffs.iter().chain(stage2_coeffs.iter()) {
        let xxt = x.matmul(&x.t()?)?;
        let xxt2 = xxt.matmul(&xxt)?;
        x = ((x * *a)? + (xxt.matmul(&x)? * *b)? + (xxt2.matmul(&x)? * *c)?)?;
    }
    Ok(x)
}
```

### Falsifier test

K9 RustCode 4 rounds × {AdamW, Muon} 비교:
- 같은 seed checkpoint
- 같은 LR schedule
- final eval + train loss curve

**Pass 기준**:
- final eval ≥ AdamW의 11/24 (regression 없음)
- 또는 train loss 더 빠르게 수렴 (5K steps 내 수렴 시점 비교)

**Fail 기준**: final eval < 11/24 또는 학습 발산.

### Risk

낮음. Optimizer 교체는 기존 학습 루프 외 변경 없음. AdamW로 즉시 rollback 가능.

### 비용

~250 LOC + 4-6 unit tests + 1 K9 측정 (~10 min). 한 세션 내 끝남.

### Decision gate

- Muon이 AdamW와 *동등하거나 우월* → Phase 13 search space에 13번째 axis (`optimizer: Adam | Muon`)로 추가. NAS가 자동으로 골라보게 함.
- Muon이 worse → honest negative + Muon은 deeper model 필요 가설로 보류.

---

## Phase 12 S2 — OPD 인프라 (scaffolding)

### Scope

3개 specialist + 1개 student를 동시 보유하는 actor 인프라:

#### `nanogpt-rs/src/distill_opd.rs` (또는 train.rs 확장)

```rust
pub fn train_opd(
    student_cfg: &GPTConfig,
    student_init_from: &Path,
    specialists: &[(SpecialistTag, PathBuf, GPTConfig)],
    teacher_weights: &[f32],  // Σ w_i = 1
    rollout_prompts: &[String],
    cfg: &TrainConfig,
    ...
) -> Result<TrainOutcome> {
    // For each step:
    //   1. Sample rollout from student (own-policy, on-the-fly)
    //   2. For each specialist: forward on (prompt + rollout) → logits
    //   3. KL(student logits || teacher logits) over full vocab, per token
    //   4. Σ_i w_i · KL_i → loss
    //   5. backward + step
}
```

#### `llm-actors/src/opd_supervisor.rs` (선택)

여러 specialist actor + 1 student actor를 동시 spawn. Round 단위로
specialist set은 frozen.

#### Specialist 정의

3개 도메인을 우리 기존 자산으로 매핑:
- **Code specialist**: K9 RustCode self-improve 결과 (`checkpoints/rust_round.r9.safetensors`)
- **Math specialist**: ArithmeticDomain 최대 학습 checkpoint
- **Korean specialist**: K8 30K KoWiki checkpoint

**중요**: DeepSeek는 specialist를 SFT + **GRPO**로 학습. 우리는
GRPO 미보유. **단순화**: 기존 SFT-trained checkpoint를 specialist로
가정. 진짜 GRPO는 Phase 13+ deferred.

### Falsifier test (정의만 — 측정은 S3에서)

OPD-trained student vs SFT baseline on multi-domain eval:
- K9 eval + Arithmetic eval + Korean eval 모두에서 측정
- OPD student가 *각 specialist에 가까운 성능* 보여야 함 ("merging works")

### Risk

중간. 구현 복잡도 높지만 framework는 잘 정의됨 (Hinton-style distillation
의 generalization). Multi-model GPU 메모리가 binding constraint —
1M 모델 4개 = 4M params, A100에 fit.

### 비용

~500 LOC (train_opd + multi-model loading + KL 계산). 인프라 한
세션, 측정 한 세션 = Phase 12 S2 + S3.

### Decision gate

S2 구현 완료 후 단순 smoke (3 specialists, 5 prompts × 50 steps,
loss 감소 확인). 그 후 S3로.

---

## Phase 12 S3 — OPD 첫 측정 vs Phase 11 SFT/DPO

### Scope

S2 인프라 위에서 첫 비교 측정:

```
Specialist set: {K9 SFT, Korean K8, Arithmetic 5K-step}
Student init: scratch 1M GPT (또는 K9 SFT 시작점)
Eval metrics:
  - K9 RustCode pass rate (24 prompts)
  - Korean fluency (heuristic)
  - Arithmetic (single-digit add/sub)
  - sum-AUC on each domain (Shape C 계열)
```

baseline 비교군:
1. Phase 11 SFT (current K9 best, 11/24)
2. Phase 11 S5 hybrid α=0.3 (single-round peak 18/24)
3. **OPD student** (이 세션의 새 측정)

### 가설

> OPD student가 K9에서 **≥ 11/24** (SFT 동등) AND Korean과
> Arithmetic에서 baseline above-chance.

이건 RL-replacement 가설 — DeepSeek가 RL을 OPD로 통째로 대체한
것이 우리 Phase 11 SFT/DPO matrix를 능가하는지 검증.

### Risk

높음. OPD 자체가 우리 toy scale에서 작동할지 미지수. DeepSeek
1.6T 결과가 1M에 transfer 안 될 수 있음 (Phase 9 S4의 1.5B vs 0.5B
역전 패턴). Falsifier로 honest negative 가능.

### Decision gate

- OPD ≥ SFT 11/24 across all 3 domains → **승리.** Phase 12 S4(mHC)로
- OPD < SFT → mechanism 분석. 가능성: (a) GRPO 안 한 specialist가 약함, (b) toy scale 한계, (c) full-vocab KL 잘못 구현
- OPD ≈ SFT but multi-domain → partial 승리. integrating capability 자체는 검증

---

## Phase 12 S4 — mHC (deferred contingent on S3)

### Scope

`nanogpt-rs/src/model.rs::Block`을 multi-stream으로 확장:

```rust
pub struct Block {
    cfg: GPTConfig,
    attn: CausalSelfAttention,
    mlp: FeedForward,
    ln1: Norm,
    ln2: Norm,
    /// Phase 12 S4: mHC streams (≥ 2 enables hyper-connection mode).
    hc_streams: usize,
    /// (hc_streams × hc_streams) doubly-stochastic mixing matrix,
    /// projected onto Birkhoff polytope each step via Sinkhorn-Knopp.
    hc_mixing: Option<Var>,
}
```

수학적 연산:
1. Block 시작에 `streams = vec![x; hc_streams]` (단일 입력 → N stream)
2. Attention/MLP는 stream별 독립 적용
3. 각 layer 끝에 `streams = mixing_matrix @ streams` (mixing)
4. 마지막 block 끝에 `streams.sum() / hc_streams`로 단일 stream 회수

Sinkhorn-Knopp:
```rust
fn sinkhorn_knopp(m: &Tensor, n_iter: usize) -> CResult<Tensor> {
    let mut m = m.exp()?;  // ensure positive
    for _ in 0..n_iter {
        // row normalize
        let row_sum = m.sum_keepdim(1)?;
        m = m.broadcast_div(&row_sum)?;
        // col normalize
        let col_sum = m.sum_keepdim(0)?;
        m = m.broadcast_div(&col_sum)?;
    }
    Ok(m)
}
```

### Falsifier test

Phase 10 패턴: K8 5K + 30K 두 scale 측정.

| metric | baseline | mHC streams=2 |
|---|---|---|
| top-1 mass | TBD | TBD |
| sum-AUC | TBD | TBD |
| signal magnitude across layers (max norm ratio) | TBD | TBD |

### Risk

매우 높음. DeepSeek가 검증한 건 3B/9B/27B. 우리 50M에서 underpowered
HC가 의미 있을 가능성 < 50%. Phase 10 JEPA처럼 honest negative 가능.

### Decision gate

S3 결과가 좋으면 진행. 안 좋으면 mHC도 보류 (DeepSeek 기술 모음을
3개에서 1-2개로 줄이는 게 정직).

---

## Phase 12 S5 (deferred indefinitely) — CSA/HCA hybrid attention

block_size 256-1024가 우리 default. 1M context 사용 사례 없음.
KoWiki / RustCode 모두 short prompt. 다음 사용 사례 후보:
- 다문서 retrieval-augmented self-improve (Phase 13+)
- 더 큰 Korean corpus (multi-source)
- 외부 모델과의 long-context 통합

이 사용 사례가 생기기 전에는 sparse attention 학습은 *over-engineering*.

---

## Phase 12 first session scope (S1 시작)

이 design doc 다음으로 진행할 한 세션:

**Phase 12 S1**:
1. `nanogpt-rs/src/optim/mod.rs` + `muon.rs` 작성
2. Newton-Schulz iteration unit tests (orthogonalization 검증, edge cases)
3. AdamW vs Muon `train_from_full` 호환 인터페이스
4. K9 RustCode 4-round 비교 (AdamW vs Muon)
5. 결과 → `docs/phase12-s1-muon.md`
6. Decision: NAS 13번째 axis로 승격 vs honest negative

비용: 한 세션 (~3-5 hours of coding + 1 GPU run).

## 우리 프로젝트의 falsifier-test workflow와의 정합성

DeepSeek 기술 4개를 일괄 도입하는 게 아니라 *각각 falsifier test로
검증* 하는 것이 우리 패턴. Phase 5/6/7/8/9/10/11 다 그 패턴.
가능한 결과 분기:

- **모두 작동**: Phase 12 끝에 Muon + OPD + mHC 다 채택. 14개 commit, 4개 honest positive.
- **OPD만 작동**: S1 Muon ✓, S3 OPD ✓, S4 mHC ✗. 결과 하나라도 메인 win.
- **다 실패**: 4개 negative. DeepSeek 기술이 우리 toy scale에 transfer 안 됨. Phase 9 S4 1.5B 패턴과 일관.
- **OPD가 우리 Phase 11 행렬을 흡수**: 가장 큰 win. Phase 11 5세션 작업이 OPD로 통합/대체. RL training infrastructure 단순화 가능.

각 단계마다 commit + risk register update + memory entry. 4-12개
commit 사이.

## See also

- `docs/phase7-design.md` — 13 risks register
- `docs/phase11-s5-hybrid-dpo.md` — Phase 11 RL 한계 결론
- `book/llm_book.md` — 12 phase narrative (Phase 12 추가 예정)
- DeepSeek V4 paper sources (이 doc 외부 — Notion 참조)
