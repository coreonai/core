"""Phase 14 C2 — Muon optimizer for PyTorch (LoRA-adapter use case).

Mirror of nanogpt-rs/src/muon.rs. Same Newton-Schulz iteration
schedule (4 fast + 1 stabilize), same per-parameter dispatch (2-D
matrices → NS orthogonalize, 1-D → SGD-momentum fallback),
decoupled weight decay.

Designed for PEFT LoRA adapters: q_proj/v_proj LoRA delta_A
(in_dim × r) and delta_B (r × out_dim) are exactly the small-dense
2-D matrices Muon's NS targets.
"""

from typing import Iterable

import torch


def newton_schulz(x: torch.Tensor, n_steps: int = 5) -> torch.Tensor:
    """Approximate the orthogonal polar factor of `x` (X = Q · S → Q).

    Pre-normalize by Frobenius norm so σ_max ≤ 1, then 5 NS iterations
    with DeepSeek V4's coefficient schedule (4 fast + 1 stabilize).
    """
    if x.dim() != 2:
        raise ValueError(f"newton_schulz expects 2-D matrix, got {x.dim()}-D")
    frob = x.norm().clamp(min=1e-7)
    x = x / frob
    stage1 = (3.4445, -4.7750, 2.0315)
    stage2 = (2.0, -1.5, 0.5)
    n_stage1 = max(0, n_steps - 1)
    for i in range(n_steps):
        a, b, c = stage1 if i < n_stage1 else stage2
        xxt = x @ x.T
        xxt_x = xxt @ x
        xxt2_x = xxt @ xxt_x
        x = a * x + b * xxt_x + c * xxt2_x
    return x


class Muon(torch.optim.Optimizer):
    """Newton-Schulz orthogonalized SGD-momentum.

    For 2-D parameters: gradient → momentum → NS orthogonalize → step.
    For 1-D parameters: gradient → momentum → step (no NS).
    Decoupled weight decay (AdamW-style).

    Drop-in for `torch.optim.AdamW` for LoRA adapter training.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        momentum: float = 0.95,
        fallback_momentum: float = 0.9,
        weight_decay: float = 0.01,
        ns_steps: int = 5,
    ):
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if not 0 <= momentum < 1:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        defaults = dict(
            lr=lr, momentum=momentum, fallback_momentum=fallback_momentum,
            weight_decay=weight_decay, ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            mom = group["momentum"]
            mom_1d = group["fallback_momentum"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "momentum_buf" not in state:
                    state["momentum_buf"] = torch.zeros_like(p)
                m = state["momentum_buf"]
                # momentum update (β depends on rank — 2-D params get
                # the higher Muon momentum; 1-D fall back).
                beta = mom if p.dim() == 2 else mom_1d
                m.mul_(beta).add_(grad)

                # 2-D: NS orthogonalize the momentum direction.
                # 1-D: use raw momentum (SGD-mom fallback).
                if p.dim() == 2:
                    # Cast to fp32 for the NS computation; LoRA adapters
                    # may live in fp16.
                    direction = newton_schulz(m.float(), ns_steps).to(m.dtype)
                else:
                    direction = m

                # Decoupled weight decay (AdamW-style):
                #   θ ← θ · (1 − lr·wd) − lr · direction
                p.data.mul_(1.0 - lr * wd).add_(direction, alpha=-lr)

        return loss


# ---- Self-tests (run as `python muon.py`) ----
if __name__ == "__main__":
    torch.manual_seed(0)

    # 1. NS on identity → identity
    x = torch.eye(4)
    q = newton_schulz(x, 5)
    assert torch.allclose(q, x, atol=0.1), f"NS(I) ≠ I: {q}"
    print("[ok] newton_schulz(I) ≈ I")

    # 2. NS produces near-orthogonal columns
    x = torch.randn(6, 6)
    q = newton_schulz(x, 5)
    qtq = q.T @ q
    err = (qtq - torch.eye(6)).abs().max().item()
    assert err < 0.15, f"Q^T Q far from I: max_err={err}"
    print(f"[ok] newton_schulz random Q^T Q ≈ I (max_err {err:.4f})")

    # 3. Muon optimizer reduces a quadratic loss
    target = torch.randn(4, 4) * 1.0  # bigger target so loss starts higher
    w = torch.zeros(4, 4, requires_grad=True)
    opt = Muon([w], lr=0.1)
    losses = []
    for _ in range(80):
        loss = ((w - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5, (
        f"Muon failed to reduce loss: start={losses[0]:.4f} end={losses[-1]:.4f}"
    )
    print(f"[ok] Muon: loss {losses[0]:.4f} → {losses[-1]:.4f}")

    # 4. 1-D fallback works
    target_1d = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = torch.zeros(4, requires_grad=True)
    opt = Muon([b], lr=0.05)
    for _ in range(40):
        loss = ((b - target_1d) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert (b.detach() - target_1d).abs().max() < 0.5, (
        f"1-D fallback failed: {b.detach()}"
    )
    print(f"[ok] 1-D fallback: b ≈ target ({b.detach().tolist()})")

    print("\nall muon.py self-tests passed")
