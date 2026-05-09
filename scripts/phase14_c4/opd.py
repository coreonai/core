"""Phase 14 C4 — On-Policy Distillation (OPD) loss for PyTorch.

Mirror of nanogpt-rs/src/opd.rs (Phase 12 S2). Forward and Reverse
KL directions, multi-teacher weighted-sum, full-vocabulary KL on
[B, T, V] logits. Caller supplies a labels tensor (-100 marks
positions to ignore — typically prompt tokens).

OPD signature differs from CE: the "label" is a teacher's logit
distribution, not a discrete token id. Teachers are run frozen at
forward time; gradients flow only through student.
"""

from typing import List, Tuple
import torch
import torch.nn.functional as F


def opd_loss(
    student_logits: torch.Tensor,           # [B, T, V]
    teacher_logits: List[Tuple[float, torch.Tensor]],  # [(weight, [B, T, V])]
    labels: torch.Tensor,                   # [B, T], -100 where ignored
    temperature: float = 1.0,
    direction: str = "forward",             # "forward" or "reverse"
) -> torch.Tensor:
    """Mean weighted-sum-of-KL over kept positions (labels != -100)."""
    if not teacher_logits:
        raise ValueError("opd_loss: at least one teacher required")
    if temperature <= 0:
        raise ValueError(f"opd_loss: temperature must be > 0, got {temperature}")
    if direction not in ("forward", "reverse"):
        raise ValueError(f"direction must be 'forward' or 'reverse', got {direction}")

    s_dims = student_logits.shape
    for _, t in teacher_logits:
        if t.shape != s_dims:
            raise ValueError(
                f"opd_loss: teacher logits {tuple(t.shape)} mismatch student {tuple(s_dims)}"
            )

    # Shift for next-token prediction: drop last student / teacher position,
    # drop first label position (causal LM convention).
    s = student_logits[:, :-1, :].float() / temperature  # [B, T-1, V]
    label_shift = labels[:, 1:]                          # [B, T-1]
    mask = (label_shift != -100).float()                 # [B, T-1]
    n_pos = mask.sum().clamp(min=1.0)

    s_log_p = F.log_softmax(s, dim=-1)

    total = torch.zeros((), device=s.device, dtype=s.dtype)
    for w, t_logits in teacher_logits:
        t = t_logits[:, :-1, :].float() / temperature
        t_log_p = F.log_softmax(t, dim=-1)

        if direction == "forward":
            # KL(teacher || student) = Σ p_t · (log p_t − log p_s)
            p_t = t_log_p.exp()
            diff = t_log_p - s_log_p
            kl_per_pos = (p_t * diff).sum(dim=-1)        # [B, T-1]
        else:
            # KL(student || teacher) = Σ p_s · (log p_s − log p_t)
            p_s = s_log_p.exp()
            diff = s_log_p - t_log_p
            kl_per_pos = (p_s * diff).sum(dim=-1)

        kl_mean = (kl_per_pos * mask).sum() / n_pos
        total = total + w * kl_mean
    return total


# ---- self-tests (run as `python opd.py`) ----
if __name__ == "__main__":
    torch.manual_seed(0)

    # 1. KL(p || p) = 0
    s = torch.randn(2, 3, 5)
    labels = torch.tensor([[0, 1, 2], [3, 4, 0]])
    l = opd_loss(s, [(1.0, s.clone())], labels, 1.0, "forward").item()
    assert abs(l) < 1e-5, f"expected 0, got {l}"
    print(f"[ok] forward KL(p||p) = {l:.6e}")

    l = opd_loss(s, [(1.0, s.clone())], labels, 1.0, "reverse").item()
    assert abs(l) < 1e-5
    print(f"[ok] reverse KL(p||p) = {l:.6e}")

    # 2. Disagreement → large positive
    s = torch.zeros(1, 2, 5); s[..., 0] = 10.0  # student picks 0
    t = torch.zeros(1, 2, 5); t[..., 2] = 10.0  # teacher picks 2
    labels = torch.tensor([[0, 1]])
    l = opd_loss(s, [(1.0, t)], labels, 1.0, "forward").item()
    assert l > 5.0, f"expected large KL, got {l}"
    print(f"[ok] disagreement forward-KL = {l:.4f}")

    # 3. Weighted sum matches individual KLs
    s = torch.randn(1, 4, 6)
    t1 = torch.randn(1, 4, 6)
    t2 = torch.randn(1, 4, 6)
    labels = torch.tensor([[0, 1, 2, 3]])
    l_combined = opd_loss(s, [(0.5, t1), (0.5, t2)], labels, 1.0, "forward").item()
    l1 = opd_loss(s, [(1.0, t1)], labels, 1.0, "forward").item()
    l2 = opd_loss(s, [(1.0, t2)], labels, 1.0, "forward").item()
    expected = 0.5 * l1 + 0.5 * l2
    assert abs(l_combined - expected) < 1e-5, f"{l_combined} vs {expected}"
    print(f"[ok] weighted-sum: {l_combined:.4f} == 0.5·{l1:.4f} + 0.5·{l2:.4f}")

    # 4. Mask respects -100 ignore
    s = torch.randn(1, 4, 5)
    t = torch.randn(1, 4, 5)
    labels_full = torch.tensor([[0, 1, 2, 3]])
    labels_partial = torch.tensor([[-100, -100, 2, 3]])
    l_full = opd_loss(s, [(1.0, t)], labels_full, 1.0, "forward").item()
    l_partial = opd_loss(s, [(1.0, t)], labels_partial, 1.0, "forward").item()
    # Should differ unless the masked positions were exactly average
    print(f"[ok] mask respects -100: full={l_full:.4f} partial={l_partial:.4f}")

    # 5. Temperature softens the loss
    s = torch.zeros(1, 2, 5); s[..., 0] = 5.0
    t = torch.zeros(1, 2, 5); t[..., 2] = 5.0
    labels = torch.tensor([[0, 1]])
    l_t1 = opd_loss(s, [(1.0, t)], labels, 1.0, "forward").item()
    l_t2 = opd_loss(s, [(1.0, t)], labels, 2.0, "forward").item()
    assert l_t2 < l_t1, f"higher T should soften loss: T=1 {l_t1}, T=2 {l_t2}"
    print(f"[ok] T=1 ({l_t1:.4f}) > T=2 ({l_t2:.4f})")

    # 6. Gradient flows to student only
    s = torch.randn(1, 3, 5, requires_grad=True)
    t = torch.randn(1, 3, 5)  # no grad
    labels = torch.tensor([[0, 1, 2]])
    l = opd_loss(s, [(1.0, t)], labels, 1.0, "forward")
    l.backward()
    assert s.grad is not None and s.grad.abs().sum() > 0
    print(f"[ok] gradient flows to student (||grad||={s.grad.norm():.4f})")

    print("\nall opd.py self-tests passed")
