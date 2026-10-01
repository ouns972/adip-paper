"""Teacher-forcing CE losses for autoregressive bit models."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from adip.ar_model import AutoregressiveBitModel


def incumbent_teacher_forcing_ce_loss(
    model: AutoregressiveBitModel,
    x: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Mean (or sum) over sites ``t = 0 … N-1`` of cross-entropy between
    next-bit logits given prefix ``x_{<t}`` and target ``x[:, t]``.

    Uses a single causal :meth:`AutoregressiveBitModel.forward` pass (teacher forcing),
    which is mathematically the same as calling :meth:`~AutoregressiveBitModel.forward_next`
    at each site but far cheaper in memory.
    """
    x = x.long()
    if x.ndim != 2:
        raise ValueError(f"x must be (B, N), got {x.shape}")
    _, n = x.shape
    if n != model.num_sites:
        raise ValueError(f"x has N={n}, ADIP num_sites={model.num_sites}")
    logits = model.forward(x)  # (B, N, 2)
    # cross_entropy expects (N, C) class dim; flatten batch×site
    per = F.cross_entropy(
        logits.reshape(-1, 2),
        x.reshape(-1),
        reduction="none",
    ).view_as(x)
    if reduction == "mean":
        return per.mean()
    if reduction == "sum":
        return per.sum()
    raise ValueError(f"Unknown reduction={reduction!r}")


def incumbent_teacher_forcing_soft_ce_loss(
    model: AutoregressiveBitModel,
    x: torch.Tensor,
    *,
    prob_correct: float = 0.999,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Same teacher-forcing schedule as :func:`incumbent_teacher_forcing_ce_loss`, but each site uses a
    **soft** 2-way target: the correct bit class gets probability ``prob_correct``, the other
    ``1 - prob_correct``. Keeps softmax margins slightly open so later soft-Lagrangian training
    (``forward_next_soft``) still receives gradients.

    One-pass :meth:`AutoregressiveBitModel.forward` (same as hard CE).

    Parameters
    ----------
    prob_correct : float in ``(0.5, 1)``, typically ``0.999``.
    """
    x = x.long()
    if x.ndim != 2:
        raise ValueError(f"x must be (B, N), got {x.shape}")
    b, n = x.shape
    if n != model.num_sites:
        raise ValueError(f"x has N={n}, ADIP num_sites={model.num_sites}")
    p = float(prob_correct)
    if not (0.5 < p < 1.0):
        raise ValueError(f"prob_correct must be in (0.5, 1), got {p}")

    logits = model.forward(x)  # (B, N, 2)
    log_probs = F.log_softmax(logits, dim=-1)
    # Soft targets (B, N, 2)
    high = torch.full((b, n), p, dtype=logits.dtype, device=logits.device)
    low = 1.0 - high
    bit1 = x == 1
    prob0 = torch.where(bit1, low, high)
    prob1 = torch.where(bit1, high, low)
    target = torch.stack([prob0, prob1], dim=-1)
    per = -(target * log_probs).sum(dim=-1)  # (B, N)
    if reduction == "mean":
        return per.mean()
    if reduction == "sum":
        return per.sum()
    raise ValueError(f"Unknown reduction={reduction!r}")
