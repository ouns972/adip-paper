"""Equivalence / smoke tests for one-pass teacher-forcing CE."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from adip.ar_model import AutoregressiveBitModel
from adip.teacher_ce import (
    incumbent_teacher_forcing_ce_loss,
    incumbent_teacher_forcing_soft_ce_loss,
)


def _loop_hard_ce(model: AutoregressiveBitModel, x: torch.Tensor) -> torch.Tensor:
    losses = []
    for t in range(x.shape[1]):
        logits = model.forward_next(x[:, :t])
        losses.append(F.cross_entropy(logits, x[:, t], reduction="none"))
    return torch.stack(losses, dim=1).mean()


def _loop_soft_ce(
    model: AutoregressiveBitModel, x: torch.Tensor, *, prob_correct: float
) -> torch.Tensor:
    p = float(prob_correct)
    low = 1.0 - p
    high_t = torch.tensor(p, dtype=torch.float32, device=x.device)
    low_t = torch.tensor(low, dtype=torch.float32, device=x.device)
    losses = []
    for t in range(x.shape[1]):
        logits = model.forward_next(x[:, :t])
        log_probs = F.log_softmax(logits, dim=-1)
        bit = x[:, t]
        prob0 = torch.where(bit == 0, high_t, low_t)
        prob1 = torch.where(bit == 1, high_t, low_t)
        target = torch.stack([prob0, prob1], dim=1).to(dtype=logits.dtype)
        losses.append(-(target * log_probs).sum(dim=-1))
    return torch.stack(losses, dim=1).mean()


def test_one_pass_hard_ce_matches_loop() -> None:
    torch.manual_seed(0)
    n = 32
    model = AutoregressiveBitModel(
        num_sites=n,
        d_model=32,
        nhead=4,
        nlayers=2,
        sparse_attn_prev_fraction=0.1,
        sparse_attn_pattern="band",
    )
    x = torch.randint(0, 2, (2, n), dtype=torch.long)
    one = incumbent_teacher_forcing_ce_loss(model, x)
    loop = _loop_hard_ce(model, x)
    assert torch.allclose(one, loop, rtol=1e-5, atol=1e-6), (float(one), float(loop))


def test_one_pass_soft_ce_matches_loop() -> None:
    torch.manual_seed(1)
    n = 32
    model = AutoregressiveBitModel(
        num_sites=n,
        d_model=32,
        nhead=4,
        nlayers=2,
        sparse_attn_prev_fraction=0.1,
        sparse_attn_pattern="band",
    )
    x = torch.randint(0, 2, (2, n), dtype=torch.long)
    one = incumbent_teacher_forcing_soft_ce_loss(model, x, prob_correct=0.999)
    loop = _loop_soft_ce(model, x, prob_correct=0.999)
    assert torch.allclose(one, loop, rtol=1e-5, atol=1e-6), (float(one), float(loop))
