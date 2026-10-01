"""Gradient-checkpointed phase-2 soft rollout: same grads as full graph."""

from __future__ import annotations

import copy

import torch

from adip.ar_model import AutoregressiveBitModel
from adip.soft_lagrangian import (
    soft_rollout_lagrangian_qkp_max_objective_loss,
    soft_rollout_lagrangian_qkp_max_objective_loss_gumbel,
)


def _tiny_qkp(n: int):
    w = torch.ones(n, dtype=torch.float32)
    A = w.unsqueeze(0)
    lb = torch.tensor([0.0], dtype=torch.float32)
    ub = torch.tensor([float(n // 2)], dtype=torch.float32)
    Q = torch.eye(n, dtype=torch.float32)
    return A, lb, ub, Q


def _grad_vec(model: AutoregressiveBitModel) -> torch.Tensor:
    parts = []
    for p in model.parameters():
        if p.grad is None:
            parts.append(torch.zeros(p.numel()))
        else:
            parts.append(p.grad.detach().reshape(-1).cpu())
    return torch.cat(parts)


def test_softmax_checkpoint_matches_full_grads() -> None:
    torch.manual_seed(0)
    n = 24
    model = AutoregressiveBitModel(
        num_sites=n,
        d_model=32,
        nhead=4,
        nlayers=2,
        sparse_attn_prev_fraction=0.1,
        sparse_attn_pattern="band",
    )
    A, lb, ub, Q = _tiny_qkp(n)
    m_full = copy.deepcopy(model)
    m_ckpt = copy.deepcopy(model)

    loss_full = soft_rollout_lagrangian_qkp_max_objective_loss(
        m_full, A, lb, ub, Q, tau=0.5, lambda_penalty=1.0, gradient_checkpointing=False
    )
    loss_full.backward()
    g_full = _grad_vec(m_full)

    loss_ckpt = soft_rollout_lagrangian_qkp_max_objective_loss(
        m_ckpt, A, lb, ub, Q, tau=0.5, lambda_penalty=1.0, gradient_checkpointing=True
    )
    loss_ckpt.backward()
    g_ckpt = _grad_vec(m_ckpt)

    assert torch.allclose(loss_full, loss_ckpt, rtol=1e-5, atol=1e-6)
    assert torch.allclose(g_full, g_ckpt, rtol=1e-4, atol=1e-5), (
        float((g_full - g_ckpt).abs().max()),
    )


def test_gumbel_ste_checkpoint_matches_full_grads() -> None:
    n = 16
    A, lb, ub, Q = _tiny_qkp(n)
    base = AutoregressiveBitModel(
        num_sites=n,
        d_model=32,
        nhead=4,
        nlayers=2,
        sparse_attn_prev_fraction=0.1,
        sparse_attn_pattern="band",
    )
    m_full = copy.deepcopy(base)
    m_ckpt = copy.deepcopy(base)

    torch.manual_seed(123)
    loss_full = soft_rollout_lagrangian_qkp_max_objective_loss_gumbel(
        m_full,
        A,
        lb,
        ub,
        Q,
        tau=0.5,
        lambda_penalty=1.0,
        straight_through=True,
        gradient_checkpointing=False,
    )
    loss_full.backward()
    g_full = _grad_vec(m_full)

    torch.manual_seed(123)
    loss_ckpt = soft_rollout_lagrangian_qkp_max_objective_loss_gumbel(
        m_ckpt,
        A,
        lb,
        ub,
        Q,
        tau=0.5,
        lambda_penalty=1.0,
        straight_through=True,
        gradient_checkpointing=True,
    )
    loss_ckpt.backward()
    g_ckpt = _grad_vec(m_ckpt)

    assert torch.allclose(loss_full, loss_ckpt, rtol=1e-5, atol=1e-6)
    assert torch.allclose(g_full, g_ckpt, rtol=1e-4, atol=1e-5), (
        float((g_full - g_ckpt).abs().max()),
    )


def test_n1000_gumbel_ste_one_step_survives() -> None:
    """Smoke: one checkpointed phase-2 step at n=1000 must not OOM."""
    import resource

    torch.manual_seed(0)
    n = 1000
    model = AutoregressiveBitModel(
        num_sites=n,
        d_model=64,
        nhead=4,
        nlayers=2,
        sparse_attn_prev_fraction=0.1,
        sparse_attn_pattern="band",
    )
    A = torch.ones(1, n, dtype=torch.float32)
    lb = torch.tensor([0.0])
    ub = torch.tensor([250.0])
    Q = torch.eye(n, dtype=torch.float32)  # cheap stand-in
    loss = soft_rollout_lagrangian_qkp_max_objective_loss_gumbel(
        model,
        A,
        lb,
        ub,
        Q,
        tau=0.5,
        lambda_penalty=100.0,
        straight_through=True,
        gradient_checkpointing=True,
    )
    loss.backward()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS: bytes; Linux: KiB — either way stay well under ~8 GB equivalent
    rss_gb = rss / 1e9 if rss > 1e7 else rss / (1024**2)
    assert float(loss.detach()) == float(loss.detach())  # finite
    assert rss_gb < 8.0, f"peak RSS too high: {rss_gb:.2f} GB"
