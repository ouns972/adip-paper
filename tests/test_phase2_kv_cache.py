"""KV-cache rollout matches the full-prefix soft forward, including chunked checkpoint grads."""

from __future__ import annotations

import copy

import torch

from adip.ar_model import AutoregressiveBitModel
from adip.soft_lagrangian import soft_rollout_lagrangian_qkp_max_objective_loss_gumbel


def _model(n: int, *, pattern: str, fraction: float) -> AutoregressiveBitModel:
    return AutoregressiveBitModel(
        num_sites=n,
        d_model=32,
        nhead=4,
        nlayers=2,
        dropout=0.0,
        sparse_attn_prev_fraction=fraction,
        sparse_attn_pattern=pattern,
    )


def _cached_logits(model: AutoregressiveBitModel, prefix: torch.Tensor) -> torch.Tensor:
    caches = model.init_soft_kv_cache(batch=prefix.shape[0], device=prefix.device)
    logits = None
    p = prefix[:, :0]
    # Walk the given prefix one bit at a time, then score the next bit.
    for t in range(prefix.shape[1] + 1):
        logits, caches = model.forward_next_soft_cached(p, caches)
        if t < prefix.shape[1]:
            p = prefix[:, : t + 1]
    assert logits is not None
    return logits


def test_band_cache_matches_full_prefix_logits() -> None:
    torch.manual_seed(0)
    n = 20
    model = _model(n, pattern="band", fraction=0.1).eval()
    prefix = torch.rand(2, 12)
    full = model.forward_next_soft(prefix)
    cached = _cached_logits(model, prefix)
    assert torch.allclose(full, cached, rtol=1e-4, atol=1e-5), (full - cached).abs().max().item()


def test_chunk_checkpoint_matches_uncached_gumbel_grads() -> None:
    n = 16
    base = _model(n, pattern="band", fraction=0.1)
    A = torch.ones(1, n)
    lb = torch.tensor([0.0])
    ub = torch.tensor([float(n // 2)])
    Q = torch.eye(n)
    m_full = copy.deepcopy(base)
    m_ckpt = copy.deepcopy(base)

    torch.manual_seed(7)
    loss_full = soft_rollout_lagrangian_qkp_max_objective_loss_gumbel(
        m_full,
        A,
        lb,
        ub,
        Q,
        tau=0.5,
        straight_through=True,
        gradient_checkpointing=False,
    )
    loss_full.backward()

    torch.manual_seed(7)
    loss_ckpt = soft_rollout_lagrangian_qkp_max_objective_loss_gumbel(
        m_ckpt,
        A,
        lb,
        ub,
        Q,
        tau=0.5,
        straight_through=True,
        gradient_checkpointing=True,
        checkpoint_chunk_size=4,
    )
    loss_ckpt.backward()

    assert torch.allclose(loss_full, loss_ckpt, rtol=1e-4, atol=1e-5)
    g_full = torch.cat([p.grad.reshape(-1) for p in m_full.parameters()])
    g_ckpt = torch.cat([p.grad.reshape(-1) for p in m_ckpt.parameters()])
    assert torch.allclose(g_full, g_ckpt, rtol=1e-4, atol=1e-5), (g_full - g_ckpt).abs().max().item()
