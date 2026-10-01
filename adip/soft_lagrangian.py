"""Soft autoregressive rollouts with a knapsack Lagrangian penalty."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from adip.ar_model import AutoregressiveBitModel


def _flatten_kv_cache(
    caches: list,
) -> tuple:
    flat = []
    for k, v in caches:
        flat.append(k)
        flat.append(v)
    return tuple(flat)


def _unflatten_kv_cache(flat: tuple) -> list:
    return [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]


def _append_bit_from_logits(
    prefix_probs: torch.Tensor,
    logits: torch.Tensor,
    *,
    kind: str,
    tau: float,
    straight_through: bool,
) -> torch.Tensor:
    if kind == "softmax":
        p = F.softmax(logits / float(tau), dim=-1)
        bit = p[:, 1:2]
    elif kind == "gumbel":
        y = F.gumbel_softmax(
            logits,
            tau=float(tau),
            hard=bool(straight_through),
            dim=-1,
        )
        bit = y[:, 1:2]
    else:
        raise ValueError(f"unknown rollout kind {kind!r}")
    return torch.cat([prefix_probs, bit], dim=1)


def _rollout_prefix(
    model: AutoregressiveBitModel,
    *,
    dtype: torch.dtype,
    device: torch.device,
    kind: str,
    tau: float,
    straight_through: bool,
    gradient_checkpointing: bool,
    checkpoint_chunk_size: int,
) -> torch.Tensor:
    """
    Build soft ``x̃`` with an incremental K/V cache (one new token per site).

    When ``gradient_checkpointing`` is on, the rollout is checkpointed in chunks of
    ``checkpoint_chunk_size`` steps: activations inside a chunk are recomputed on
    backward, cache tensors at chunk boundaries stay in the graph.
    """
    n = model.num_sites
    prefix_probs = torch.zeros(1, 0, device=device, dtype=dtype)
    caches = model.init_soft_kv_cache(batch=1, device=device)
    chunk = max(1, int(checkpoint_chunk_size))
    if not gradient_checkpointing:
        chunk = n

    def _run_steps(prefix_in: torch.Tensor, flat_cache: tuple, n_steps: int):
        p = prefix_in
        cache = _unflatten_kv_cache(flat_cache)
        for _ in range(int(n_steps)):
            logits, cache = model.forward_next_soft_cached(p, cache)
            p = _append_bit_from_logits(
                p,
                logits,
                kind=kind,
                tau=float(tau),
                straight_through=bool(straight_through),
            )
        return (p, *_flatten_kv_cache(cache))

    done = 0
    flat = _flatten_kv_cache(caches)
    while done < n:
        n_steps = min(chunk, n - done)

        def chunk_fn(
            prefix_in: torch.Tensor,
            *flat_cache: torch.Tensor,
            _n_steps: int = n_steps,
        ) -> tuple:
            return _run_steps(prefix_in, flat_cache, _n_steps)

        if gradient_checkpointing:
            out = checkpoint(chunk_fn, prefix_probs, *flat, use_reentrant=False)
        else:
            out = chunk_fn(prefix_probs, *flat)
        prefix_probs = out[0]
        flat = out[1:]
        done += n_steps
    return prefix_probs


def _qkp_surrogate_from_xsoft(
    x_soft: torch.Tensor,
    A: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
    Q: torch.Tensor,
    *,
    cost_weight: float,
    lambda_penalty: float,
    reduction: str,
    c_lin: Optional[torch.Tensor],
    x_soft_out: Optional[List[torch.Tensor]],
) -> torch.Tensor:
    quad = torch.einsum("bi,ij,bj->b", x_soft, Q, x_soft)
    if c_lin is not None:
        quad = quad + (x_soft * c_lin.unsqueeze(0)).sum(dim=1)
    flux = x_soft @ A.T
    viol_lo = F.relu(lb.unsqueeze(0) - flux)
    viol_hi = F.relu(flux - ub.unsqueeze(0))
    penalty = (viol_lo + viol_hi).sum(dim=1)
    per_sample = -float(cost_weight) * quad + float(lambda_penalty) * penalty
    if reduction == "mean":
        out = per_sample.mean()
    elif reduction == "sum":
        out = per_sample.sum()
    else:
        raise ValueError(f"Unknown reduction={reduction!r}")
    if x_soft_out is not None:
        x_soft_out.clear()
        x_soft_out.append(x_soft.detach().clone())
    return out


def soft_rollout_lagrangian_qkp_max_objective_loss(
    model: AutoregressiveBitModel,
    A: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
    Q: torch.Tensor,
    *,
    tau: float = 1.0,
    lambda_penalty: float = 1.0,
    cost_weight: float = 1.0,
    reduction: str = "mean",
    x_soft_out: Optional[List[torch.Tensor]] = None,
    c_lin: Optional[torch.Tensor] = None,
    gradient_checkpointing: bool = True,
    checkpoint_chunk_size: int = 32,
) -> torch.Tensor:
    """
    Relaxed-trajectory loss for the **quadratic** knapsack: **maximize** :math:`x^{\\top} Q x + c^{\\top}x` subject
    to the same box constraints as :func:`soft_rollout_lagrangian_loss` on :math:`A\\tilde{x}`.

    Uses a mean-field surrogate :math:`\\tilde{x}^{\\top} Q \\tilde{x} + c^{\\top}\\tilde{x}` with
    :math:`\\tilde{x}_t = p_t[1]` (same soft chain as the linear case).

    We **minimize**
    ``- cost_weight * (x̃ᵀ Q x̃ + cᵀx̃) + lambda_penalty * (sum of ReLU slacks)`` so that reducing the
    scalar encourages **higher** quadratic value when feasible (same slack structure as the linear
    version).
    """
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    device = A.device
    n = model.num_sites
    m = A.shape[0]
    if A.shape[1] != n or lb.shape[0] != m or ub.shape[0] != m:
        raise ValueError("Shape mismatch among A, lb, ub, and model.num_sites.")
    if Q.shape != (n, n):
        raise ValueError(f"Q must be ({n}, {n}), got {tuple(Q.shape)}")
    if c_lin is not None and tuple(c_lin.shape) != (n,):
        raise ValueError(f"c_lin must be ({n},), got {tuple(c_lin.shape)}")

    x_soft = _rollout_prefix(
        model,
        dtype=A.dtype,
        device=device,
        kind="softmax",
        tau=float(tau),
        straight_through=False,
        gradient_checkpointing=bool(gradient_checkpointing),
        checkpoint_chunk_size=int(checkpoint_chunk_size),
    )
    return _qkp_surrogate_from_xsoft(
        x_soft,
        A,
        lb,
        ub,
        Q,
        cost_weight=cost_weight,
        lambda_penalty=lambda_penalty,
        reduction=reduction,
        c_lin=c_lin,
        x_soft_out=x_soft_out,
    )


def soft_rollout_lagrangian_qkp_max_objective_loss_gumbel(
    model: AutoregressiveBitModel,
    A: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
    Q: torch.Tensor,
    *,
    tau: float = 1.0,
    lambda_penalty: float = 1.0,
    cost_weight: float = 1.0,
    reduction: str = "mean",
    straight_through: bool = False,
    x_soft_out: Optional[List[torch.Tensor]] = None,
    c_lin: Optional[torch.Tensor] = None,
    gradient_checkpointing: bool = True,
    checkpoint_chunk_size: int = 32,
) -> torch.Tensor:
    """
    Same surrogate as :func:`soft_rollout_lagrangian_qkp_max_objective_loss`, but each bond uses
    **Gumbel–Softmax** on the two next-bit logits instead of plain ``softmax(logits/τ)``.

    Parameters
    ----------
    straight_through
        If ``False``, standard Gumbel–Softmax (fully differentiable sample).
        If ``True``, PyTorch ``hard=True``: **forward** one-hot from the Gumbel–Softmax draw,
        **backward** through the continuous relaxation (STE).
    gradient_checkpointing
        If ``True`` (default), checkpoint the cached rollout in chunks of
        ``checkpoint_chunk_size`` steps (recompute those activations on backward).
    checkpoint_chunk_size
        Autoregressive steps per checkpoint segment. Larger chunks are faster and use more RAM.
    """
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    device = A.device
    n = model.num_sites
    m = A.shape[0]
    if A.shape[1] != n or lb.shape[0] != m or ub.shape[0] != m:
        raise ValueError("Shape mismatch among A, lb, ub, and model.num_sites.")
    if Q.shape != (n, n):
        raise ValueError(f"Q must be ({n}, {n}), got {tuple(Q.shape)}")
    if c_lin is not None and tuple(c_lin.shape) != (n,):
        raise ValueError(f"c_lin must be ({n},), got {tuple(c_lin.shape)}")

    x_soft = _rollout_prefix(
        model,
        dtype=A.dtype,
        device=device,
        kind="gumbel",
        tau=float(tau),
        straight_through=bool(straight_through),
        gradient_checkpointing=bool(gradient_checkpointing),
        checkpoint_chunk_size=int(checkpoint_chunk_size),
    )
    return _qkp_surrogate_from_xsoft(
        x_soft,
        A,
        lb,
        ub,
        Q,
        cost_weight=cost_weight,
        lambda_penalty=lambda_penalty,
        reduction=reduction,
        c_lin=c_lin,
        x_soft_out=x_soft_out,
    )
