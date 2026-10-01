"""Feasibility check for binary vectors under linear inequality constraints."""

from __future__ import annotations

import numpy as np
import torch
from scipy import sparse


def check_feasibility(samples: torch.Tensor, A, lb, ub) -> torch.Tensor:
    """
    samples: [B, N] with 0/1 entries
    A: [m, N]
    lb, ub: [m]

    returns: BoolTensor [B], True iff each sample satisfies lb <= A x <= ub
    """
    lb = torch.as_tensor(lb, dtype=torch.int64)
    ub = torch.as_tensor(ub, dtype=torch.int64)
    x = samples.to(torch.int64)

    if sparse.issparse(A):
        x_np = x.cpu().numpy().astype(np.int64, copy=False)
        fluxes_np = (A @ x_np.T).T
        fluxes = torch.as_tensor(fluxes_np, dtype=torch.int64)
    else:
        A_t = torch.as_tensor(A, dtype=torch.int64)
        fluxes = x @ A_t.transpose(0, 1)
    ok = (fluxes >= lb.unsqueeze(0)) & (fluxes <= ub.unsqueeze(0))
    return ok.all(dim=1)
