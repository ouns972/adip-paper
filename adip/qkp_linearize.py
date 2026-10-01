"""Fortet / McCormick linearization of binary quadratic knapsack.

HiGHS native QP requires a positive-semidefinite Q; CBC has no quadratic MIP
interface in OR-Tools. The default ADIP instances use indefinite Q (entries in
``[-5, 5]``), so those solvers see the equivalent MILP

    maximize  sum_i (Q_ii + c_i) x_i + sum_{i<j} (Q_ij + Q_ji) y_ij
    s.t.      w^T x <= W
              y_ij <= x_i,  y_ij <= x_j,  y_ij >= x_i + x_j - 1
              x, y binary.

On {0,1}^n this is identical to ``x^T Q x + c^T x``. SCIP keeps the native
quadratic model (it accepts nonconvex MIQP).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class FortetQKP:
    """Linearized max-QKP: objective coeffs on ``x`` and on pair binaries ``y``."""

    n: int
    lin_x: np.ndarray  # (n,) int64,  Q_ii + c_i
    pair_i: np.ndarray  # (m,) int32, i < j
    pair_j: np.ndarray  # (m,) int32
    pair_coeff: np.ndarray  # (m,) int64, Q_ij + Q_ji for nonzero pairs
    w: np.ndarray  # (n,) int64
    W: int

    @property
    def n_pairs(self) -> int:
        return int(self.pair_i.shape[0])

    @property
    def n_cols(self) -> int:
        return int(self.n) + self.n_pairs


def fortet_qkp(
    Q: np.ndarray,
    w: np.ndarray,
    W: int,
    c: Optional[np.ndarray] = None,
) -> FortetQKP:
    """Build Fortet data for ``maximize xᵀ Q x + cᵀ x`` over binary knapsack."""
    Qd = np.asarray(Q, dtype=np.int64)
    if Qd.ndim != 2 or int(Qd.shape[0]) != int(Qd.shape[1]):
        raise ValueError(f"Q must be square, got shape {Qd.shape}")
    n = int(Qd.shape[0])
    wv = np.asarray(w, dtype=np.int64).ravel()
    if int(wv.shape[0]) != n:
        raise ValueError(f"w length {int(wv.shape[0])} != n={n}")
    lin = np.diag(Qd).astype(np.int64, copy=True)
    if c is not None:
        cv = np.asarray(c, dtype=np.int64).ravel()
        if int(cv.shape[0]) != n:
            raise ValueError(f"c length {int(cv.shape[0])} != n={n}")
        lin = lin + cv
    # (Q + Qᵀ)_ij = Q_ij + Q_ji for the product x_i x_j (i < j).
    qt = Qd + Qd.T
    iu, ju = np.triu_indices(n, k=1)
    coeff = qt[iu, ju]
    mask = coeff != 0
    return FortetQKP(
        n=n,
        lin_x=lin,
        pair_i=iu[mask].astype(np.int32, copy=False),
        pair_j=ju[mask].astype(np.int32, copy=False),
        pair_coeff=coeff[mask].astype(np.int64, copy=False),
        w=wv,
        W=int(W),
    )


def fortet_objective(model: FortetQKP, x: np.ndarray) -> int:
    """Integer linearized objective at a binary ``x`` (equals ``xᵀQx + cᵀx``)."""
    xv = np.asarray(x, dtype=np.int64).ravel()
    if int(xv.shape[0]) != int(model.n):
        raise ValueError(f"x length {int(xv.shape[0])} != n={model.n}")
    acc = int(np.dot(model.lin_x, xv))
    if model.n_pairs:
        y = xv[model.pair_i] * xv[model.pair_j]
        acc += int(np.dot(model.pair_coeff, y))
    return acc


def fortet_col_primal(model: FortetQKP, x: np.ndarray) -> np.ndarray:
    """Full MILP column vector ``(x, y)`` with ``y_ij = x_i x_j``."""
    xv = np.asarray(x, dtype=np.float64).ravel()
    if int(xv.shape[0]) != int(model.n):
        raise ValueError(f"x length {int(xv.shape[0])} != n={model.n}")
    out = np.empty(model.n_cols, dtype=np.float64)
    out[: model.n] = xv
    if model.n_pairs:
        out[model.n :] = xv[model.pair_i] * xv[model.pair_j]
    return out


def relative_mip_gap(primal: Optional[float], dual: Optional[float]) -> Optional[float]:
    """``|primal − dual| / |primal|``, matching SCIP-style reporting of a large gap."""
    if primal is None or dual is None:
        return None
    p = float(primal)
    d = float(dual)
    denom = abs(p)
    if denom < 1e-12:
        denom = 1.0
    return abs(d - p) / denom
