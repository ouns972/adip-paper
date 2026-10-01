"""Continuous knapsack-boundary relaxation from binary warm starts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from adip.pipeline import QKPInstance


def _q_float(inst: QKPInstance) -> np.ndarray:
    return np.asarray(inst.Q, dtype=np.float64)


def _c_float(inst: QKPInstance) -> Optional[np.ndarray]:
    if inst.c is None:
        return None
    return np.asarray(inst.c, dtype=np.float64).ravel()


def _w_float(inst: QKPInstance) -> np.ndarray:
    return np.asarray(inst.w, dtype=np.float64).ravel()


def _sanitize_box_x(x: np.ndarray, n: int) -> np.ndarray:
    """Clip to ``[0,1]^n`` and replace non-finite values (avoids matmul warnings)."""
    xf = np.asarray(x, dtype=np.float64).ravel()
    if int(xf.shape[0]) != int(n):
        raise ValueError(f"x length {int(xf.shape[0])} != n={n}")
    xf = np.nan_to_num(xf, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(xf, 0.0, 1.0)


def knapsack_weight(inst: QKPInstance, x: np.ndarray) -> float:
    """``wᵀ x`` for continuous ``x``."""
    n = int(inst.n)
    xf = _sanitize_box_x(x, n)
    return float(np.dot(_w_float(inst), xf))


def continuous_qkp_value(
    inst: QKPInstance,
    x: np.ndarray,
) -> float:
    """``xᵀ Q x + cᵀ x`` for continuous ``x`` (float64)."""
    n = int(inst.n)
    xf = _sanitize_box_x(x, n)
    Qf = _q_float(inst)
    val = float(np.einsum("i,ij,j", xf, Qf, xf, optimize=True))
    cf = _c_float(inst)
    if cf is not None:
        val += float(np.dot(cf, xf))
    if not np.isfinite(val):
        return float("nan")
    return val


def continuous_qkp_gradient(
    inst: QKPInstance,
    x: np.ndarray,
) -> np.ndarray:
    """Gradient of ``xᵀ Q x + cᵀ x`` w.r.t. ``x`` (general non-symmetric ``Q``)."""
    n = int(inst.n)
    xf = _sanitize_box_x(x, n)
    Qf = _q_float(inst)
    g = np.einsum("ij,j->i", Qf + Qf.T, xf, optimize=True)
    cf = _c_float(inst)
    if cf is not None:
        g = g + cf
    g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(g, dtype=np.float64)


def _projected_gradient_box(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Project gradient for box constraints ``0 <= x <= 1`` (maximization)."""
    g_proj = np.asarray(g, dtype=np.float64).copy()
    at_lo = x <= 1e-12
    at_hi = x >= 1.0 - 1e-12
    g_proj[at_lo & (g < 0.0)] = 0.0
    g_proj[at_hi & (g > 0.0)] = 0.0
    return g_proj


def _max_step_along_direction_box(x: np.ndarray, d: np.ndarray) -> float:
    """Largest ``alpha >= 0`` with ``x + alpha d`` still in ``[0, 1]ⁿ``."""
    alpha = float("inf")
    for xi, di in zip(x, d):
        if di > 1e-15:
            alpha = min(alpha, (1.0 - xi) / di)
        elif di < -1e-15:
            alpha = min(alpha, xi / (-di))
    if not np.isfinite(alpha):
        return 0.0
    return max(0.0, float(alpha))


def binary_start_vector(x: np.ndarray) -> np.ndarray:
    """Embed a binary (or near-binary) vector in ``[0, 1]ⁿ`` as floats."""
    xf = np.asarray(x, dtype=np.float64).ravel()
    if xf.size and np.max(np.abs(xf - np.rint(xf))) < 1e-3:
        return np.rint(xf).astype(np.float64)
    return (xf > 0.5).astype(np.float64)


@dataclass
class ContinuousRelaxGDResult:
    label: str
    x0: np.ndarray
    x_star: np.ndarray
    f0: float
    f_star: float
    w0: float
    w_star: float
    n_iters: int
    converged: bool
    stop_reason: str

    def to_json_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["x0"] = np.asarray(self.x0, dtype=np.float64).tolist()
        d["x_star"] = np.asarray(self.x_star, dtype=np.float64).tolist()
        return d


def knapsack_boundary_gradient_ascent(
    inst: QKPInstance,
    x0: np.ndarray,
    *,
    label: str = "start",
    max_iters: int = 3000,
    lr: float = 0.05,
    tol_knapsack: float = 1e-6,
    tol_grad: float = 1e-8,
) -> ContinuousRelaxGDResult:
    """
    Gradient **ascent** on ``max xᵀ Q x + cᵀ x`` over ``x ∈ [0, 1]ⁿ``.

    Starting from a knapsack-feasible point, take projected-ascent steps until
    ``wᵀ x`` reaches ``W`` (knapsack boundary) or progress stops.
    """
    n = int(inst.n)
    W = float(inst.W)
    w = _w_float(inst)
    tol_w = max(float(tol_knapsack), 1e-9 * max(1.0, abs(W)))

    x = _sanitize_box_x(binary_start_vector(x0), n)
    f = continuous_qkp_value(inst, x)
    if not np.isfinite(f):
        return ContinuousRelaxGDResult(
            label=str(label),
            x0=np.asarray(x0, dtype=np.float64).ravel(),
            x_star=x,
            f0=float("nan"),
            f_star=float("nan"),
            w0=float("nan"),
            w_star=float("nan"),
            n_iters=0,
            converged=False,
            stop_reason="nonfinite_f0",
        )

    f0 = float(f)
    w0 = knapsack_weight(inst, x)
    stop = "max_iters"
    converged = False

    if w0 >= W - tol_w:
        return ContinuousRelaxGDResult(
            label=str(label),
            x0=np.asarray(x0, dtype=np.float64).ravel(),
            x_star=x,
            f0=f0,
            f_star=f0,
            w0=w0,
            w_star=w0,
            n_iters=0,
            converged=True,
            stop_reason="knapsack_already_tight",
        )

    base_lr = float(lr)
    it = -1

    for it in range(int(max_iters)):
        w_cur = knapsack_weight(inst, x)
        if w_cur >= W - tol_w:
            stop = "knapsack_boundary"
            converged = True
            break

        g = continuous_qkp_gradient(inst, x)
        if not np.all(np.isfinite(g)):
            stop = "nonfinite_grad"
            break

        g_proj = _projected_gradient_box(x, g)
        g_inf = float(np.max(np.abs(g_proj))) if g_proj.size else 0.0
        if g_inf < float(tol_grad):
            stop = "zero_projected_grad"
            break

        d = g_proj / g_inf
        w_dot_d = float(np.dot(w, d))
        if w_dot_d <= 1e-12:
            stop = "knapsack_not_reachable"
            break

        alpha_box = _max_step_along_direction_box(x, d)
        alpha_knap = (W - w_cur) / w_dot_d
        alpha = min(base_lr, alpha_box, alpha_knap)
        if alpha <= 1e-15:
            stop = "step_too_small"
            break

        x = _sanitize_box_x(x + alpha * d, n)
        f = continuous_qkp_value(inst, x)

        if knapsack_weight(inst, x) >= W - tol_w:
            stop = "knapsack_boundary"
            converged = True
            break

    x = _sanitize_box_x(x, n)
    f_fin = continuous_qkp_value(inst, x)
    if not np.isfinite(f_fin):
        f_fin = float(f)
    w_star = knapsack_weight(inst, x)

    return ContinuousRelaxGDResult(
        label=str(label),
        x0=np.asarray(x0, dtype=np.float64).ravel(),
        x_star=x,
        f0=f0,
        f_star=float(f_fin),
        w0=float(w0),
        w_star=float(w_star),
        n_iters=it + 1,
        converged=bool(converged),
        stop_reason=stop,
    )


def pairwise_l2_distances(
    results: Sequence[ContinuousRelaxGDResult],
) -> Dict[str, float]:
    """L2 distances between converged ``x_star`` vectors, keyed ``label_a|label_b``."""
    out: Dict[str, float] = {}
    m = len(results)
    for i in range(m):
        for j in range(i + 1, m):
            a = results[i].label
            b = results[j].label
            d = float(np.linalg.norm(results[i].x_star - results[j].x_star))
            out[f"{a}|{b}"] = d
    return out


def summarize_gd_results(
    results: Sequence[ContinuousRelaxGDResult],
) -> Dict[str, Any]:
    """Best continuous objective and spread across starts."""
    if not results:
        return {"n_starts": 0}
    f_vals = [float(r.f_star) for r in results]
    best_i = int(np.argmax(f_vals))
    return {
        "n_starts": len(results),
        "f_star_min": float(np.min(f_vals)),
        "f_star_max": float(np.max(f_vals)),
        "f_star_mean": float(np.mean(f_vals)),
        "f_star_spread": float(np.max(f_vals) - np.min(f_vals)),
        "best_label": results[best_i].label,
        "best_f_star": f_vals[best_i],
        "pairwise_l2": pairwise_l2_distances(results),
    }


def adip_binary_start_from_pipeline(
    inst: QKPInstance,
    *,
    rounded_x: Optional[np.ndarray],
    rounded_feasible: Optional[bool],
) -> Tuple[Optional[np.ndarray], str]:
    """
    Pick the ADIP binary start for continuous relaxation.

    Uses knapsack-feasible rounded ``x̃`` at lowest phase-2 loss when available.
    """
    w = inst.w
    W = int(inst.W)
    if rounded_x is not None and rounded_feasible:
        xr = np.asarray(rounded_x, dtype=np.int64).ravel()
        if int(xr.shape[0]) == inst.n and int(np.dot(w, xr)) <= W:
            return xr, "adip_rounded_min_loss"
    return None, "none"
