"""Phase-2 differentiable soft QKP training with knapsack Lagrangian penalty."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from tqdm.auto import tqdm

from adip.ar_model import AutoregressiveBitModel
from adip.phase1_ce import tensors_from_problem
from adip.soft_lagrangian import (
    soft_rollout_lagrangian_qkp_max_objective_loss,
    soft_rollout_lagrangian_qkp_max_objective_loss_gumbel,
)
from adip.binary_mip import BinaryMIPSpec

# Relaxed loss is ``w_soft * L``; ``L`` is nonnegative when ``cost_weight == 0``.
_NUMERICAL_ZERO_TOL = 1e-10
# Max ``|Δ(w_soft·L)|`` between consecutive Adam steps to count as "flat" for plateau early stopping.
DEFAULT_SOFT_PLATEAU_MIN_DELTA = 1e-5


def _dense_A_numpy(A_any: Any) -> np.ndarray:
    from scipy import sparse as sp

    if sp.issparse(A_any):
        return A_any.toarray()
    return np.asarray(A_any, dtype=np.float32)


def _binary_x_box_feasible(A_any: Any, lb: np.ndarray, ub: np.ndarray, x_np: np.ndarray) -> bool:
    """``True`` iff ``lb <= A x <= ub`` for 0/1 ``x`` (same rule as :func:`check_feasibility`)."""
    from adip.feasibility import check_feasibility

    xv = np.asarray(x_np, dtype=np.float64).ravel()
    x = torch.as_tensor(xv.reshape(1, -1), dtype=torch.float32)
    return bool(check_feasibility(x, A_any, lb, ub)[0].item())


@dataclass
class Phase2AdipStats:
    n_soft_steps: int
    #: ``None`` if training ran all ``max_optimizer_steps``; else why we stopped early.
    early_stop_reason: Optional[str]
    #: At the Adam step with lowest ``w_soft·L``: ``round(x̃)`` (``int64`` 0/1), discrete ``xᵀQx`` with
    #: integer ``Q``, and surrogate ``x̃ᵀQ x̃`` from that same forward (``None`` if not captured).
    best_loss_soft_x_rounded: Optional[np.ndarray] = None
    best_loss_rounded_xT_Q_x: Optional[float] = None
    best_loss_surrogate_xTilde_Q_xTilde: Optional[float] = None
    #: Whether ``best_loss_soft_x_rounded`` satisfies the training MIP ``lb <= A x <= ub`` (``None`` if
    #: no snapshot was recorded).
    best_loss_rounded_feasible: Optional[bool] = None
    #: Optional per-step relaxed ``x̃`` (``float`` ``[0,1]ⁿ``) when phase-2 tracing is enabled.
    soft_x_trace: Optional[List[np.ndarray]] = None
    #: Wall seconds from phase-2 start, aligned with ``soft_x_trace``.
    soft_x_trace_times_s: Optional[List[float]] = None

    def summary_lines(self) -> List[str]:
        if self.early_stop_reason == "threshold":
            stop_detail = "yes — w_soft·L ≤ loss_stop_threshold"
        elif self.early_stop_reason == "plateau":
            stop_detail = (
                "yes — consecutive flat steps: "
                "|w_soft·L − previous step| ≤ soft_loss_plateau_min_delta "
                "for plateau_patience iterations"
            )
        elif self.early_stop_reason == "numerical_zero":
            stop_detail = "yes — relaxed Lagrangian reached numerical zero (no reason to continue)"
        elif self.early_stop_reason == "time_limit":
            stop_detail = "yes — phase-2 wall-clock time limit reached"
        else:
            stop_detail = "no — exhausted max_optimizer_steps without trigger"
        return [
            (
                "  Training objective: w_soft · soft_rollout_lagrangian_loss only "
                "(relaxed softmax chain — no discrete samples in training)."
            ),
            (
                f"  Soft Lagrangian Adam steps: {int(self.n_soft_steps)}. "
                f"Stopped early: {stop_detail}"
            ),
        ]






# --- Phase-1 reference: soft Lagrangian (no cost) → extract a feasible discrete x ------------








# --- QKP: phase-2 soft Lagrangian with quadratic surrogate x̃ᵀ Q x̃ ------------------------------


def _qkp_relaxation_surrogate_loss(
    model: AutoregressiveBitModel,
    A_t: torch.Tensor,
    lb_t: torch.Tensor,
    ub_t: torch.Tensor,
    Q_t: torch.Tensor,
    *,
    tau: float,
    lambda_penalty: float,
    cost_weight: float,
    relaxation: str,
    x_soft_out: Optional[List[torch.Tensor]] = None,
    c_lin: Optional[torch.Tensor] = None,
    gradient_checkpointing: bool = True,
    checkpoint_chunk_size: int = 32,
) -> torch.Tensor:
    r = str(relaxation).strip().lower()
    if r == "softmax":
        return soft_rollout_lagrangian_qkp_max_objective_loss(
            model,
            A_t,
            lb_t,
            ub_t,
            Q_t,
            tau=float(tau),
            lambda_penalty=float(lambda_penalty),
            cost_weight=float(cost_weight),
            reduction="mean",
            x_soft_out=x_soft_out,
            c_lin=c_lin,
            gradient_checkpointing=bool(gradient_checkpointing),
            checkpoint_chunk_size=int(checkpoint_chunk_size),
        )
    if r == "gumbel_soft":
        return soft_rollout_lagrangian_qkp_max_objective_loss_gumbel(
            model,
            A_t,
            lb_t,
            ub_t,
            Q_t,
            tau=float(tau),
            lambda_penalty=float(lambda_penalty),
            cost_weight=float(cost_weight),
            reduction="mean",
            straight_through=False,
            x_soft_out=x_soft_out,
            c_lin=c_lin,
            gradient_checkpointing=bool(gradient_checkpointing),
            checkpoint_chunk_size=int(checkpoint_chunk_size),
        )
    if r == "gumbel_ste":
        return soft_rollout_lagrangian_qkp_max_objective_loss_gumbel(
            model,
            A_t,
            lb_t,
            ub_t,
            Q_t,
            tau=float(tau),
            lambda_penalty=float(lambda_penalty),
            cost_weight=float(cost_weight),
            reduction="mean",
            straight_through=True,
            x_soft_out=x_soft_out,
            c_lin=c_lin,
            gradient_checkpointing=bool(gradient_checkpointing),
            checkpoint_chunk_size=int(checkpoint_chunk_size),
        )
    raise ValueError(
        "relaxation must be 'softmax', 'gumbel_soft', or 'gumbel_ste', "
        f"got {relaxation!r}"
    )


@dataclass
class Phase2AdipConfig:
    """
    Minimize ``w_soft * (-q_w · x̃ᵀQ x̃ + λ · slack)`` (same slack as linear Type-B) to **increase**
    the quadratic value while satisfying knapsack in the soft sense.
    """

    lr: float = 1e-3
    device: str = "cpu"
    w_soft: float = 0.5
    softmax_tau: float = 0.3
    lambda_lagrangian: float = 1.0
    q_objective_weight: float = 1.0
    max_optimizer_steps: int = 2000
    loss_stop_threshold: Optional[float] = None
    #: Consecutive-step flatness on ``w_soft·L`` (see :class:`SoftSampleLoopConfig`).
    soft_loss_plateau_patience: int = 20
    soft_loss_plateau_min_delta: float = DEFAULT_SOFT_PLATEAU_MIN_DELTA
    #: ``softmax`` (default) — ``softmax(logits/τ)`` chain;
    #: ``gumbel_soft`` / ``gumbel_ste`` — :func:`soft_rollout_lagrangian_qkp_max_objective_loss_gumbel`
    #: with ``straight_through`` False / True (``torch.nn.functional.gumbel_softmax(..., hard=…)``).
    relaxation: str = "softmax"
    #: Checkpoint the cached rollout in chunks (recompute activations on backward).
    gradient_checkpointing: bool = True
    #: Autoregressive steps per checkpoint chunk. Larger is faster and uses more RAM.
    checkpoint_chunk_size: int = 32
    #: Wall-clock cap on the phase-2 Adam loop (seconds). ``None`` or nonpositive = no limit.
    max_wall_time_sec: Optional[float] = None
    #: If ``True``, record relaxed ``x̃`` after each Adam step (subsample with ``soft_x_trace_stride``).
    record_soft_x_trace: bool = False
    #: Record every ``k``-th step (``1`` = every step). Only used when ``record_soft_x_trace``.
    soft_x_trace_stride: int = 1


def _clone_state_dict_cpu(module: AutoregressiveBitModel) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _soft_x_rounded_quadratic_traces(
    x_soft_row: torch.Tensor,
    Q_int: np.ndarray,
    c_int: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    From one relaxed row ``x̃`` (values in ``[0,1]`` typically), return
    ``(round(x̃), objective on rounding, surrogate at x̃)`` for ``x̃ᵀQ x̃ + cᵀx̃`` (omit ``c`` when ``None``).

    Uses torch ``einsum`` on CPU to avoid NumPy matmul runtime warnings on non-finite /
    borderline inputs.
    """
    xs = x_soft_row.float().reshape(-1).detach().cpu()
    n = int(xs.numel())
    Qd = torch.as_tensor(
        np.asarray(Q_int, dtype=np.float64).reshape(n, n),
        dtype=torch.float64,
        device=xs.device,
    )
    if not bool(torch.isfinite(xs).all().item()):
        xr = torch.zeros(n, dtype=torch.int64, device=xs.device)
        return xr.numpy().copy(), float("nan"), float("nan")
    xd = xs.double()
    surr_v = torch.einsum("i,ij,j->", xd, Qd, xd)
    if c_int is not None:
        cd = torch.as_tensor(
            np.asarray(c_int, dtype=np.float64).ravel(),
            dtype=torch.float64,
            device=xs.device,
        )
        surr_v = surr_v + torch.dot(cd, xd)
    surr = float(surr_v.item()) if bool(torch.isfinite(surr_v).item()) else float("nan")
    xr = torch.clip(torch.round(xs), 0.0, 1.0).to(torch.int64)
    xrf = xr.double()
    disc_v = torch.einsum("i,ij,j->", xrf, Qd, xrf)
    if c_int is not None:
        disc_v = disc_v + torch.dot(cd, xrf)
    disc = float(disc_v.item()) if bool(torch.isfinite(disc_v).item()) else float("nan")
    return xr.numpy().copy(), disc, surr


def _run_adip_scaled_surrogate_adam(
    model: AutoregressiveBitModel,
    opt: optim.Optimizer,
    *,
    scaled_surrogate_loss: Callable[[], torch.Tensor],
    max_steps: int,
    loss_stop_threshold: Optional[float],
    plateau_patience: int,
    plateau_min_delta: float,
    max_wall_time_sec: Optional[float],
    desc: str,
    show_progress: bool,
    x_soft_holder: Optional[List[torch.Tensor]] = None,
    Q_int_for_best_soft_trace: Optional[np.ndarray] = None,
    c_int_for_best_soft_trace: Optional[np.ndarray] = None,
    soft_x_trace: Optional[List[np.ndarray]] = None,
    soft_x_trace_times_s: Optional[List[float]] = None,
    soft_x_trace_stride: int = 1,
) -> Tuple[List[float], Optional[str], Optional[Tuple[np.ndarray, float, float]]]:
    """
    Adam on ``scaled_surrogate_loss`` (typically ``w_soft * L``). Tracks minimum reported loss and restores
    post-step weights from the iteration that achieved that minimum before returning.

    When ``x_soft_holder`` and ``Q_int_for_best_soft_trace`` are set, the caller's ``scaled_surrogate_loss``
    must pass ``x_soft_out=x_soft_holder`` into :func:`_qkp_relaxation_surrogate_loss` so each forward
    stores ``x̃``; we then record ``round(x̃)``, objective on that rounding, and surrogate at ``x̃`` for the
    step that achieved the lowest ``w_soft·L``. Optional ``c_int_for_best_soft_trace`` adds the same
    linear term as SCIP / discrete eval.
    """
    loss_history: List[float] = []
    early_reason: Optional[str] = None
    best_soft_snap: Optional[Tuple[np.ndarray, float, float]] = None

    ms = max(1, int(max_steps))
    thresh = loss_stop_threshold
    use_loss_stop = thresh is not None and float(thresh) >= 0.0
    thr_val = float(thresh) if use_loss_stop else None
    patience = int(plateau_patience)
    min_delta = float(plateau_min_delta)
    use_plateau = patience > 0
    prev_lv: Optional[float] = None
    plateau_counter = 0

    use_time = max_wall_time_sec is not None and float(max_wall_time_sec) > 0.0
    t_limit = float(max_wall_time_sec) if use_time else None
    t0 = time.perf_counter()

    best_lv = float("inf")
    best_sd: Optional[Dict[str, torch.Tensor]] = None

    pbar = tqdm(
        range(ms),
        total=ms,
        desc=desc,
        unit="step",
        disable=not show_progress,
        leave=True,
    )
    for _ in pbar:
        loss = scaled_surrogate_loss()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        lv = float(loss.detach().cpu())
        loss_history.append(lv)
        if (
            soft_x_trace is not None
            and x_soft_holder is not None
            and len(x_soft_holder) == 1
            and int(soft_x_trace_stride) > 0
        ):
            step_i = len(loss_history)
            if step_i % int(soft_x_trace_stride) == 0 or step_i == 1:
                soft_x_trace.append(
                    x_soft_holder[0].detach().float().cpu().numpy().astype(np.float64).ravel().copy()
                )
                if soft_x_trace_times_s is not None:
                    soft_x_trace_times_s.append(float(time.perf_counter() - t0))
        # Use scientific notation: plain ``g`` can look like "0" for tiny |w_soft·L| or −0.0.
        pbar.set_postfix(loss=f"{lv:.5e}", refresh=False)

        if lv < best_lv:
            xr_snap: Optional[np.ndarray] = None
            disc_snap: Optional[float] = None
            surr_snap: Optional[float] = None
            if (
                x_soft_holder is not None
                and Q_int_for_best_soft_trace is not None
                and len(x_soft_holder) == 1
            ):
                xr_snap, disc_snap, surr_snap = _soft_x_rounded_quadratic_traces(
                    x_soft_holder[0],
                    Q_int_for_best_soft_trace,
                    c_int_for_best_soft_trace,
                )
            best_lv = lv
            best_sd = _clone_state_dict_cpu(model)
            if xr_snap is not None:
                best_soft_snap = (xr_snap.copy(), float(disc_snap), float(surr_snap))

        if thr_val is not None and lv <= thr_val:
            early_reason = "threshold"
            break
        if use_plateau:
            if prev_lv is not None:
                if abs(lv - prev_lv) <= min_delta:
                    plateau_counter += 1
                    if plateau_counter >= patience:
                        early_reason = "plateau"
                        break
                else:
                    plateau_counter = 0
            prev_lv = lv

        if use_time and t_limit is not None:
            if (time.perf_counter() - t0) >= t_limit:
                early_reason = "time_limit"
                break

    if best_sd is not None:
        model.load_state_dict(best_sd)

    return loss_history, early_reason, best_soft_snap


def run_phase2_adip_training(
    model: AutoregressiveBitModel,
    spec: BinaryMIPSpec,
    Q: np.ndarray,
    config: Phase2AdipConfig,
    *,
    show_progress: bool = True,
    qkp_linear_c: Optional[np.ndarray] = None,
) -> Tuple[List[float], Phase2AdipStats]:
    """
    Minimize the soft QKP surrogate (see :class:`Phase2AdipConfig`).

    After training, reloads weights from the Adam step with the lowest recorded ``w_soft·L`` (not
    necessarily the last step). Respects optional ``max_wall_time_sec`` early stop.

    If ``show_progress`` is True (default), shows a tqdm bar over ``max_optimizer_steps`` Adam steps.
    """
    from adip.phase1_ce import tensors_from_problem

    device = torch.device(config.device)
    model = model.to(device)
    A_dense = _dense_A_numpy(spec.A)
    A_t, lb_t, ub_t, _c = tensors_from_problem(A_dense, spec.lb, spec.ub, spec.c, device)
    Q_t = torch.as_tensor(np.asarray(Q, dtype=np.float32), device=device, dtype=torch.float32)
    c_lin_t: Optional[torch.Tensor] = None
    c_int_trace: Optional[np.ndarray] = None
    if qkp_linear_c is not None:
        c_int_trace = np.asarray(qkp_linear_c, dtype=np.int64)
        c_lin_t = torch.as_tensor(
            np.asarray(qkp_linear_c, dtype=np.float32),
            device=device,
            dtype=torch.float32,
        )

    opt = optim.Adam(model.parameters(), lr=float(config.lr))
    model.train()

    x_soft_holder: List[torch.Tensor] = []
    Q_int = np.asarray(Q, dtype=np.int64)

    def scaled_surrogate_loss() -> torch.Tensor:
        loss_b = _qkp_relaxation_surrogate_loss(
            model,
            A_t,
            lb_t,
            ub_t,
            Q_t,
            tau=float(config.softmax_tau),
            lambda_penalty=float(config.lambda_lagrangian),
            cost_weight=float(config.q_objective_weight),
            relaxation=str(config.relaxation),
            x_soft_out=x_soft_holder,
            c_lin=c_lin_t,
            gradient_checkpointing=bool(config.gradient_checkpointing),
            checkpoint_chunk_size=int(config.checkpoint_chunk_size),
        )
        return float(config.w_soft) * loss_b

    soft_trace_box: Optional[List[np.ndarray]] = None
    soft_trace_times: Optional[List[float]] = None
    if bool(config.record_soft_x_trace):
        soft_trace_box = []
        soft_trace_times = []

    loss_history, early_reason, best_soft_snap = _run_adip_scaled_surrogate_adam(
        model,
        opt,
        scaled_surrogate_loss=scaled_surrogate_loss,
        max_steps=int(config.max_optimizer_steps),
        loss_stop_threshold=config.loss_stop_threshold,
        plateau_patience=int(config.soft_loss_plateau_patience),
        plateau_min_delta=float(config.soft_loss_plateau_min_delta),
        max_wall_time_sec=config.max_wall_time_sec,
        desc="Phase-2 soft QKP",
        show_progress=show_progress,
        x_soft_holder=x_soft_holder,
        Q_int_for_best_soft_trace=Q_int,
        c_int_for_best_soft_trace=c_int_trace,
        soft_x_trace=soft_trace_box,
        soft_x_trace_times_s=soft_trace_times,
        soft_x_trace_stride=int(config.soft_x_trace_stride or 1),
    )

    n_done = len(loss_history)
    stats = Phase2AdipStats(
        n_soft_steps=int(n_done),
        early_stop_reason=early_reason,
        best_loss_soft_x_rounded=(
            best_soft_snap[0].copy() if best_soft_snap is not None else None
        ),
        best_loss_rounded_xT_Q_x=(
            float(best_soft_snap[1]) if best_soft_snap is not None else None
        ),
        best_loss_surrogate_xTilde_Q_xTilde=(
            float(best_soft_snap[2]) if best_soft_snap is not None else None
        ),
        best_loss_rounded_feasible=(
            _binary_x_box_feasible(spec.A, spec.lb, spec.ub, best_soft_snap[0])
            if best_soft_snap is not None
            else None
        ),
        soft_x_trace=(
            [row.copy() for row in soft_trace_box]
            if soft_trace_box is not None
            else None
        ),
        soft_x_trace_times_s=(
            [float(t) for t in soft_trace_times]
            if soft_trace_times is not None
            else None
        ),
    )
    return loss_history, stats


