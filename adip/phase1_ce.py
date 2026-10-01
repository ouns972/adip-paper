"""Phase-1 teacher-forcing cross-entropy on a feasible binary incumbent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

from adip.ar_model import AutoregressiveBitModel
from adip.teacher_ce import (
    incumbent_teacher_forcing_ce_loss,
    incumbent_teacher_forcing_soft_ce_loss,
)


@dataclass
class Phase1CEConfig:
    """Teacher-forcing CE to match a fixed feasible integer path (phase 1 of two-phase training)."""

    lr: float = 1e-3
    device: str = "cpu"
    #: Adam steps. ``0`` skips phase-1 entirely (model left at random init).
    max_optimizer_steps: int = 200
    #: Stop after this many consecutive steps without CE improving by more than ``ce_plateau_min_delta``.
    #: ``0`` disables plateau early stopping (run up to ``max_optimizer_steps`` only).
    ce_plateau_patience: int = 5
    #: Improvement requires CE loss ``< best − ce_plateau_min_delta`` (lower is better).
    ce_plateau_min_delta: float = 1e-8
    #: If in ``(0.5, 1)``, phase-1 CE uses soft targets (correct class prob = this). Use ``1.0`` or
    #: ``None`` for standard hard CE on ``{0,1}``.
    ce_target_prob_correct: Optional[float] = 0.999


def run_phase1_ce_training(
    model: AutoregressiveBitModel,
    x_target: np.ndarray,
    config: Phase1CEConfig,
    *,
    show_progress: bool = True,
) -> Tuple[List[float], Optional[str]]:
    """
    Adam steps minimizing mean next-token CE to a single target bitstring ``x_target`` (shape ``(n,)``).
    No soft Lagrangian — only :func:`incumbent_teacher_forcing_ce_loss`.

    If ``show_progress`` is True (default), shows a tqdm bar for Adam steps (``max_optimizer_steps``).
    Set False for clean logs in CI or when stdout is not a TTY.

    Returns training loss history and optional early-stop reason (``"plateau"`` or ``None``).
    """
    device = torch.device(config.device)
    model = model.to(device)
    xv = np.asarray(x_target, dtype=np.int64).ravel()
    x_b = torch.as_tensor(xv, dtype=torch.long, device=device).unsqueeze(0)
    max_steps = int(config.max_optimizer_steps)
    if max_steps <= 0:
        return [], "skipped"
    opt = optim.Adam(model.parameters(), lr=float(config.lr))
    model.train()
    loss_history: List[float] = []
    patience = int(config.ce_plateau_patience)
    min_delta = float(config.ce_plateau_min_delta)
    use_plateau = patience > 0
    best_ce = float("inf")
    plateau_counter = 0
    early_reason: Optional[str] = None

    ctp = config.ce_target_prob_correct
    use_soft = ctp is not None and float(ctp) < 1.0 - 1e-12

    pbar = tqdm(
        range(max_steps),
        total=max_steps,
        desc="Phase-1 CE",
        unit="step",
        disable=not show_progress,
        leave=True,
    )
    for _ in pbar:
        if use_soft:
            loss = incumbent_teacher_forcing_soft_ce_loss(
                model, x_b, prob_correct=float(ctp), reduction="mean"
            )
        else:
            loss = incumbent_teacher_forcing_ce_loss(model, x_b, reduction="mean")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        lv = float(loss.detach().cpu())
        loss_history.append(lv)
        pbar.set_postfix(loss=f"{lv:.4g}", refresh=False)

        if use_plateau:
            improved = lv < best_ce - min_delta
            if improved:
                best_ce = lv
                plateau_counter = 0
            else:
                plateau_counter += 1
                if plateau_counter >= patience:
                    early_reason = "plateau"
                    break

    return loss_history, early_reason


def tensors_from_problem(
    A: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    c: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    A_t = torch.as_tensor(np.asarray(A, dtype=np.float32), device=device)
    lb_t = torch.as_tensor(np.asarray(lb, dtype=np.float32).ravel(), device=device)
    ub_t = torch.as_tensor(np.asarray(ub, dtype=np.float32).ravel(), device=device)
    c_t = torch.as_tensor(np.asarray(c, dtype=np.float32).ravel(), device=device)
    return A_t, lb_t, ub_t, c_t


