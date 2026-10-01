"""Anytime objective vs wall time: SCIP-only vs SCIP+ADIP."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from adip.pipeline import QKPInstance, QKPIncumbentSnapshot, _qkp_obj


def _step_xy(
    points: Sequence[Tuple[float, float]],
    *,
    t_end: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Monotone best-so-far step curve, held until ``t_end``."""
    cleaned: List[Tuple[float, float]] = []
    best = -np.inf
    for t, v in sorted(((float(t), float(v)) for t, v in points), key=lambda z: z[0]):
        if not np.isfinite(v):
            continue
        if v > best:
            best = v
            cleaned.append((max(0.0, t), v))
    if not cleaned:
        return np.array([0.0, float(t_end)], dtype=np.float64), np.array(
            [0.0, 0.0], dtype=np.float64
        )
    xs: List[float] = []
    ys: List[float] = []
    if cleaned[0][0] > 1e-12:
        xs.append(0.0)
        ys.append(0.0)
    for t, v in cleaned:
        xs.append(t)
        ys.append(v)
    if xs[-1] < float(t_end):
        xs.append(float(t_end))
        ys.append(ys[-1])
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def scip_anytime_from_snapshots(
    snaps: Sequence[QKPIncumbentSnapshot],
    *,
    t_end: float,
    final_obj: Optional[float] = None,
    final_t: Optional[float] = None,
    obj_trace: Optional[Sequence[Tuple[float, float]]] = None,
) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    if obj_trace:
        pts.extend((float(t), float(v)) for t, v in obj_trace)
    pts.extend((float(s.wall_time_s), float(s.xTQx)) for s in snaps)
    if final_obj is not None:
        pts.append((float(final_t if final_t is not None else t_end), float(final_obj)))
    return pts


def phase2_rounded_anytime(
    inst: QKPInstance,
    *,
    t_hand: float,
    p2_xs: Sequence[np.ndarray],
    p2_times_s: Sequence[float],
    rounded_final: Optional[float],
    rounded_feasible: Optional[bool],
    phase2_wall_s: float,
) -> List[Tuple[float, float]]:
    """Running-best feasible rounded objective during phase-2 only (CE-free clock)."""
    pts: List[Tuple[float, float]] = []
    w = np.asarray(inst.w, dtype=np.int64).ravel()
    W = int(inst.W)
    for x_soft, dt in zip(p2_xs, p2_times_s):
        xr = np.clip(np.rint(np.asarray(x_soft, dtype=np.float64).ravel()), 0.0, 1.0).astype(
            np.int64
        )
        if int(np.dot(w, xr)) > W:
            continue
        pts.append((float(t_hand) + float(dt), float(_qkp_obj(inst, xr))))
    if rounded_final is not None and bool(rounded_feasible):
        pts.append((float(t_hand) + float(phase2_wall_s), float(rounded_final)))
    return pts


def adip_anytime_exclude_ce(
    *,
    scip_pts: Sequence[Tuple[float, float]],
    warm_budget_s: float,
    phase2_pts: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """
    Combined SCIP+ADIP anytime on a CE-free clock.

    SCIP warm on ``[0, warm_budget]``, then phase-2 discoveries on
    ``[warm_budget, …]``. Combined incumbent never drops below the SCIP warm best.
    """
    t_hand = float(warm_budget_s)
    pts: List[Tuple[float, float]] = []
    warm_best = 0.0
    for t, v in scip_pts:
        tt = min(float(t), t_hand)
        pts.append((tt, float(v)))
        warm_best = max(warm_best, float(v))
    pts.append((t_hand, float(warm_best)))
    for t, v in phase2_pts:
        pts.append((float(t), max(float(v), float(warm_best))))
    return pts


def adip_anytime_from_traces(
    inst: QKPInstance,
    *,
    scip_pts: Sequence[Tuple[float, float]],
    warm_wall_s: float,
    phase1_wall_s: float,
    p2_xs: Sequence[np.ndarray],
    p2_times_s: Sequence[float],
    rounded_final: Optional[float],
    rounded_feasible: Optional[bool],
    pipeline_wall_s: float,
) -> List[Tuple[float, float]]:
    """Running-best feasible objective along SCIP warm-start then ADIP (includes CE time)."""
    pts: List[Tuple[float, float]] = list(scip_pts)
    t0_p2 = float(warm_wall_s) + float(phase1_wall_s)
    w = np.asarray(inst.w, dtype=np.int64).ravel()
    W = int(inst.W)
    for x_soft, dt in zip(p2_xs, p2_times_s):
        xr = np.clip(np.rint(np.asarray(x_soft, dtype=np.float64).ravel()), 0.0, 1.0).astype(
            np.int64
        )
        if int(np.dot(w, xr)) > W:
            continue
        pts.append((t0_p2 + float(dt), float(_qkp_obj(inst, xr))))
    if rounded_final is not None and bool(rounded_feasible):
        pts.append((float(pipeline_wall_s), float(rounded_final)))
    return pts


def _draw_y_break_marks(fig: Any, ax_top: Any, ax_bot: Any) -> None:
    """Parallel diagonal marks in figure coordinates (same angle on both panels)."""
    from matplotlib.lines import Line2D

    # Identical rise/run in figure space so marks stay parallel despite unequal panel heights.
    run, rise = 0.007, 0.012
    for ax, y_frac in ((ax_top, 0.0), (ax_bot, 1.0)):
        bbox = ax.get_position()
        for x_frac in (0.0, 1.0):
            x0 = bbox.x0 + x_frac * bbox.width
            y0 = bbox.y0 + y_frac * bbox.height
            fig.add_artist(
                Line2D(
                    [x0 - run, x0 + run],
                    [y0 - rise, y0 + rise],
                    transform=fig.transFigure,
                    color="k",
                    lw=1.0,
                    clip_on=False,
                    solid_capstyle="butt",
                )
            )


def _apply_invlog_remaining_x(
    ax: Any,
    *,
    t_max: float,
    t_min: float,
    eps: float = 1.0,
) -> None:
    """X-scale ``-log(t_max + eps - t)``: expands late times (inverse-log of remaining time)."""
    t_hi = float(t_max) + float(eps)

    def _forward(t: Any) -> Any:
        tt = np.asarray(t, dtype=np.float64)
        return -np.log(np.clip(t_hi - tt, 1e-12, None))

    def _inverse(u: Any) -> Any:
        uu = np.asarray(u, dtype=np.float64)
        return t_hi - np.exp(-uu)

    ax.set_xscale("function", functions=(_forward, _inverse))
    ax.set_xlim(float(t_min), float(t_max))


def plot_obj_vs_time(
    *,
    scip_pts: Sequence[Tuple[float, float]],
    adip_pts: Sequence[Tuple[float, float]],
    t_end: float,
    out_path: Path,
    title: str,
    t_handoff: Optional[float] = None,
    phase2_only_pts: Optional[Sequence[Tuple[float, float]]] = None,
    xlabel: str = "Wall time (s)",
    log_x: bool = False,
    invlog_remaining_x: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    y_break: Optional[Tuple[float, float, float, float]] = None,
) -> Dict[str, Any]:
    """
    Parameters
    ----------
    y_break
        If set ``(y_low_min, y_low_max, y_high_min, y_high_max)``, draw a broken
        y-axis (lower panel for the near-zero band, upper for the zoomed range).
    log_x
        Use a logarithmic x-axis (times ``<= 0`` are clipped to a small positive floor).
        Prefer linear / ``invlog_remaining_x`` when the interesting dynamics are late.
    invlog_remaining_x
        Map time with ``-log(t_end + eps - t)`` so late wall times are visually expanded.
    xlim
        Optional ``(t_min, t_max)`` window. Useful to zoom the improvement region.
    """
    from matplotlib import pyplot as plt

    # Single SCIP+ADIP anytime curve: SCIP warm, then phase-2 discoveries,
    # never dropping below the warm incumbent.
    if t_handoff is not None and phase2_only_pts is not None:
        warm = [(t, v) for t, v in scip_pts if float(t) <= float(t_handoff) + 1e-9]
        warm_best = max((float(v) for _, v in warm), default=0.0)
        combined: List[Tuple[float, float]] = list(warm)
        combined.append((float(t_handoff), warm_best))
        for t, v in phase2_only_pts:
            combined.append((float(t), max(float(v), warm_best)))
        adip_pts = combined

    x_s, y_s = _step_xy(scip_pts, t_end=t_end)
    x_a, y_a = _step_xy(adip_pts, t_end=t_end)

    t_floor = 1e-2
    if log_x:
        x_s = np.maximum(x_s, t_floor)
        x_a = np.maximum(x_a, t_floor)

    x_lo = float(xlim[0]) if xlim is not None else (t_floor if log_x else 0.0)
    x_hi = float(xlim[1]) if xlim is not None else float(t_end)

    def _style_axis(
        ax: Any,
        *,
        show_xlabel: bool,
        show_legend: bool,
        show_handoff_label: bool,
    ) -> None:
        ax.step(x_s, y_s, where="post", color="tab:blue", lw=2.0, label="SCIP (matched time)")
        ax.step(x_a, y_a, where="post", color="tab:green", lw=2.0, label="SCIP + ADIP")
        if t_handoff is not None and float(t_handoff) >= x_lo - 1e-9:
            ax.axvline(float(t_handoff), color="0.45", ls="--", lw=1.0, alpha=0.75)
            if show_handoff_label:
                ax.text(
                    float(t_handoff),
                    0.05,
                    "phase-2 starts",
                    color="0.35",
                    fontsize=8,
                    rotation=90,
                    va="bottom",
                    ha="right",
                    transform=ax.get_xaxis_transform(),
                )
        if invlog_remaining_x:
            _apply_invlog_remaining_x(ax, t_max=x_hi, t_min=x_lo)
        elif log_x:
            ax.set_xscale("log")
            ax.set_xlim(x_lo, x_hi)
        else:
            ax.set_xlim(x_lo, x_hi)
        ax.grid(True, alpha=0.3, which="both" if (log_x or invlog_remaining_x) else "major")
        if show_xlabel:
            ax.set_xlabel(xlabel)
        if show_legend:
            ax.legend(loc="lower right")

    if y_break is None:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        _style_axis(ax, show_xlabel=True, show_legend=True, show_handoff_label=True)
        ax.set_ylabel(r"$x^\top Q x$")
        ax.set_title(title)
    else:
        y0a, y0b, y1a, y1b = (float(v) for v in y_break)
        fig, (ax_hi, ax_lo) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(8.5, 5.6),
            gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.05},
        )
        _style_axis(ax_hi, show_xlabel=False, show_legend=True, show_handoff_label=True)
        _style_axis(ax_lo, show_xlabel=True, show_legend=False, show_handoff_label=False)
        ax_hi.set_ylim(y1a, y1b)
        ax_lo.set_ylim(y0a, y0b)
        ax_hi.spines["bottom"].set_visible(False)
        ax_lo.spines["top"].set_visible(False)
        ax_hi.tick_params(bottom=False, labelbottom=False)
        ax_hi.set_title(title)
        ax_hi.set_ylabel(r"$x^\top Q x$")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if y_break is None:
        fig.tight_layout()
    else:
        fig.subplots_adjust(hspace=0.05, left=0.12, right=0.98, top=0.92, bottom=0.12)
        _draw_y_break_marks(fig, ax_hi, ax_lo)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return {
        "out_path": str(out_path),
        "scip_final": None if y_s.size == 0 else float(y_s[-1]),
        "adip_final": None if y_a.size == 0 else float(y_a[-1]),
        "t_end": float(t_end),
    }
