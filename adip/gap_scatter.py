"""Scatter plots of warm-start MIP gap vs ADIP objective improvement."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# --- Paste your numbers here (one row per instance / experiment) -----------------

# SCIP MIP optimality gap at termination (unitless, e.g. 7.94 means ~794% in some logs).
GAP: List[float] = [ 
    6.62,
    7.119,
    8.786,
    7.387,
    5.569,
    6.834,
    9.778,
    6.618,
    9.332,
    5.644,
    6.114,
    9.21,
    8.257,
    6.852,
    8.446,
]

# Best feasible objective from your method (e.g. ADIP rollout); use the same units as SCIP z.
BEST_OBJ: List[float] = [
    2376.0,
    2341.0,
    2029.0,
    2148.0,
    2951.0,
    2471.0,
    1904.0,
    2376.0,
    2046.0,
    2639.0,
    2885.0,
    2126.0,
    1841.0,
    2300.0,
    1783.0,
]

# SCIP incumbent objective (max xᵀQx / epigraph z).
SCIP_OBJ: List[float] = [
    2376.0,
    2322.0,
    1919.0,
    2148.0,
    2951.0,
    2326.0,
    1862.0,
    2376.0,
    1801.0,
    2639.0,
    2885.0,
    1971.0,
    1738.0,
    2300.0,
    1708.0,
]

# Optional short labels for hover / annotate (same length as GAP, or leave empty).
LABELS: List[str] = [
    # "N=50 exp=0",
]


@dataclass(frozen=True)
class GapImprovementPoint:
    gap: float
    improvement_pct: float
    label: str = ""


def improvement_pct_vs_scip(*, best: float, scip: float) -> float:
    """Percent change of ``best`` relative to ``scip`` (maximization: positive => best > scip)."""
    if abs(scip) <= 1e-15:
        raise ValueError(f"|scip| too small for relative improvement: scip={scip!r}")
    return 100.0 * (float(best) - float(scip)) / abs(float(scip))


def reported_obj_and_improvement_pct(*, adip: float, scip: float) -> Tuple[float, float]:
    """If ADIP never beats the feasible SCIP incumbent, keep that incumbent (0% improvement)."""
    scip_f = float(scip)
    adip_f = float(adip)
    if adip_f < scip_f:
        return scip_f, 0.0
    return adip_f, improvement_pct_vs_scip(best=adip_f, scip=scip_f)


def points_from_lists(
    gap: Sequence[float],
    best_obj: Sequence[float],
    scip_obj: Sequence[float],
    labels: Optional[Sequence[str]] = None,
) -> List[GapImprovementPoint]:
    n = len(gap)
    if len(best_obj) != n or len(scip_obj) != n:
        raise ValueError(
            f"GAP ({len(gap)}), BEST_OBJ ({len(best_obj)}), SCIP_OBJ ({len(scip_obj)}) "
            "must have the same length."
        )
    lbls: List[str]
    if not labels:
        lbls = [""] * n
    else:
        if len(labels) != n:
            raise ValueError(f"LABELS length {len(labels)} != {n}")
        lbls = [str(x) for x in labels]
    out: List[GapImprovementPoint] = []
    for i in range(n):
        out.append(
            GapImprovementPoint(
                gap=float(gap[i]),
                improvement_pct=improvement_pct_vs_scip(
                    best=float(best_obj[i]), scip=float(scip_obj[i])
                ),
                label=lbls[i],
            )
        )
    return out


def points_from_improvement_pct(
    gap: Sequence[float],
    improvement_pct: Sequence[float],
    labels: Optional[Sequence[str]] = None,
) -> List[GapImprovementPoint]:
    """Use when you already computed y by hand from the terminal."""
    n = len(gap)
    if len(improvement_pct) != n:
        raise ValueError(
            f"GAP length {n} != IMPROVEMENT_PCT length {len(improvement_pct)}"
        )
    lbls: List[str]
    if not labels:
        lbls = [""] * n
    else:
        if len(labels) != n:
            raise ValueError(f"LABELS length {len(labels)} != {n}")
        lbls = [str(x) for x in labels]
    return [
        GapImprovementPoint(gap=float(gap[i]), improvement_pct=float(improvement_pct[i]), label=lbls[i])
        for i in range(n)
    ]


# If you prefer to type y directly, set USE_DIRECT_IMPROVEMENT = True and fill these:
USE_DIRECT_IMPROVEMENT = False
IMPROVEMENT_PCT: List[float] = [
    # 4.5,
]


def build_points() -> List[GapImprovementPoint]:
    if not GAP:
        raise ValueError(
            "No data: fill GAP and either (BEST_OBJ, SCIP_OBJ) or "
            "(USE_DIRECT_IMPROVEMENT=True, IMPROVEMENT_PCT) in "
            "adip/qkp_gap_improvement_scatter.py"
        )
    if USE_DIRECT_IMPROVEMENT:
        return points_from_improvement_pct(GAP, IMPROVEMENT_PCT, LABELS or None)
    return points_from_lists(GAP, BEST_OBJ, SCIP_OBJ, LABELS or None)


def plot_scatter(
    points: Sequence[GapImprovementPoint],
    *,
    out_path: Optional[str] = None,
    title: str = "Best feasible solution vs SCIP (by terminal gap)",
    show: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    xs = [p.gap for p in points]
    ys = [p.improvement_pct for p in points]

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.scatter(xs, ys, s=56, alpha=0.85, edgecolors="k", linewidths=0.4)
    ax.axhline(0.0, color="0.4", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xlabel("SCIP MIP gap at termination")
    ax.set_ylabel("% improvement of best feasible vs SCIP objective")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    for p in points:
        if not p.label:
            continue
        ax.annotate(
            p.label,
            (p.gap, p.improvement_pct),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Wrote {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional PNG path (e.g. qkp_benchmark_out/gap_vs_improvement.png).",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Only save --out; do not open an interactive window.",
    )
    p.add_argument("--title", type=str, default="", help="Plot title override.")
    args = p.parse_args()

    pts = build_points()
    for i, pt in enumerate(pts):
        extra = f"  {pt.label!r}" if pt.label else ""
        print(
            f"  [{i}]  gap={pt.gap:g}  improvement={pt.improvement_pct:+.2f}%{extra}"
        )

    title = args.title.strip() or "Best feasible solution vs SCIP (by terminal gap)"
    out = args.out.strip() or None
    plot_scatter(
        pts,
        out_path=out,
        title=title,
        show=not bool(args.no_show),
    )


if __name__ == "__main__":
    main()
