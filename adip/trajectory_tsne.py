"""
t-SNE + 3D objective visualization for QKP warm-start SCIP incumbents vs phase-2 soft ``x̃``.

Fits **one** t-SNE on a stacked matrix of ``n``-dimensional vectors (binary / soft in ``[0,1]ⁿ``),
then plots in 3D: t-SNE coordinates as ``x,y`` and continuous objective ``xᵀQx+cᵀx`` as ``z``:

* SCIP incumbent polyline (chronological order of recorded snapshots),
* phase-2 relaxed ``x̃`` polyline (Adam step order),
* line segments from knapsack-boundary relaxation (each sampled start → GD endpoint).

**Why few SCIP incumbents?** If ``incumbent_trace_meta.n_bestsol_callbacks`` is large but
``len(incumbent_snapshots)`` is small, callbacks may not have been able to read ``x`` reliably
(``getBestSol`` / ``getSolVal``); if both are small, SCIP may have found few improving primals
within the time limit. See warm-start diagnostics printed by the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from adip.continuous_relax import (
    continuous_qkp_value,
    knapsack_boundary_gradient_ascent,
)
from adip.pipeline import (
    AdipPipelineResult,
    QKPInstance,
    QKPIncumbentSnapshot,
)


def _objectives(inst: QKPInstance, rows: np.ndarray) -> np.ndarray:
    out = np.zeros(int(rows.shape[0]), dtype=np.float64)
    for i in range(int(rows.shape[0])):
        out[i] = float(continuous_qkp_value(inst, rows[i]))
    return out


def _subsample_uniform(n: int, cap: int) -> np.ndarray:
    if n <= cap:
        return np.arange(n, dtype=np.int64)
    return np.unique(
        np.linspace(0, n - 1, num=min(cap, n), dtype=np.int64)
    )


def count_distinct_l2(points: List[np.ndarray], *, threshold: float) -> int:
    """Greedy unique count: a point is new if its ℓ₂ distance to all reps exceeds ``threshold``."""
    thr = float(threshold)
    reps: List[np.ndarray] = []
    for p in points:
        x = np.asarray(p, dtype=np.float64).ravel()
        if all(float(np.linalg.norm(x - r)) > thr for r in reps):
            reps.append(x)
    return int(len(reps))


def replot_tsne_from_cache(
    cache_path: Path,
    out_path: Path,
) -> Dict[str, Any]:
    """Redraw an existing t-SNE cache with current styling (no GD / t-SNE recompute)."""
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    data = np.load(Path(cache_path), allow_pickle=True)
    emb_scip = np.asarray(data["emb_scip"], dtype=np.float64)
    emb_p2 = np.asarray(data["emb_p2"], dtype=np.float64)
    emb_r0 = np.asarray(data["emb_r0"], dtype=np.float64)
    emb_r1 = np.asarray(data["emb_r1"], dtype=np.float64)
    z_scip = np.asarray(data["z_scip"], dtype=np.float64)
    z_p2 = np.asarray(data["z_p2"], dtype=np.float64)
    z_r0 = np.asarray(data["z_r0"], dtype=np.float64)
    z_r1 = np.asarray(data["z_r1"], dtype=np.float64)
    base_seed = int(np.asarray(data["base_seed"]).item())
    n = int(np.asarray(data["n"]).item())
    note = str(np.asarray(data["caption"]).item()) if "caption" in data.files else ""

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if emb_scip.shape[0]:
        ax.scatter(
            emb_scip[:, 0], emb_scip[:, 1], z_scip, c="tab:blue", s=40, label="SCIP incumbents"
        )
        if emb_scip.shape[0] >= 2:
            ax.plot(emb_scip[:, 0], emb_scip[:, 1], z_scip, c="tab:blue", lw=1.3, alpha=0.9)

    if emb_p2.shape[0]:
        ax.scatter(
            emb_p2[:, 0], emb_p2[:, 1], z_p2, c="tab:green", s=18, label="Phase-2 x̃ (plot subset)"
        )
        if emb_p2.shape[0] >= 2:
            ax.plot(emb_p2[:, 0], emb_p2[:, 1], z_p2, c="tab:green", lw=1.0, alpha=0.75)

    for i in range(int(emb_r0.shape[0])):
        ax.plot(
            [emb_r0[i, 0], emb_r1[i, 0]],
            [emb_r0[i, 1], emb_r1[i, 1]],
            [z_r0[i], z_r1[i]],
            c="tab:red",
            lw=1.1,
            alpha=0.7,
            linestyle=(0, (1.2, 1.8)),
        )
    if int(emb_r1.shape[0]) > 0:
        ax.scatter(
            emb_r1[:, 0],
            emb_r1[:, 1],
            z_r1,
            c="tab:red",
            s=22,
            depthshade=False,
            label="Relax to knapsack boundary",
            zorder=5,
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_zlabel("xᵀQx + cᵀx")
    ax.set_title(f"QKP incumbent / phase-2 / relax rays  (n={n})")
    ax.legend(loc="upper left", fontsize=8)
    if note:
        fig.text(0.02, 0.02, note, fontsize=7, family="monospace")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return {"out_path": str(out_path), "cache_path": str(cache_path), "caption": note}


def plot_qkp_trajectories_tsne(
    inst: QKPInstance,
    outcome: AdipPipelineResult,
    out_path: Path,
    *,
    base_seed: int,
    gd_lr: float = 0.05,
    gd_max_iters: int = 3000,
    gd_tol_knapsack: float = 1e-6,
    gd_tol_grad: float = 1e-8,
    tsne_perplexity: float = 30.0,
    tsne_seed: int = 0,
    max_tsne_rows: int = 500,
    max_phase2_plot: int = 220,
    max_relax_rays: int = 96,
    endpoint_l2_threshold: float = 1e-3,
) -> Dict[str, Any]:
    """
    Fit t-SNE on stacked trajectory + ray endpoints; save a 3D matplotlib PNG.

    Also records basin stats: trajectory-point count, distinct GD endpoints under
    ``endpoint_l2_threshold``, incumbent-ray endpoint objective, and best endpoint objective.
    """
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from sklearn.manifold import TSNE

    n = int(inst.n)
    snaps: List[QKPIncumbentSnapshot] = list(outcome.incumbent_snapshots or [])
    scip_X = (
        np.stack([np.asarray(s.x, dtype=np.float64) for s in snaps], axis=0)
        if snaps
        else np.zeros((0, n), dtype=np.float64)
    )

    p2_raw = list(outcome.phase2_soft_x_trace or [])
    p2_all = (
        np.stack([np.asarray(r, dtype=np.float64).ravel() for r in p2_raw], axis=0)
        if p2_raw
        else np.zeros((0, n), dtype=np.float64)
    )
    if int(p2_all.shape[0]) > int(max_phase2_plot):
        idx = _subsample_uniform(int(p2_all.shape[0]), int(max_phase2_plot))
        p2_X = p2_all[idx].copy()
    else:
        p2_X = p2_all

    ray_starts: List[np.ndarray] = []
    if int(scip_X.shape[0]) > 0:
        for i in range(int(scip_X.shape[0])):
            ray_starts.append(scip_X[i].copy())
    budget = max(0, int(max_relax_rays) - len(ray_starts))
    if int(p2_X.shape[0]) > 0 and budget > 0:
        idx = _subsample_uniform(int(p2_X.shape[0]), budget)
        for j in idx:
            ray_starts.append(p2_X[int(j)].copy())
    ray_starts = ray_starts[: int(max_relax_rays)]

    ray_ends: List[np.ndarray] = []
    ray_f_stars: List[float] = []
    n_scip_starts = int(scip_X.shape[0])
    for k, x0 in enumerate(ray_starts):
        res = knapsack_boundary_gradient_ascent(
            inst,
            x0,
            label=f"ray_{k}",
            max_iters=int(gd_max_iters),
            lr=float(gd_lr),
            tol_knapsack=float(gd_tol_knapsack),
            tol_grad=float(gd_tol_grad),
        )
        ray_ends.append(np.asarray(res.x_star, dtype=np.float64).ravel().copy())
        ray_f_stars.append(float(res.f_star))

    r0 = np.stack(ray_starts, axis=0) if ray_starts else np.zeros((0, n), dtype=np.float64)
    r1 = np.stack(ray_ends, axis=0) if ray_ends else np.zeros((0, n), dtype=np.float64)
    thr = float(endpoint_l2_threshold)
    n_distinct = count_distinct_l2(ray_ends, threshold=thr) if ray_ends else 0
    # Incumbent endpoint: GD from the chronologically last SCIP snapshot (if any).
    if n_scip_starts > 0 and ray_f_stars:
        incumbent_endpoint_obj = float(ray_f_stars[n_scip_starts - 1])
    else:
        incumbent_endpoint_obj = None
    best_endpoint_obj = float(max(ray_f_stars)) if ray_f_stars else None
    basin = {
        "trajectory_points": int(p2_all.shape[0]),
        "n_relax_rays": int(len(ray_starts)),
        "distinct_endpoints": int(n_distinct),
        "endpoint_l2_threshold": thr,
        "incumbent_endpoint_obj": incumbent_endpoint_obj,
        "best_endpoint_obj": best_endpoint_obj,
    }

    p2_use = p2_X.copy()
    rng = np.random.default_rng(int(tsne_seed))
    X_fit: np.ndarray
    while True:
        parts: List[np.ndarray] = []
        if int(scip_X.shape[0]) > 0:
            parts.append(scip_X)
        if int(p2_use.shape[0]) > 0:
            parts.append(p2_use)
        if int(r0.shape[0]) > 0:
            parts.append(r0)
            parts.append(r1)
        if not parts:
            if outcome.warm_x is not None:
                X_fit = np.clip(
                    np.asarray(outcome.warm_x, dtype=np.float64), 0.0, 1.0
                ).reshape(1, -1)
            else:
                X_fit = np.zeros((1, n), dtype=np.float64)
            break
        X_fit = np.vstack(parts)
        if int(X_fit.shape[0]) <= int(max_tsne_rows):
            break
        if int(p2_use.shape[0]) > 8:
            p2_use = p2_use[: max(8, int(p2_use.shape[0]) // 2)].copy()
            continue
        n_all = int(X_fit.shape[0])
        pick = np.sort(rng.choice(n_all, size=int(max_tsne_rows), replace=False))
        X_fit = X_fit[pick].copy()
        break

    p2_plot = np.asarray(p2_use, dtype=np.float64)
    n_fit = int(X_fit.shape[0])
    perplexity = float(tsne_perplexity)
    perplexity = max(5.0, min(perplexity, max(5.0, (n_fit - 1) / 3.0)))

    emb = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=int(tsne_seed),
        init="pca",
        learning_rate="auto",
    ).fit_transform(X_fit.astype(np.float64, copy=False))

    # Map each original row to nearest row in X_fit (t-SNE has no transform).
    def _row_emb(row: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(X_fit - row.reshape(1, -1), axis=1)
        return emb[int(np.argmin(d))].copy()

    emb_scip = (
        np.stack([_row_emb(scip_X[i]) for i in range(int(scip_X.shape[0]))], axis=0)
        if int(scip_X.shape[0]) > 0
        else np.zeros((0, 2), dtype=np.float64)
    )
    emb_p2 = (
        np.stack([_row_emb(p2_plot[i]) for i in range(int(p2_plot.shape[0]))], axis=0)
        if int(p2_plot.shape[0]) > 0
        else np.zeros((0, 2), dtype=np.float64)
    )
    emb_r0 = (
        np.stack([_row_emb(a) for a in ray_starts], axis=0) if ray_starts else np.zeros((0, 2), dtype=np.float64)
    )
    emb_r1 = (
        np.stack([_row_emb(b) for b in ray_ends], axis=0) if ray_ends else np.zeros((0, 2), dtype=np.float64)
    )

    z_scip = _objectives(inst, scip_X) if int(scip_X.shape[0]) > 0 else np.array([])
    z_p2 = _objectives(inst, p2_plot) if int(p2_plot.shape[0]) > 0 else np.array([])
    z_r0 = _objectives(inst, np.asarray(ray_starts)) if ray_starts else np.array([])
    z_r1 = _objectives(inst, np.asarray(ray_ends)) if ray_ends else np.array([])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if emb_scip.shape[0]:
        ax.scatter(
            emb_scip[:, 0], emb_scip[:, 1], z_scip, c="tab:blue", s=40, label="SCIP incumbents"
        )
        if emb_scip.shape[0] >= 2:
            ax.plot(emb_scip[:, 0], emb_scip[:, 1], z_scip, c="tab:blue", lw=1.3, alpha=0.9)

    if emb_p2.shape[0]:
        ax.scatter(
            emb_p2[:, 0], emb_p2[:, 1], z_p2, c="tab:green", s=18, label="Phase-2 x̃ (plot subset)"
        )
        if emb_p2.shape[0] >= 2:
            ax.plot(emb_p2[:, 0], emb_p2[:, 1], z_p2, c="tab:green", lw=1.0, alpha=0.75)

    for i in range(int(emb_r0.shape[0])):
        ax.plot(
            [emb_r0[i, 0], emb_r1[i, 0]],
            [emb_r0[i, 1], emb_r1[i, 1]],
            [z_r0[i], z_r1[i]],
            c="tab:red",
            lw=1.1,
            alpha=0.7,
            linestyle=(0, (1.2, 1.8)),
        )
    if int(emb_r1.shape[0]) > 0:
        ax.scatter(
            emb_r1[:, 0],
            emb_r1[:, 1],
            z_r1,
            c="tab:red",
            s=22,
            depthshade=False,
            label="Relax to knapsack boundary",
            zorder=5,
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_zlabel("xᵀQx + cᵀx")
    ax.set_title(f"QKP incumbent / phase-2 / relax rays  (n={n})")
    ax.legend(loc="upper left", fontsize=8)

    meta = outcome.incumbent_trace_meta
    note = ""
    if meta is not None:
        note = (
            f"SCIP: {meta.n_snapshots_after_harvest} snapshots | "
            f"{meta.n_bestsol_callbacks} BESTSOL callbacks | "
            f"n_best_found={meta.n_bestsol_found} | stored_sols={meta.n_stored_sols}"
        )
    fig.text(0.02, 0.02, note, fontsize=7, family="monospace")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    cache_path = out_path.with_suffix(".npz")
    np.savez_compressed(
        cache_path,
        emb_scip=emb_scip,
        emb_p2=emb_p2,
        emb_r0=emb_r0,
        emb_r1=emb_r1,
        z_scip=np.asarray(z_scip, dtype=np.float64),
        z_p2=np.asarray(z_p2, dtype=np.float64),
        z_r0=np.asarray(z_r0, dtype=np.float64),
        z_r1=np.asarray(z_r1, dtype=np.float64),
        base_seed=np.int64(base_seed),
        n=np.int64(n),
        caption=np.asarray(note),
    )

    return {
        "out_path": str(out_path),
        "cache_path": str(cache_path),
        "n_scip": int(scip_X.shape[0]),
        "n_phase2_plot": int(p2_plot.shape[0]),
        "n_phase2_raw": int(p2_all.shape[0]),
        "n_relax_rays": int(len(ray_starts)),
        "tsne_rows": int(X_fit.shape[0]),
        "caption": note,
        "basin_analysis": basin,
    }
