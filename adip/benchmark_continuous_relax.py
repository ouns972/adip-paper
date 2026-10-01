"""Continuous relaxation trajectories from SCIP incumbents and ADIP starts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from adip.continuous_relax import (
    ContinuousRelaxGDResult,
    continuous_qkp_value,
    adip_binary_start_from_pipeline,
    knapsack_boundary_gradient_ascent,
    summarize_gd_results,
)
from adip.pipeline import (
    AdipPipelineConfig,
    _apply_reproducibility,
    _derive_torch_run_seed,
    _parse_scip_param_kv_list,
    _instance_identity_message,
    _qkp_obj,
    distinct_incumbent_snapshots_by_x,
    last_k_distinct_incumbent_snapshots,
    run_adip_pipeline,
    sample_qkp_instance,
)
from adip.obj_time_plot import adip_anytime_from_traces, scip_anytime_from_snapshots
from adip.trajectory_tsne import plot_qkp_trajectories_tsne
from adip.warm_solvers import WARM_SOLVER_CHOICES, normalize_warm_solver


def _build_adip_cfg(
    *,
    args: argparse.Namespace,
    scip_base: Optional[Dict[str, Any]],
    gurobi_lic: Optional[str],
    base_seed: int,
) -> AdipPipelineConfig:
    tag = normalize_warm_solver(args.warm_solver)
    pipeline = f"adip_{tag}_gumbel_ste"
    return AdipPipelineConfig(
        pipeline_name=pipeline,
        use_gurobi_warm_start=False,
        device=str(args.device),
        base_seed=int(base_seed),
        q_kind=str(args.q_kind),
        ar_lr=float(args.ar_lr),
        ar_d_model=int(args.ar_d_model),
        ar_nhead=int(args.ar_nhead),
        ar_nlayers=int(args.ar_nlayers),
        ar_sparse_attn_prev_fraction=float(args.ar_sparse_attn_prev_fraction),
        ar_sparse_attn_pattern=str(args.ar_sparse_attn_pattern),
        scip_time_limit=float(args.scip_warm_limit),
        unused_gurobi_time_limit=float(args.gurobi_warm_limit),
        scip_warm_start_seed_shift=int(args.scip_warm_start_seed_shift),
        gurobi_warm_start_seed=int(args.gurobi_warm_start_seed),
        gurobi_license=gurobi_lic,
        scip_extra_params=scip_base,
        phase1_ce_target_prob=float(args.phase1_ce_target_prob),
        phase1_ce_max_steps=int(args.phase1_ce_max_steps),
        phase1_ce_plateau_patience=int(args.phase1_ce_plateau_patience),
        phase1_ce_plateau_min_delta=float(args.phase1_ce_plateau_min_delta),
        phase2_max_steps=int(args.phase2_max_steps),
        phase2_w_soft=float(args.phase2_w_soft),
        phase2_tau=float(args.phase2_tau),
        phase2_lambda=float(args.phase2_lambda),
        phase2_q_weight=float(args.phase2_q_weight),
        phase2_plateau_patience=int(args.phase2_plateau_patience),
        phase2_plateau_min_delta=float(args.phase2_plateau_min_delta),
        phase2_time_limit_s=(
            None if bool(args.no_phase2_time_limit) else float(args.phase2_time_limit)
        ),
        gumbel_use_ste=bool(args.gumbel_ste),
        phase2_lr=(None if getattr(args, "phase2_lr", None) is None else float(args.phase2_lr)),
        reproducible=True,
        train_show_progress=not bool(args.no_progress),
        record_incumbent_snapshots=True,
        record_phase2_soft_x_trace=(
            getattr(args, "plot_trajectories", None) is not None
            and str(getattr(args, "plot_trajectories", "")).strip() != ""
        ),
        phase2_soft_x_trace_stride=max(1, int(getattr(args, "tsne_phase2_stride", 5))),
        warm_solver=tag,
        warm_threads=int(getattr(args, "warm_threads", 1)),
    )


def run_one_instance(
    *,
    base_seed: int,
    exp_id: int,
    n: int,
    q_kind: str,
    args: argparse.Namespace,
    scip_base: Optional[Dict[str, Any]],
    gurobi_lic: Optional[str],
) -> Dict[str, Any]:
    inst = sample_qkp_instance(n, exp_id, base_seed=base_seed, q_kind=q_kind)
    print(
        f"\n{'=' * 72}\n"
        f"continuous relax GD  N={n}  base_seed={base_seed}  exp_id={exp_id}\n"
        f"{_instance_identity_message(base_seed=base_seed, n=n, exp_id=exp_id, q_kind=q_kind, Q=inst.Q, w=inst.w, W=int(inst.W), c=inst.c)}",
        flush=True,
    )
    _ts = _derive_torch_run_seed(base_seed, n, exp_id, q_kind)
    _apply_reproducibility(_ts, device=str(args.device))

    cfg = _build_adip_cfg(
        args=args, scip_base=scip_base, gurobi_lic=gurobi_lic, base_seed=base_seed
    )
    t0 = time.perf_counter()
    outcome = run_adip_pipeline(inst, cfg=cfg, t_instance_start=t0)
    pipeline_wall = time.perf_counter() - t0

    if str(outcome.status) != "ok":
        return {
            "base_seed": base_seed,
            "status": f"pipeline_{outcome.status}",
            "pipeline_wall_s": pipeline_wall,
        }

    snaps = list(outcome.incumbent_snapshots or [])
    snaps_distinct = distinct_incumbent_snapshots_by_x(snaps)
    last_k = last_k_distinct_incumbent_snapshots(snaps, int(args.last_k_scip))
    n_dup = len(snaps) - len(snaps_distinct)
    _meta = outcome.incumbent_trace_meta
    _meta_s = ""
    if _meta is not None:
        _meta_s = (
            f"  |  SCIP: {_meta.n_bestsol_callbacks} BESTSOLFOUND callbacks, "
            f"n_best_found={_meta.n_bestsol_found}, obj_trace={_meta.n_obj_trace_points}"
        )
    print(
        f"  SCIP incumbent trace: {len(snaps)} raw, {len(snaps_distinct)} distinct x, "
        f"GD from last {len(last_k)} distinct (K={int(args.last_k_scip)})"
        + (f"  [{n_dup} duplicate x dropped]" if n_dup else "")
        + _meta_s,
        flush=True,
    )

    adip_x, adip_tag = adip_binary_start_from_pipeline(
        inst,
        rounded_x=outcome.rounded_x,
        rounded_feasible=outcome.rounded_feasible,
    )

    starts: List[tuple[str, np.ndarray]] = []
    for i, snap in enumerate(last_k):
        starts.append((f"scip_incumbent_{i}_t{snap.wall_time_s:.0f}s", snap.x))
    if adip_x is not None:
        starts.append((adip_tag, adip_x))
    if outcome.warm_x is not None:
        wx = np.asarray(outcome.warm_x, dtype=np.int64).ravel()
        if not any(np.array_equal(wx, np.asarray(s.x).ravel()) for s in last_k):
            starts.append(("scip_final_warm", wx))

    gd_results: List[ContinuousRelaxGDResult] = []
    print(
        f"\n  Knapsack-boundary relax ascent ({len(starts)} starts, lr={args.gd_lr}, "
        f"max_iters={args.gd_max_iters}, W={int(inst.W)})",
        flush=True,
    )
    for label, x_bin in starts:
        res = knapsack_boundary_gradient_ascent(
            inst,
            x_bin,
            label=label,
            max_iters=int(args.gd_max_iters),
            lr=float(args.gd_lr),
            tol_knapsack=float(args.gd_tol_knapsack),
            tol_grad=float(args.gd_tol_grad),
        )
        gd_results.append(res)
        print(
            f"    {label:28s}  f0={res.f0:10.4g} → f*={res.f_star:10.4g}  "
            f"Δf={res.f_star - res.f0:+.4g}  w: {res.w0:.4g}→{res.w_star:.4g}  "
            f"iters={res.n_iters}  stop={res.stop_reason}  "
            f"binary_xTQx={_qkp_obj(inst, x_bin):.4g}",
            flush=True,
        )

    summary = summarize_gd_results(gd_results)
    print(f"\n  GD summary: {summary}", flush=True)

    plot_meta: Optional[Dict[str, Any]] = None
    _pt = getattr(args, "plot_trajectories", None)
    if _pt is not None and str(_pt).strip() != "":
        plot_meta = plot_qkp_trajectories_tsne(
            inst,
            outcome,
            Path(str(_pt)),
            base_seed=int(base_seed),
            gd_lr=float(args.gd_lr),
            gd_max_iters=int(args.gd_max_iters),
            gd_tol_knapsack=float(args.gd_tol_knapsack),
            gd_tol_grad=float(args.gd_tol_grad),
            tsne_perplexity=float(getattr(args, "tsne_perplexity", 30.0)),
            tsne_seed=int(getattr(args, "tsne_seed", 0)),
            max_tsne_rows=int(getattr(args, "tsne_max_rows", 500)),
            max_phase2_plot=int(getattr(args, "tsne_max_phase2", 220)),
            max_relax_rays=int(getattr(args, "tsne_max_rays", 96)),
            endpoint_l2_threshold=float(getattr(args, "endpoint_l2_threshold", 1e-3)),
        )
        print(f"  t-SNE trajectory plot: {plot_meta.get('out_path')}", flush=True)
        _basin = plot_meta.get("basin_analysis") if isinstance(plot_meta, dict) else None
        if isinstance(_basin, dict):
            print(
                "  basin_analysis: "
                f"traj_pts={_basin.get('trajectory_points')}  "
                f"rays={_basin.get('n_relax_rays')}  "
                f"distinct_endpoints={_basin.get('distinct_endpoints')}  "
                f"(ℓ₂>{_basin.get('endpoint_l2_threshold')})  "
                f"incumbent_end={_basin.get('incumbent_endpoint_obj')}  "
                f"best_end={_basin.get('best_endpoint_obj')}",
                flush=True,
            )

    scip_pts = scip_anytime_from_snapshots(
        snaps,
        t_end=float(pipeline_wall),
        final_obj=outcome.warm_xTQx,
        final_t=outcome.warm_wall_s,
        obj_trace=(
            list(outcome.incumbent_trace_meta.obj_trace)
            if outcome.incumbent_trace_meta is not None
            else None
        ),
    )
    adip_pts = adip_anytime_from_traces(
        inst,
        scip_pts=scip_pts,
        warm_wall_s=float(outcome.warm_wall_s or 0.0),
        phase1_wall_s=float(outcome.phase1_wall_s or 0.0),
        p2_xs=list(outcome.phase2_soft_x_trace or []),
        p2_times_s=list(outcome.phase2_soft_x_trace_times_s or []),
        rounded_final=outcome.rounded_xTQx,
        rounded_feasible=outcome.rounded_feasible,
        pipeline_wall_s=float(pipeline_wall),
    )
    obj_time = {
        "scip_warm_pts": [{"t": t, "obj": v} for t, v in scip_pts],
        "adip_pts": [{"t": t, "obj": v} for t, v in adip_pts],
        "warm_wall_s": outcome.warm_wall_s,
        "phase1_wall_s": outcome.phase1_wall_s,
        "phase2_wall_s": outcome.phase2_wall_s,
        "pipeline_wall_s": pipeline_wall,
        "warm_xTQx": outcome.warm_xTQx,
        "rounded_xTQx": outcome.rounded_xTQx,
        "rounded_feasible": outcome.rounded_feasible,
    }

    return {
        "base_seed": base_seed,
        "exp_id": exp_id,
        "n": n,
        "q_kind": q_kind,
        "status": "ok",
        "pipeline_wall_s": pipeline_wall,
        "warm_gap": outcome.warm_gap,
        "warm_xTQx": outcome.warm_xTQx,
        "rounded_xTQx": outcome.rounded_xTQx,
        "rounded_feasible": outcome.rounded_feasible,
        "n_scip_incumbent_snapshots_raw": len(snaps),
        "n_scip_incumbent_snapshots_distinct_x": len(snaps_distinct),
        "n_scip_duplicate_x_in_trace": n_dup,
        "scip_incumbent_trace_meta": (
            None
            if _meta is None
            else {
                "n_bestsol_callbacks": _meta.n_bestsol_callbacks,
                "n_bestsol_found": _meta.n_bestsol_found,
                "n_obj_trace_points": _meta.n_obj_trace_points,
                "n_stored_sols": _meta.n_stored_sols,
                "n_snapshots_after_harvest": _meta.n_snapshots_after_harvest,
                "obj_trace": [
                    {"t": float(t), "obj": float(v)} for t, v in (_meta.obj_trace or ())
                ],
            }
        ),
        "n_starts_gd": len(starts),
        "scip_incumbents": [
            {
                "wall_time_s": s.wall_time_s,
                "mip_obj": s.mip_obj,
                "xTQx": s.xTQx,
                "x": s.x.tolist(),
            }
            for s in last_k
        ],
        "adip_start_tag": adip_tag,
        "adip_start_xTQx": (
            float(_qkp_obj(inst, adip_x)) if adip_x is not None else None
        ),
        "gd_summary": summary,
        "gd_runs": [r.to_json_dict() for r in gd_results],
        "f_continuous_at_binary_starts": {
            r.label: float(continuous_qkp_value(inst, r.x0)) for r in gd_results
        },
        "tsne_plot": plot_meta,
        "obj_time": obj_time,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-seed", type=int, default=None, help="Single instance seed.")
    p.add_argument("--base-seed-start", type=int, default=40)
    p.add_argument("--num-runs", type=int, default=1)
    p.add_argument("--exp-id", type=int, default=0)
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--q-kind", type=str, default="indefinite")
    p.add_argument(
        "--warm-solver",
        choices=WARM_SOLVER_CHOICES,
        default="scip",
        help=(
            "Warm-start solver. SCIP uses native MIQP; "
            "highs/cbc/cpsat use Fortet linearization."
        ),
    )
    p.add_argument("--last-k-scip", type=int, default=5)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "qkp_benchmark_out"),
    )
    p.add_argument(
        "--scip-warm-limit",
        type=float,
        default=500.0,
        help="Warm-start time limit in seconds (SCIP and OSS solvers).",
    )
    p.add_argument("--warm-threads", type=int, default=1)
    p.add_argument("--scip-warm-start-seed-shift", type=int, default=424267)
    p.add_argument("--scip-param", action="append", default=None)
    p.add_argument("--gurobi-warm-limit", type=float, default=500.0)
    p.add_argument("--gurobi-warm-start-seed", type=int, default=424267)
    p.add_argument("--gurobi-lic", type=str, default=None)
    p.add_argument("--phase2-time-limit", type=float, default=500.0)
    p.add_argument("--no-phase2-time-limit", action="store_true")
    # ADIP recipe (match gap sweep)
    p.add_argument("--ar-lr", type=float, default=1e-3)
    p.add_argument(
        "--phase2-lr",
        type=float,
        default=None,
        help="Adam LR for phase-2 only (default: same as --ar-lr / phase-1 CE).",
    )
    p.add_argument("--ar-d-model", type=int, default=64)
    p.add_argument("--ar-nhead", type=int, default=4)
    p.add_argument("--ar-nlayers", type=int, default=2)
    p.add_argument("--ar-sparse-attn-prev-fraction", type=float, default=0.1)
    p.add_argument("--ar-sparse-attn-pattern", choices=("band", "topk"), default="band")
    p.add_argument("--phase1-ce-target-prob", type=float, default=0.999)
    p.add_argument("--phase1-ce-max-steps", type=int, default=150)
    p.add_argument("--phase1-ce-plateau-patience", type=int, default=12)
    p.add_argument("--phase1-ce-plateau-min-delta", type=float, default=1e-5)
    p.add_argument("--phase2-max-steps", type=int, default=2000)
    p.add_argument("--phase2-w-soft", type=float, default=1.0)
    p.add_argument("--phase2-tau", type=float, default=0.5)
    p.add_argument("--phase2-lambda", type=float, default=100.0)
    p.add_argument("--phase2-q-weight", type=float, default=1.0)
    p.add_argument("--phase2-plateau-patience", type=int, default=45)
    p.add_argument("--phase2-plateau-min-delta", type=float, default=1e-5)
    p.add_argument("--gumbel-ste", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--plot-trajectories",
        type=str,
        default=None,
        metavar="PNG",
        help="If set, record phase-2 x̃ trace and save t-SNE 3D plot to this path.",
    )
    p.add_argument(
        "--tsne-phase2-stride",
        type=int,
        default=5,
        help="Store every k-th phase-2 step when plotting (smaller = heavier).",
    )
    p.add_argument("--tsne-perplexity", type=float, default=30.0)
    p.add_argument("--tsne-seed", type=int, default=0)
    p.add_argument("--tsne-max-rows", type=int, default=500)
    p.add_argument("--tsne-max-phase2", type=int, default=220)
    p.add_argument("--tsne-max-rays", type=int, default=96)
    p.add_argument(
        "--endpoint-l2-threshold",
        type=float,
        default=1e-3,
        help="ℓ₂ threshold for counting distinct GD endpoints in basin_analysis.",
    )
    # Relaxed ascent knobs (stop when ``wᵀx`` reaches ``W``)
    p.add_argument("--gd-max-iters", type=int, default=3000)
    p.add_argument("--gd-lr", type=float, default=0.05)
    p.add_argument("--gd-tol-knapsack", type=float, default=1e-6)
    p.add_argument("--gd-tol-grad", type=float, default=1e-8)
    args = p.parse_args(list(argv) if argv is not None else None)

    if args.base_seed is not None:
        seeds = [int(args.base_seed)]
    else:
        seeds = [int(args.base_seed_start) + i for i in range(int(args.num_runs))]

    if bool(args.smoke_test):
        args.n = min(int(args.n), 32)
        seeds = seeds[:2]
        args.scip_warm_limit = min(float(args.scip_warm_limit), 10.0)
        args.gurobi_warm_limit = min(float(args.gurobi_warm_limit), 10.0)
        args.phase1_ce_max_steps = min(int(args.phase1_ce_max_steps), 20)
        if not bool(args.no_phase2_time_limit):
            args.phase2_time_limit = min(float(args.phase2_time_limit), 15.0)
        args.gd_max_iters = min(int(args.gd_max_iters), 200)

    for k, v in (
        ("OMP_NUM_THREADS", "1"),
        ("MKL_NUM_THREADS", "1"),
        ("OPENBLAS_NUM_THREADS", "1"),
    ):
        os.environ.setdefault(k, v)

    try:
        scip_items = list(args.scip_param) if args.scip_param else ["timing/clocktype=wall"]
        scip_base = _parse_scip_param_kv_list(scip_items)
    except ValueError as err:
        print(str(err), file=sys.stderr)
        return 2

    gurobi_lic = args.gurobi_lic

    out_dir = Path(str(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"cont_relax_{args.warm_solver}_n{args.n}_k{args.last_k_scip}"
    if len(seeds) == 1:
        out_path = out_dir / f"{tag}_seed{seeds[0]}.json"
    else:
        out_path = out_dir / f"{tag}_s{seeds[0]}_n{len(seeds)}.json"

    rows: List[Dict[str, Any]] = []
    for base_seed in seeds:
        rows.append(
            run_one_instance(
                base_seed=base_seed,
                exp_id=int(args.exp_id),
                n=int(args.n),
                q_kind=str(args.q_kind),
                args=args,
                scip_base=scip_base,
                gurobi_lic=gurobi_lic,
            )
        )

    out_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
