"""Matched-time SCIP vs SCIP+ADIP objective curves (CE time excluded)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from adip.obj_time_plot import (
    adip_anytime_exclude_ce,
    plot_obj_vs_time,
)
from adip.pipeline import (
    _apply_reproducibility,
    _derive_torch_run_seed,
    _merge_scip_params_with_seed_shift,
    _parse_scip_param_kv_list,
    sample_qkp_instance,
    solve_max_qkp_scip_with_incumbent_x_trace,
)


def _remap_adip_pts_exclude_ce(
    adip_pts: Sequence[Dict[str, Any]],
    *,
    warm_wall_s: float,
    phase1_wall_s: float,
    warm_budget_s: float,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Split remapped SCIP-warm vs phase-2 points from a saved ``obj_time.adip_pts`` list."""
    t_hand = float(warm_budget_s)
    t_p2_wall0 = float(warm_wall_s) + float(phase1_wall_s)
    scip_pts: List[Tuple[float, float]] = []
    p2_pts: List[Tuple[float, float]] = []
    for row in adip_pts:
        t = float(row["t"])
        v = float(row["obj"])
        if t <= float(warm_wall_s) + 1e-6:
            scip_pts.append((min(t, t_hand), v))
        elif t >= t_p2_wall0 - 1e-6:
            p2_pts.append((t_hand + (t - t_p2_wall0), v))
    return scip_pts, p2_pts


def _parse_y_break(s: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    if s is None or str(s).strip() == "":
        return None
    parts = [float(x) for x in str(s).split(",")]
    if len(parts) != 4:
        raise ValueError("--y-break expects y_lo_min,y_lo_max,y_hi_min,y_hi_max")
    return (parts[0], parts[1], parts[2], parts[3])


def _parse_xlim(s: Optional[str]) -> Optional[Tuple[float, float]]:
    if s is None or str(s).strip() == "":
        return None
    parts = [float(x) for x in str(s).split(",")]
    if len(parts) != 2:
        raise ValueError("--xlim expects t_min,t_max")
    return (parts[0], parts[1])


def _plot_from_saved(
    data: Dict[str, Any],
    *,
    out_png: Path,
    invlog_remaining_x: bool,
    y_break: Optional[Tuple[float, float, float, float]],
    xlim: Optional[Tuple[float, float]],
) -> Dict[str, Any]:
    n = int(data["n"])
    base_seed = int(data["base_seed"])
    t_end = float(data["scip_time_limit"])
    warm_budget = float(data["warm_budget_s"])
    scip_pts = [(float(p["t"]), float(p["obj"])) for p in data["scip_pts"]]
    adip_pts = [(float(p["t"]), float(p["obj"])) for p in data["adip_pts_remapped"]]
    p2_only = [(float(p["t"]), float(p["obj"])) for p in (data.get("phase2_only_pts") or [])]
    return plot_obj_vs_time(
        scip_pts=scip_pts,
        adip_pts=adip_pts,
        t_end=t_end,
        out_path=out_png,
        title=f"Anytime objective  (n={n})",
        t_handoff=warm_budget,
        phase2_only_pts=p2_only or None,
        xlabel="Wall time (s)",
        invlog_remaining_x=bool(invlog_remaining_x),
        xlim=xlim,
        y_break=y_break,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--adip-json",
        type=str,
        default=None,
        help="cont_relax JSON with obj_time (required unless --from-json).",
    )
    p.add_argument(
        "--from-json",
        type=str,
        default=None,
        help="Replot from a previously written obj_vs_time JSON (no SCIP re-solve).",
    )
    p.add_argument(
        "--scip-time-limit",
        type=float,
        default=1000.0,
        help="Matched SCIP-only budget (SCIP warm + phase-2; CE excluded).",
    )
    p.add_argument("--warm-budget", type=float, default=500.0)
    p.add_argument("--out-png", type=str, default="results/obj_vs_time_seed102.png")
    p.add_argument("--out-json", type=str, default="results/obj_vs_time_seed102.json")
    p.add_argument("--scip-warm-start-seed-shift", type=int, default=424267)
    p.add_argument(
        "--y-break",
        type=str,
        default="-20,60,1500,1800",
        help="Broken y-axis as y_lo_min,y_lo_max,y_hi_min,y_hi_max (empty to disable).",
    )
    p.add_argument(
        "--xlim",
        type=str,
        default="0,1000",
        help="X window t_min,t_max (empty for full [0, scip-time-limit]).",
    )
    p.add_argument(
        "--invlog-remaining-x",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Expand late times via -log(T - t) (default on for the paper figure).",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    for k, v in (
        ("OMP_NUM_THREADS", "1"),
        ("MKL_NUM_THREADS", "1"),
        ("OPENBLAS_NUM_THREADS", "1"),
    ):
        os.environ.setdefault(k, v)

    y_break = _parse_y_break(args.y_break)
    xlim = _parse_xlim(args.xlim)
    out_png = Path(str(args.out_png))

    if args.from_json:
        data = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
        plot_meta = _plot_from_saved(
            data,
            out_png=out_png,
            invlog_remaining_x=bool(args.invlog_remaining_x),
            y_break=y_break,
            xlim=xlim,
        )
        print(f"Replotted {out_png} from {args.from_json}", flush=True)
        print(f"  plot={plot_meta}", flush=True)
        return 0

    if not args.adip_json:
        print("Provide --adip-json or --from-json", file=sys.stderr)
        return 2

    rows = json.loads(Path(args.adip_json).read_text(encoding="utf-8"))
    rec = rows[0] if isinstance(rows, list) else rows
    ot = rec.get("obj_time") or {}
    if not ot:
        print("obj_time missing in ADIP JSON", file=sys.stderr)
        return 2

    n = int(rec["n"])
    base_seed = int(rec["base_seed"])
    exp_id = int(rec.get("exp_id", 0))
    q_kind = str(rec.get("q_kind", "indefinite"))
    inst = sample_qkp_instance(n, exp_id, base_seed=base_seed, q_kind=q_kind)
    _apply_reproducibility(
        _derive_torch_run_seed(base_seed, n, exp_id, q_kind),
        device="cpu",
    )

    warm_budget = float(args.warm_budget)
    scip_from_adip, p2_from_adip = _remap_adip_pts_exclude_ce(
        ot.get("adip_pts") or [],
        warm_wall_s=float(ot.get("warm_wall_s") or warm_budget),
        phase1_wall_s=float(ot.get("phase1_wall_s") or 0.0),
        warm_budget_s=warm_budget,
    )
    for row in ot.get("scip_warm_pts") or []:
        scip_from_adip.append((min(float(row["t"]), warm_budget), float(row["obj"])))
    meta_warm = rec.get("scip_incumbent_trace_meta") or {}
    for row in meta_warm.get("obj_trace") or []:
        scip_from_adip.append((min(float(row["t"]), warm_budget), float(row["obj"])))
    for snap in rec.get("scip_incumbents") or []:
        scip_from_adip.append(
            (min(float(snap["wall_time_s"]), warm_budget), float(snap.get("xTQx", snap.get("mip_obj", 0.0))))
        )

    scip_params = _parse_scip_param_kv_list(["timing/clocktype=wall"])
    scip_params = _merge_scip_params_with_seed_shift(
        scip_params, int(args.scip_warm_start_seed_shift)
    )
    print(
        f"SCIP matched-time solve  n={n} seed={base_seed}  limit={args.scip_time_limit:g}s",
        flush=True,
    )
    res, snaps, meta = solve_max_qkp_scip_with_incumbent_x_trace(
        inst,
        time_limit_sec=float(args.scip_time_limit),
        quiet=True,
        scip_params=scip_params,
    )
    print(
        f"  SCIP status={res.status!r} obj={res.obj_value} wall={res.wall_time_s:.3f}s  "
        f"snapshots={len(snaps)} callbacks={meta.n_bestsol_callbacks} "
        f"obj_trace={meta.n_obj_trace_points}",
        flush=True,
    )

    scip_pts: List[Tuple[float, float]] = list(meta.obj_trace)
    scip_pts.extend((float(s.wall_time_s), float(s.xTQx)) for s in snaps)
    if res.obj_value is not None:
        scip_pts.append((float(res.wall_time_s), float(res.obj_value)))

    t_end = float(args.scip_time_limit)
    adip_pts = adip_anytime_exclude_ce(
        scip_pts=scip_from_adip,
        warm_budget_s=warm_budget,
        phase2_pts=p2_from_adip,
    )
    p2_only = list(p2_from_adip)

    plot_meta = plot_obj_vs_time(
        scip_pts=scip_pts,
        adip_pts=adip_pts,
        t_end=t_end,
        out_path=out_png,
        title=f"Anytime objective  (n={n})",
        t_handoff=warm_budget,
        phase2_only_pts=p2_only,
        xlabel="Wall time (s)",
        invlog_remaining_x=bool(args.invlog_remaining_x),
        xlim=xlim,
        y_break=y_break,
    )
    out = {
        "n": n,
        "base_seed": base_seed,
        "scip_time_limit": float(args.scip_time_limit),
        "warm_budget_s": warm_budget,
        "ce_excluded": True,
        "scip_status": res.status,
        "scip_obj": res.obj_value,
        "scip_wall_s": res.wall_time_s,
        "scip_gap": res.gap,
        "scip_n_snapshots": len(snaps),
        "scip_obj_trace": [{"t": t, "obj": v} for t, v in meta.obj_trace],
        "scip_pts": [{"t": t, "obj": v} for t, v in scip_pts],
        "adip_pts_remapped": [{"t": t, "obj": v} for t, v in adip_pts],
        "phase2_only_pts": [{"t": t, "obj": v} for t, v in p2_only],
        "adip_pipeline_wall_s": ot.get("pipeline_wall_s"),
        "adip_phase1_wall_s": ot.get("phase1_wall_s"),
        "adip_phase2_wall_s": ot.get("phase2_wall_s"),
        "adip_warm_xTQx": ot.get("warm_xTQx"),
        "adip_rounded_xTQx": ot.get("rounded_xTQx"),
        "plot": plot_meta,
    }
    out_json = Path(str(args.out_json))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_png} and {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
