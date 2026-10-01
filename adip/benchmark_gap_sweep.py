"""Sweep base_seed and plot warm-start MIP gap vs ADIP improvement."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from adip.gap_scatter import (
    plot_scatter,
    points_from_improvement_pct,
    reported_obj_and_improvement_pct,
)
from adip.pipeline import (
    AdipPipelineConfig,
    AdipPipelineResult,
    _apply_reproducibility,
    _derive_torch_run_seed,
    _parse_scip_param_kv_list,
    _instance_identity_message,
    run_adip_pipeline,
    sample_qkp_instance,
)
from adip.warm_solvers import WARM_SOLVER_CHOICES, normalize_warm_solver

WarmSolver = str


@dataclass(frozen=True)
class GapImprovementTrialResult:
    base_seed: int
    exp_id: int
    n: int
    q_kind: str
    warm_solver: str
    status: str
    warm_gap: Optional[float] = None
    warm_mip_obj: Optional[float] = None
    warm_xTQx: Optional[float] = None
    warm_wall_s: Optional[float] = None
    rounded_xTQx: Optional[float] = None
    rounded_feasible: Optional[bool] = None
    improvement_pct: Optional[float] = None
    wall_total_s: Optional[float] = None

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _trial_from_pipeline(
    *,
    base_seed: int,
    exp_id: int,
    n: int,
    q_kind: str,
    warm_solver: str,
    outcome: AdipPipelineResult,
    wall_total_s: float,
) -> GapImprovementTrialResult:
    improvement: Optional[float] = None
    status = str(outcome.status)
    warm_x = outcome.warm_xTQx
    rounded_q = outcome.rounded_xTQx
    rounded_feas = outcome.rounded_feasible

    if status == "ok" and warm_x is not None:
        scip_f = float(warm_x)
        if rounded_q is not None and rounded_feas:
            adip_f = float(rounded_q)
            if adip_f < scip_f:
                rounded_q, improvement = scip_f, 0.0
            elif abs(scip_f) <= 1e-15:
                # Relative % undefined from a zero warm incumbent; keep absolute ADIP obj.
                rounded_q, improvement = adip_f, None
            else:
                rounded_q, improvement = reported_obj_and_improvement_pct(
                    adip=adip_f, scip=scip_f
                )
        else:
            improvement = 0.0

    return GapImprovementTrialResult(
        base_seed=int(base_seed),
        exp_id=int(exp_id),
        n=int(n),
        q_kind=str(q_kind),
        warm_solver=str(warm_solver),
        status=status,
        warm_gap=outcome.warm_gap,
        warm_mip_obj=outcome.warm_mip_obj,
        warm_xTQx=warm_x,
        warm_wall_s=outcome.warm_wall_s,
        rounded_xTQx=rounded_q,
        rounded_feasible=rounded_feas,
        improvement_pct=improvement,
        wall_total_s=float(wall_total_s),
    )


def _build_pipeline_cfg(
    *,
    warm_solver: str,
    args: argparse.Namespace,
    scip_base: Optional[Dict[str, Any]],
    gurobi_lic: Optional[str],
) -> AdipPipelineConfig:
    use_gurobi = False
    tag = normalize_warm_solver(warm_solver)
    suffix = "gumbel_ste" if bool(args.gumbel_ste) else "gumbel"
    pipeline = f"adip_{tag}_{suffix}"
    return AdipPipelineConfig(
        pipeline_name=pipeline,
        use_gurobi_warm_start=use_gurobi,
        device=str(args.device),
        base_seed=0,  # filled per trial
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
        warm_solver=tag,
        warm_threads=int(getattr(args, "warm_threads", 1)),
    )


def _load_completed_seeds(jsonl_path: Path) -> set[int]:
    if not jsonl_path.is_file():
        return set()
    done: set[int] = set()
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            done.add(int(json.loads(line)["base_seed"]))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return done


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--warm-solver",
        choices=WARM_SOLVER_CHOICES,
        default="scip",
        help=(
            "Warm-start solver. SCIP uses the native (possibly nonconvex) MIQP; "
            "highs/cbc/cpsat use a Fortet linearization of x_i x_j (indefinite Q)."
        ),
    )
    p.add_argument("--base-seed-start", type=int, default=7)
    p.add_argument("--num-runs", type=int, default=100)
    p.add_argument("--exp-id", type=int, default=0)
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--q-kind", type=str, default="indefinite")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "qkp_benchmark_out"),
    )
    p.add_argument("--no-progress", action="store_true")
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
    p.add_argument("--plot-title", type=str, default="")
    p.add_argument("--no-show-plot", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)

    if int(args.num_runs) < 1:
        print("--num-runs must be >= 1", file=sys.stderr)
        return 2

    for k, v in (
        ("OMP_NUM_THREADS", "1"),
        ("MKL_NUM_THREADS", "1"),
        ("OPENBLAS_NUM_THREADS", "1"),
        ("VECLIB_MAXIMUM_THREADS", "1"),
        ("NUMEXPR_NUM_THREADS", "1"),
    ):
        os.environ.setdefault(k, v)

    warm_solver = normalize_warm_solver(args.warm_solver)
    n = int(args.n)
    exp_id = int(args.exp_id)
    q_kind = str(args.q_kind)
    seed_start = int(args.base_seed_start)
    num_runs = int(args.num_runs)
    if bool(args.smoke_test):
        n = min(n, 32)
        num_runs = min(num_runs, 2)

    try:
        scip_items = list(args.scip_param) if args.scip_param else ["timing/clocktype=wall"]
        scip_base = _parse_scip_param_kv_list(scip_items)
    except ValueError as err:
        print(str(err), file=sys.stderr)
        return 2

    gurobi_lic = args.gurobi_lic

    cfg_template = _build_pipeline_cfg(
        warm_solver=warm_solver,
        args=args,
        scip_base=scip_base,
        gurobi_lic=gurobi_lic,
    )

    out_dir = Path(str(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{warm_solver}_n{n}_s{seed_start}_k{num_runs}"
    jsonl_path = out_dir / f"gap_improvement_sweep_{tag}.jsonl"
    summary_path = out_dir / f"gap_improvement_sweep_{tag}.json"
    plot_path = out_dir / f"gap_vs_improvement_{tag}.png"

    done_seeds = _load_completed_seeds(jsonl_path) if bool(args.resume) else set()
    results: List[GapImprovementTrialResult] = []
    if bool(args.resume) and jsonl_path.is_file():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    results.append(GapImprovementTrialResult(**json.loads(line)))
                except (json.JSONDecodeError, TypeError):
                    pass

    print(
        f"Gap/improvement sweep → run_adip_pipeline  "
        f"warm_solver={warm_solver!r}  n={n}  exp_id={exp_id}  "
        f"base_seed ∈ [{seed_start}, {seed_start + num_runs - 1}]  log={jsonl_path}",
        flush=True,
    )

    for run_idx in range(num_runs):
        base_seed = seed_start + run_idx
        if base_seed in done_seeds:
            print(f"  [{run_idx + 1}/{num_runs}] base_seed={base_seed}  (resume: skip)", flush=True)
            continue

        inst = sample_qkp_instance(n, exp_id, base_seed=base_seed, q_kind=q_kind)
        print(
            f"\n{'=' * 72}\n"
            f"QKP soft-AR  N={n}  exp={exp_id}  W={int(inst.W)}  q_kind={q_kind!r}  "
            f"base_seed={base_seed}\n"
            f"{_instance_identity_message(base_seed=base_seed, n=n, exp_id=exp_id, q_kind=q_kind, Q=inst.Q, w=inst.w, W=int(inst.W), c=inst.c)}",
            flush=True,
        )

        _ts = _derive_torch_run_seed(base_seed, n, exp_id, q_kind)
        _apply_reproducibility(_ts, device=str(args.device))
        print(f"  reproducibility:  torch/python/numpy seed={_ts}", flush=True)

        t0 = time.perf_counter()
        cfg = replace(cfg_template, base_seed=int(base_seed))
        try:
            outcome = run_adip_pipeline(
                inst,
                cfg=cfg,
                t_instance_start=t0,
            )
            trial = _trial_from_pipeline(
                base_seed=base_seed,
                exp_id=exp_id,
                n=n,
                q_kind=q_kind,
                warm_solver=warm_solver,
                outcome=outcome,
                wall_total_s=time.perf_counter() - t0,
            )
        except Exception as err:
            trial = GapImprovementTrialResult(
                base_seed=base_seed,
                exp_id=exp_id,
                n=n,
                q_kind=q_kind,
                warm_solver=warm_solver,
                status=f"error:{type(err).__name__}",
                wall_total_s=time.perf_counter() - t0,
            )
            print(f"  ERROR base_seed={base_seed}: {err}", flush=True)

        results.append(trial)
        _append_jsonl(jsonl_path, trial.to_json_dict())
        if trial.improvement_pct is not None and trial.warm_gap is not None:
            print(
                f"  [sweep summary] gap={trial.warm_gap:g}  warm_xTQx={trial.warm_xTQx:g}  "
                f"improvement={trial.improvement_pct:+.3f}%  status={trial.status!r}",
                flush=True,
            )

    summary_path.write_text(
        json.dumps([r.to_json_dict() for r in results], indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {jsonl_path}\nWrote {summary_path}")

    plot_rows = [r for r in results if r.warm_gap is not None and r.improvement_pct is not None]
    if not plot_rows:
        print("No plottable points.", file=sys.stderr)
        return 1

    pts = points_from_improvement_pct(
        [float(r.warm_gap) for r in plot_rows],
        [float(r.improvement_pct) for r in plot_rows],
        [f"s={r.base_seed}" for r in plot_rows],
    )
    ws = warm_solver.upper()
    title = (
        args.plot_title.strip()
        or f"% improvement (rounded x̃ vs warm {ws}) vs {ws} MIP gap  (N={n})"
    )
    try:
        plot_scatter(pts, out_path=str(plot_path), title=title, show=not bool(args.no_show_plot))
    except Exception as err:  # noqa: BLE001
        print(f"Plot failed (results already written): {err}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
