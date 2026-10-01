"""Open-source MIP warm-starts for QKP (HiGHS, CBC, CP-SAT).

These solvers do not accept indefinite quadratic objectives, so the model is the
Fortet linearization in :mod:`adip.qkp_linearize`. SCIP stays on the native
quadratic epigraph in :mod:`adip.pipeline`.
"""

from __future__ import annotations

import time
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from adip.qkp_linearize import (
    FortetQKP,
    fortet_col_primal,
    fortet_qkp,
    relative_mip_gap,
)

WARM_SOLVER_CHOICES: Tuple[str, ...] = ("scip", "highs", "cbc", "cpsat", "zeros")
OSS_WARM_SOLVERS: Tuple[str, ...] = ("highs", "cbc", "cpsat")

WARM_SOLVER_TAGS = {
    "scip": "SCIP",
    "highs": "HiGHS",
    "cbc": "CBC",
    "cpsat": "CP-SAT",
    "gurobi": "Gurobi",
    "zeros": "zeros",
}


def normalize_warm_solver(name: str) -> str:
    key = str(name).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "scip": "scip",
        "highs": "highs",
        "highspy": "highs",
        "cbc": "cbc",
        "coincbc": "cbc",
        "cpsat": "cpsat",
        "sat": "cpsat",
        "ortools": "cpsat",
        "gurobi": "gurobi",
        "zeros": "zeros",
        "zero": "zeros",
        "null": "zeros",
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown warm solver {name!r}. Choose from {list(WARM_SOLVER_CHOICES)}."
        )
    return aliases[key]


def _qkp_obj(inst: Any, x: np.ndarray) -> float:
    from adip.pipeline import _qkp_obj as _obj

    return float(_obj(inst, x))


def _trace_meta(n_cb: int, n_snaps: int) -> Any:
    from adip.pipeline import ScipIncumbentTraceMeta

    return ScipIncumbentTraceMeta(
        n_bestsol_callbacks=int(n_cb),
        n_bestsol_found=int(n_cb),
        n_stored_sols=int(n_snaps),
        n_obj_trace_points=int(n_cb),
        n_snapshots_after_harvest=int(n_snaps),
    )


def _empty_result(
    *,
    status: str,
    wall: float,
    obj: Optional[float] = None,
    gap: Optional[float] = None,
    n_nodes: Optional[int] = None,
    n_sols: int = 0,
    x_opt: Optional[np.ndarray] = None,
) -> Any:
    from adip.pipeline import QKPScipSolveResult

    return QKPScipSolveResult(
        status=str(status),
        obj_value=obj,
        wall_time_s=float(wall),
        gap=gap,
        n_nodes=n_nodes,
        n_sols=int(n_sols),
        x_opt=x_opt,
    )


def _append_snapshot(
    snapshots: List[Any],
    *,
    inst: Any,
    origin: float,
    x: np.ndarray,
    mip_obj: float,
    wall_time_s: Optional[float] = None,
) -> None:
    from adip.pipeline import QKPIncumbentSnapshot

    xv = np.asarray(x, dtype=np.int64).ravel()
    if snapshots and np.array_equal(snapshots[-1].x, xv):
        return
    t_snap = (
        float(wall_time_s)
        if wall_time_s is not None
        else float(time.perf_counter() - origin)
    )
    snapshots.append(
        QKPIncumbentSnapshot(
            wall_time_s=t_snap,
            mip_obj=float(mip_obj),
            x=xv.copy(),
            xTQx=float(_qkp_obj(inst, xv)),
        )
    )


def _fortet_from_inst(inst: Any) -> FortetQKP:
    return fortet_qkp(inst.Q, inst.w, int(inst.W), c=inst.c)


def _round_binary_prefix(col_value: Sequence[float], n: int) -> np.ndarray:
    xv = np.zeros(n, dtype=np.int64)
    for j in range(n):
        xv[j] = int(round(float(col_value[j])))
    return xv


def solve_max_qkp_highs(
    inst: Any,
    *,
    time_limit_sec: float,
    quiet: bool = True,
    threads: int = 1,
    random_seed: Optional[int] = None,
    initial_binary_x: Optional[np.ndarray] = None,
    trace: bool = False,
) -> Tuple[Any, List[Any], Any]:
    """Max-QKP via HiGHS MIP on the Fortet linearization."""
    try:
        import highspy
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "HiGHS (highspy) is required for --warm-solver highs. "
            "Install with `poetry install` or `pip install highspy`."
        ) from err

    model = _fortet_from_inst(inst)
    n = model.n
    snapshots: List[Any] = []
    n_cb = 0
    origin = time.perf_counter()

    h = highspy.Highs()
    if quiet:
        h.setOptionValue("output_flag", False)
        h.setOptionValue("log_to_console", False)
    h.setOptionValue("time_limit", float(time_limit_sec))
    h.setOptionValue("threads", int(max(1, threads)))
    h.setOptionValue("mip_rel_gap", 0.0)
    if random_seed is not None:
        h.setOptionValue("random_seed", int(random_seed) % (2**31 - 1))

    n_col = model.n_cols
    costs = np.zeros(n_col, dtype=np.float64)
    costs[:n] = model.lin_x.astype(np.float64, copy=False)
    if model.n_pairs:
        costs[n:] = model.pair_coeff.astype(np.float64, copy=False)
    lower = np.zeros(n_col, dtype=np.float64)
    upper = np.ones(n_col, dtype=np.float64)
    h.addCols(
        n_col,
        costs,
        lower,
        upper,
        0,
        np.zeros(1, dtype=np.int32),
        np.zeros(0, dtype=np.int32),
        np.zeros(0, dtype=np.float64),
    )
    integ = np.full(n_col, int(highspy.HighsVarType.kInteger), dtype=np.int32)
    h.changeColsIntegrality(n_col, np.arange(n_col, dtype=np.int32), integ)
    _set_highs_maximize(h, highspy)

    inf = float(highspy.kHighsInf)
    n_pairs = model.n_pairs
    n_rows = 1 + 3 * n_pairs
    # knapsack nnz + 2+2+3 per Fortet triple
    nnz_knap = int(np.count_nonzero(model.w))
    n_nz = nnz_knap + 7 * n_pairs
    starts = np.empty(n_rows, dtype=np.int32)
    indices = np.empty(n_nz, dtype=np.int32)
    values = np.empty(n_nz, dtype=np.float64)
    row_lo = np.empty(n_rows, dtype=np.float64)
    row_up = np.empty(n_rows, dtype=np.float64)

    nz = 0
    starts[0] = 0
    for i in range(n):
        wi = int(model.w[i])
        if wi == 0:
            continue
        indices[nz] = i
        values[nz] = float(wi)
        nz += 1
    row_lo[0] = -inf
    row_up[0] = float(model.W)

    pi = model.pair_i
    pj = model.pair_j
    for k in range(n_pairs):
        ycol = n + k
        i = int(pi[k])
        j = int(pj[k])
        # y - x_i <= 0
        starts[1 + 3 * k] = nz
        indices[nz] = i
        values[nz] = -1.0
        nz += 1
        indices[nz] = ycol
        values[nz] = 1.0
        nz += 1
        row_lo[1 + 3 * k] = -inf
        row_up[1 + 3 * k] = 0.0
        # y - x_j <= 0
        starts[2 + 3 * k] = nz
        indices[nz] = j
        values[nz] = -1.0
        nz += 1
        indices[nz] = ycol
        values[nz] = 1.0
        nz += 1
        row_lo[2 + 3 * k] = -inf
        row_up[2 + 3 * k] = 0.0
        # x_i + x_j - y <= 1
        starts[3 + 3 * k] = nz
        indices[nz] = i
        values[nz] = 1.0
        nz += 1
        indices[nz] = j
        values[nz] = 1.0
        nz += 1
        indices[nz] = ycol
        values[nz] = -1.0
        nz += 1
        row_lo[3 + 3 * k] = -inf
        row_up[3 + 3 * k] = 1.0

    h.addRows(n_rows, row_lo, row_up, int(nz), starts, indices, values)

    if initial_binary_x is not None:
        col0 = fortet_col_primal(model, initial_binary_x)
        _try_highs_mip_start(h, col0)

    if trace:
        def _cb(callback_type: Any, message: Any, data_out: Any, data_in: Any, user_data: Any) -> None:  # noqa: ARG001
            nonlocal n_cb
            cb_improve = _highs_cb_type(highspy, "kCallbackMipImprovingSolution")
            if cb_improve is None or int(callback_type) != int(cb_improve):
                return
            sol = _highs_cb_mip_solution(data_out)
            if sol is None or len(sol) < n:
                return
            xv = _round_binary_prefix(sol, n)
            n_cb += 1
            mip_obj = float(_qkp_obj(inst, xv))
            _append_snapshot(snapshots, inst=inst, origin=origin, x=xv, mip_obj=mip_obj)

        try:
            h.setCallback(_cb)
            cb_improve = _highs_cb_type(highspy, "kCallbackMipImprovingSolution")
            if cb_improve is not None and hasattr(h, "startCallback"):
                h.startCallback(cb_improve)
        except Exception:  # noqa: BLE001
            pass

    h.run()
    wall = time.perf_counter() - origin

    status = str(h.modelStatusToString(h.getModelStatus())).strip().lower()
    info = h.getInfo()
    sol = h.getSolution()
    col_value = list(sol.col_value) if sol is not None else []
    x_opt: Optional[np.ndarray] = None
    obj: Optional[float] = None
    n_sols = 0
    if col_value and len(col_value) >= n:
        # A MIP incumbent is present when primal values are binary-looking.
        xv = _round_binary_prefix(col_value, n)
        if int(np.dot(model.w, xv)) <= int(model.W):
            x_opt = xv
            obj = float(_qkp_obj(inst, xv))
            n_sols = 1
    if obj is None:
        try:
            raw = float(info.objective_function_value)
            if np.isfinite(raw):
                obj = raw
        except Exception:  # noqa: BLE001
            obj = None

    gap = _highs_gap(info, obj)
    n_nodes = _highs_nodes(info)
    if x_opt is not None and obj is not None:
        _append_snapshot(snapshots, inst=inst, origin=origin, x=x_opt, mip_obj=float(obj), wall_time_s=wall)

    res = _empty_result(
        status=status,
        wall=wall,
        obj=obj,
        gap=gap,
        n_nodes=n_nodes,
        n_sols=n_sols,
        x_opt=x_opt,
    )
    return res, snapshots, _trace_meta(n_cb, len(snapshots))


def _set_highs_maximize(h: Any, highspy: Any) -> None:
    sense = None
    obj_sense = getattr(highspy, "ObjSense", None)
    if obj_sense is not None:
        sense = getattr(obj_sense, "kMaximize", None)
    if sense is None:
        sense = getattr(highspy, "kObjSenseMaximize", None)
    if sense is None:
        raise RuntimeError("highspy ObjSense.kMaximize not found")
    h.changeObjectiveSense(sense)


def _try_highs_mip_start(h: Any, col0: np.ndarray) -> None:
    try:
        h.setSolution(col0.astype(np.float64, copy=False))
        return
    except Exception:  # noqa: BLE001
        pass
    try:
        h.setSolution(list(map(float, col0)))
    except Exception:  # noqa: BLE001
        pass


def _highs_cb_type(highspy: Any, name: str) -> Optional[int]:
    for owner in (
        getattr(highspy, "HighsCallbackType", None),
        getattr(getattr(highspy, "cb", None), "HighsCallbackType", None),
        highspy,
    ):
        if owner is None:
            continue
        val = getattr(owner, name, None)
        if val is not None:
            return int(val)
    return None


def _highs_cb_mip_solution(data_out: Any) -> Optional[Sequence[float]]:
    if data_out is None:
        return None
    sol = getattr(data_out, "mip_solution", None)
    if sol is None:
        return None
    try:
        return list(sol)
    except TypeError:
        return None


def _highs_gap(info: Any, obj: Optional[float]) -> Optional[float]:
    try:
        g = float(info.mip_gap)
        if np.isfinite(g):
            return g
    except Exception:  # noqa: BLE001
        pass
    dual = None
    for attr in ("mip_dual_bound", "mip_best_bound"):
        if hasattr(info, attr):
            try:
                dual = float(getattr(info, attr))
            except Exception:  # noqa: BLE001
                dual = None
            if dual is not None and np.isfinite(dual):
                break
            dual = None
    return relative_mip_gap(obj, dual)


def _highs_nodes(info: Any) -> Optional[int]:
    for attr in ("mip_node_count", "node_count"):
        if hasattr(info, attr):
            try:
                return int(getattr(info, attr))
            except Exception:  # noqa: BLE001
                return None
    return None


def solve_max_qkp_cbc(
    inst: Any,
    *,
    time_limit_sec: float,
    quiet: bool = True,
    threads: int = 1,
    random_seed: Optional[int] = None,
    initial_binary_x: Optional[np.ndarray] = None,
    trace: bool = False,
) -> Tuple[Any, List[Any], Any]:
    """Max-QKP via COIN-OR CBC on the Fortet linearization (OR-Tools pywraplp)."""
    try:
        from ortools.linear_solver import pywraplp
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "OR-Tools is required for --warm-solver cbc. "
            "Install with `poetry install` or `pip install ortools`."
        ) from err

    model = _fortet_from_inst(inst)
    n = model.n
    snapshots: List[Any] = []
    origin = time.perf_counter()

    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("OR-Tools CBC solver is not available in this build.")
    solver.SetTimeLimit(int(max(1.0, float(time_limit_sec)) * 1000.0))
    try:
        solver.SetNumThreads(int(max(1, threads)))
    except Exception:  # noqa: BLE001
        pass
    if quiet:
        solver.SuppressOutput()
    if random_seed is not None:
        try:
            solver.SetSolverSpecificParametersAsString(f"randomSeed={int(random_seed)}")
        except Exception:  # noqa: BLE001
            pass

    x_vars = [solver.BoolVar(f"x_{i}") for i in range(n)]
    y_vars = [solver.BoolVar(f"y_{k}") for k in range(model.n_pairs)]
    knap_terms = [int(model.w[i]) * x_vars[i] for i in range(n) if int(model.w[i]) != 0]
    if knap_terms:
        solver.Add(solver.Sum(knap_terms) <= int(model.W))
    for k in range(model.n_pairs):
        i = int(model.pair_i[k])
        j = int(model.pair_j[k])
        y = y_vars[k]
        solver.Add(y <= x_vars[i])
        solver.Add(y <= x_vars[j])
        solver.Add(y >= x_vars[i] + x_vars[j] - 1)

    obj = solver.Objective()
    for i in range(n):
        ci = int(model.lin_x[i])
        if ci:
            obj.SetCoefficient(x_vars[i], float(ci))
    for k in range(model.n_pairs):
        ck = int(model.pair_coeff[k])
        if ck:
            obj.SetCoefficient(y_vars[k], float(ck))
    obj.SetMaximization()

    if initial_binary_x is not None:
        xv0 = np.asarray(initial_binary_x, dtype=np.int64).ravel()
        if int(xv0.shape[0]) == n and int(np.dot(model.w, xv0)) <= int(model.W):
            try:
                hints_v = list(x_vars)
                hints_val = [float(xv0[i]) for i in range(n)]
                for k in range(model.n_pairs):
                    hints_v.append(y_vars[k])
                    hints_val.append(float(xv0[int(model.pair_i[k])] * xv0[int(model.pair_j[k])]))
                solver.SetHint(hints_v, hints_val)
            except Exception:  # noqa: BLE001
                pass

    status_code = solver.Solve()
    wall = time.perf_counter() - origin
    status_map = {
        pywraplp.Solver.OPTIMAL: "optimal",
        pywraplp.Solver.FEASIBLE: "timelimit",
        pywraplp.Solver.INFEASIBLE: "infeasible",
        pywraplp.Solver.UNBOUNDED: "unbounded",
        pywraplp.Solver.ABNORMAL: "abnormal",
        pywraplp.Solver.NOT_SOLVED: "notsolved",
    }
    status = status_map.get(int(status_code), f"status_{int(status_code)}")

    x_opt: Optional[np.ndarray] = None
    obj_v: Optional[float] = None
    n_sols = 0
    if status_code in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        xv = np.zeros(n, dtype=np.int64)
        for i in range(n):
            xv[i] = int(round(x_vars[i].solution_value()))
        x_opt = xv
        obj_v = float(_qkp_obj(inst, xv))
        n_sols = 1
        if trace:
            _append_snapshot(snapshots, inst=inst, origin=origin, x=xv, mip_obj=obj_v, wall_time_s=wall)

    dual = None
    try:
        dual = float(solver.BestObjectiveBound())
    except Exception:  # noqa: BLE001
        dual = None
    gap = relative_mip_gap(obj_v, dual)
    n_nodes: Optional[int] = None
    try:
        n_nodes = int(solver.nodes())
    except Exception:  # noqa: BLE001
        n_nodes = None

    res = _empty_result(
        status=status,
        wall=wall,
        obj=obj_v,
        gap=gap,
        n_nodes=n_nodes,
        n_sols=n_sols,
        x_opt=x_opt,
    )
    return res, snapshots, _trace_meta(len(snapshots), len(snapshots))


def solve_max_qkp_cpsat(
    inst: Any,
    *,
    time_limit_sec: float,
    quiet: bool = True,
    threads: int = 1,
    random_seed: Optional[int] = None,
    initial_binary_x: Optional[np.ndarray] = None,
    trace: bool = False,
) -> Tuple[Any, List[Any], Any]:
    """Max-QKP via OR-Tools CP-SAT (boolean products + knapsack)."""
    try:
        from ortools.sat.python import cp_model
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "OR-Tools is required for --warm-solver cpsat. "
            "Install with `poetry install` or `pip install ortools`."
        ) from err

    model = _fortet_from_inst(inst)
    n = model.n
    snapshots: List[Any] = []
    n_cb = 0
    origin = time.perf_counter()

    cp = cp_model.CpModel()
    x_vars = [cp.NewBoolVar(f"x_{i}") for i in range(n)]
    y_vars = [cp.NewBoolVar(f"y_{k}") for k in range(model.n_pairs)]
    cp.Add(
        sum(int(model.w[i]) * x_vars[i] for i in range(n) if int(model.w[i]) != 0)
        <= int(model.W)
    )
    for k in range(model.n_pairs):
        i = int(model.pair_i[k])
        j = int(model.pair_j[k])
        # Native boolean product (equivalent to Fortet on {0,1}).
        cp.AddMultiplicationEquality(y_vars[k], [x_vars[i], x_vars[j]])

    obj_vars: List[Any] = []
    obj_coef: List[int] = []
    for i in range(n):
        ci = int(model.lin_x[i])
        if ci:
            obj_vars.append(x_vars[i])
            obj_coef.append(ci)
    for k in range(model.n_pairs):
        ck = int(model.pair_coeff[k])
        if ck:
            obj_vars.append(y_vars[k])
            obj_coef.append(ck)
    if obj_vars:
        cp.Maximize(cp_model.LinearExpr.WeightedSum(obj_vars, obj_coef))

    if initial_binary_x is not None:
        xv0 = np.asarray(initial_binary_x, dtype=np.int64).ravel()
        if int(xv0.shape[0]) == n and int(np.dot(model.w, xv0)) <= int(model.W):
            for i in range(n):
                cp.AddHint(x_vars[i], int(xv0[i]))
            for k in range(model.n_pairs):
                cp.AddHint(
                    y_vars[k],
                    int(xv0[int(model.pair_i[k])] * xv0[int(model.pair_j[k])]),
                )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_sec)
    solver.parameters.num_search_workers = int(max(1, threads))
    if quiet:
        solver.parameters.log_search_progress = False
    if random_seed is not None:
        solver.parameters.random_seed = int(random_seed) % (2**31 - 1)

    class _IncumbentCb(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self) -> None:
            nonlocal n_cb
            if not trace:
                return
            xv = np.zeros(n, dtype=np.int64)
            for i in range(n):
                xv[i] = int(self.Value(x_vars[i]))
            n_cb += 1
            mip_obj = float(self.ObjectiveValue())
            _append_snapshot(
                snapshots,
                inst=inst,
                origin=origin,
                x=xv,
                mip_obj=mip_obj,
                wall_time_s=float(self.WallTime()),
            )

    cb = _IncumbentCb()
    status_code = solver.Solve(cp, cb) if trace else solver.Solve(cp)
    wall = time.perf_counter() - origin

    status_map = {
        cp_model.OPTIMAL: "optimal",
        cp_model.FEASIBLE: "timelimit",
        cp_model.INFEASIBLE: "infeasible",
        cp_model.MODEL_INVALID: "invalid",
        cp_model.UNKNOWN: "unknown",
    }
    status = status_map.get(int(status_code), solver.StatusName(status_code).lower())

    x_opt: Optional[np.ndarray] = None
    obj_v: Optional[float] = None
    n_sols = 0
    if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        xv = np.zeros(n, dtype=np.int64)
        for i in range(n):
            xv[i] = int(solver.Value(x_vars[i]))
        x_opt = xv
        obj_v = float(_qkp_obj(inst, xv))
        n_sols = 1
        _append_snapshot(snapshots, inst=inst, origin=origin, x=xv, mip_obj=obj_v, wall_time_s=wall)

    dual = None
    try:
        dual = float(solver.BestObjectiveBound())
    except Exception:  # noqa: BLE001
        dual = None
    gap = relative_mip_gap(obj_v, dual)
    n_nodes: Optional[int] = None
    try:
        n_nodes = int(solver.NumBranches())
    except Exception:  # noqa: BLE001
        n_nodes = None

    res = _empty_result(
        status=status,
        wall=wall,
        obj=obj_v,
        gap=gap,
        n_nodes=n_nodes,
        n_sols=n_sols,
        x_opt=x_opt,
    )
    return res, snapshots, _trace_meta(n_cb, len(snapshots))


_OSS_DISPATCH: dict[str, Callable[..., Tuple[Any, List[Any], Any]]] = {
    "highs": solve_max_qkp_highs,
    "cbc": solve_max_qkp_cbc,
    "cpsat": solve_max_qkp_cpsat,
}


def solve_max_qkp_zeros(
    inst: Any,
    *,
    time_limit_sec: float = 0.0,
    quiet: bool = True,
    threads: int = 1,
    random_seed: Optional[int] = None,
    initial_binary_x: Optional[np.ndarray] = None,
    trace: bool = False,
) -> Tuple[Any, List[Any], Any]:
    """Synthetic warm-start: feasible all-zero ``x`` (objective 0), no MIP solve."""
    del time_limit_sec, quiet, threads, random_seed, initial_binary_x
    n = int(getattr(inst, "n"))
    x0 = np.zeros(n, dtype=np.int64)
    origin = time.perf_counter()
    res = _empty_result(
        status="optimal",
        wall=time.perf_counter() - origin,
        obj=0.0,
        gap=None,
        n_nodes=0,
        n_sols=1,
        x_opt=x0,
    )
    snaps: List[Any] = []
    if trace:
        from adip.pipeline import QKPIncumbentSnapshot

        snaps.append(
            QKPIncumbentSnapshot(
                wall_time_s=0.0,
                mip_obj=0.0,
                x=x0.copy(),
                xTQx=0.0,
            )
        )
    return res, snaps, _trace_meta(len(snaps), len(snaps))


def solve_max_qkp_oss(
    solver: str,
    inst: Any,
    *,
    time_limit_sec: float,
    quiet: bool = True,
    threads: int = 1,
    random_seed: Optional[int] = None,
    initial_binary_x: Optional[np.ndarray] = None,
    trace: bool = False,
) -> Tuple[Any, List[Any], Any]:
    """Dispatch to an open-source linearized QKP solver (not SCIP)."""
    key = normalize_warm_solver(solver)
    if key == "zeros":
        return solve_max_qkp_zeros(
            inst,
            time_limit_sec=float(time_limit_sec),
            quiet=bool(quiet),
            threads=int(threads),
            random_seed=random_seed,
            initial_binary_x=initial_binary_x,
            trace=bool(trace),
        )
    fn = _OSS_DISPATCH.get(key)
    if fn is None:
        raise ValueError(f"{solver!r} is not an OSS linearized solver.")
    return fn(
        inst,
        time_limit_sec=float(time_limit_sec),
        quiet=bool(quiet),
        threads=int(threads),
        random_seed=random_seed,
        initial_binary_x=initial_binary_x,
        trace=bool(trace),
    )
