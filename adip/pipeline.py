"""ADIP pipeline for Quadratic Knapsack: MIP warm-start, CE, soft Lagrangian phase-2."""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from adip.ar_model import (
    AutoregressiveBitModel,
)
from adip.phase1_ce import (
    Phase1CEConfig,
    run_phase1_ce_training,
)
from adip.phase2_train import (
    Phase2AdipConfig,
    Phase2AdipStats,
    run_phase2_adip_training,
)
from adip.binary_mip import BinaryMIPSpec

# --- Defaults (SCIP table suite) ------------------------------------------------
DEFAULT_N_VALUES: Tuple[int, ...] = (50, 100, 200, 400)
DEFAULT_EXPERIMENTS_PER_N = 10
DEFAULT_BASE_SEED = 0
DEFAULT_TIME_LIMIT_SEC = 300.0


def _fmt_adip_phase2_loss_trace(hist: Sequence[float]) -> str:
    """One-line ``w_soft·L`` trace for :func:`run_phase2_adip_training` history.

    The trainer restores weights at the step that achieved ``min`` (not ``terminal_last``).
    """
    if not hist:
        return ""
    i_best = min(range(len(hist)), key=lambda j: hist[j])
    return (
        f"first={hist[0]:.6g}  min={hist[i_best]:.6g} @ step {i_best} "
        f"(weights restored)  terminal_last={hist[-1]:.6g}"
    )


def _parse_scip_cli_param_value(raw: str) -> Any:
    """Parse a single token from ``--scip-param NAME=VALUE`` (VALUE side)."""
    s = raw.strip()
    sl = s.lower()
    if sl in ("true", "yes", "on"):
        return True
    if sl in ("false", "no", "off"):
        return False
    try:
        return int(s, 10)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def _parse_scip_param_kv_list(items: Sequence[str]) -> Dict[str, Any]:
    """Parse repeatable ``NAME=VALUE`` strings into a dict for :meth:`Model.setParam`."""
    out: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected NAME=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        key = k.strip()
        if not key:
            raise ValueError(f"empty parameter name in {item!r}")
        out[key] = _parse_scip_cli_param_value(v)
    return out


def _normalize_scip_param_key(name: str) -> str:
    return str(name).strip().lower().replace("\\", "/")


def _coerce_scip_param_for_setparam(name: str, val: Any) -> Any:
    """
    PySCIPOpt ``setParam`` expects typed values; some SCIP params are integer enums that do not
    accept symbolic names. Map common strings for those.
    """
    key = _normalize_scip_param_key(str(name))
    if key == "timing/clocktype" and isinstance(val, str):
        sl = val.strip().lower()
        if sl in ("wall", "wallclock", "w", "elapsed"):
            return 2  # SCIP_CLOCKTYPE_WALL
        if sl in ("cpu", "cputime", "c", "processor"):
            return 1  # SCIP_CLOCKTYPE_CPU
        if sl in ("default", "auto", "d"):
            return 0  # SCIP_CLOCKTYPE_DEFAULT
    return val


def _apply_scip_params(m: Any, params: Mapping[str, Any]) -> None:
    for name, val in params.items():
        m.setParam(str(name), _coerce_scip_param_for_setparam(str(name), val))


def _split_scip_params_ordered(
    params: Optional[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Split params so we can apply:

    * ``timing/clocktype`` first,
    * ``randomization/randomseedshift`` before ``limits/time``,
    * everything else (including overrides) last.
    """
    if not params:
        return {}, {}, {}
    clock: Dict[str, Any] = {}
    seed: Dict[str, Any] = {}
    rest: Dict[str, Any] = {}
    for name, val in params.items():
        nk = _normalize_scip_param_key(str(name))
        if nk == "timing/clocktype":
            clock[str(name)] = val
        elif nk == "randomization/randomseedshift":
            seed[str(name)] = val
        else:
            rest[str(name)] = val
    return clock, seed, rest


def _wall_clock_requested(clock_params: Mapping[str, Any]) -> bool:
    for name, val in clock_params.items():
        if _normalize_scip_param_key(str(name)) != "timing/clocktype":
            continue
        coerced = _coerce_scip_param_for_setparam(str(name), val)
        try:
            return int(coerced) == 2  # SCIP_CLOCKTYPE_WALL
        except (TypeError, ValueError):
            return False
    return False


def _parallel_max_threads_explicit(params: Optional[Mapping[str, Any]]) -> bool:
    if not params:
        return False
    for name in params:
        if _normalize_scip_param_key(str(name)) == "parallel/maxnthreads":
            return True
    return False


def _apply_scip_params_pre_limit_then_limit_then_rest(
    m: Any,
    *,
    time_limit_sec: float,
    quiet: bool,
    scip_params: Optional[Mapping[str, Any]],
) -> None:
    """
    Order matters for SCIP:

    * ``timing/clocktype`` before ``limits/time`` so the limit uses the intended clock.
    * ``randomization/randomseedshift`` before ``limits/time`` (matches multi-run seed semantics).
    * With **wall** clock, default ``parallel/maxnthreads`` (>1) can still stop after
      ~(limits/time)/(threads) **wall** seconds on multi-core machines; we set
      ``parallel/maxnthreads=1`` unless you already passed that parameter (then rest overrides).
    * ``limits/time`` from the caller, then verbosity, then remaining params (may override time).
    """
    clock_p, seed_p, rest_p = _split_scip_params_ordered(scip_params)
    if clock_p:
        _apply_scip_params(m, clock_p)
    if (
        clock_p
        and _wall_clock_requested(clock_p)
        and not _parallel_max_threads_explicit(scip_params)
    ):
        m.setParam("parallel/maxnthreads", 1)
    if seed_p:
        _apply_scip_params(m, seed_p)
    m.setParam("limits/time", float(time_limit_sec))
    if quiet:
        m.setParam("display/verblevel", 0)
        m.hideOutput()
    if rest_p:
        _apply_scip_params(m, rest_p)


@dataclass(frozen=True)
class QKPInstance:
    n: int
    exp_id: int
    Q: np.ndarray  # (n, n) int64
    w: np.ndarray  # (n,) int64, nonnegative
    W: int  # knapsack capacity (integer)
    #: Optional linear term in ``maximize xᵀ Q x + cᵀ x`` (same ``c`` for SCIP, discrete eval, soft surrogate).
    c: Optional[np.ndarray] = None  # (n,) int64 when present


@dataclass(frozen=True)
class QKPScipSolveResult:
    status: str
    obj_value: Optional[float]
    wall_time_s: float
    gap: Optional[float]
    n_nodes: Optional[int]
    n_sols: int
    #: Incumbent binary ``x`` when ``n_sols > 0`` (same order as :class:`QKPInstance`); ``None`` otherwise.
    x_opt: Optional[np.ndarray] = None


@dataclass(frozen=True)
class QKPIncumbentSnapshot:
    """One improving SCIP incumbent during a traced max-QKP solve."""

    wall_time_s: float
    mip_obj: float
    x: np.ndarray
    xTQx: float


@dataclass(frozen=True)
class ScipIncumbentTraceMeta:
    """Diagnostics for :func:`solve_max_qkp_scip_with_incumbent_x_trace`."""

    n_bestsol_callbacks: int
    n_bestsol_found: Optional[int]
    n_stored_sols: Optional[int]
    n_obj_trace_points: int
    n_snapshots_after_harvest: int
    #: ``(wall_s, mip_obj)`` for each improving BESTSOLFOUND callback (may lack ``x``).
    obj_trace: Tuple[Tuple[float, float], ...] = ()


def sample_qkp_instance(
    n: int,
    exp_id: int,
    *,
    base_seed: int = DEFAULT_BASE_SEED,
    q_kind: str = "indefinite",
) -> QKPInstance:
    """
    Draw a single QKP instance; identical ``(n, exp_id, base_seed)`` always yields the same ``Q, w, W``.

    ``q_kind``:

    * ``"indefinite"`` (default) — i.i.d. integer entries in ``[-5, 5]`` (typical QKP; ``x^T Q x`` is
      not convex on ``R^n``).
    * ``"psd"`` — ``Q = B B^T`` with integer ``B`` in ``[-3, 3]`` (same RNG stream), so ``Q`` is positive
      semidefinite and ``x \\mapsto x^T Q x`` is **convex** on ``R^n`` (maximizing it over the binary
      hypercube is still a hard combinatorial problem, but the quadratic form is convex). Smaller
      ``B`` than ``\\pm 5`` keeps ``Q`` entries from growing as large.
    * ``"concave_planted"`` — strictly concave planted **continuous** KKT point at ``x = (1/2,\\dots,1/2)``
      on ``[0,1]^n`` with active knapsack ``w^T x = W = (1/2)\\sum_i w_i`` (integer ``w_i \\ge 2``,
      even ``\\sum w``). Uses ``Q = -I - \\alpha \\mathbf{1}\\mathbf{1}^T`` and integer ``c`` with
      ``Q\\mathbf{1} + c = \\lambda w`` so the relaxed objective ``x^T Q x + c^T x`` has the intended
      stationary point; the **binary** MIP maximizes the same expression (non-trivial via ``\\lambda w``).
    """
    rng = np.random.default_rng(
        np.random.SeedSequence([int(base_seed), int(n), int(exp_id)])
    )
    kind = str(q_kind).strip().lower()
    if kind == "indefinite":
        Q = rng.integers(-5, 6, size=(n, n), dtype=np.int64)
        w = rng.integers(0, 6, size=(n,), dtype=np.int64)
        W = int(n // 4)
        return QKPInstance(n=n, exp_id=exp_id, Q=Q, w=w, W=W)
    if kind == "psd":
        B = rng.integers(-3, 4, size=(n, 1), dtype=np.int64)
        Q = (B @ B.T).astype(np.int64)
        w = rng.integers(0, 6, size=(n,), dtype=np.int64)
        W = int(n // 4)
        return QKPInstance(n=n, exp_id=exp_id, Q=Q, w=w, W=W)
    if kind == "concave_planted":
        if int(n) < 1:
            raise ValueError("concave_planted requires n >= 1.")
        w = rng.integers(2, 8, size=(n,), dtype=np.int64)
        if int(np.sum(w)) % 2 != 0:
            w = w.copy()
            w[0] = int(w[0]) + 1
        W = int(np.sum(w) // 2)
        alpha = int(rng.integers(1, 4))  # 1..3
        lam = int(rng.integers(2, 8))  # 2..7
        one = np.ones(int(n), dtype=np.int64)
        jmat = np.outer(one, one)
        Q = (-np.eye(int(n), dtype=np.int64) - int(alpha) * jmat).astype(np.int64, copy=False)
        c = (int(lam) * w + (1 + int(alpha) * int(n)) * one).astype(np.int64, copy=False)
        return QKPInstance(n=n, exp_id=exp_id, Q=Q, w=w, W=W, c=c)
    raise ValueError(
        f"Unknown q_kind {q_kind!r} (use indefinite, psd, or concave_planted)."
    )


def _instance_identity_message(
    *,
    base_seed: int,
    n: int,
    exp_id: int,
    q_kind: str,
    Q: np.ndarray,
    w: np.ndarray,
    W: int,
    c: Optional[np.ndarray] = None,
) -> str:
    """
    Human-readable key + short hash so two runs can be checked for the same ``(Q, w, W)``.

    :func:`sample_qkp_instance` is deterministic from ``(base_seed, n, exp_id, q_kind)``; the hash
    is a quick equality check on the actual arrays.
    """
    h = hashlib.sha256()
    h.update(np.asarray(Q, dtype=np.int64).tobytes())
    h.update(np.asarray(w, dtype=np.int64).tobytes())
    h.update(int(W).to_bytes(8, "little", signed=True))
    if c is not None:
        h.update(np.asarray(c, dtype=np.int64).tobytes())
    fp = h.hexdigest()[:16]
    return (
        f"  instance identity (fix Q, w, W):  base_seed={int(base_seed)}  n={int(n)}  "
        f"exp_id={int(exp_id)}  q_kind={q_kind!r}  |  sha256-prefix={fp}"
    )


def _derive_torch_run_seed(
    base_seed: int, n: int, exp_id: int, q_kind: str
) -> int:
    """
    Integer seed for torch / python random / numpy global RNG, fixed by the same instance key
    that determines ``(Q, w, W)``. Two runs with the same key and the same code path are intended
    to be bitwise-identical (CPU; CUDA also uses deterministic cuDNN and no TF32).
    """
    h = hashlib.sha256(
        f"adip_torch|{int(base_seed)}|{int(n)}|{int(exp_id)}|{str(q_kind).strip().lower()}".encode()
    ).digest()
    u = int.from_bytes(h[:8], "big", signed=False)
    return u % (2**31 - 1) or 1


def _apply_reproducibility(seed: int, *, device: str) -> None:
    """
    Best-effort reproducible training: same ``seed`` before any ADIP / torch work for an instance
    should match a second run (same device, same hyperparameters, same branch structure).
    """
    s = int(seed)
    if s <= 0:
        s = 1
    random.seed(s)
    np.random.seed(s & 0xFFFFFFFFFFFFFFFF)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    try:
        mps_mod = getattr(torch, "mps", None)
        if mps_mod is not None and hasattr(mps_mod, "manual_seed"):
            mps_mod.manual_seed(s)
    except Exception:
        pass
    dev = str(device).lower()
    if dev.startswith("cuda"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.backends.cuda.matmul.allow_tf32 = False  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass


def _build_quadratic_expr(n: int, Q: np.ndarray, x: Any) -> object:
    """Build ``xᵀ Q x``. Prefer matrix multiply when ``x`` is a MatrixVariable."""
    from pyscipopt import quicksum

    Qf = np.asarray(Q, dtype=np.float64).reshape(int(n), int(n))
    # MatrixVariable path: O(nnz)-ish in C++/Cython; avoids a Python list of n² Exprs.
    if hasattr(x, "T") and not isinstance(x, (list, tuple)):
        return x.T @ Qf @ x

    terms: List[Any] = []
    for i in range(n):
        for j in range(n):
            c = float(Qf[i, j])
            if c != 0.0:
                terms.append(c * x[i] * x[j])
    if not terms:
        return 0.0
    if len(terms) == 1:
        return terms[0]
    return quicksum(terms)


def _build_qkp_epigraph_rhs_pyscip(
    n: int,
    Q: np.ndarray,
    x: Any,
    c: Optional[np.ndarray],
) -> object:
    """
    PySCIPOpt expression for ``xᵀ Q x + cᵀ x`` (used as RHS in ``z <= ...``).
    """
    from pyscipopt import quicksum

    quad = _build_quadratic_expr(n, Q, x)
    if c is None:
        return quad
    cv = np.asarray(c, dtype=np.float64).ravel()
    if int(cv.shape[0]) != int(n):
        raise ValueError(f"c length {int(cv.shape[0])} != n={int(n)}")
    lin_terms = [float(cv[i]) * x[i] for i in range(n) if float(cv[i]) != 0.0]
    if not lin_terms:
        return quad
    lin = lin_terms[0] if len(lin_terms) == 1 else quicksum(lin_terms)
    if isinstance(quad, (int, float)) and float(quad) == 0.0:
        return lin
    return quad + lin


def _scip_add_binary_x(m: Any, n: int) -> Any:
    """Binary decision vector; MatrixVariable when available (faster quadratic build)."""
    if hasattr(m, "addMatrixVar"):
        return m.addMatrixVar(shape=int(n), vtype="B", name="x", lb=0.0, ub=1.0)
    return [m.addVar(vtype="B", name=f"x_{i}") for i in range(int(n))]


def _qkp_pyscip_try_heuristic_start(
    m: object,
    x_vars: list,
    z_var: object,
    *,
    x0: np.ndarray,
    Q: np.ndarray,
    w: np.ndarray,
    W: int,
    c: Optional[np.ndarray] = None,
) -> bool:
    """
    Inject a full primal start ``(x, z)`` into the epigraph max-QKP SCIP model
    (``z = xᵀQx + cᵀx`` at the given binary ``x`` when ``c`` is set). Returns whether SCIP accepted the solution.

    If ``wᵀx > W`` the start is skipped (returns ``False``) without calling SCIP.
    """
    xv = np.asarray(x0, dtype=np.int64).ravel()
    n = int(len(x_vars))
    if int(xv.shape[0]) != n:
        raise ValueError(f"initial_binary_x length {int(xv.shape[0])} != n={n}")
    wv = np.asarray(w, dtype=np.int64).ravel()
    if int(np.dot(wv, xv)) > int(W):
        return False
    Qm = torch.as_tensor(
        np.asarray(Q, dtype=np.float64).reshape(n, n),
        dtype=torch.float64,
        device="cpu",
    )
    xvd = torch.as_tensor(xv, dtype=torch.float64, device="cpu")
    z_t = torch.einsum("i,ij,j->", xvd, Qm, xvd)
    if c is not None:
        cd = torch.as_tensor(
            np.asarray(c, dtype=np.float64).ravel(),
            dtype=torch.float64,
            device="cpu",
        )
        z_t = z_t + torch.dot(cd, xvd)
    if not bool(torch.isfinite(z_t).item()):
        return False
    z_val = float(z_t.item())
    sol = m.createSol()
    for j in range(n):
        m.setSolVal(sol, x_vars[j], float(int(xv[j])))
    m.setSolVal(sol, z_var, z_val)
    try:
        try:
            rc = m.addSol(sol)
        except TypeError:
            rc = m.addSol(sol, free=True)  # type: ignore[call-arg]
    except Exception:
        return False
    if rc is False:
        return False
    return True


def solve_max_qkp_scip(
    inst: QKPInstance,
    *,
    time_limit_sec: float = DEFAULT_TIME_LIMIT_SEC,
    quiet: bool = True,
    scip_params: Optional[Mapping[str, Any]] = None,
    initial_binary_x: Optional[np.ndarray] = None,
) -> QKPScipSolveResult:
    """
    Maximize ``x^T Q x + c^T x`` s.t. ``w^T x <= W``, ``x`` binary, using SCIP.

    Epigraph: ``max z`` s.t. ``z <= x^T Q x + c^T x`` (omit ``c`` when ``c is None``).

    ``scip_params`` are ordered internally: ``timing/clocktype`` and ``randomization/randomseedshift``
    are applied **before** ``limits/time``; other keys last (and may override ``limits/time``).
    When wall clock is requested, ``parallel/maxnthreads=1`` is set unless you pass that parameter
    yourself—parallel SCIP can otherwise stop after ~(limits/time)/(threads) wall seconds.

    ``initial_binary_x`` (optional length-``n`` 0/1 vector with ``wᵀx <= W``): passed to SCIP as a
    heuristic primal start (``createSol`` / ``setSolVal`` / ``addSol``) before ``optimize()``.
    If infeasible for the knapsack or rejected by SCIP, the solve still runs from scratch.
    """
    try:
        from pyscipopt import Model, quicksum
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "PySCIPOpt is required. Install the dev extra: `poetry install` "
            "with `pyscipopt` in group dev, or `pip install pyscipopt`."
        ) from err

    n = inst.n
    Q = inst.Q
    w = inst.w
    W = int(inst.W)
    c = inst.c

    import time

    t0 = time.perf_counter()
    m = Model("QKP")
    _apply_scip_params_pre_limit_then_limit_then_rest(
        m,
        time_limit_sec=float(time_limit_sec),
        quiet=bool(quiet),
        scip_params=scip_params,
    )

    x = _scip_add_binary_x(m, n)
    z = m.addVar(vtype="C", name="z_epigraph", lb=None, ub=None)

    rhs = _build_qkp_epigraph_rhs_pyscip(n, Q, x, c)
    m.addCons(z <= rhs, name="epigraph")
    m.addCons(quicksum(int(w[i]) * x[i] for i in range(n)) <= float(W), name="knapsack")
    m.setObjective(z, sense="maximize")
    if initial_binary_x is not None:
        x_list = x if isinstance(x, list) else [x[i] for i in range(n)]
        _qkp_pyscip_try_heuristic_start(
            m, x_list, z, x0=initial_binary_x, Q=Q, w=w, W=W, c=c,
        )
    m.optimize()
    wall = time.perf_counter() - t0

    status = str(m.getStatus())
    n_sols = int(m.getNSols()) if hasattr(m, "getNSols") else 0
    obj: Optional[float] = None
    if n_sols > 0:
        try:
            obj = float(m.getObjVal())
        except Exception:  # noqa: BLE001
            obj = None

    gap: Optional[float] = None
    if hasattr(m, "getGap"):
        try:
            gap = float(m.getGap())
        except Exception:  # noqa: BLE001
            gap = None

    n_nodes: Optional[int] = None
    for attr in ("getNTotalNodes", "getNNodes", "getNtotalNodes"):
        if hasattr(m, attr):
            try:
                n_nodes = int(getattr(m, attr)())
            except Exception:  # noqa: BLE001
                n_nodes = None
            break

    x_opt: Optional[np.ndarray] = None
    if n_sols > 0:
        try:
            xv = np.zeros(n, dtype=np.int64)
            for j in range(n):
                xv[j] = int(round(m.getVal(x[j])))
            x_opt = xv
        except Exception:  # noqa: BLE001
            x_opt = None

    return QKPScipSolveResult(
        status=status,
        obj_value=obj,
        wall_time_s=wall,
        gap=gap,
        n_nodes=n_nodes,
        n_sols=n_sols,
        x_opt=x_opt,
    )








def _enable_scip_solution_pool(m: Any) -> None:
    """Increase retained feasible solutions (helps ``getSols()`` after solve)."""
    for key, val in (
        ("limits/maxsol", 512),
        ("constraints/countsols/collect", True),
    ):
        try:
            if isinstance(val, bool):
                m.setBoolParam(key, val)
            else:
                m.setIntParam(key, int(val))
        except Exception:  # noqa: BLE001
            pass


def solve_max_qkp_scip_with_incumbent_x_trace(
    inst: QKPInstance,
    *,
    time_limit_sec: float = DEFAULT_TIME_LIMIT_SEC,
    quiet: bool = True,
    scip_params: Optional[Mapping[str, Any]] = None,
    time_origin_perf_counter: Optional[float] = None,
) -> Tuple[QKPScipSolveResult, List[QKPIncumbentSnapshot], ScipIncumbentTraceMeta]:
    """
    Like :func:`solve_max_qkp_scip_with_incumbent_trace`, but stores the **binary** incumbent
    ``x`` at each ``BESTSOLFOUND`` event (for continuous-relaxation experiments).

    Consecutive events with the same ``x`` are dropped. Use
    :func:`last_k_distinct_incumbent_snapshots` to take the last ``K`` **pairwise-distinct**
    incumbents for GD.

    The returned :class:`ScipIncumbentTraceMeta` helps diagnose sparse traces: many
    ``n_bestsol_callbacks`` but few snapshots suggests ``x`` was not readable in some callbacks;
    few callbacks and few snapshots usually means SCIP found few improving primals before the
    time limit. ``limits/maxsol`` is raised so ``getSols()`` may list extra feasible solutions.
    """
    try:
        from pyscipopt import Model, SCIP_EVENTTYPE, quicksum
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "PySCIPOpt is required. Install the dev extra: `poetry install` "
            "with `pyscipopt` in group dev, or `pip install pyscipopt`."
        ) from err

    import time as _time

    n = inst.n
    Q = inst.Q
    w = inst.w
    W = int(inst.W)
    c = inst.c

    snapshots: List[QKPIncumbentSnapshot] = []
    obj_trace: List[Tuple[float, float]] = []
    n_bestsol_callbacks = 0
    origin = float(time_origin_perf_counter) if time_origin_perf_counter is not None else None

    def _elapsed() -> float:
        assert origin is not None
        return float(_time.perf_counter() - origin)

    def _incumbent_x_from_model(model: Model) -> Optional[np.ndarray]:
        """Read incumbent ``x`` (``getBestSol`` first — required in ``BESTSOLFOUND`` callbacks)."""
        xv = np.zeros(n, dtype=np.int64)
        try:
            sol = model.getBestSol()
        except Exception:  # noqa: BLE001
            sol = None
        if sol is not None:
            try:
                for j in range(n):
                    xv[j] = int(round(model.getSolVal(sol, x[j])))
                return xv
            except Exception:  # noqa: BLE001
                pass
        try:
            for j in range(n):
                xv[j] = int(round(model.getVal(x[j])))
            return xv
        except Exception:  # noqa: BLE001
            return None

    def _append_snapshot(
        model: Model,
        obj_v: float,
        *,
        wall_time_s: Optional[float] = None,
    ) -> None:
        if origin is None:
            return
        xv = _incumbent_x_from_model(model)
        if xv is None:
            return
        xTQx = float(_qkp_obj(inst, xv))
        if snapshots and np.array_equal(snapshots[-1].x, xv):
            return
        t_snap = float(wall_time_s) if wall_time_s is not None else _elapsed()
        snapshots.append(
            QKPIncumbentSnapshot(
                wall_time_s=t_snap,
                mip_obj=float(obj_v),
                x=xv.copy(),
                xTQx=xTQx,
            )
        )

    def _on_best_sol(model: Model, event: object) -> None:  # noqa: ARG001
        nonlocal n_bestsol_callbacks
        n_bestsol_callbacks += 1
        t_now = _elapsed() if origin is not None else None
        obj_v: Optional[float] = None
        try:
            obj_v = float(model.getObjVal())
        except Exception:  # noqa: BLE001
            obj_v = None
        if obj_v is None:
            try:
                sol = model.getBestSol()
                if sol is not None:
                    obj_v = float(model.getSolOrigObj(sol))
            except Exception:  # noqa: BLE001
                obj_v = None
        if t_now is not None and obj_v is not None:
            obj_trace.append((float(t_now), float(obj_v)))
        if obj_v is not None:
            _append_snapshot(model, float(obj_v), wall_time_s=t_now)

    t_build = _time.perf_counter()
    if origin is None:
        origin = float(t_build)

    m = Model("QKP_x_trace")
    _apply_scip_params_pre_limit_then_limit_then_rest(
        m,
        time_limit_sec=float(time_limit_sec),
        quiet=bool(quiet),
        scip_params=scip_params,
    )
    _enable_scip_solution_pool(m)

    x = _scip_add_binary_x(m, n)
    z = m.addVar(vtype="C", name="z_epigraph", lb=None, ub=None)

    rhs = _build_qkp_epigraph_rhs_pyscip(n, Q, x, c)
    m.addCons(z <= rhs, name="epigraph")
    m.addCons(quicksum(int(w[i]) * x[i] for i in range(n)) <= float(W), name="knapsack")
    m.setObjective(z, sense="maximize")

    m.attachEventHandlerCallback(
        _on_best_sol,
        [SCIP_EVENTTYPE.BESTSOLFOUND],
        name="qkp_incumbent_x_trace",
        description="record incumbent x and objective vs wall time",
    )

    m.optimize()
    wall = _time.perf_counter() - t_build

    status = str(m.getStatus())
    n_sols = int(m.getNSols()) if hasattr(m, "getNSols") else 0
    obj: Optional[float] = None
    if n_sols > 0:
        try:
            obj = float(m.getObjVal())
        except Exception:  # noqa: BLE001
            obj = None

    gap: Optional[float] = None
    if hasattr(m, "getGap"):
        try:
            gap = float(m.getGap())
        except Exception:  # noqa: BLE001
            gap = None

    n_nodes: Optional[int] = None
    for attr in ("getNTotalNodes", "getNNodes", "getNtotalNodes"):
        if hasattr(m, attr):
            try:
                n_nodes = int(getattr(m, attr)())
            except Exception:  # noqa: BLE001
                n_nodes = None
            break

    x_opt: Optional[np.ndarray] = None
    if n_sols > 0:
        x_opt = _incumbent_x_from_model(m)

    if origin is not None and obj is not None and x_opt is not None:
        _append_snapshot(m, float(obj))

    def _x_already_recorded(xv: np.ndarray) -> bool:
        return any(np.array_equal(s.x, xv) for s in snapshots)

    try:
        stored_sols = list(m.getSols())
    except Exception:  # noqa: BLE001
        stored_sols = []
    for sol in stored_sols:
        try:
            xv = np.zeros(n, dtype=np.int64)
            for j in range(n):
                xv[j] = int(round(m.getSolVal(sol, x[j])))
        except Exception:  # noqa: BLE001
            continue
        if _x_already_recorded(xv):
            continue
        try:
            mip_obj = float(m.getSolOrigObj(sol))
        except Exception:  # noqa: BLE001
            mip_obj = float(_qkp_obj(inst, xv))
        try:
            t_sol = float(m.getSolTime(sol))
        except Exception:  # noqa: BLE001
            t_sol = _elapsed()
        snapshots.append(
            QKPIncumbentSnapshot(
                wall_time_s=t_sol,
                mip_obj=mip_obj,
                x=xv.copy(),
                xTQx=float(_qkp_obj(inst, xv)),
            )
        )

    snapshots.sort(key=lambda s: float(s.wall_time_s))

    n_bestsol_found: Optional[int] = None
    if hasattr(m, "getNBestSolsFound"):
        try:
            n_bestsol_found = int(m.getNBestSolsFound())
        except Exception:  # noqa: BLE001
            n_bestsol_found = None

    trace_meta = ScipIncumbentTraceMeta(
        n_bestsol_callbacks=int(n_bestsol_callbacks),
        n_bestsol_found=n_bestsol_found,
        n_stored_sols=int(len(stored_sols)) if stored_sols is not None else None,
        n_obj_trace_points=int(len(obj_trace)),
        n_snapshots_after_harvest=int(len(snapshots)),
        obj_trace=tuple((float(t), float(v)) for t, v in obj_trace),
    )

    res = QKPScipSolveResult(
        status=status,
        obj_value=obj,
        wall_time_s=wall,
        gap=gap,
        n_nodes=n_nodes,
        n_sols=n_sols,
        x_opt=x_opt,
    )
    return res, snapshots, trace_meta


def distinct_incumbent_snapshots_by_x(
    snapshots: Sequence[QKPIncumbentSnapshot],
) -> List[QKPIncumbentSnapshot]:
    """Chronological list with one entry per distinct binary incumbent ``x``."""
    out: List[QKPIncumbentSnapshot] = []
    for s in snapshots:
        if any(np.array_equal(s.x, t.x) for t in out):
            continue
        out.append(s)
    return out


def last_k_distinct_incumbent_snapshots(
    snapshots: Sequence[QKPIncumbentSnapshot],
    k: int,
) -> List[QKPIncumbentSnapshot]:
    """
    Return up to ``k`` snapshots with **pairwise-distinct** ``x``, chosen from the
    **end** of the trace (most recent occurrence of each ``x``).
    """
    kk = max(0, int(k))
    if kk == 0 or not snapshots:
        return []
    picked_rev: List[QKPIncumbentSnapshot] = []
    seen: List[np.ndarray] = []
    for s in reversed(snapshots):
        if any(np.array_equal(s.x, ux) for ux in seen):
            continue
        seen.append(np.asarray(s.x))
        picked_rev.append(s)
        if len(picked_rev) >= kk:
            break
    return list(reversed(picked_rev))












# --- ADIP (two-phase) ------------------------------------------------------------


def knapsack_binary_mip_spec(w: np.ndarray, W: int) -> BinaryMIPSpec:
    """``lb <= wᵀx <= ub`` with one row: ``0 <= wᵀx <= W`` (integer-feasible semantics)."""
    w = np.asarray(w, dtype=np.float64).ravel()
    n = int(w.shape[0])
    A = w.reshape(1, -1)
    lb = np.array([0.0], dtype=np.float64)
    ub = np.array([float(W)], dtype=np.float64)
    c = np.zeros(n, dtype=np.float64)
    return BinaryMIPSpec(A=A, lb=lb, ub=ub, c=c)




def _discrete_q_value(
    Q: np.ndarray,
    x: np.ndarray,
    *,
    c: Optional[np.ndarray] = None,
) -> float:
    """``xᵀ Q x + cᵀ x`` for 0/1 (or near-binary) ``x``; integer matmul avoids float matmul overflow."""
    xf = np.asarray(x, dtype=np.float64).ravel()
    if not np.all(np.isfinite(xf)):
        xf = np.nan_to_num(xf, nan=0.0, posinf=1.0, neginf=0.0)
    Qd = np.asarray(Q, dtype=np.int64)
    # Treat as binary if close to 0/1, else binarize for safety.
    if xf.size and np.max(np.abs(xf - np.rint(xf))) < 1e-3:
        xi = np.rint(xf).astype(np.int64)
    else:
        xi = (xf > 0.5).astype(np.int64)
    base = float(xi @ Qd @ xi)
    if c is None:
        return base
    cc = np.asarray(c, dtype=np.int64).ravel()
    if int(cc.shape[0]) != int(xi.shape[0]):
        raise ValueError("c length must match x and Q.")
    return float(base + int(np.dot(cc, xi)))


def _qkp_obj(inst: QKPInstance, x: np.ndarray) -> float:
    """Discrete QKP objective for ``inst`` (includes linear term when present)."""
    return _discrete_q_value(inst.Q, x, c=inst.c)




def _merge_scip_params_with_seed_shift(
    base: Optional[Mapping[str, Any]],
    seed_shift: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base) if base else {}
    out["randomization/randomseedshift"] = int(seed_shift)
    return out






























@dataclass(frozen=True)
class AdipPipelineConfig:
    """Hyperparameters for :func:`run_adip_pipeline` (``scip_ce_*`` / ``gurobi_ce_*``)."""

    pipeline_name: str
    use_gurobi_warm_start: bool
    device: str
    base_seed: int
    q_kind: str
    ar_lr: float
    ar_d_model: int
    ar_nhead: int
    ar_nlayers: int
    ar_sparse_attn_prev_fraction: float
    ar_sparse_attn_pattern: str
    scip_time_limit: float
    unused_gurobi_time_limit: float
    scip_warm_start_seed_shift: Optional[int]
    gurobi_warm_start_seed: Optional[int]
    gurobi_license: Optional[str]
    scip_extra_params: Optional[Mapping[str, Any]]
    phase1_ce_target_prob: float
    phase1_ce_max_steps: int
    phase1_ce_plateau_patience: int
    phase1_ce_plateau_min_delta: float
    phase2_max_steps: int
    phase2_w_soft: float
    phase2_tau: float
    phase2_lambda: float
    phase2_q_weight: float
    phase2_plateau_patience: int
    phase2_plateau_min_delta: float
    phase2_time_limit_s: Optional[float]
    gumbel_use_ste: bool
    #: Adam LR for phase-2 only. ``None`` → use ``ar_lr`` (same as phase-1 CE).
    phase2_lr: Optional[float] = None
    post_p2_scip_time_limit_sec: Optional[float] = None
    post_p2_scip_seed_shift: Optional[int] = None
    reproducible: bool = True
    train_show_progress: bool = True
    record_incumbent_snapshots: bool = False
    record_phase2_soft_x_trace: bool = False
    phase2_soft_x_trace_stride: int = 5
    #: ``scip`` (native MIQP), or linearized MILP/CP: ``highs``, ``cbc``, ``cpsat``.
    warm_solver: str = "scip"
    warm_threads: int = 1


@dataclass
class AdipPipelineResult:
    """Outcome of one ``adip_scip_gumbel*`` / ``adip_gurobi_gumbel*`` instance run."""

    status: str
    warm_solver_tag: str
    warm_gap: Optional[float] = None
    warm_mip_obj: Optional[float] = None
    warm_xTQx: Optional[float] = None
    warm_wall_s: Optional[float] = None
    rounded_xTQx: Optional[float] = None
    rounded_feasible: Optional[bool] = None
    rounded_x: Optional[np.ndarray] = None
    warm_x: Optional[np.ndarray] = None
    incumbent_snapshots: Optional[List[QKPIncumbentSnapshot]] = None
    incumbent_trace_meta: Optional[ScipIncumbentTraceMeta] = None
    phase2_soft_x_trace: Optional[List[np.ndarray]] = None
    phase2_soft_x_trace_times_s: Optional[List[float]] = None
    phase1_wall_s: Optional[float] = None
    phase2_wall_s: Optional[float] = None
    cumulative_time_s: Optional[float] = None


def run_adip_pipeline(
    inst: QKPInstance,
    *,
    cfg: AdipPipelineConfig,
    t_instance_start: Optional[float] = None,
) -> AdipPipelineResult:
    """
    Shared implementation for ``adip_*_gumbel*`` pipelines.

    Warm-start MIP → teacher CE → Gumbel soft QKP phase-2.
    Prints progress for warm-start, CE, and phase-2.
    """
    from adip.warm_solvers import (
        WARM_SOLVER_TAGS,
        normalize_warm_solver,
        solve_max_qkp_oss,
    )

    n = int(inst.n)
    Q = inst.Q
    w = inst.w
    W = int(inst.W)
    exp_id = int(inst.exp_id)
    dev = torch.device(str(cfg.device))
    mip_spec = knapsack_binary_mip_spec(w, W)

    _warm_scip_params: Optional[Mapping[str, Any]] = cfg.scip_extra_params
    _gmode = (
        "gumbel_ste (forward hard one-hot, backward through soft)"
        if cfg.gumbel_use_ste
        else "gumbel_soft (standard)"
    )
    pipeline = str(cfg.pipeline_name)
    _scip_snaps: List[QKPIncumbentSnapshot] = []
    _scip_trace_meta: Optional[ScipIncumbentTraceMeta] = None
    _warm_key = (
        "gurobi"
        if bool(cfg.use_gurobi_warm_start)
        else normalize_warm_solver(getattr(cfg, "warm_solver", "scip"))
    )
    _ws_tag = WARM_SOLVER_TAGS.get(_warm_key, _warm_key.upper())
    _warm_limit = (
        float(cfg.unused_gurobi_time_limit)
        if _warm_key == "gurobi"
        else float(cfg.scip_time_limit)
    )
    _p2_chain = (
        "(soft targets from --phase1-ce-target-prob) → phase-2 minimize "
        "w_soft·(−q_w·x̃ᵀQx̃ + λ·knapsack slack) with **Gumbel–Softmax** relax chain;  "
        f"τ={float(cfg.phase2_tau):g}  λ={float(cfg.phase2_lambda):g}  mode={_gmode}."
    )

    if _warm_key == "gurobi":
        raise RuntimeError(
            "Gurobi warm-start is not supported in this release; "
            "use --warm-solver scip, highs, cbc, cpsat, or zeros."
        )
    if _warm_key == "zeros":
        print(
            f"  {pipeline} pipeline:  "
            f"zeros warm-start (all-zero incumbent, no MIP) → teacher CE "
            f"{_p2_chain}"
        )
        r_gu, _scip_snaps, _scip_trace_meta = solve_max_qkp_oss(
            "zeros",
            inst,
            time_limit_sec=0.0,
            quiet=True,
            threads=1,
            random_seed=cfg.scip_warm_start_seed_shift,
            trace=bool(cfg.record_incumbent_snapshots),
        )
    elif _warm_key == "scip":
        _seed_note = (
            f"  warm-start SCIP randomseedshift={int(cfg.scip_warm_start_seed_shift)}"
            if cfg.scip_warm_start_seed_shift is not None
            else ""
        )
        if cfg.scip_warm_start_seed_shift is not None:
            _warm_scip_params = _merge_scip_params_with_seed_shift(
                cfg.scip_extra_params, int(cfg.scip_warm_start_seed_shift)
            )
        print(
            f"  {pipeline} pipeline:  "
            f"SCIP warm-start ({_warm_limit:g}s limit){_seed_note} → teacher CE "
            f"{_p2_chain}"
        )
        if cfg.record_incumbent_snapshots:
            r_gu, _scip_snaps, _scip_trace_meta = solve_max_qkp_scip_with_incumbent_x_trace(
                inst,
                time_limit_sec=_warm_limit,
                quiet=True,
                scip_params=_warm_scip_params,
            )
        else:
            r_gu = solve_max_qkp_scip(
                inst,
                time_limit_sec=_warm_limit,
                quiet=True,
                scip_params=_warm_scip_params,
            )
    else:
        _seed_note = (
            f"  seed={int(cfg.scip_warm_start_seed_shift)}"
            if cfg.scip_warm_start_seed_shift is not None
            else ""
        )
        print(
            f"  {pipeline} pipeline:  "
            f"{_ws_tag} warm-start ({_warm_limit:g}s limit, Fortet linearization, "
            f"threads={int(cfg.warm_threads)}){_seed_note} → teacher CE "
            f"{_p2_chain}"
        )
        r_gu, _scip_snaps, _scip_trace_meta = solve_max_qkp_oss(
            _warm_key,
            inst,
            time_limit_sec=_warm_limit,
            quiet=True,
            threads=int(cfg.warm_threads),
            random_seed=cfg.scip_warm_start_seed_shift,
            trace=bool(cfg.record_incumbent_snapshots),
        )

    print(
        "  (no default post-SCIP soft-AR follow-up; use --post-p2-scip-time-limit for an extra SCIP "
        "solve from best-loss x̃_round.)"
    )

    _scip_snaps_out: List[QKPIncumbentSnapshot] = list(_scip_snaps)
    _scip_trace_meta_out: Optional[ScipIncumbentTraceMeta] = _scip_trace_meta
    if cfg.record_incumbent_snapshots and _scip_trace_meta_out is not None:
        _m = _scip_trace_meta_out
        print(
            f"  {_ws_tag} incumbent trace (warm):  {_m.n_snapshots_after_harvest} snapshot(s)  "
            f"({_m.n_bestsol_callbacks} improving callbacks,  "
            f"n_best_found={_m.n_bestsol_found},  "
            f"obj_trace={_m.n_obj_trace_points},  "
            f"stored_sols={_m.n_stored_sols})",
            flush=True,
        )

    def _skip(status: str) -> AdipPipelineResult:
        return AdipPipelineResult(
            status=status,
            warm_solver_tag=_ws_tag,
            warm_gap=r_gu.gap,
            warm_mip_obj=r_gu.obj_value,
            warm_wall_s=float(r_gu.wall_time_s),
            incumbent_snapshots=list(_scip_snaps_out) if _scip_snaps_out else None,
            incumbent_trace_meta=_scip_trace_meta_out,
            phase2_soft_x_trace=None,
        )

    if r_gu.x_opt is None:
        print(
            f"  {pipeline}:  skip — {_ws_tag} returned no incumbent binary vector."
        )
        return _skip("skip_no_warm_incumbent")

    x_ref_g = np.asarray(r_gu.x_opt, dtype=np.int64).ravel()
    if int(x_ref_g.shape[0]) != int(n):
        print(f"  {pipeline}:  skip — incumbent length ≠ N.")
        return _skip("skip_bad_incumbent_length")
    if int(np.dot(w, x_ref_g)) > int(W):
        print(f"  {pipeline}:  skip — incumbent violates knapsack.")
        return _skip("skip_bad_incumbent_knapsack")

    _obj_g = f"{r_gu.obj_value:.6g}" if r_gu.obj_value is not None else "—"
    _gap_g = f"{r_gu.gap:.6g}" if r_gu.gap is not None else "—"
    warm_xTQx = float(_qkp_obj(inst, x_ref_g))
    print(
        f"  {_ws_tag} warm-start incumbent:  xᵀQx={warm_xTQx:.6g}  "
        f"MIP objective={_obj_g}  gap={_gap_g}  time={r_gu.wall_time_s:.3f}s"
    )

    def _new_model() -> AutoregressiveBitModel:
        return AutoregressiveBitModel(
            num_sites=n,
            d_model=int(cfg.ar_d_model),
            nhead=int(cfg.ar_nhead),
            nlayers=int(cfg.ar_nlayers),
            sparse_attn_prev_fraction=float(cfg.ar_sparse_attn_prev_fraction),
            sparse_attn_pattern=str(cfg.ar_sparse_attn_pattern),
        )

    p1_cfg_ce = Phase1CEConfig(
        lr=float(cfg.ar_lr),
        device=str(cfg.device),
        max_optimizer_steps=int(cfg.phase1_ce_max_steps),
        ce_plateau_patience=int(cfg.phase1_ce_plateau_patience),
        ce_plateau_min_delta=float(cfg.phase1_ce_plateau_min_delta),
        ce_target_prob_correct=(
            None
            if float(cfg.phase1_ce_target_prob) >= 1.0
            else float(cfg.phase1_ce_target_prob)
        ),
    )
    p2cfg = Phase2AdipConfig(
        lr=float(cfg.ar_lr if cfg.phase2_lr is None else cfg.phase2_lr),
        device=str(cfg.device),
        w_soft=float(cfg.phase2_w_soft),
        softmax_tau=float(cfg.phase2_tau),
        lambda_lagrangian=float(cfg.phase2_lambda),
        q_objective_weight=float(cfg.phase2_q_weight),
        max_optimizer_steps=int(cfg.phase2_max_steps),
        loss_stop_threshold=None,
        soft_loss_plateau_patience=int(cfg.phase2_plateau_patience),
        soft_loss_plateau_min_delta=float(cfg.phase2_plateau_min_delta),
        relaxation="gumbel_ste" if cfg.gumbel_use_ste else "gumbel_soft",
        max_wall_time_sec=(
            None if cfg.phase2_time_limit_s is None else float(cfg.phase2_time_limit_s)
        ),
        record_soft_x_trace=bool(cfg.record_phase2_soft_x_trace),
        soft_x_trace_stride=int(cfg.phase2_soft_x_trace_stride),
    )

    g_gu = _new_model()
    t_p1g0 = time.perf_counter()
    hist_ce_g, p1e_g = run_phase1_ce_training(
        g_gu,
        x_ref_g,
        p1_cfg_ce,
        show_progress=bool(cfg.train_show_progress),
    )
    t_p1g1 = time.perf_counter()
    if p1e_g == "skipped" or int(cfg.phase1_ce_max_steps) <= 0:
        print("  phase-1 CE: skipped (random init → phase-2)")
    elif hist_ce_g:
        _ctp_g = float(cfg.phase1_ce_target_prob)
        _ce_mode_g = (
            "hard CE"
            if _ctp_g >= 1.0
            else f"soft CE p(correct)={_ctp_g:g} p(wrong)={1.0 - _ctp_g:g}"
        )
        print(
            f"  phase-1 CE ({_ce_mode_g}): steps={len(hist_ce_g)}/"
            f"{cfg.phase1_ce_max_steps}  "
            f"first={hist_ce_g[0]:.6g}  last={hist_ce_g[-1]:.6g}  "
            f"early={p1e_g!r}"
        )

    t_p2g0 = time.perf_counter()
    hist_p2g, st_p2g = run_phase2_adip_training(
        g_gu,
        mip_spec,
        Q,
        p2cfg,
        qkp_linear_c=inst.c,
        show_progress=bool(cfg.train_show_progress),
    )
    t_p2g1 = time.perf_counter()
    if hist_p2g:
        print(
            f"  phase-2 soft QKP (Gumbel): steps={len(hist_p2g)}/"
            f"{int(cfg.phase2_max_steps)}  "
            f"{_fmt_adip_phase2_loss_trace(hist_p2g)}  "
            f"early={st_p2g.early_stop_reason!r}  "
            f"time={t_p2g1 - t_p2g0:.3f}s"
        )

    _br = st_p2g.best_loss_rounded_xT_Q_x
    _bs = st_p2g.best_loss_surrogate_xTilde_Q_xTilde
    _bx = st_p2g.best_loss_soft_x_rounded
    _bfeas = st_p2g.best_loss_rounded_feasible
    _feas_s = "—" if _bfeas is None else str(bool(_bfeas))
    _xprev = (
        "".join(str(int(_bx[i])) for i in range(min(64, int(_bx.shape[0]))))
        + ("…" if _bx is not None and int(_bx.shape[0]) > 64 else "")
        if _bx is not None
        else "—"
    )
    print(
        f"  at lowest w_soft·L (same step as weight restore):  "
        f"x̃_round[0:64]={_xprev}  "
        f"xᵀQx(rounded)={_br if _br is not None else '—'}  "
        f"x̃ᵀQ x̃={_bs if _bs is not None else '—'}  "
        f"MIP-feasible(rounded)={_feas_s}"
    )

    if (
        cfg.post_p2_scip_time_limit_sec is not None
        and float(cfg.post_p2_scip_time_limit_sec) > 0.0
    ):
        _tl_pp = float(cfg.post_p2_scip_time_limit_sec)
        if cfg.post_p2_scip_seed_shift is not None:
            _p2_scip_p = _merge_scip_params_with_seed_shift(
                cfg.scip_extra_params, int(cfg.post_p2_scip_seed_shift)
            )
        else:
            _p2_scip_p = _warm_scip_params
        if _bx is None:
            print(
                "  post-phase2 SCIP:  skip — no x̃_round snapshot at best w_soft·L.",
                flush=True,
            )
        else:
            _kn_ok = int(np.dot(w, _bx)) <= int(W)
            if not _kn_ok:
                print(
                    f"  post-phase2 SCIP:  rounded x violates knapsack "
                    f"(wᵀx={int(np.dot(w, _bx))} > W={int(W)}); "
                    f"solve runs **without** that warm start.",
                    flush=True,
                )
            _hs = (
                "primal warm-start: best-loss x̃_round"
                if _kn_ok
                else "cold start (infeasible rounded x)"
            )
            print(
                f"  post-phase2 SCIP ({_hs}):  limits/time={_tl_pp:g}s …",
                flush=True,
            )
            t_pp0 = time.perf_counter()
            r_pp = solve_max_qkp_scip(
                inst,
                time_limit_sec=_tl_pp,
                quiet=True,
                scip_params=_p2_scip_p,
                initial_binary_x=(_bx if _kn_ok else None),
            )
            t_pp1 = time.perf_counter()
            _ov_pp = f"{r_pp.obj_value:.6g}" if r_pp.obj_value is not None else "—"
            if r_pp.x_opt is not None:
                _xw_pp = f"xᵀQx(incumbent)={_qkp_obj(inst, r_pp.x_opt):.6g}"
            else:
                _xw_pp = "x_opt=—"
            print(
                f"  post-phase2 SCIP done:  status={r_pp.status!r}  MIP obj={_ov_pp}  "
                f"{_xw_pp}  wall={r_pp.wall_time_s:.3f}s  "
                f"(local {t_pp1 - t_pp0:.3f}s)  nodes="
                f"{r_pp.n_nodes if r_pp.n_nodes is not None else '—'}",
                flush=True,
            )

    t_end = time.perf_counter()
    _cum_label = f"{_ws_tag}+CE+Gumbel"
    if t_instance_start is not None:
        print(
            f"  {_cum_label} soft-AR time:  cumulative (instance) = "
            f"{t_end - float(t_instance_start):.3f}s"
        )

    return AdipPipelineResult(
        status="ok",
        warm_solver_tag=_ws_tag,
        warm_gap=r_gu.gap,
        warm_mip_obj=r_gu.obj_value,
        warm_xTQx=warm_xTQx,
        warm_wall_s=float(r_gu.wall_time_s),
        warm_x=x_ref_g.copy(),
        rounded_xTQx=_br,
        rounded_feasible=_bfeas,
        rounded_x=(
            np.asarray(_bx, dtype=np.int64).ravel().copy()
            if _bx is not None
            else None
        ),
        incumbent_snapshots=list(_scip_snaps_out) if _scip_snaps_out else None,
        incumbent_trace_meta=_scip_trace_meta_out,
        phase2_soft_x_trace=(
            [row.copy() for row in st_p2g.soft_x_trace]
            if st_p2g.soft_x_trace is not None
            else None
        ),
        phase2_soft_x_trace_times_s=(
            [float(t) for t in st_p2g.soft_x_trace_times_s]
            if st_p2g.soft_x_trace_times_s is not None
            else None
        ),
        phase1_wall_s=float(t_p1g1 - t_p1g0),
        phase2_wall_s=float(t_p2g1 - t_p2g0),
        cumulative_time_s=(
            (t_end - float(t_instance_start))
            if t_instance_start is not None
            else None
        ),
    )

