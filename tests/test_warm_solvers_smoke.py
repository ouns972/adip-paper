"""Tiny feasible QKP solves for HiGHS / CBC / CP-SAT (skip if extras missing)."""

from __future__ import annotations

import numpy as np
import pytest

from adip.pipeline import QKPInstance, sample_qkp_instance
from adip.qkp_linearize import fortet_objective, fortet_qkp
from adip.warm_solvers import solve_max_qkp_oss


def _tiny_inst() -> QKPInstance:
    return sample_qkp_instance(8, 0, base_seed=1, q_kind="indefinite")


def _assert_feasible_incumbent(inst: QKPInstance, x: np.ndarray, obj: float) -> None:
    xv = np.asarray(x, dtype=np.int64).ravel()
    assert xv.shape == (inst.n,)
    assert set(np.unique(xv)).issubset({0, 1})
    assert int(np.dot(inst.w, xv)) <= int(inst.W)
    model = fortet_qkp(inst.Q, inst.w, inst.W, c=inst.c)
    assert abs(float(obj) - float(fortet_objective(model, xv))) < 1e-6


@pytest.mark.parametrize("solver", ["highs", "cbc", "cpsat"])
def test_oss_solvers_find_feasible_qkp(solver: str) -> None:
    if solver == "highs":
        pytest.importorskip("highspy")
    else:
        pytest.importorskip("ortools")
    inst = _tiny_inst()
    res, _, _ = solve_max_qkp_oss(solver, inst, time_limit_sec=5.0, quiet=True, threads=1)
    assert res.x_opt is not None, res.status
    assert res.obj_value is not None
    _assert_feasible_incumbent(inst, res.x_opt, float(res.obj_value))
