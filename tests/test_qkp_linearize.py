"""Fortet linearization matches discrete ``xᵀ Q x + cᵀ x`` on {0,1}."""

from __future__ import annotations

import numpy as np

from adip.qkp_linearize import fortet_objective, fortet_qkp


def test_fortet_matches_quadratic_indefinite() -> None:
    rng = np.random.default_rng(0)
    n = 12
    Q = rng.integers(-5, 6, size=(n, n), dtype=np.int64)
    c = rng.integers(-3, 4, size=(n,), dtype=np.int64)
    w = rng.integers(0, 6, size=(n,), dtype=np.int64)
    W = int(n // 4)
    model = fortet_qkp(Q, w, W, c=c)
    for _ in range(40):
        x = rng.integers(0, 2, size=(n,), dtype=np.int64)
        quad = int(x @ Q @ x + np.dot(c, x))
        assert fortet_objective(model, x) == quad


def test_fortet_skips_zero_pairs() -> None:
    Q = np.zeros((3, 3), dtype=np.int64)
    Q[0, 0] = 2
    Q[1, 2] = 4
    Q[2, 1] = -1
    model = fortet_qkp(Q, np.ones(3, dtype=np.int64), 2)
    assert model.n_pairs == 1
    assert int(model.pair_i[0]) == 1 and int(model.pair_j[0]) == 2
    assert int(model.pair_coeff[0]) == 3
    x = np.array([1, 1, 1], dtype=np.int64)
    assert fortet_objective(model, x) == int(x @ Q @ x)
