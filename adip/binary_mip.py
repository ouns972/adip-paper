"""Binary MIP problem specification for knapsack-constrained QKP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class BinaryMIPSpec:
    """``A`` dense :class:`numpy.ndarray` or scipy sparse ``(m, n)``; ``c`` length ``n``."""

    A: Any
    lb: np.ndarray
    ub: np.ndarray
    c: np.ndarray
