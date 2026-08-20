"""惩罚与熵的固定调度公式。"""

from __future__ import annotations

import math

from pace_eco_lab.constants import (
    ENTROPY_FINAL,
    ENTROPY_INITIAL,
    ENTROPY_SLOPE,
    ENTROPY_TURNOVER_ITERATION,
    PENALTY_HALF_LIFE_ITERATIONS,
)


def penalty_coefficient(iteration: int | float, half_life: float = PENALTY_HALF_LIFE_ITERATIONS) -> float:
    """返回 ``1 - 2**(-t/half_life)``。"""

    if iteration < 0 or half_life <= 0:
        raise ValueError("iteration 不得为负，half_life 必须为正。")
    return 1.0 - 2.0 ** (-float(iteration) / half_life)


def entropy_coefficient(
    iteration: int | float,
    *,
    initial: float = ENTROPY_INITIAL,
    final: float = ENTROPY_FINAL,
    turnover: float = ENTROPY_TURNOVER_ITERATION,
    slope: float = ENTROPY_SLOPE,
) -> float:
    """PACE tanh 熵系数调度。"""

    if iteration < 0 or initial < final or final < 0 or slope <= 0:
        raise ValueError("熵调度参数非法。")
    epsilon = 0.5 - 0.5 * math.tanh(slope * (float(iteration) - turnover))
    return final + epsilon * (initial - final)
