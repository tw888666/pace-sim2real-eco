"""不依赖 Isaac Sim 的 Terrain20s 碰撞表面审计判据。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pace_eco_lab.multi_terrain_protocol import (
    MAX_AUDITED_FORWARD_M,
    TERRAIN_ACTIVE_DISTANCE_M,
)


def audit_difficulty_table(table: np.ndarray, lower: float, upper: float) -> dict[str, object]:
    """确认几何分层的每一行落入独立难度带，且同列严格递增。"""

    values = np.asarray(table, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or not lower < upper:
        raise ValueError("难度表必须是至少两行的二维数组，且范围递增。")
    rows = values.shape[0]
    width = (upper - lower) / rows
    records = []
    all_in_band = True
    for row in range(rows):
        band_lower = lower + row * width
        band_upper = band_lower + width
        row_values = values[row]
        passed = bool(np.all(row_values >= band_lower) and np.all(row_values < band_upper))
        all_in_band &= passed
        records.append(
            {
                "等级": row,
                "冻结下界": float(band_lower),
                "冻结上界": float(band_upper),
                "实际最小": float(row_values.min()),
                "实际最大": float(row_values.max()),
                "通过": passed,
            }
        )
    return {
        "逐行难度带": all_in_band,
        "同列严格递增": bool(np.all(np.diff(values, axis=0) > 0.0)),
        "各等级": records,
    }


def audit_surface_profiles(
    category: str,
    directions: Sequence[str],
    x_m: np.ndarray,
    y_m: np.ndarray,
    relative_height_m: np.ndarray,
) -> dict[str, object]:
    """审计出生点至30米前向审计线内的方向、平台和类别特性。"""

    x = np.asarray(x_m, dtype=float)
    y = np.asarray(y_m, dtype=float)
    heights = np.asarray(relative_height_m, dtype=float)
    if heights.shape != (len(directions), len(x), len(y)):
        raise ValueError("高度数组必须为 [实例, x采样, y采样]。")
    if not np.any(np.isclose(x, 0.0)) or not np.any(np.isclose(x, TERRAIN_ACTIVE_DISTANCE_M)):
        raise ValueError("采样必须包含出生点 0 m 和有效地形末端 29 m。")
    if float(np.min(np.abs(y))) > 1.0e-5:
        raise ValueError("横向采样必须包含中心线。")

    audited = (x >= 0.0) & (x <= MAX_AUDITED_FORWARD_M)
    start = (x >= 0.0) & (x <= 1.0)
    terminal = x >= TERRAIN_ACTIVE_DISTANCE_M
    active = (x >= 1.75) & (x <= 27.25)
    center_index = int(np.argmin(np.abs(y)))
    records: list[dict[str, object]] = []
    all_checks: dict[str, bool] = {
        "全部射线命中有限高度": bool(np.all(np.isfinite(heights))),
        "覆盖出生点至30m审计线": bool(x.min() <= 0.0 and x.max() >= MAX_AUDITED_FORWARD_M),
        "覆盖29m有效段后平台": bool(x.max() >= TERRAIN_ACTIVE_DISTANCE_M),
    }
    start_index = np.isclose(x, 0.0)
    end_index = np.isclose(x, TERRAIN_ACTIVE_DISTANCE_M)
    for index, direction in enumerate(directions):
        profile = heights[index]
        center = profile[:, center_index]
        delta = float(np.median(profile[end_index]) - np.median(profile[start_index]))
        checks = {
            "出生平台平整": bool(np.ptp(profile[start]) <= 0.015),
            "29m后为有限平台": bool(np.ptp(profile[terminal]) <= 0.015),
        }
        if direction == "up":
            checks["方向符号正确"] = delta > 0.10
        elif direction == "down":
            checks["方向符号正确"] = delta < -0.10
        elif direction == "level":
            checks["方向符号正确"] = abs(delta) <= 0.02
        else:
            raise ValueError(f"未知方向：{direction}")
        if category in ("stairs", "slope"):
            differences = np.diff(center[active])
            checks["有效段单调"] = bool(
                np.all(differences >= -0.02) if direction == "up" else np.all(differences <= 0.02)
            )
        elif category == "flat":
            checks["平地高度恒定"] = bool(np.ptp(profile[audited]) <= 0.005)
        elif category == "rough":
            checks["粗糙相对高度不越界"] = bool(np.ptp(profile[audited]) <= 0.13)
        elif category == "boxes":
            values = profile[audited]
            checks["箱块高度不越界"] = bool(np.ptp(values) <= 0.17)
            baseline = np.median(profile[start])
            checks["每个纵向采样存在通道"] = bool(
                np.all(np.min(np.abs(values - baseline), axis=1) <= 0.005)
            )
        else:
            raise ValueError(f"未知地形类别：{category}")
        for name, passed in checks.items():
            all_checks[f"实例{index}:{name}"] = passed
        records.append(
            {
                "实例": index,
                "方向": direction,
                "出生点相对高度_m": float(np.median(profile[start_index])),
                "29m处相对高度_m": float(np.median(profile[end_index])),
                "净高度变化_m": delta,
                "审计段最低相对高度_m": float(profile[audited].min()),
                "审计段最高相对高度_m": float(profile[audited].max()),
                "检查": checks,
            }
        )
    return {"全部通过": all(all_checks.values()), "检查": all_checks, "逐实例": records}


__all__ = ["audit_difficulty_table", "audit_surface_profiles"]
