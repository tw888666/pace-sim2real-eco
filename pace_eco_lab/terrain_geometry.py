"""不依赖 Isaac Sim 的固定20秒长地形生成与审计核心。"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import trimesh

from pace_eco_lab.multi_terrain_protocol import (
    GEOMETRY_RANGES,
    TERRAIN_ACTIVE_DISTANCE_M,
    TERRAIN_ACTIVE_END_X_M,
    TERRAIN_LENGTH_M,
    TERRAIN_ORIGIN_X_M,
    TERRAIN_WIDTH_M,
)


TerrainKind = Literal["flat", "rough", "stairs", "boxes", "slope"]
TerrainDirection = Literal["level", "up", "down"]


class LongTerrainGeometryCfg(Protocol):
    """生成函数实际读取的最小配置接口。"""

    size: tuple[float, float]
    category: TerrainKind
    direction: TerrainDirection
    terrain_origin_x_m: float
    active_distance_m: float
    spawn_platform_m: float
    seed: int | None


@dataclass
class LongTerrainAuditCfg:
    """供纯 CPU 审计使用，不参与 Isaac Lab 配置实例化。"""

    category: TerrainKind
    direction: TerrainDirection = "level"
    seed: int = 0
    size: tuple[float, float] = (TERRAIN_LENGTH_M, TERRAIN_WIDTH_M)
    terrain_origin_x_m: float = TERRAIN_ORIGIN_X_M
    active_distance_m: float = TERRAIN_ACTIVE_DISTANCE_M
    spawn_platform_m: float = 1.5


def _rng(cfg: LongTerrainGeometryCfg, difficulty: float) -> np.random.Generator:
    payload = f"{cfg.seed}|{cfg.category}|{cfg.direction}|{difficulty:.17g}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(value)


def _lerp(bounds: tuple[float, float], difficulty: float) -> float:
    return float(bounds[0] + difficulty * (bounds[1] - bounds[0]))


def _box_specs(cfg: LongTerrainGeometryCfg, difficulty: float) -> list[dict[str, float]]:
    """在冻结间距范围内顺序采样箱块，保证不会越过有效地形末端。"""

    rng = _rng(cfg, difficulty)
    count = int(round(5 + 4 * difficulty))
    usable_start = cfg.terrain_origin_x_m + cfg.spawn_platform_m + 0.5
    usable_end = cfg.terrain_origin_x_m + cfg.active_distance_m - 1.5
    for _ in range(200):
        forward_sizes = rng.uniform(*GEOMETRY_RANGES.box_forward_size_m, size=count)
        gaps = rng.uniform(*GEOMETRY_RANGES.box_gap_m, size=max(0, count - 1))
        span = float(forward_sizes.sum() + gaps.sum())
        if span <= usable_end - usable_start:
            break
    else:
        raise RuntimeError("冻结箱块尺寸和间距无法放入有效路线。")
    cursor = usable_start + rng.uniform(0.0, usable_end - usable_start - span)
    nominal_height = _lerp(GEOMETRY_RANGES.box_height_m, difficulty)
    specs: list[dict[str, float]] = []
    for index, forward in enumerate(forward_sizes):
        lateral = float(rng.uniform(*GEOMETRY_RANGES.box_lateral_size_m))
        height = nominal_height * float(rng.uniform(0.55, 1.0))
        specs.append(
            {
                "x_center_m": float(cursor + forward / 2.0),
                "y_center_m": float(cfg.size[1] / 2.0 + rng.uniform(-0.9, 0.9)),
                "forward_size_m": float(forward),
                "lateral_size_m": lateral,
                "height_m": height,
            }
        )
        cursor += float(forward)
        if index < len(gaps):
            cursor += float(gaps[index])
    return specs


def resolved_parameters(cfg: LongTerrainGeometryCfg, difficulty: float) -> dict[str, object]:
    """给出某个实例的可审计实际几何参数。"""

    if not 0.0 <= difficulty <= 1.0:
        raise ValueError("difficulty 必须位于 [0, 1]。")
    rng = _rng(cfg, difficulty)
    result: dict[str, object] = {
        "category": cfg.category,
        "direction": cfg.direction,
        "difficulty": float(difficulty),
        "seed": int(cfg.seed or 0),
    }
    if cfg.category == "rough":
        result["height_m"] = _lerp(GEOMETRY_RANGES.rough_height_m, difficulty)
        result["correlation_m"] = rng.uniform(*GEOMETRY_RANGES.rough_correlation_m)
    elif cfg.category == "stairs":
        result["step_height_m"] = _lerp(GEOMETRY_RANGES.stair_height_m, difficulty)
        nominal_width = float(rng.uniform(*GEOMETRY_RANGES.stair_width_m))
        stair_length = cfg.active_distance_m - cfg.spawn_platform_m - 1.5
        step_count = max(1, int(math.ceil(stair_length / nominal_width)))
        result["sampled_nominal_step_width_m"] = nominal_width
        result["actual_step_width_m"] = stair_length / step_count
        result["step_count"] = step_count
        result["total_elevation_change_m"] = (
            step_count * float(result["step_height_m"]) * (1.0 if cfg.direction == "up" else -1.0)
        )
    elif cfg.category == "boxes":
        specs = _box_specs(cfg, difficulty)
        result["height_m"] = _lerp(GEOMETRY_RANGES.box_height_m, difficulty)
        result["count"] = len(specs)
        result["instances"] = specs
    elif cfg.category == "slope":
        nominal = _lerp(GEOMETRY_RANGES.slope_degrees, difficulty)
        result["slope_degrees"] = nominal
        signed = nominal if cfg.direction == "up" else -nominal
        result["total_elevation_change_m"] = math.tan(math.radians(signed)) * (
            cfg.active_distance_m - cfg.spawn_platform_m - 1.5
        )
    return result


def _surface_mesh(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> trimesh.Trimesh:
    if z.shape != (len(x), len(y)):
        raise ValueError("高度矩阵形状与 x/y 网格不一致。")
    xx, yy = np.meshgrid(x, y, indexing="ij")
    vertices = np.column_stack((xx.ravel(), yy.ravel(), z.ravel()))
    ny = len(y)
    faces: list[tuple[int, int, int]] = []
    for ix in range(len(x) - 1):
        for iy in range(len(y) - 1):
            a = ix * ny + iy
            b = (ix + 1) * ny + iy
            c = (ix + 1) * ny + iy + 1
            d = ix * ny + iy + 1
            faces.extend(((a, b, c), (a, c, d)))
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)


def _piecewise_surface(cfg: LongTerrainGeometryCfg, start_z: float, end_z: float) -> trimesh.Trimesh:
    x = np.asarray(
        (0.0, cfg.terrain_origin_x_m, cfg.terrain_origin_x_m + cfg.active_distance_m, cfg.size[0])
    )
    y = np.asarray((0.0, cfg.size[1]))
    heights = np.asarray((start_z, start_z, end_z, end_z))
    z = np.repeat(heights[:, None], len(y), axis=1)
    return _surface_mesh(x, y, z)


def _rough_surface(cfg: LongTerrainGeometryCfg, difficulty: float) -> trimesh.Trimesh:
    params = resolved_parameters(cfg, difficulty)
    horizontal = 0.20
    x = np.linspace(0.0, cfg.size[0], round(cfg.size[0] / horizontal) + 1)
    y = np.linspace(0.0, cfg.size[1], round(cfg.size[1] / horizontal) + 1)
    rng = _rng(cfg, difficulty)
    z = rng.normal(size=(len(x), len(y)))
    correlation = float(params["correlation_m"])
    passes = max(1, round(correlation / horizontal))
    for _ in range(passes):
        padded = np.pad(z, 1, mode="edge")
        z = (
            padded[1:-1, 1:-1] * 4.0
            + padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
        ) / 8.0
    z -= z.mean()
    z *= float(params["height_m"]) / max(float(np.max(np.abs(z))), 1.0e-9)
    # 起点周围必须是平面；路线末端也平滑回到零，避免出生穿透和终点台阶。
    ramp_in = np.clip((x - cfg.terrain_origin_x_m - cfg.spawn_platform_m) / 0.8, 0.0, 1.0)
    ramp_out = np.clip((cfg.terrain_origin_x_m + cfg.active_distance_m - x) / 0.8, 0.0, 1.0)
    z *= np.minimum(ramp_in, ramp_out)[:, None]
    return _surface_mesh(x, y, z)


def _stairs_meshes(cfg: LongTerrainGeometryCfg, difficulty: float) -> tuple[list[trimesh.Trimesh], float]:
    params = resolved_parameters(cfg, difficulty)
    step_height = float(params["step_height_m"])
    nominal_width = float(params["sampled_nominal_step_width_m"])
    terminal_platform_m = 1.5
    stair_start = cfg.terrain_origin_x_m + cfg.spawn_platform_m
    stair_end = cfg.terrain_origin_x_m + cfg.active_distance_m - terminal_platform_m
    stair_length = stair_end - stair_start
    count = max(1, int(math.ceil(stair_length / nominal_width)))
    step_width = stair_length / count
    top_levels = np.arange(count + 1, dtype=float) * step_height
    if cfg.direction == "down":
        top_levels = top_levels[::-1]
    elif cfg.direction != "up":
        raise ValueError("楼梯方向必须为 up 或 down。")
    bottom = -0.5
    meshes: list[trimesh.Trimesh] = []

    def add_band(x0: float, x1: float, top: float) -> None:
        height = top - bottom
        center = ((x0 + x1) / 2.0, cfg.size[1] / 2.0, bottom + height / 2.0)
        meshes.append(
            trimesh.creation.box(
                (x1 - x0, cfg.size[1], height),
                trimesh.transformations.translation_matrix(center),
            )
        )

    add_band(0.0, stair_start, float(top_levels[0]))
    for index in range(count):
        x0 = stair_start + index * step_width
        x1 = stair_start + (index + 1) * step_width
        add_band(x0, x1, float(top_levels[index]))
    add_band(stair_end, cfg.size[0], float(top_levels[-1]))
    return meshes, float(top_levels[0])


def _box_meshes(cfg: LongTerrainGeometryCfg, difficulty: float) -> list[trimesh.Trimesh]:
    meshes = [_piecewise_surface(cfg, 0.0, 0.0)]
    for spec in _box_specs(cfg, difficulty):
        meshes.append(
            trimesh.creation.box(
                (spec["forward_size_m"], spec["lateral_size_m"], spec["height_m"]),
                trimesh.transformations.translation_matrix(
                    (spec["x_center_m"], spec["y_center_m"], spec["height_m"] / 2.0)
                ),
            )
        )
    return meshes


def long_terrain(
    difficulty: float, cfg: LongTerrainGeometryCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """生成固定20秒长地形，并把地形 origin 放在历史任务的出生参考点。"""

    if tuple(cfg.size) != (TERRAIN_LENGTH_M, TERRAIN_WIDTH_M):
        raise ValueError(
            f"多地形协议冻结长地形尺寸为 {(TERRAIN_LENGTH_M, TERRAIN_WIDTH_M)}，实际为 {cfg.size}。"
        )
    if cfg.terrain_origin_x_m + cfg.active_distance_m > cfg.size[0] - 5.0:
        raise ValueError("有效地形末端与长地形块前缘的安全余量不足 5 m。")
    if cfg.category == "flat":
        meshes = [_piecewise_surface(cfg, 0.0, 0.0)]
        spawn_z = 0.0
    elif cfg.category == "rough":
        meshes = [_rough_surface(cfg, difficulty)]
        spawn_z = 0.0
    elif cfg.category == "stairs":
        meshes, spawn_z = _stairs_meshes(cfg, difficulty)
    elif cfg.category == "boxes":
        meshes = _box_meshes(cfg, difficulty)
        spawn_z = 0.0
    elif cfg.category == "slope":
        if cfg.direction not in ("up", "down"):
            raise ValueError("斜坡方向必须为 up 或 down。")
        angle = math.radians(float(resolved_parameters(cfg, difficulty)["slope_degrees"]))
        terminal_platform_m = 1.5
        slope_start = cfg.terrain_origin_x_m + cfg.spawn_platform_m
        slope_end = cfg.terrain_origin_x_m + cfg.active_distance_m - terminal_platform_m
        delta_z = math.tan(angle) * (slope_end - slope_start)
        start_z, end_z = (0.0, delta_z) if cfg.direction == "up" else (delta_z, 0.0)
        x = np.asarray((0.0, slope_start, slope_end, cfg.size[0]))
        y = np.asarray((0.0, cfg.size[1]))
        heights = np.asarray((start_z, start_z, end_z, end_z))
        meshes = [_surface_mesh(x, y, np.repeat(heights[:, None], len(y), axis=1))]
        spawn_z = start_z
    else:
        raise ValueError(f"未知长地形类别：{cfg.category}")
    origin = np.asarray((cfg.terrain_origin_x_m, cfg.size[1] / 2.0, spawn_z), dtype=float)
    return meshes, origin


def geometry_audit(cfg: LongTerrainGeometryCfg, difficulty: float) -> dict[str, object]:
    """纯 CPU 几何审计，用于静态测试而不启动 Isaac Sim。"""

    meshes, origin = long_terrain(difficulty, cfg)
    merged = trimesh.util.concatenate(meshes)
    bounds = np.asarray(merged.bounds)
    return {
        "origin": origin.tolist(),
        "bounds_min": bounds[0].tolist(),
        "bounds_max": bounds[1].tolist(),
        "vertex_count": int(len(merged.vertices)),
        "face_count": int(len(merged.faces)),
        "watertight": bool(merged.is_watertight),
        "parameters": resolved_parameters(cfg, difficulty),
        "active_end_x_m": TERRAIN_ACTIVE_END_X_M,
    }


def curriculum_difficulty_table(
    seed: int,
    num_rows: int,
    num_cols: int,
    difficulty_range: tuple[float, float],
) -> np.ndarray:
    """复现 Isaac Lab curriculum 逐列逐行的难度采样，不生成 mesh。"""

    rng = np.random.default_rng(seed)
    lower, upper = difficulty_range
    table = np.zeros((num_rows, num_cols), dtype=float)
    for column in range(num_cols):
        for row in range(num_rows):
            normalized = (row + rng.uniform()) / num_rows
            table[row, column] = lower + (upper - lower) * normalized
    return table


__all__ = [
    "LongTerrainAuditCfg",
    "TerrainDirection",
    "TerrainKind",
    "long_terrain",
    "curriculum_difficulty_table",
    "geometry_audit",
    "resolved_parameters",
]
