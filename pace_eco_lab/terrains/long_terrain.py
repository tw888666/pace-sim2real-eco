"""Isaac Lab 固定20秒长地形配置；几何实现位于纯 CPU 模块。"""

from __future__ import annotations

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass

from pace_eco_lab.multi_terrain_protocol import TERRAIN_ACTIVE_DISTANCE_M, TERRAIN_ORIGIN_X_M
from pace_eco_lab.terrain_geometry import (
    TerrainDirection,
    TerrainKind,
    long_terrain,
    curriculum_difficulty_table,
    geometry_audit,
    resolved_parameters,
)


@configclass
class LongTerrainCfg(SubTerrainBaseCfg):
    """单一地形类别的固定20秒长地形配置。"""

    function = long_terrain
    category: TerrainKind = "flat"
    direction: TerrainDirection = "level"
    terrain_origin_x_m: float = TERRAIN_ORIGIN_X_M
    active_distance_m: float = TERRAIN_ACTIVE_DISTANCE_M
    spawn_platform_m: float = 1.5


__all__ = [
    "LongTerrainCfg",
    "long_terrain",
    "curriculum_difficulty_table",
    "geometry_audit",
    "resolved_parameters",
]
