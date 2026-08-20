"""方向条件复杂地形运动 v2.1 的冻结协议常量。"""

from __future__ import annotations

from typing import Literal

from pace_eco_lab.multi_terrain_protocol import (
    EVAL_BATCHES,
    EVAL_EPISODES,
    EVAL_NUM_ENVS,
    EVAL_TERRAIN_COLS,
    EVAL_TERRAIN_ROWS,
    EVALUATION_WARMUP_S,
    FOOT_BOUNDARY_AUDIT_VERSION,
    MAX_EPISODE_S,
    TERRAIN_LABELS,
    TERRAIN_NAMES,
    TARGET_SPEED_M_S,
)


DirectionVariant = Literal["observation_control", "directional"]
DirectionMethod = Literal["task_only", "eco"]

PROTOCOL_VERSION = "gpt-direction-conditioned-v2.1"
VARIANT_NAMES: tuple[DirectionVariant, ...] = ("observation_control", "directional")
VARIANT_LABELS = {
    "observation_control": "E1方向观测架构对照",
    "directional": "E2方向条件主要方法",
}
METHOD_NAMES: tuple[DirectionMethod, ...] = ("task_only", "eco")
METHOD_LABELS = {
    "task_only": "任务型PPO",
    "eco": "PACE-ECO",
}

DESIRED_DIRECTION_W: tuple[float, float] = (1.0, 0.0)
TARGET_SPEED_M_S_V2 = TARGET_SPEED_M_S
MIN_DIRECTIONAL_PROGRESS_M = 16.0
MAX_CROSS_TRACK_DEVIATION_M = 3.0

_VARIANT_TOKENS = {
    "observation_control": "DirectionObsControl",
    "directional": "DirectionConditioned",
}
_METHOD_TOKENS = {"task_only": "TaskOnly", "eco": "ECO"}
_TERRAIN_TOKENS = {
    "flat": "Flat",
    "rough": "Rough",
    "stairs": "Stairs",
    "boxes": "Boxes",
    "slope": "Slope",
    "mixed": "MixedTerrain",
}

TASK_IDS: dict[tuple[DirectionVariant, DirectionMethod, str], str] = {
    (variant, method, terrain): (
        f"Isaac-PACE-{_VARIANT_TOKENS[variant]}-{_METHOD_TOKENS[method]}-"
        f"{_TERRAIN_TOKENS[terrain]}-Terrain20sWide-Anymal-D-v0"
    )
    for variant in VARIANT_NAMES
    for method in METHOD_NAMES
    for terrain in (*TERRAIN_NAMES, "mixed")
}
DIRECTION_CONDITIONED_TASK_IDS: tuple[str, ...] = tuple(TASK_IDS.values())

# 与 v1.4 的 310000--450999、930000--940999 完全分离。E1/E2 在同一
# stage/terrain/PPO seed 下复用同一地形 seed，形成严格配对。
_TERRAIN_SEED_BASE = {
    "stage1_smoke_train": 950_000,
    "stage1_budget_train": 510_000,
    "stage1_formal_train": 530_000,
    "stage1_calibration": 540_000,
    "stage1_holdout": 550_000,
    "stage2_budget_train": 610_000,
    "stage2_formal_train": 630_000,
    "stage2_calibration": 640_000,
    "stage2_holdout": 650_000,
    "stage2_smoke_train": 960_000,
}
_TERRAIN_SEED_ROLE_ALIASES = {
    "stage1_capacity_smoke_train": "stage1_smoke_train",
    "stage2_capacity_smoke_train": "stage2_smoke_train",
}

PPO_SEEDS = {
    "stage1_smoke": (900,),
    "stage1_budget": (0,),
    "stage1_formal": (1, 2, 3, 4, 5),
    "stage2_budget": (0,),
    "stage2_formal": (1, 2, 3, 4, 5),
    "stage2_smoke": (901,),
}


def terrain_seed(role: str, terrain: str, ppo_seed: int | None = None) -> int:
    """返回 v2 独立且可审计的地形 seed。"""

    role = _TERRAIN_SEED_ROLE_ALIASES.get(role, role)
    if role not in _TERRAIN_SEED_BASE:
        raise ValueError(f"未知 v2 地形 seed 角色：{role}")
    terrains = (*TERRAIN_NAMES, "mixed")
    if terrain not in terrains:
        raise ValueError(f"未知地形类别：{terrain}")
    seed_offset = 0 if ppo_seed is None else int(ppo_seed)
    if not 0 <= seed_offset < 1_000:
        raise ValueError("PPO seed 必须位于 [0, 999]。")
    return _TERRAIN_SEED_BASE[role] + terrains.index(terrain) * 1_000 + seed_offset


def evaluation_batch_seed(base_seed: int, batch_index: int) -> int:
    if not 0 <= int(batch_index) < EVAL_BATCHES:
        raise ValueError(f"评估批次必须位于 [0, {EVAL_BATCHES - 1}]。")
    return int(base_seed) + int(batch_index)


def evaluation_batch_offset(batch_index: int) -> int:
    evaluation_batch_seed(0, batch_index)
    return int(batch_index) * EVAL_NUM_ENVS


def evaluation_global_id(batch_index: int, local_index: int) -> int:
    if not 0 <= int(local_index) < EVAL_NUM_ENVS:
        raise ValueError(f"评估批次内编号必须位于 [0, {EVAL_NUM_ENVS - 1}]。")
    return evaluation_batch_offset(batch_index) + int(local_index)


def validate_protocol() -> None:
    if EVAL_BATCHES * EVAL_NUM_ENVS != EVAL_EPISODES:
        raise ValueError("v2 评估批次设计与总回合数不一致。")
    if DESIRED_DIRECTION_W != (1.0, 0.0):
        raise ValueError("v2.1 只研究固定世界 +x 地形主方向。")
    if TARGET_SPEED_M_S_V2 * MAX_EPISODE_S != 20.0:
        raise ValueError("v2.1 的名义20秒任务距离必须为20m。")
    if not 0.0 < MIN_DIRECTIONAL_PROGRESS_M < 20.0:
        raise ValueError("方向穿越最低进度必须位于 (0, 20)m。")
    if MAX_CROSS_TRACK_DEVIATION_M <= 0.0:
        raise ValueError("最大横向偏移阈值必须为正。")
    if set(TASK_IDS) != {
        (variant, method, terrain)
        for variant in VARIANT_NAMES
        for method in METHOD_NAMES
        for terrain in (*TERRAIN_NAMES, "mixed")
    }:
        raise ValueError("v2 任务 ID 矩阵不完整。")


validate_protocol()


__all__ = [
    "DESIRED_DIRECTION_W",
    "DIRECTION_CONDITIONED_TASK_IDS",
    "EVAL_BATCHES",
    "EVAL_EPISODES",
    "EVAL_NUM_ENVS",
    "EVAL_TERRAIN_COLS",
    "EVAL_TERRAIN_ROWS",
    "EVALUATION_WARMUP_S",
    "FOOT_BOUNDARY_AUDIT_VERSION",
    "MAX_CROSS_TRACK_DEVIATION_M",
    "METHOD_LABELS",
    "METHOD_NAMES",
    "MIN_DIRECTIONAL_PROGRESS_M",
    "PPO_SEEDS",
    "PROTOCOL_VERSION",
    "TASK_IDS",
    "TARGET_SPEED_M_S_V2",
    "TERRAIN_LABELS",
    "VARIANT_LABELS",
    "VARIANT_NAMES",
    "evaluation_batch_offset",
    "evaluation_batch_seed",
    "evaluation_global_id",
    "terrain_seed",
]
