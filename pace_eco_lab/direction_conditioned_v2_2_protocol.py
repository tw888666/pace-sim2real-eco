"""方向条件 v2.2 的三方法精简实验协议。"""

from __future__ import annotations

from typing import Literal

from pace_eco_lab.direction_conditioned_protocol import (
    FOOT_BOUNDARY_AUDIT_VERSION,
    PROTOCOL_VERSION as V2_1_PROTOCOL_VERSION,
    TASK_IDS as V2_1_TASK_IDS,
    terrain_seed as v2_1_terrain_seed,
)
from pace_eco_lab.multi_terrain_protocol import (
    PACE_PAPER_FIXED_WEIGHT,
    PACE_PAPER_FIXED_WEIGHT_LABEL,
    TERRAIN_LABELS,
    TERRAIN_NAMES,
)


DirectionV22Method = Literal["task_only", "fixed_weight", "eco"]

PROTOCOL_VERSION = "gpt-direction-conditioned-v2.2"
EVALUATION_PROTOCOL_VERSION = "gpt-direction-conditioned-v2.2-45model-holdout-v1"
METRIC_PROTOCOL_VERSION = "gpt-direction-conditioned-v2.2-table1-table2-v1"
AMENDED_METRIC_PROTOCOL_VERSION = "gpt-direction-conditioned-v2.2-table1-table2-v2"
SOURCE_PROTOCOL_VERSION = V2_1_PROTOCOL_VERSION
VARIANT = "directional"
METHOD_NAMES: tuple[DirectionV22Method, ...] = ("task_only", "fixed_weight", "eco")
METHOD_LABELS = {
    "task_only": "任务型PPO",
    "fixed_weight": "固定权重PPO",
    "eco": "PACE-ECO PPO-Lagrangian",
}

# W100 是项目 v1/v1.4 已冻结的固定权重，不在 v2.2 重新搜索。固定权重
# PPO 的能耗奖励项需要负号；论文表述中的固定 lambda 使用其正幅值。
FIXED_WEIGHT_LABEL = PACE_PAPER_FIXED_WEIGHT_LABEL
FIXED_ENERGY_REWARD_WEIGHT = PACE_PAPER_FIXED_WEIGHT
FIXED_LAMBDA = abs(FIXED_ENERGY_REWARD_WEIGHT)
SMOKE_SEED = 902

_TERRAIN_TOKENS = {
    "flat": "Flat",
    "rough": "Rough",
    "stairs": "Stairs",
    "boxes": "Boxes",
    "slope": "Slope",
}

# task_only/ECO 任务 ID 与 v2.1 E2 完全相同，确保环境和算法定义不漂移；
# fixed_weight 使用同一 E2 环境的独立任务 ID。
TASK_IDS: dict[tuple[str, DirectionV22Method, str], str] = {}
for _terrain in TERRAIN_NAMES:
    TASK_IDS[(VARIANT, "task_only", _terrain)] = V2_1_TASK_IDS[
        (VARIANT, "task_only", _terrain)
    ]
    TASK_IDS[(VARIANT, "eco", _terrain)] = V2_1_TASK_IDS[(VARIANT, "eco", _terrain)]
    TASK_IDS[(VARIANT, "fixed_weight", _terrain)] = (
        f"Isaac-PACE-DirectionConditioned-FixedWeight-{_TERRAIN_TOKENS[_terrain]}-"
        "Terrain20sWide-Anymal-D-v0"
    )

DIRECTION_CONDITIONED_V2_2_TASK_IDS: tuple[str, ...] = tuple(TASK_IDS.values())

# 论文主矩阵：flat/rough 的 task-only 与 ECO 复用已完成 seed1--5，固定权重
# 新训 seed1--3；复杂地形三方法统一 seed1--3。
TARGET_SEEDS: dict[tuple[DirectionV22Method, str], tuple[int, ...]] = {
    (method, terrain): (
        (1, 2, 3, 4, 5)
        if terrain in ("flat", "rough") and method in ("task_only", "eco")
        else (1, 2, 3)
    )
    for method in METHOD_NAMES
    for terrain in TERRAIN_NAMES
}

REUSED_V2_1_MODELS: frozenset[tuple[DirectionV22Method, str, int]] = frozenset(
    {
        (method, terrain, seed)
        for method in ("task_only", "eco")
        for terrain in ("flat", "rough")
        for seed in range(1, 6)
    }
    | {
        (method, "stairs", 1)
        for method in ("task_only", "eco")
    }
)

TARGET_MATRIX: tuple[tuple[DirectionV22Method, str, int], ...] = tuple(
    (method, terrain, seed)
    for terrain in TERRAIN_NAMES
    for method in METHOD_NAMES
    for seed in TARGET_SEEDS[(method, terrain)]
)
# 53 项仍是不可改写的训练/归档矩阵；论文正式评估只取每个组合共同的
# PPO seed1--3。seed4/5 保留在 TARGET_MATRIX 中，但不进入以下独立矩阵。
EVALUATION_SEEDS: tuple[int, ...] = (1, 2, 3)
EVALUATION_MATRIX: tuple[tuple[DirectionV22Method, str, int], ...] = tuple(
    (method, terrain, seed)
    for terrain in TERRAIN_NAMES
    for method in METHOD_NAMES
    for seed in EVALUATION_SEEDS
)
EVALUATION_REUSED_V2_1_MODELS: frozenset[tuple[DirectionV22Method, str, int]] = frozenset(
    item for item in EVALUATION_MATRIX if item in REUSED_V2_1_MODELS
)
EXCLUDED_FROM_EVALUATION: tuple[tuple[DirectionV22Method, str, int], ...] = tuple(
    item for item in TARGET_MATRIX if item not in EVALUATION_MATRIX
)
NEW_TRAINING_MATRIX: tuple[tuple[DirectionV22Method, str, int], ...] = tuple(
    item for item in TARGET_MATRIX if item not in REUSED_V2_1_MODELS
)


def terrain_seed(role: str, terrain: str, ppo_seed: int | None = None) -> int:
    """复用 v2.1 的 seed 映射，使三方法在相同 seed 下共享同一地形。"""

    return v2_1_terrain_seed(role, terrain, ppo_seed)


def is_target(method: str, terrain: str, seed: int) -> bool:
    return (method, terrain) in TARGET_SEEDS and seed in TARGET_SEEDS[(method, terrain)]


def is_new_training_target(method: str, terrain: str, seed: int) -> bool:
    return (method, terrain, seed) in NEW_TRAINING_MATRIX


def is_evaluation_target(method: str, terrain: str, seed: int) -> bool:
    return (method, terrain, seed) in EVALUATION_MATRIX


def validate_protocol() -> None:
    if set(METHOD_NAMES) != {"task_only", "fixed_weight", "eco"}:
        raise ValueError("v2.2 必须且只能包含三种主比较方法。")
    if len(TARGET_MATRIX) != 53:
        raise ValueError(f"v2.2 论文主矩阵应为 53 项，实际 {len(TARGET_MATRIX)}。")
    if len(REUSED_V2_1_MODELS) != 22 or len(NEW_TRAINING_MATRIX) != 31:
        raise ValueError("v2.2 复用/新增模型数量不符合冻结协议。")
    if len(EVALUATION_MATRIX) != 45 or len(set(EVALUATION_MATRIX)) != 45:
        raise ValueError("v2.2 正式评估矩阵必须是 45 个唯一模型。")
    if len(EVALUATION_REUSED_V2_1_MODELS) != 14:
        raise ValueError("v2.2 正式评估应复用 14 个 v2.1 模型。")
    if len(EXCLUDED_FROM_EVALUATION) != 8 or any(seed not in (4, 5) for _, _, seed in EXCLUDED_FROM_EVALUATION):
        raise ValueError("正式评估排除项必须且只能是 flat/rough 的 seed4、5。")
    if FIXED_ENERGY_REWARD_WEIGHT >= 0.0 or FIXED_LAMBDA <= 0.0:
        raise ValueError("固定权重奖励系数与 lambda 符号约定错误。")
    if len(DIRECTION_CONDITIONED_V2_2_TASK_IDS) != len(set(DIRECTION_CONDITIONED_V2_2_TASK_IDS)):
        raise ValueError("v2.2 任务 ID 不唯一。")


validate_protocol()


__all__ = [
    "DIRECTION_CONDITIONED_V2_2_TASK_IDS",
    "EVALUATION_MATRIX",
    "EVALUATION_PROTOCOL_VERSION",
    "EVALUATION_REUSED_V2_1_MODELS",
    "EVALUATION_SEEDS",
    "EXCLUDED_FROM_EVALUATION",
    "FIXED_ENERGY_REWARD_WEIGHT",
    "FIXED_LAMBDA",
    "FIXED_WEIGHT_LABEL",
    "FOOT_BOUNDARY_AUDIT_VERSION",
    "METHOD_LABELS",
    "METHOD_NAMES",
    "AMENDED_METRIC_PROTOCOL_VERSION",
    "METRIC_PROTOCOL_VERSION",
    "NEW_TRAINING_MATRIX",
    "PROTOCOL_VERSION",
    "REUSED_V2_1_MODELS",
    "SOURCE_PROTOCOL_VERSION",
    "SMOKE_SEED",
    "TARGET_MATRIX",
    "TARGET_SEEDS",
    "TASK_IDS",
    "TERRAIN_LABELS",
    "TERRAIN_NAMES",
    "VARIANT",
    "is_new_training_target",
    "is_evaluation_target",
    "is_target",
    "terrain_seed",
]
