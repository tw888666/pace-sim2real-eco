"""方向条件 v2.3 Mixed 单模型实验的冻结协议。"""

from __future__ import annotations

from typing import Literal

from pace_eco_lab.direction_conditioned_protocol import (
    DESIRED_DIRECTION_W,
    MAX_CROSS_TRACK_DEVIATION_M,
    MIN_DIRECTIONAL_PROGRESS_M,
    TARGET_SPEED_M_S_V2,
)
from pace_eco_lab.multi_terrain_protocol import (
    EVAL_BATCHES,
    EVAL_EPISODES,
    EVAL_NUM_ENVS,
    EVAL_TERRAIN_COLS,
    EVAL_TERRAIN_ROWS,
    FOOT_BOUNDARY_AUDIT_VERSION,
    PACE_PAPER_FIXED_WEIGHT,
    PACE_PAPER_FIXED_WEIGHT_LABEL,
    TERRAIN_DISTRIBUTIONS,
    terrain_column_metadata,
)


MixedMethod = Literal["task_only", "fixed_weight", "eco"]

PROTOCOL_VERSION = "gpt-direction-conditioned-v2.3-mixed-v1"
MANIFEST_VERSION = "gpt-direction-conditioned-v2.3-mixed-manifest-v1"
VARIANT = "directional"
TERRAIN = "mixed"
METHOD_NAMES: tuple[MixedMethod, ...] = ("task_only", "fixed_weight", "eco")
METHOD_LABELS = {
    "task_only": "任务型PPO",
    "fixed_weight": "固定权重PPO W100",
    "eco": "PACE-ECO PPO-Lagrangian",
}
FIXED_WEIGHT_LABEL = PACE_PAPER_FIXED_WEIGHT_LABEL
FIXED_ENERGY_REWARD_WEIGHT = PACE_PAPER_FIXED_WEIGHT
SMOKE_SEED = 903
BUDGET_SEED = 0
FORMAL_SEEDS = (1, 2, 3)
NUM_ENVS = 4_096
TRAINING_UPDATES = 3_000
SMOKE_UPDATES = 2

TASK_IDS: dict[MixedMethod, str] = {
    "task_only": "Isaac-PACE-DirectionConditioned-V23Mixed-TaskOnly-Terrain20sWide-Anymal-D-v0",
    "fixed_weight": "Isaac-PACE-DirectionConditioned-V23Mixed-FixedWeight-Terrain20sWide-Anymal-D-v0",
    "eco": "Isaac-PACE-DirectionConditioned-V23Mixed-ECO-Terrain20sWide-Anymal-D-v0",
}
DIRECTION_CONDITIONED_V2_3_MIXED_TASK_IDS = tuple(TASK_IDS.values())

SUBTASKS = (
    "flat",
    "rough",
    "boxes",
    "stairs-up",
    "stairs-down",
    "slope-up",
    "slope-down",
)
SUBTASK_COUNTS_PER_BATCH = {
    "flat": 10,
    "rough": 10,
    "boxes": 10,
    "stairs-up": 5,
    "stairs-down": 5,
    "slope-up": 5,
    "slope-down": 5,
}
SUBTASK_SUCCESS_MINIMUM = {
    name: int(count * 0.95 + 0.999999)
    for name, count in {
        key: value * EVAL_BATCHES for key, value in SUBTASK_COUNTS_PER_BATCH.items()
    }.items()
}

# 与 v2.1/v2.2 的 510000--650999、950000--960999 完全分离。
_TERRAIN_SEED_BASE = {
    "smoke_train": 970_000,
    "budget_train": 710_000,
    "formal_train": 730_000,
    "calibration": 740_000,
    "holdout": 750_000,
}


def terrain_seed(role: str, ppo_seed: int | None = None) -> int:
    if role not in _TERRAIN_SEED_BASE:
        raise ValueError(f"未知 v2.3 Mixed seed 角色：{role}")
    offset = 0 if ppo_seed is None else int(ppo_seed)
    if not 0 <= offset < 1_000:
        raise ValueError("PPO seed 必须位于 [0, 999]。")
    return _TERRAIN_SEED_BASE[role] + offset


def evaluation_batch_seed(split: str, batch_index: int) -> int:
    if split not in ("calibration", "holdout"):
        raise ValueError(f"未知评估拆分：{split}")
    if not 0 <= int(batch_index) < EVAL_BATCHES:
        raise ValueError("评估批次必须位于 [0, 3]。")
    return terrain_seed(split) + int(batch_index)


def training_target(method: str, seed: int, role: str) -> bool:
    if role == "smoke_train":
        return method == "fixed_weight" and seed == SMOKE_SEED
    if role == "budget_train":
        return method == "task_only" and seed == BUDGET_SEED
    if role == "formal_train":
        return method in METHOD_NAMES and seed in FORMAL_SEEDS
    return False


def subtask_name(category: str, direction: str) -> str:
    return category if direction == "level" else f"{category}-{direction}"


def validate_protocol() -> None:
    if EVAL_BATCHES != 4 or EVAL_NUM_ENVS != 50 or EVAL_EPISODES != 200:
        raise ValueError("v2.3 Mixed 评估必须为 4×50=200 回合。")
    if EVAL_TERRAIN_ROWS != 5 or EVAL_TERRAIN_COLS != 10:
        raise ValueError("v2.3 Mixed 地形网格必须为 5×10。")
    labels = terrain_column_metadata("mixed", EVAL_TERRAIN_COLS)
    expected = (
        "flat:level", "flat:level", "rough:level", "rough:level",
        "stairs:up", "stairs:down", "boxes:level", "boxes:level",
        "slope:up", "slope:down",
    )
    if labels != expected:
        raise ValueError(f"Mixed 列分配漂移：{labels}")
    if sum(SUBTASK_COUNTS_PER_BATCH.values()) != EVAL_NUM_ENVS:
        raise ValueError("Mixed 每批子任务数量之和不是50。")
    if set(SUBTASKS) != set(SUBTASK_COUNTS_PER_BATCH):
        raise ValueError("Mixed 七方向子任务不完整。")
    if TERRAIN_DISTRIBUTIONS["mixed"] != (
        ("flat", "level", 0.2), ("rough", "level", 0.2),
        ("stairs", "up", 0.1), ("stairs", "down", 0.1),
        ("boxes", "level", 0.2), ("slope", "up", 0.1),
        ("slope", "down", 0.1),
    ):
        raise ValueError("底层 Mixed 分布漂移。")
    if FIXED_ENERGY_REWARD_WEIGHT != -1.6e-4 or FIXED_WEIGHT_LABEL != "W100":
        raise ValueError("v2.3 固定权重必须为 W100=-1.6e-4。")
    if DESIRED_DIRECTION_W != (1.0, 0.0) or TARGET_SPEED_M_S_V2 != 1.0:
        raise ValueError("v2.3 方向指令或目标速度漂移。")
    if MIN_DIRECTIONAL_PROGRESS_M != 16.0 or MAX_CROSS_TRACK_DEVIATION_M != 3.0:
        raise ValueError("v2.3 成功阈值漂移。")
    ranges = [
        (terrain_seed(role, 0), terrain_seed(role, 999), role)
        for role in _TERRAIN_SEED_BASE
    ]
    for left, right in zip(sorted(ranges), sorted(ranges)[1:]):
        if left[1] >= right[0]:
            raise ValueError(f"v2.3 seed 段重叠：{left[2]}/{right[2]}")


validate_protocol()


__all__ = [name for name in globals() if name.isupper()] + [
    "evaluation_batch_seed", "subtask_name", "terrain_seed", "training_target",
]
