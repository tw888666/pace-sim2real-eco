"""PACE-ECO 多地形补充实验的冻结协议常量与纯 Python 校验。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TerrainName = Literal["flat", "rough", "stairs", "boxes", "slope", "mixed"]
MethodName = Literal["task_only", "fixed_weight", "eco"]

PROTOCOL_VERSION = "gpt-multi-terrain-v1.4"
FOOT_BOUNDARY_AUDIT_VERSION = "gpt-四足真实地形边界-v4"
TERRAIN_NAMES: tuple[str, ...] = ("flat", "rough", "stairs", "boxes", "slope")
TERRAIN_LABELS = {
    "flat": "平地",
    "rough": "粗糙地形",
    "stairs": "楼梯",
    "boxes": "箱块地形",
    "slope": "斜坡",
    "mixed": "五类混合地形",
}
METHOD_NAMES: tuple[str, ...] = ("task_only", "fixed_weight", "eco")
METHOD_LABELS = {
    "task_only": "任务型PPO",
    "fixed_weight": "固定权重PPO",
    "eco": "PACE-ECO",
}

TERRAIN_DISTRIBUTIONS: dict[str, tuple[tuple[str, str, float], ...]] = {
    "flat": (("flat", "level", 1.0),),
    "rough": (("rough", "level", 1.0),),
    "stairs": (("stairs", "up", 0.5), ("stairs", "down", 0.5)),
    "boxes": (("boxes", "level", 1.0),),
    "slope": (("slope", "up", 0.5), ("slope", "down", 0.5)),
    "mixed": (
        ("flat", "level", 0.2),
        ("rough", "level", 0.2),
        ("stairs", "up", 0.1),
        ("stairs", "down", 0.1),
        ("boxes", "level", 0.2),
        ("slope", "up", 0.1),
        ("slope", "down", 0.1),
    ),
}

TASK_IDS: dict[tuple[str, str], str] = {
    (method, terrain): (
        f"Isaac-PACE-{method_token}-{terrain_token}-Terrain20sWide-Anymal-D-v0"
    )
    for method, method_token in {
        "task_only": "TaskOnly",
        "fixed_weight": "FixedWeight",
        "eco": "ECO",
    }.items()
    for terrain, terrain_token in {
        "flat": "Flat",
        "rough": "Rough",
        "stairs": "Stairs",
        "boxes": "Boxes",
        "slope": "Slope",
        "mixed": "MixedTerrain",
    }.items()
}
MULTI_TERRAIN_TASK_IDS: tuple[str, ...] = tuple(TASK_IDS.values())

# v1.4 复用既有平地协议的 PPO seed 角色，以形成直观配对：seed0 标定，seed1--5 正式。
# 阶段与地形的数据隔离由独立任务 ID、地形 seed 段和输出根保证。
PPO_SEEDS = {
    "stage1_smoke": (900,),
    "stage1_budget": (0,),
    "stage1_formal": (1, 2, 3, 4, 5),
    "stage2_budget": (0,),
    "stage2_formal": (1, 2, 3, 4, 5),
    "stage2_smoke": (901,),
}

# v1.4 主实验直接使用 PACE 论文为 ANYmal 报告的固定能耗奖励系数。
# 网格候选只保留为未来新协议的设计参考，不能在 v1.4 中训练或选择。
PACE_PAPER_FIXED_WEIGHT_LABEL = "W100"
PACE_PAPER_FIXED_WEIGHT = -1.6e-4
DEFERRED_FIXED_WEIGHT_CANDIDATES = {
    "W025": -4.0e-5,
    "W050": -8.0e-5,
    "W100": -1.6e-4,
    "W200": -3.2e-4,
    "W400": -6.4e-4,
}

TERRAIN_LENGTH_M = 65.0
TERRAIN_WIDTH_M = 60.0
TERRAIN_ORIGIN_X_M = 30.0
TERRAIN_ACTIVE_DISTANCE_M = 29.0
TERRAIN_ACTIVE_END_X_M = TERRAIN_ORIGIN_X_M + TERRAIN_ACTIVE_DISTANCE_M
MAX_AUDITED_FORWARD_M = 30.0
MAX_AUDITED_LATERAL_M = 3.0
MAX_AUDITED_BACKWARD_M = 2.0
MAX_EPISODE_S = 20.0
TARGET_SPEED_M_S = 1.0
SUCCESS_SPEED_RANGE_M_S = (0.8, 1.2)
EVALUATION_WARMUP_S = 5.0

TRAIN_TERRAIN_ROWS = 5
TRAIN_TERRAIN_COLS = 10
EVAL_TERRAIN_ROWS = 5
EVAL_TERRAIN_COLS = 10
EVAL_NUM_ENVS = EVAL_TERRAIN_ROWS * EVAL_TERRAIN_COLS
EVAL_BATCHES = 4
EVAL_EPISODES = EVAL_BATCHES * EVAL_NUM_ENVS

# 训练/标定/留出使用不相交的六位地形种子段。
_TERRAIN_SEED_BASE = {
    "stage1_smoke_train": 930_000,
    "stage1_budget_train": 310_000,
    "stage1_formal_train": 330_000,
    "stage1_calibration": 340_000,
    "stage1_holdout": 350_000,
    "stage2_budget_train": 410_000,
    "stage2_formal_train": 430_000,
    "stage2_calibration": 440_000,
    "stage2_holdout": 450_000,
    "stage2_smoke_train": 940_000,
}

# 容量冒烟只改变并行环境数，不新增数据拆分；因此复用对应普通冒烟的
# PPO seed 与地形生成序列，但使用独立 protocol_role 和输出目录。
_TERRAIN_SEED_ROLE_ALIASES = {
    "stage1_capacity_smoke_train": "stage1_smoke_train",
    "stage2_capacity_smoke_train": "stage2_smoke_train",
}


@dataclass(frozen=True)
class LongTerrainGeometryRanges:
    rough_height_m: tuple[float, float] = (0.015, 0.060)
    rough_correlation_m: tuple[float, float] = (0.45, 0.90)
    stair_height_m: tuple[float, float] = (0.030, 0.080)
    stair_width_m: tuple[float, float] = (0.75, 1.10)
    box_height_m: tuple[float, float] = (0.040, 0.160)
    box_forward_size_m: tuple[float, float] = (0.35, 0.80)
    box_lateral_size_m: tuple[float, float] = (0.80, 2.40)
    box_gap_m: tuple[float, float] = (0.45, 1.10)
    slope_degrees: tuple[float, float] = (3.0, 8.0)


GEOMETRY_RANGES = LongTerrainGeometryRanges()


def terrain_seed(role: str, terrain: str, ppo_seed: int | None = None) -> int:
    """返回冻结且可审计的独立地形种子。"""

    role = _TERRAIN_SEED_ROLE_ALIASES.get(role, role)
    if role not in _TERRAIN_SEED_BASE:
        raise ValueError(f"未知地形种子角色：{role}")
    if terrain not in (*TERRAIN_NAMES, "mixed"):
        raise ValueError(f"未知地形类别：{terrain}")
    terrain_offset = (*TERRAIN_NAMES, "mixed").index(terrain) * 1_000
    seed_offset = 0 if ppo_seed is None else int(ppo_seed)
    if seed_offset < 0 or seed_offset >= 1_000:
        raise ValueError("PPO seed 必须位于 [0, 999]，以保持地形种子段不重叠。")
    return _TERRAIN_SEED_BASE[role] + terrain_offset + seed_offset


def direction_names(terrain: str) -> tuple[str, ...]:
    if terrain in ("stairs", "slope"):
        return ("up", "down")
    return ("level",)


def evaluation_batch_seed(base_seed: int, batch_index: int) -> int:
    """从数据拆分的冻结基准 seed 派生四个互异评估批次 seed。"""

    if not 0 <= int(batch_index) < EVAL_BATCHES:
        raise ValueError(f"评估批次必须位于 [0, {EVAL_BATCHES - 1}]。")
    return int(base_seed) + int(batch_index)


def evaluation_batch_offset(batch_index: int) -> int:
    """返回当前评估批次在200回合全局设计中的起始编号。"""

    evaluation_batch_seed(0, batch_index)
    return int(batch_index) * EVAL_NUM_ENVS


def evaluation_global_id(batch_index: int, local_index: int) -> int:
    """把批次内环境或实例编号映射到冻结的0--199全局编号。"""

    offset = evaluation_batch_offset(batch_index)
    if not 0 <= int(local_index) < EVAL_NUM_ENVS:
        raise ValueError(f"评估批次内编号必须位于 [0, {EVAL_NUM_ENVS - 1}]。")
    return offset + int(local_index)


def terrain_column_metadata(category: str, num_cols: int) -> tuple[str, ...]:
    """复现 Isaac Lab curriculum 按比例把子地形固定分配到列的规则。"""

    if category not in TERRAIN_DISTRIBUTIONS or num_cols <= 0:
        raise ValueError(f"非法地形类别或列数：{category}/{num_cols}")
    specs = TERRAIN_DISTRIBUTIONS[category]
    total = sum(spec[2] for spec in specs)
    cumulative: list[float] = []
    running = 0.0
    for _, _, proportion in specs:
        running += proportion / total
        cumulative.append(running)
    labels: list[str] = []
    for column in range(num_cols):
        position = column / num_cols + 0.001
        index = next(i for i, boundary in enumerate(cumulative) if position < boundary)
        terrain, direction, _ = specs[index]
        labels.append(f"{terrain}:{direction}")
    return tuple(labels)


def difficulty_label(level: int, num_rows: int) -> str:
    """把五个冻结难度层映射为中文标签。"""

    if not 0 <= level < num_rows:
        raise ValueError(f"难度层越界：level={level}, num_rows={num_rows}")
    if num_rows == 5:
        return ("低", "较低", "中", "较高", "高")[level]
    return f"L{level}"


def validate_protocol() -> None:
    if TERRAIN_ACTIVE_END_X_M >= TERRAIN_LENGTH_M:
        raise ValueError("有效地形末端必须位于长地形块内部。")
    forward_margin = TERRAIN_LENGTH_M - (TERRAIN_ORIGIN_X_M + MAX_AUDITED_FORWARD_M)
    if forward_margin < 4.0:
        raise ValueError("1.5 m/s×20 s 审计位置与地形块前缘的余量不足 4 m。")
    lateral_margin = TERRAIN_WIDTH_M / 2.0 - MAX_AUDITED_LATERAL_M
    if lateral_margin < 1.5:
        raise ValueError("侧向审计边界与相邻地形块的余量不足 1.5 m。")
    backward_margin = TERRAIN_ORIGIN_X_M - MAX_AUDITED_BACKWARD_M
    if backward_margin < 2.5:
        raise ValueError("后向审计边界与长地形块后缘的余量不足 2.5 m。")
    nominal_success_travel_m = SUCCESS_SPEED_RANGE_M_S[1] * MAX_EPISODE_S
    centered_clearance_m = min(TERRAIN_ORIGIN_X_M, TERRAIN_WIDTH_M / 2.0)
    if centered_clearance_m - nominal_success_travel_m < 4.0:
        raise ValueError("宽地形对称余量不足以容纳成功速度上限的20秒名义路程。")
    if EVAL_EPISODES != EVAL_BATCHES * EVAL_TERRAIN_ROWS * EVAL_TERRAIN_COLS:
        raise ValueError("评估总回合数必须等于批次数×每批地形实例数。")
    for stage in ("stage1", "stage2"):
        budget = set(PPO_SEEDS[f"{stage}_budget"])
        formal = set(PPO_SEEDS[f"{stage}_formal"])
        if budget & formal:
            raise ValueError(f"{stage} 的 B_ref 与正式 PPO seed 发生重叠。")
    if PPO_SEEDS["stage1_budget"] != PPO_SEEDS["stage2_budget"]:
        raise ValueError("阶段一和阶段二必须复用相同的 B_ref PPO seed。")
    if PPO_SEEDS["stage1_formal"] != PPO_SEEDS["stage2_formal"]:
        raise ValueError("阶段一和阶段二必须复用相同的正式 PPO seed。")
    reserved_seed_ranges = sorted(
        (
            terrain_seed(role, terrain, 0),
            terrain_seed(role, terrain, 999),
            f"{role}/{terrain}",
        )
        for role in _TERRAIN_SEED_BASE
        for terrain in (*TERRAIN_NAMES, "mixed")
    )
    for left, right in zip(reserved_seed_ranges, reserved_seed_ranges[1:]):
        if left[1] >= right[0]:
            raise ValueError(f"地形种子保留段发生重叠：{left[2]} 与 {right[2]}。")


validate_protocol()
