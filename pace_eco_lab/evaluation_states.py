"""预算标定和最终留出评估使用的纯数据协议。"""

from __future__ import annotations

import json
from hashlib import sha256

# 每行依次为 x/y/z、roll/pitch/yaw、六维根速度。
# calibration_v1 已参与种子 0 先导实验，只用于开发验证和预算标定。
_CALIBRATION_ROOT_OFFSETS = (
    (0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00),
    (0.00, 0.00, 0.00, 0.03, 0.00, 0.00, 0.15, 0.00, 0.00, 0.00, 0.00, 0.00),
    (0.00, 0.00, 0.00, -0.03, 0.00, 0.00, -0.15, 0.00, 0.00, 0.00, 0.00, 0.00),
    (0.00, 0.00, 0.00, 0.00, 0.03, 0.00, 0.00, 0.15, 0.00, 0.00, 0.00, 0.00),
    (0.00, 0.00, 0.00, 0.00, -0.03, 0.00, 0.00, -0.15, 0.00, 0.00, 0.00, 0.00),
    (0.00, 0.00, 0.00, 0.00, 0.00, 0.06, 0.10, -0.10, 0.00, 0.00, 0.00, 0.10),
    (0.00, 0.00, 0.00, 0.00, 0.00, -0.06, -0.10, 0.10, 0.00, 0.00, 0.00, -0.10),
    (0.00, 0.00, 0.00, 0.02, -0.02, 0.04, 0.05, 0.05, 0.00, 0.05, -0.05, 0.05),
)
_CALIBRATION_JOINT_SCALES = (1.00, 0.97, 1.03, 0.95, 1.05, 0.98, 1.02, 1.00)

CALIBRATION_STATE_SET = "calibration_v1"
HOLDOUT_STATE_SET = "holdout_v1"
EVALUATION_STATE_SETS = (CALIBRATION_STATE_SET, HOLDOUT_STATE_SET)

# v1.4 Terrain20sWide 主协议复用历史 Flat 的八个 calibration/holdout 初始状态数值，
# 只使用独立名称隔离结果文件；因此状态定义哈希分别与历史集合完全相同。
MULTI_TERRAIN_CALIBRATION_STATE_SET = "multi_terrain20s_wide_calibration_v1"
MULTI_TERRAIN_HOLDOUT_STATE_SET = "multi_terrain20s_wide_holdout_v1"
MULTI_TERRAIN_EVALUATION_STATE_SETS = (
    MULTI_TERRAIN_CALIBRATION_STATE_SET,
    MULTI_TERRAIN_HOLDOUT_STATE_SET,
)

# holdout_v1 在训练随机化范围内，但与 calibration_v1 的根状态和关节缩放均不重复。
# 该集合在正式多种子配置冻结后定义，禁止用于预算或超参数调整。
_HOLDOUT_ROOT_OFFSETS = (
    (0.04, -0.03, 0.00, 0.015, 0.025, -0.045, 0.08, -0.12, 0.04, 0.06, -0.09, 0.11),
    (-0.04, 0.03, 0.00, -0.015, -0.025, 0.045, -0.08, 0.12, -0.04, -0.06, 0.09, -0.11),
    (0.07, 0.02, 0.00, -0.04, 0.01, 0.075, -0.18, -0.04, 0.06, 0.14, -0.05, -0.13),
    (-0.07, -0.02, 0.00, 0.04, -0.01, -0.075, 0.18, 0.04, -0.06, -0.14, 0.05, 0.13),
    (0.01, 0.08, 0.00, 0.025, -0.04, 0.02, -0.06, 0.20, 0.02, -0.10, 0.16, -0.04),
    (-0.01, -0.08, 0.00, -0.025, 0.04, -0.02, 0.06, -0.20, -0.02, 0.10, -0.16, 0.04),
    (0.09, -0.06, 0.00, -0.035, -0.03, 0.09, 0.22, 0.07, -0.08, 0.18, 0.12, 0.17),
    (-0.09, 0.06, 0.00, 0.035, 0.03, -0.09, -0.22, -0.07, 0.08, -0.18, -0.12, -0.17),
)
_HOLDOUT_JOINT_SCALES = (0.93, 1.07, 0.96, 1.04, 0.99, 1.01, 0.92, 1.08)

_STATE_SET_TABLES = {
    CALIBRATION_STATE_SET: (_CALIBRATION_ROOT_OFFSETS, _CALIBRATION_JOINT_SCALES),
    HOLDOUT_STATE_SET: (_HOLDOUT_ROOT_OFFSETS, _HOLDOUT_JOINT_SCALES),
    MULTI_TERRAIN_CALIBRATION_STATE_SET: (
        _CALIBRATION_ROOT_OFFSETS,
        _CALIBRATION_JOINT_SCALES,
    ),
    MULTI_TERRAIN_HOLDOUT_STATE_SET: (
        _HOLDOUT_ROOT_OFFSETS,
        _HOLDOUT_JOINT_SCALES,
    ),
}


def evaluation_state_definition(
    state_set: str,
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
    """返回不可变状态表。"""

    try:
        return _STATE_SET_TABLES[state_set]
    except KeyError as error:
        raise ValueError(f"未知评估状态集：{state_set}；允许值={tuple(_STATE_SET_TABLES)}") from error


def evaluation_state_count(state_set: str) -> int:
    root_offsets, _ = evaluation_state_definition(state_set)
    return len(root_offsets)


def evaluation_state_definition_sha256(state_set: str) -> str:
    """返回只由状态数值和顺序决定的稳定哈希。"""

    definition = evaluation_state_definition(state_set)
    return sha256(json.dumps(definition, separators=(",", ":")).encode("utf-8")).hexdigest()


__all__ = [
    "CALIBRATION_STATE_SET",
    "EVALUATION_STATE_SETS",
    "HOLDOUT_STATE_SET",
    "MULTI_TERRAIN_CALIBRATION_STATE_SET",
    "MULTI_TERRAIN_EVALUATION_STATE_SETS",
    "MULTI_TERRAIN_HOLDOUT_STATE_SET",
    "evaluation_state_count",
    "evaluation_state_definition",
    "evaluation_state_definition_sha256",
]
