"""预算标定和最终留出评估使用的确定性初始状态集合。"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

from pace_eco_lab.evaluation_states import (
    CALIBRATION_STATE_SET,
    EVALUATION_STATE_SETS,
    HOLDOUT_STATE_SET,
    evaluation_state_count,
    evaluation_state_definition,
    evaluation_state_definition_sha256,
)


def fixed_evaluation_state_count() -> int:
    """兼容旧入口：返回 calibration_v1 的状态数。"""

    return evaluation_state_count(CALIBRATION_STATE_SET)


def _state_indices(
    env_ids: torch.Tensor, state_set: str, state_index_offset: int = 0
) -> torch.Tensor:
    return torch.remainder(
        env_ids + int(state_index_offset), evaluation_state_count(state_set)
    ).long()


def reset_joints_training(
    env,
    env_ids: torch.Tensor,
    position_scale_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """随机缩放默认姿态并直接采样初始关节速度。"""

    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    joint_pos *= math_utils.sample_uniform(*position_scale_range, joint_pos.shape, asset.device)
    joint_vel += math_utils.sample_uniform(*velocity_range, joint_vel.shape, asset.device)
    position_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos.clamp_(position_limits[..., 0], position_limits[..., 1])
    velocity_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel.clamp_(-velocity_limits, velocity_limits)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def reset_root_state_from_evaluation_set(
    env,
    env_ids: torch.Tensor,
    state_set: str,
    state_index_offset: int = 0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """按环境编号确定性选取指定协议的根状态之一。"""

    asset: Articulation = env.scene[asset_cfg.name]
    default = asset.data.default_root_state[env_ids].clone()
    root_offsets, _ = evaluation_state_definition(state_set)
    table = torch.tensor(root_offsets, device=asset.device, dtype=default.dtype)
    offsets = table[_state_indices(env_ids, state_set, state_index_offset)]
    positions = default[:, :3] + env.scene.env_origins[env_ids] + offsets[:, :3]
    rotation = math_utils.quat_from_euler_xyz(offsets[:, 3], offsets[:, 4], offsets[:, 5])
    orientations = math_utils.quat_mul(default[:, 3:7], rotation)
    velocities = default[:, 7:13] + offsets[:, 6:12]
    asset.write_root_pose_to_sim(torch.cat((positions, orientations), dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_joints_from_evaluation_set(
    env,
    env_ids: torch.Tensor,
    state_set: str,
    state_index_offset: int = 0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """用与根状态相同的集合编号确定性缩放默认关节姿态。"""

    asset: Articulation = env.scene[asset_cfg.name]
    _, joint_scales = evaluation_state_definition(state_set)
    scales = torch.tensor(joint_scales, device=asset.device, dtype=asset.data.default_joint_pos.dtype)
    joint_pos = (
        asset.data.default_joint_pos[env_ids].clone()
        * scales[_state_indices(env_ids, state_set, state_index_offset), None]
    )
    limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos.clamp_(limits[..., 0], limits[..., 1])
    joint_vel = torch.zeros_like(asset.data.default_joint_vel[env_ids])
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def reset_root_state_from_fixed_set(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """兼容旧入口：使用 calibration_v1 根状态。"""

    reset_root_state_from_evaluation_set(
        env,
        env_ids,
        state_set=CALIBRATION_STATE_SET,
        asset_cfg=asset_cfg,
    )


def reset_joints_from_fixed_set(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """兼容旧入口：使用 calibration_v1 关节状态。"""

    reset_joints_from_evaluation_set(
        env,
        env_ids,
        state_set=CALIBRATION_STATE_SET,
        asset_cfg=asset_cfg,
    )


__all__ = [
    "CALIBRATION_STATE_SET",
    "EVALUATION_STATE_SETS",
    "HOLDOUT_STATE_SET",
    "evaluation_state_count",
    "evaluation_state_definition",
    "evaluation_state_definition_sha256",
    "fixed_evaluation_state_count",
    "reset_joints_training",
    "reset_joints_from_evaluation_set",
    "reset_joints_from_fixed_set",
    "reset_root_state_from_evaluation_set",
    "reset_root_state_from_fixed_set",
]
