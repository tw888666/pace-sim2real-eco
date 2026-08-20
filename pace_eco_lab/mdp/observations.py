"""PACE Critic 特权观察项。"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def ground_friction(
    env: ManagerBasedEnv,
    value: float,
) -> torch.Tensor:
    """返回固定地面动摩擦系数，形状为 ``[num_envs, 1]``。"""

    return torch.full((env.num_envs, 1), value, device=env.device)


def binary_foot_contacts(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """返回四足当前接触指示，不使用历史最大值。"""

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    return (torch.linalg.vector_norm(forces, dim=-1) > threshold).to(torch.float32)


def joint_observation_in_pace_order(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    *,
    velocity: bool,
) -> torch.Tensor:
    """按显式解析后的 PACE 关节名称顺序返回相对状态。"""

    asset: Articulation = env.scene[asset_cfg.name]
    if velocity:
        return asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]
    return asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
