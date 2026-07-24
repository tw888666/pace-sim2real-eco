# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


class foot_touchdown_velocity(ManagerTermBase):
    """PACE foot-touchdown (FTD) speed over a short policy-step history."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        history_length = int(cfg.params.get("history_length", 3))
        if history_length <= 0:
            raise ValueError("history_length must be positive")
        num_feet = len(asset_cfg.body_ids)
        self._history = torch.zeros(
            env.num_envs,
            num_feet,
            history_length,
            device=env.device,
        )

    def reset(self, env_ids=None):
        if env_ids is None:
            self._history.zero_()
        else:
            self._history[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        history_length: int = 3,
    ) -> torch.Tensor:
        del history_length  # shape was fixed during term construction
        asset: Articulation = env.scene[asset_cfg.name]
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        foot_speed = torch.linalg.vector_norm(asset.data.body_com_lin_vel_w[:, asset_cfg.body_ids, :], dim=-1)
        self._history = torch.roll(self._history, shifts=-1, dims=-1)
        self._history[..., -1] = foot_speed
        touchdown = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
        return torch.sum(torch.max(self._history, dim=-1).values * touchdown, dim=-1)


def pace_collision_indicator(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """PACE binary collision term for joint limits or undesired body contact."""
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    body_collision = torch.linalg.vector_norm(contact_force, dim=-1).amax(dim=(1, 2)) > threshold

    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    joint_limit_collision = ((joint_pos < limits[..., 0]) | (joint_pos > limits[..., 1])).any(dim=-1)
    return (body_collision | joint_limit_collision).float()
