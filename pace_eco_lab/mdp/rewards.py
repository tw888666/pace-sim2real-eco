"""PACE 任务奖励、FTD 和碰撞项。"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor

from pace_eco_lab.rl.schedules import penalty_coefficient

from .ftd import FootTouchdownHistory
from .joint_limits import joint_limit_collision_indicator


def pace_velocity_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """PACE 的平面线速度和偏航速度双指数核，并抵消 RewardManager 的 dt。"""

    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    linear_error = torch.sum(torch.square(command[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=-1)
    yaw_error = torch.square(command[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-linear_error / sigma**2) + torch.exp(-yaw_error / sigma**2)
    return reward / env.step_dt


def pace_directional_velocity_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """跟踪固定世界方向的二维目标速度，并抵消 RewardManager 的 dt。"""

    if sigma <= 0.0:
        raise ValueError("方向速度奖励 sigma 必须为正。")
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    if not hasattr(command_term, "direction_w") or not hasattr(command_term, "target_speed"):
        raise TypeError(f"command {command_name} 不是 DirectionCommand。")
    target_velocity_w = command_term.direction_w * command_term.target_speed
    error = asset.data.root_lin_vel_w[:, :2] - target_velocity_w
    return torch.exp(-torch.sum(error.square(), dim=-1) / sigma**2) / env.step_dt


def pace_yaw_rate_tracking(
    env: ManagerBasedRLEnv,
    sigma: float,
    target_yaw_rate: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """独立保留 PACE 的机身偏航角速度指数核。"""

    if sigma <= 0.0:
        raise ValueError("偏航角速度奖励 sigma 必须为正。")
    asset: Articulation = env.scene[asset_cfg.name]
    error = asset.data.root_ang_vel_b[:, 2] - float(target_yaw_rate)
    return torch.exp(-error.square() / sigma**2) / env.step_dt


def pace_collision_indicator(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    thigh_sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
    joint_limit_tolerance_rad: float = 1.0e-3,
) -> torch.Tensor:
    """关节硬限位或任一大腿接触时返回 1。"""

    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[thigh_sensor_cfg.name]
    joint_collision = joint_limit_collision_indicator(
        asset.data.joint_pos[:, asset_cfg.joint_ids],
        asset.data.joint_pos_limits[:, asset_cfg.joint_ids],
        tolerance_rad=joint_limit_tolerance_rad,
    )
    thigh_forces = sensor.data.net_forces_w_history[:, :, thigh_sensor_cfg.body_ids]
    thigh_contact = torch.any(
        torch.max(torch.linalg.vector_norm(thigh_forces, dim=-1), dim=1).values > contact_threshold,
        dim=-1,
    )
    return (joint_collision | thigh_contact).to(torch.float32) / env.step_dt


class PaceFootTouchdownPenalty(ManagerTermBase):
    """最近三个策略步的足端速度触地惩罚。"""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset_body_ids = cfg.params["asset_cfg"].body_ids
        self.sensor_body_ids = cfg.params["sensor_cfg"].body_ids
        self.history = FootTouchdownHistory.create(
            env.num_envs,
            num_feet=len(self.asset_body_ids),
            history_length=int(cfg.params.get("history_length", 3)),
            device=env.device,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.history.reset(env_ids)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        history_length: int = 3,
        half_life_iterations: float = 500.0,
    ) -> torch.Tensor:
        del asset_cfg, sensor_cfg, history_length
        foot_speed = torch.linalg.vector_norm(
            self.asset.data.body_lin_vel_w[:, self.asset_body_ids],
            dim=-1,
        )
        self.history.push(foot_speed)
        touchdown = self.sensor.compute_first_contact(env.step_dt)[:, self.sensor_body_ids]
        raw_penalty = self.history.penalty(touchdown)
        schedule = penalty_coefficient(
            int(getattr(env, "pace_learning_iteration", 0)),
            half_life_iterations,
        )
        return raw_penalty * schedule / env.step_dt


def scheduled_energy_reward(
    env: ManagerBasedRLEnv,
    half_life_iterations: float = 500.0,
) -> torch.Tensor:
    """固定权重基线使用的 PACE 能耗调度项。"""

    schedule = penalty_coefficient(
        int(getattr(env, "pace_learning_iteration", 0)),
        half_life_iterations,
    )
    return env.pace_energy_step * schedule / env.step_dt
