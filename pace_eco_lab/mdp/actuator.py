"""PACE 显式 PD、电机饱和、编码器偏置与力矩延迟执行器。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.actuators import DCMotor
from isaaclab.actuators.actuator_pd_cfg import DCMotorCfg
from isaaclab.utils import DelayBuffer, configclass
from isaaclab.utils.types import ArticulationActions


class PaceDelayedPDActuator(DCMotor):
    """固定力矩延迟、编码器偏置一次应用和四象限饱和的显式 PD 执行器。"""

    cfg: "PaceDelayedPDActuatorCfg"

    def __init__(self, cfg: "PaceDelayedPDActuatorCfg", *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if cfg.saturation_effort is None or cfg.saturation_effort <= 0.0:
            raise ValueError("saturation_effort 必须为正。")
        if cfg.velocity_limit is None:
            raise ValueError("PACE 执行器必须配置 velocity_limit。")
        if cfg.min_delay < 0 or cfg.max_delay < cfg.min_delay:
            raise ValueError("执行器延迟范围非法。")
        self.encoder_bias = self._parse_joint_parameter(cfg.encoder_bias, 0.0)
        self.torques_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self._all_indices = torch.arange(self._num_envs, dtype=torch.long, device=self._device)
        if cfg.min_delay == cfg.max_delay:
            self.torques_delay_buffer.set_time_lag(cfg.max_delay, self._all_indices)

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)
        if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
            env_ids = self._all_indices
        if self.cfg.min_delay == self.cfg.max_delay:
            self.torques_delay_buffer.set_time_lag(self.cfg.max_delay, env_ids)
        else:
            delay = torch.randint(
                self.cfg.min_delay,
                self.cfg.max_delay + 1,
                (len(env_ids),),
                dtype=torch.long,
                device=self._device,
            )
            self.torques_delay_buffer.set_time_lag(delay, env_ids)
        self.torques_delay_buffer.reset(env_ids)

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        # 误差为 q_des - (q - bias)，偏置只出现一次；真实状态与观察完全不改。
        action = super().compute(control_action, joint_pos - self.encoder_bias, joint_vel)
        delayed_effort = self.torques_delay_buffer.compute(action.joint_efforts)
        action.joint_efforts = delayed_effort
        # Articulation.data.applied_torque 和 PACE 能耗必须反映延迟后真正写入 PhysX 的力矩。
        self.applied_effort.copy_(delayed_effort)
        return action


@configclass
class PaceDelayedPDActuatorCfg(DCMotorCfg):
    """PACE 执行器配置。"""

    class_type: type = PaceDelayedPDActuator
    encoder_bias: dict[str, float] | float = 0.0
    min_delay: int = 0
    max_delay: int = MISSING
