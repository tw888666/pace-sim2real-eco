"""固定世界方向、机器人坐标表达的 DirectionCommand。"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, yaw_quat


class DirectionCommand(CommandTerm):
    """输出 ``[direction_b_x, direction_b_y, target_speed]``。"""

    cfg: "DirectionCommandCfg"

    def __init__(self, cfg: "DirectionCommandCfg", env):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        raw_direction = torch.tensor(cfg.direction_w, dtype=torch.float32, device=self.device)
        if raw_direction.shape != (2,) or not torch.isfinite(raw_direction).all():
            raise ValueError("世界目标方向必须是两个有限数。")
        norm = torch.linalg.vector_norm(raw_direction)
        if norm.item() <= 0.0:
            raise ValueError("世界目标方向不能是零向量。")
        if cfg.target_speed <= 0.0:
            raise ValueError("目标速度必须为正。")
        normalized = raw_direction / norm
        self.direction_w = normalized.repeat(self.num_envs, 1)
        self.direction_b = torch.zeros_like(self.direction_w)
        self.target_speed = torch.full(
            (self.num_envs, 1),
            float(cfg.target_speed),
            dtype=self.direction_w.dtype,
            device=self.device,
        )
        self._update_command()

    @property
    def command(self) -> torch.Tensor:
        return torch.cat((self.direction_b, self.target_speed), dim=-1)

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        del env_ids

    def _update_command(self) -> None:
        direction_w_3d = torch.cat(
            (self.direction_w, torch.zeros(self.num_envs, 1, device=self.device)),
            dim=-1,
        )
        direction_b_3d = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w), direction_w_3d)
        self.direction_b.copy_(direction_b_3d[:, :2])

    def _update_metrics(self) -> None:
        return

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        del debug_vis
        raise NotImplementedError


@configclass
class DirectionCommandCfg(CommandTermCfg):
    class_type: type = DirectionCommand
    asset_name: str = "robot"
    direction_w: tuple[float, float] = (1.0, 0.0)
    target_speed: float = 1.0


__all__ = ["DirectionCommand", "DirectionCommandCfg"]
