"""三步足端速度历史与触地惩罚。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def touchdown_edges(previous_contact: torch.Tensor, current_contact: torch.Tensor) -> torch.Tensor:
    """返回从未接触到接触的上升沿。"""

    if previous_contact.shape != current_contact.shape:
        raise ValueError("前后接触张量形状必须一致。")
    return current_contact.bool() & ~previous_contact.bool()


def touchdown_penalty(speed_history: torch.Tensor, touchdown: torch.Tensor) -> torch.Tensor:
    """按论文定义求各足最近历史最大速度并仅在触地时求和。

    Args:
        speed_history: ``[历史, 环境, 足]``。
        touchdown: ``[环境, 足]`` 的触地边沿。
    """

    if speed_history.ndim != 3 or touchdown.ndim != 2:
        raise ValueError("speed_history 应为 3 维，touchdown 应为 2 维。")
    if speed_history.shape[1:] != touchdown.shape:
        raise ValueError("历史中的环境/足维度与 touchdown 不一致。")
    max_speed = torch.max(speed_history, dim=0).values
    return torch.sum(max_speed * touchdown.to(max_speed.dtype), dim=-1)


@dataclass
class FootTouchdownHistory:
    """独立于模拟器的固定长度足速历史缓冲。"""

    history: torch.Tensor
    cursor: int = 0
    filled: int = 0

    @classmethod
    def create(
        cls,
        num_envs: int,
        num_feet: int = 4,
        history_length: int = 3,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "FootTouchdownHistory":
        if num_envs <= 0 or num_feet <= 0 or history_length <= 0:
            raise ValueError("环境数、足数和历史长度必须为正整数。")
        return cls(torch.zeros(history_length, num_envs, num_feet, device=device, dtype=dtype))

    def push(self, foot_speed: torch.Tensor) -> None:
        if foot_speed.shape != self.history.shape[1:]:
            raise ValueError("足速形状与缓冲不一致。")
        self.history[self.cursor].copy_(foot_speed)
        self.cursor = (self.cursor + 1) % self.history.shape[0]
        self.filled = min(self.filled + 1, self.history.shape[0])

    def penalty(self, touchdown: torch.Tensor) -> torch.Tensor:
        valid = self.history[: self.filled] if self.filled else self.history[:1]
        return touchdown_penalty(valid, touchdown)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.history[:, env_ids] = 0.0
        if isinstance(env_ids, slice) and env_ids == slice(None):
            self.cursor = 0
            self.filled = 0
