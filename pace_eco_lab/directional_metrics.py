"""方向穿越评估的纯张量几何与判定函数。"""

from __future__ import annotations

import torch


def normalized_direction(direction_w: torch.Tensor) -> torch.Tensor:
    """校验并归一化二维世界方向。"""

    if direction_w.shape[-1] != 2 or not torch.isfinite(direction_w).all():
        raise ValueError("方向张量末维必须为2且全部有限。")
    norm = torch.linalg.vector_norm(direction_w, dim=-1, keepdim=True)
    if torch.any(norm <= 0.0):
        raise ValueError("方向张量不能包含零向量。")
    return direction_w / norm


def directional_displacement_metrics(
    displacement_w_xy: torch.Tensor,
    direction_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回目标方向进度和有符号垂直方向偏移。"""

    direction = normalized_direction(direction_w)
    while direction.ndim < displacement_w_xy.ndim:
        direction = direction.unsqueeze(0)
    cross_direction = torch.stack((-direction[..., 1], direction[..., 0]), dim=-1)
    progress = torch.sum(displacement_w_xy * direction, dim=-1)
    cross_track = torch.sum(displacement_w_xy * cross_direction, dim=-1)
    return progress, cross_track


def directional_success(
    *,
    completed_20s: torch.Tensor,
    illegal_termination: torch.Tensor,
    directional_progress_m: torch.Tensor,
    max_cross_track_m: torch.Tensor,
    minimum_progress_m: float,
    maximum_cross_track_m: float,
) -> torch.Tensor:
    """按冻结阈值返回方向穿越成功掩码。"""

    return (
        completed_20s.bool()
        & ~illegal_termination.bool()
        & (directional_progress_m >= float(minimum_progress_m))
        & (max_cross_track_m <= float(maximum_cross_track_m))
    )


__all__ = [
    "directional_displacement_metrics",
    "directional_success",
    "normalized_direction",
]
