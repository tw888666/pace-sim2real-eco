"""PACE 软/硬关节限位安全目标。"""

from __future__ import annotations

import torch


def safe_joint_position_targets(
    target: torch.Tensor,
    current: torch.Tensor,
    soft_limits: torch.Tensor,
    hard_limits: torch.Tensor,
) -> torch.Tensor:
    """衰减朝硬限位外侧的 PD 目标，同时完整保留离开限位的目标。

    上限侧在 ``q_soft`` 到 ``q_hard`` 间线性插值：当前关节位于软限位时
    保持原目标，位于硬限位时把越界目标变为硬限位，使向外 PD 误差为零。
    下限侧使用完全对称的规则。
    """

    if target.shape != current.shape:
        raise ValueError("target 与 current 形状必须一致。")
    if soft_limits.shape != hard_limits.shape or soft_limits.shape[-1] != 2:
        raise ValueError("soft_limits 与 hard_limits 必须具有相同的 [..., 2] 形状。")
    if soft_limits.shape[:-1] != target.shape:
        try:
            soft_limits = torch.broadcast_to(soft_limits, target.shape + (2,))
            hard_limits = torch.broadcast_to(hard_limits, target.shape + (2,))
        except RuntimeError as exc:
            raise ValueError("限位形状与关节目标不可广播。") from exc

    hard_lower, hard_upper = hard_limits[..., 0], hard_limits[..., 1]
    soft_lower, soft_upper = soft_limits[..., 0], soft_limits[..., 1]
    if torch.any(soft_lower <= hard_lower) or torch.any(soft_upper >= hard_upper):
        raise ValueError("软限位必须严格位于硬限位内部。")

    result = target.clone()

    upper_band = (current >= soft_upper) & (current <= hard_upper) & (target > hard_upper)
    upper_alpha = ((current - soft_upper) / (hard_upper - soft_upper)).clamp(0.0, 1.0)
    upper_safe = target - upper_alpha * (target - hard_upper)
    result = torch.where(upper_band, upper_safe, result)
    result = torch.where((current > hard_upper) & (target > current), current, result)

    lower_band = (current <= soft_lower) & (current >= hard_lower) & (target < hard_lower)
    lower_alpha = ((soft_lower - current) / (soft_lower - hard_lower)).clamp(0.0, 1.0)
    lower_safe = target + lower_alpha * (hard_lower - target)
    result = torch.where(lower_band, lower_safe, result)
    result = torch.where((current < hard_lower) & (target < current), current, result)
    return result


def joint_limit_collision_indicator(
    joint_position: torch.Tensor,
    hard_limits: torch.Tensor,
    *,
    tolerance_rad: float = 1.0e-3,
) -> torch.Tensor:
    """检测任一关节到达硬限位的环境级指示量。"""

    lower = hard_limits[..., 0] + tolerance_rad
    upper = hard_limits[..., 1] - tolerance_rad
    return torch.any((joint_position <= lower) | (joint_position >= upper), dim=-1)
