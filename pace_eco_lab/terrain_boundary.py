"""Terrain20s 四足真实地形块边界的纯张量审计。"""

from __future__ import annotations

import math

import torch

from pace_eco_lab.multi_terrain_protocol import FOOT_BOUNDARY_AUDIT_VERSION

# ANYbotics 官方 ANYmal D 简化 URDF 中，base 到 FOOT 刚体中心的四段固定平移
# 长度之和为约 1.135 m。旋转不改变各段长度，三角不等式因此给出与关节角无关的
# 严格上界；1.25 m 另留约 0.115 m 数值与资产转换余量。
ANYMAL_D_FOOT_CENTER_CHAIN_BOUND_M = (
    math.hypot(0.304, 0.109)
    + math.hypot(0.069, 0.006)
    + math.sqrt(0.1805**2 + 0.285**2)
    + math.sqrt(0.1**2 + 0.02225**2 + 0.39246**2)
)
FOOT_CENTER_MAX_BASE_DISTANCE_M = 1.25


def base_near_tile_edge(
    base_displacement_w: torch.Tensor,
    *,
    terrain_length_m: float,
    terrain_width_m: float,
    terrain_origin_x_m: float,
    foot_center_max_base_distance_m: float = FOOT_CENTER_MAX_BASE_DISTANCE_M,
) -> dict[str, torch.Tensor]:
    """返回机身边缘余量和是否必须读取四足真实位置。

    当机身中心到四条边的最小余量严格大于足端中心相对机身的运动学上界时，
    任一足端中心都不可能越界；只有其余边缘邻近状态才需要昂贵的真实足端读取。
    """

    if base_displacement_w.ndim != 2 or base_displacement_w.shape[-1] != 3:
        raise ValueError("机身相对 origin 位移必须为 (环境数, 3)。")
    if terrain_length_m <= 0.0 or terrain_width_m <= 0.0:
        raise ValueError("地形长度和宽度必须为正值。")
    if not 0.0 < terrain_origin_x_m < terrain_length_m:
        raise ValueError("地形出生 origin_x 必须严格位于块内部。")
    if foot_center_max_base_distance_m <= 0.0:
        raise ValueError("足端中心相对机身的运动学上界必须为正值。")
    smallest_half_extent = min(
        terrain_origin_x_m,
        terrain_length_m - terrain_origin_x_m,
        terrain_width_m / 2.0,
    )
    if foot_center_max_base_distance_m >= smallest_half_extent:
        raise ValueError("足端运动学上界必须小于地形各方向最小半径。")

    forward = base_displacement_w[:, 0]
    lateral = base_displacement_w[:, 1]
    back_edge = -float(terrain_origin_x_m)
    front_edge = float(terrain_length_m - terrain_origin_x_m)
    lateral_edge = float(terrain_width_m / 2.0)
    edge_margin = torch.stack(
        (
            forward - back_edge,
            front_edge - forward,
            lateral_edge - lateral.abs(),
        ),
        dim=-1,
    ).amin(dim=-1)
    audit_required = edge_margin <= float(foot_center_max_base_distance_m)
    return {
        "base_edge_margin_m": edge_margin,
        "audit_required": audit_required,
        "kinematically_certified_safe": ~audit_required,
    }


def contact_foot_tile_state(
    foot_positions_w: torch.Tensor,
    foot_forces_w: torch.Tensor,
    env_origins_w: torch.Tensor,
    *,
    terrain_length_m: float,
    terrain_width_m: float,
    terrain_origin_x_m: float,
    contact_force_threshold_n: float,
) -> dict[str, torch.Tensor]:
    """返回逐环境、逐足的接触状态、边缘余量和真实块越界状态。

    足端位置使用刚体中心。边缘余量为足端中心到当前 ``65 m × 60 m``
    地形块四条平面边缘的最小有符号距离；负值表示已经越过真实块边缘。
    只有接触力严格大于传感器阈值的足端才触发硬越界，摆动足越界单独预警。
    """

    if foot_positions_w.ndim != 3 or foot_positions_w.shape[-1] != 3:
        raise ValueError("足端位置必须为 (环境数, 足数, 3)。")
    if foot_forces_w.shape != foot_positions_w.shape:
        raise ValueError("足端接触力形状必须与足端位置一致。")
    if env_origins_w.shape != (foot_positions_w.shape[0], 3):
        raise ValueError("环境 origin 必须为 (环境数, 3)。")
    if terrain_length_m <= 0.0 or terrain_width_m <= 0.0:
        raise ValueError("地形长度和宽度必须为正值。")
    if not 0.0 < terrain_origin_x_m < terrain_length_m:
        raise ValueError("地形出生 origin_x 必须严格位于块内部。")
    if contact_force_threshold_n < 0.0:
        raise ValueError("接触力阈值不能为负值。")

    relative_xy = foot_positions_w[..., :2] - env_origins_w[:, None, :2]
    forward = relative_xy[..., 0]
    lateral = relative_xy[..., 1]
    back_edge = -float(terrain_origin_x_m)
    front_edge = float(terrain_length_m - terrain_origin_x_m)
    lateral_edge = float(terrain_width_m / 2.0)
    edge_margin = torch.stack(
        (
            forward - back_edge,
            front_edge - forward,
            lateral_edge - lateral.abs(),
        ),
        dim=-1,
    ).amin(dim=-1)
    contact = torch.linalg.vector_norm(foot_forces_w, dim=-1) > float(contact_force_threshold_n)
    outside = edge_margin < 0.0
    return {
        "relative_xy": relative_xy,
        "edge_margin_m": edge_margin,
        "contact": contact,
        "outside": outside,
        "contact_outside": contact & outside,
        "swing_outside": (~contact) & outside,
    }


__all__ = [
    "ANYMAL_D_FOOT_CENTER_CHAIN_BOUND_M",
    "FOOT_BOUNDARY_AUDIT_VERSION",
    "FOOT_CENTER_MAX_BASE_DISTANCE_M",
    "base_near_tile_edge",
    "contact_foot_tile_state",
]
