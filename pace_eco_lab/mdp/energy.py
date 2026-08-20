"""PACE 策略步能耗的纯数值实现。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pace_eco_lab.constants import ELECTRICAL_COEFFICIENT, GRAVITY_M_S2


@dataclass(frozen=True)
class EnergyComponents:
    """每个环境的功率或能量三分量。"""

    electrical: torch.Tensor
    mechanical: torch.Tensor
    potential: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return self.electrical + self.mechanical + self.potential

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "electrical": self.electrical,
            "mechanical": self.mechanical,
            "potential": self.potential,
        }


def compute_energy_components(
    applied_torque: torch.Tensor,
    joint_velocity: torch.Tensor,
    body_mass: torch.Tensor,
    body_vertical_velocity_world: torch.Tensor,
    *,
    electrical_coefficient: float = ELECTRICAL_COEFFICIENT,
    gravity_m_s2: float = GRAVITY_M_S2,
    include_potential: bool = True,
) -> EnergyComponents:
    """计算一个物理子步的瞬时功率，单位 W。

    机械功先对全部关节求和，再对总和截断到非负，严禁逐关节截断。
    """

    if applied_torque.shape != joint_velocity.shape:
        raise ValueError("力矩与关节速度形状必须一致。")
    if applied_torque.device != joint_velocity.device:
        raise ValueError("力矩与关节速度必须位于同一设备。")
    if applied_torque.device != body_vertical_velocity_world.device:
        raise ValueError("关节量与刚体竖直速度必须位于同一设备。")
    if body_mass.shape != body_vertical_velocity_world.shape:
        try:
            body_mass = torch.broadcast_to(body_mass, body_vertical_velocity_world.shape)
        except RuntimeError as exc:
            raise ValueError("刚体质量与竖直速度形状不可广播。") from exc
    if applied_torque.ndim < 2 or body_vertical_velocity_world.ndim < 2:
        raise ValueError("输入至少应为 [环境, 关节/刚体] 二维张量。")

    # Isaac Sim 可能把静态 default_mass 元数据保留在 CPU，而动态状态位于 CUDA。
    # 质量不参与梯度；允许把它对齐到动态状态，避免跨设备相乘。
    body_mass = body_mass.to(
        device=body_vertical_velocity_world.device,
        dtype=body_vertical_velocity_world.dtype,
    )
    electrical = electrical_coefficient * torch.sum(applied_torque.square(), dim=-1)
    signed_mechanical = torch.sum(applied_torque * joint_velocity, dim=-1)
    mechanical = torch.clamp_min(signed_mechanical, 0.0)
    if include_potential:
        potential = -gravity_m_s2 * torch.sum(body_mass * body_vertical_velocity_world, dim=-1)
    else:
        potential = torch.zeros_like(electrical)
    return EnergyComponents(electrical=electrical, mechanical=mechanical, potential=potential)


def integrate_energy_components(
    components: list[EnergyComponents] | tuple[EnergyComponents, ...],
    physics_dt_s: float,
) -> EnergyComponents:
    """把物理子步功率积分成一个策略步能量，单位 J。"""

    if physics_dt_s <= 0.0:
        raise ValueError("physics_dt_s 必须为正数。")
    if not components:
        raise ValueError("至少需要一个物理子步。")
    return EnergyComponents(
        electrical=torch.stack([item.electrical for item in components], dim=0).sum(dim=0) * physics_dt_s,
        mechanical=torch.stack([item.mechanical for item in components], dim=0).sum(dim=0) * physics_dt_s,
        potential=torch.stack([item.potential for item in components], dim=0).sum(dim=0) * physics_dt_s,
    )
