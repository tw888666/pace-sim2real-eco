"""PACE power equations and physics-step energy accumulation.

The equations implemented here are Eqs. (12)--(14) from the PACE paper.  This
module deliberately has no Isaac Sim imports so the numerical semantics can be
unit-tested on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PowerComponents:
    """Per-environment power components in watts."""

    electrical: torch.Tensor
    mechanical: torch.Tensor
    potential: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return self.electrical + self.mechanical + self.potential


def compute_power_components(
    applied_torque: torch.Tensor,
    joint_velocity: torch.Tensor,
    body_mass: torch.Tensor,
    body_com_linear_velocity_w: torch.Tensor,
    *,
    electrical_coefficient: float | torch.Tensor,
    regeneration_coefficient: float = 0.0,
    gravity: float = 9.81,
) -> PowerComponents:
    """Evaluate PACE electrical, mechanical, and potential power.

    Args:
        applied_torque: Delayed and clipped actuator torques, shape ``(N, J)``.
        joint_velocity: Simulated joint velocities, shape ``(N, J)``.
        body_mass: Rigid-body masses, shape ``(N, B)`` or ``(1, B)``.
        body_com_linear_velocity_w: World-frame body CoM linear velocities,
            shape ``(N, B, 3)``.
        electrical_coefficient: ``R / (r^2 k_i^2)`` for each joint, or a scalar.
        regeneration_coefficient: Fraction of negative mechanical power retained.
        gravity: Positive gravitational acceleration magnitude.

    Returns:
        The three power components in watts for every environment.
    """
    if applied_torque.shape != joint_velocity.shape or applied_torque.ndim != 2:
        raise ValueError(
            "applied_torque and joint_velocity must have identical (num_envs, num_joints) shapes; "
            f"got {tuple(applied_torque.shape)} and {tuple(joint_velocity.shape)}"
        )
    if body_com_linear_velocity_w.ndim != 3 or body_com_linear_velocity_w.shape[-1] != 3:
        raise ValueError("body_com_linear_velocity_w must have shape (num_envs, num_bodies, 3)")
    if body_mass.ndim != 2 or body_mass.shape[-1] != body_com_linear_velocity_w.shape[-2]:
        raise ValueError("body_mass must have shape (num_envs|1, num_bodies)")

    alpha = torch.as_tensor(
        electrical_coefficient,
        dtype=applied_torque.dtype,
        device=applied_torque.device,
    )
    if alpha.ndim > 1 or (alpha.ndim == 1 and alpha.numel() not in (1, applied_torque.shape[-1])):
        raise ValueError("electrical_coefficient must be scalar or have one value per joint")

    electrical = torch.sum(alpha * applied_torque.square(), dim=-1)
    shaft_power = torch.sum(applied_torque * joint_velocity, dim=-1)
    mechanical = torch.where(
        shaft_power >= 0.0,
        shaft_power,
        regeneration_coefficient * shaft_power,
    )
    vertical_com_velocity = body_com_linear_velocity_w[..., 2]
    potential = torch.sum(body_mass * gravity * vertical_com_velocity, dim=-1)
    return PowerComponents(electrical=electrical, mechanical=mechanical, potential=potential)


class EnergyAccumulator:
    """Accumulate energy at the physics frequency, not the policy frequency."""

    COMPONENT_NAMES = ("electrical", "mechanical", "potential", "total")

    def __init__(self, num_envs: int, *, device: str | torch.device, dtype: torch.dtype = torch.float32):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dtype = dtype
        self.control_step = self._zeros()
        self.episode = self._zeros()
        self.physics_substeps = 0

    def _zeros(self) -> dict[str, torch.Tensor]:
        return {name: torch.zeros(self.num_envs, device=self.device, dtype=self.dtype) for name in self.COMPONENT_NAMES}

    def begin_control_step(self) -> None:
        for value in self.control_step.values():
            value.zero_()
        self.physics_substeps = 0

    def accumulate(self, power: PowerComponents, physics_dt: float) -> None:
        if physics_dt <= 0.0:
            raise ValueError(f"physics_dt must be positive, got {physics_dt}")
        energies = {
            "electrical": power.electrical * physics_dt,
            "mechanical": power.mechanical * physics_dt,
            "potential": power.potential * physics_dt,
            "total": power.total * physics_dt,
        }
        for name, energy in energies.items():
            if energy.shape != (self.num_envs,):
                raise ValueError(f"{name} power must have shape ({self.num_envs},), got {tuple(energy.shape)}")
            self.control_step[name].add_(energy)
            self.episode[name].add_(energy)
        self.physics_substeps += 1

    def reset(self, env_ids: torch.Tensor | list[int] | slice) -> None:
        for value in self.episode.values():
            value[env_ids] = 0.0

    def control_step_snapshot(self) -> dict[str, torch.Tensor]:
        return {name: value.clone() for name, value in self.control_step.items()}

    def episode_snapshot(self) -> dict[str, torch.Tensor]:
        return {name: value.clone() for name, value in self.episode.items()}
