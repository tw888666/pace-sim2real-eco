from __future__ import annotations

import pytest
import torch

from pace_eco_lab.mdp.energy import (
    compute_energy_components,
    integrate_energy_components,
    velocity_normalization,
)


def test_paper_v2_velocity_normalization():
    target_velocity = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 4.0]])
    gamma_v = velocity_normalization(target_velocity)
    torch.testing.assert_close(gamma_v, torch.tensor([1.0, 0.5, 1.0 / 26.0]))


def test_velocity_normalization_rejects_non_planar_shape():
    with pytest.raises(ValueError, match="目标平面速度"):
        velocity_normalization(torch.zeros(2, 3))


def test_zero_torque_has_only_potential_power():
    torque = torch.zeros(2, 12)
    velocity = torch.ones_like(torque)
    mass = torch.ones(2, 3)
    vertical = torch.zeros_like(mass)
    power = compute_energy_components(torque, velocity, mass, vertical)
    torch.testing.assert_close(power.total, torch.zeros(2))


def test_mechanical_power_clamps_after_joint_sum():
    torque = torch.tensor([[2.0, -1.0]])
    velocity = torch.tensor([[1.0, 3.0]])
    zeros = torch.zeros(1, 1)
    power = compute_energy_components(torque, velocity, zeros, zeros)
    assert power.mechanical.item() == 0.0

    torque = torch.tensor([[2.0, -1.0]])
    velocity = torch.tensor([[3.0, 1.0]])
    power = compute_energy_components(torque, velocity, zeros, zeros)
    assert power.mechanical.item() == pytest.approx(5.0)


def test_electrical_power_and_joule_integration():
    torque = torch.ones(1, 12)
    velocity = torch.zeros_like(torque)
    zeros = torch.zeros(1, 1)
    substep = compute_energy_components(torque, velocity, zeros, zeros)
    energy = integrate_energy_components([substep] * 8, 0.0025)
    assert substep.electrical.item() == pytest.approx(0.0192 * 12)
    assert energy.electrical.item() == pytest.approx(0.0192 * 12 * 8 * 0.0025)


def test_potential_sign_and_constant_vertical_speed_identity():
    # 静态质量元数据可能来自不同数据类型；计算时必须跟随动态状态。
    mass = torch.tensor([[2.0, 3.0]], dtype=torch.float64)
    upward_speed = torch.full((1, 2), 0.4)
    torque = torch.zeros(1, 12)
    joint_velocity = torch.zeros_like(torque)
    power = compute_energy_components(torque, joint_velocity, mass, upward_speed)
    assert power.potential.dtype == upward_speed.dtype
    assert power.potential.item() < 0.0

    duration = 2.0
    dt = 0.0025
    steps = int(duration / dt)
    energy = integrate_energy_components([power] * steps, dt)
    potential_start_minus_end = -(2.0 + 3.0) * 9.81 * 0.4 * duration
    assert energy.potential.item() == pytest.approx(potential_start_minus_end, rel=1.0e-6)


def test_potential_can_be_disabled_only_for_ablation():
    zeros = torch.zeros(1, 1)
    power = compute_energy_components(
        torch.zeros(1, 12),
        torch.zeros(1, 12),
        torch.ones(1, 1),
        torch.ones(1, 1),
        include_potential=False,
    )
    torch.testing.assert_close(power.potential, zeros[:, 0])


def test_dynamic_energy_inputs_must_share_one_device():
    torque = torch.zeros(1, 12)
    velocity = torch.zeros_like(torque)
    mass = torch.ones(1, 1)
    vertical = torch.empty(1, 1, device="meta")
    with pytest.raises(ValueError, match="同一设备"):
        compute_energy_components(torque, velocity, mass, vertical)
