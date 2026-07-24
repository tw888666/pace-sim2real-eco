import torch

from pace_sim2real.energy import EnergyAccumulator, compute_power_components


def test_pace_power_equations_and_regeneration() -> None:
    torque = torch.tensor([[2.0, -1.0], [1.0, 1.0]])
    velocity = torch.tensor([[3.0, 1.0], [-2.0, -1.0]])
    # PhysX may return masses on a different device/precision than body state.
    mass = torch.tensor([[2.0, 3.0], [2.0, 3.0]], dtype=torch.float64)
    com_velocity = torch.zeros(2, 2, 3)
    com_velocity[0, :, 2] = torch.tensor([1.0, -0.5])

    power = compute_power_components(
        torque,
        velocity,
        mass,
        com_velocity,
        electrical_coefficient=torch.tensor([0.1, 0.2]),
        regeneration_coefficient=0.5,
        gravity=10.0,
    )

    torch.testing.assert_close(power.electrical, torch.tensor([0.6, 0.3]))
    torch.testing.assert_close(power.mechanical, torch.tensor([5.0, -1.5]))
    torch.testing.assert_close(power.potential, torch.tensor([5.0, 0.0]))
    torch.testing.assert_close(power.total, torch.tensor([10.6, -1.2]))
    assert power.potential.dtype == com_velocity.dtype


def test_accumulator_integrates_every_physics_substep() -> None:
    accumulator = EnergyAccumulator(1, device="cpu")
    accumulator.begin_control_step()
    power = compute_power_components(
        torch.tensor([[1.0]]),
        torch.tensor([[2.0]]),
        torch.tensor([[1.0]]),
        torch.zeros(1, 1, 3),
        electrical_coefficient=0.5,
    )

    for _ in range(8):
        accumulator.accumulate(power, physics_dt=0.0025)

    assert accumulator.physics_substeps == 8
    torch.testing.assert_close(accumulator.control_step["electrical"], torch.tensor([0.01]))
    torch.testing.assert_close(accumulator.control_step["mechanical"], torch.tensor([0.04]))
    torch.testing.assert_close(accumulator.control_step["total"], torch.tensor([0.05]))
    torch.testing.assert_close(accumulator.episode["total"], torch.tensor([0.05]))

    accumulator.reset(torch.tensor([0]))
    torch.testing.assert_close(accumulator.episode["total"], torch.zeros(1))
