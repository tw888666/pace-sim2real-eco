from __future__ import annotations

import math

import pytest
import torch

from scripts.pace_eco.eval_metrics import CoordinationAccumulator


def _update_state(
    accumulator: CoordinationAccumulator,
    *,
    lateral: float,
    vertical: float,
    roll_pitch: tuple[float, float],
    contact: tuple[bool, bool, bool, bool],
    foot_speed: tuple[float, float, float, float],
    touchdown: tuple[bool, bool, bool, bool],
) -> None:
    accumulator.update_state(
        lateral_velocity=torch.tensor([lateral, 99.0]),
        vertical_velocity=torch.tensor([vertical, 99.0]),
        roll_pitch_angular_velocity=torch.tensor([roll_pitch, (99.0, 99.0)]),
        foot_contact=torch.tensor([contact, (True, True, True, True)]),
        foot_speed=torch.tensor([foot_speed, (99.0, 99.0, 99.0, 99.0)]),
        touchdown=torch.tensor([touchdown, (True, True, True, True)]),
        steady_mask=torch.tensor([True, False]),
    )


def test_coordination_accumulator_computes_episode_metrics():
    accumulator = CoordinationAccumulator.create(num_envs=2, action_dim=2)

    _update_state(
        accumulator,
        lateral=1.0,
        vertical=2.0,
        roll_pitch=(3.0, 4.0),
        contact=(True, False, False, True),
        foot_speed=(1.0, 2.0, 3.0, 4.0),
        touchdown=(True, False, False, False),
    )
    accumulator.update_action(torch.tensor([[0.0, 0.0], [9.0, 9.0]]), torch.tensor([True, False]))
    _update_state(
        accumulator,
        lateral=2.0,
        vertical=0.0,
        roll_pitch=(0.0, 0.0),
        contact=(False, True, True, False),
        foot_speed=(2.0, 1.0, 5.0, 2.0),
        touchdown=(False, True, False, False),
    )
    accumulator.update_action(torch.tensor([[2.0, 0.0], [8.0, 8.0]]), torch.tensor([True, False]))
    _update_state(
        accumulator,
        lateral=2.0,
        vertical=1.0,
        roll_pitch=(0.0, 0.0),
        contact=(True, False, True, False),
        foot_speed=(0.5, 6.0, 1.0, 3.0),
        touchdown=(False, False, True, False),
    )
    accumulator.update_action(torch.tensor([[2.0, 2.0], [7.0, 7.0]]), torch.tensor([True, False]))

    row = accumulator.episode_metrics(0, step_dt=0.5)
    assert row["协调性稳态样本数"] == 3
    assert row["协调性稳态时长_s"] == pytest.approx(1.5)
    assert row["机身横向速度RMS_m_s"] == pytest.approx(math.sqrt(3.0))
    assert row["机身垂向速度RMS_m_s"] == pytest.approx(math.sqrt(5.0 / 3.0))
    assert row["机身横滚俯仰角速度RMS_rad_s"] == pytest.approx(math.sqrt(25.0 / 3.0))
    assert row["动作变化RMS_归一化动作"] == pytest.approx(math.sqrt(2.0))
    assert row["三步窗触地足速均值_m_s"] == pytest.approx(8.0 / 3.0)
    assert row["触地事件数"] == 3
    assert row["左前足支撑相占比"] == pytest.approx(2.0 / 3.0)
    assert row["右前足支撑相占比"] == pytest.approx(1.0 / 3.0)
    assert row["左后足支撑相占比"] == pytest.approx(2.0 / 3.0)
    assert row["右后足支撑相占比"] == pytest.approx(1.0 / 3.0)
    assert row["支撑相占比极差"] == pytest.approx(1.0 / 3.0)
    assert row["左前足落足频率_Hz"] == pytest.approx(2.0 / 3.0)
    assert row["右前足落足频率_Hz"] == pytest.approx(2.0 / 3.0)
    assert row["左后足落足频率_Hz"] == pytest.approx(2.0 / 3.0)
    assert row["右后足落足频率_Hz"] == pytest.approx(0.0)
    assert row["落足频率变异系数"] == pytest.approx(1.0 / math.sqrt(3.0))
    assert row["对角足接触不同步率"] == pytest.approx(1.0 / 3.0)


def test_coordination_accumulator_returns_empty_metrics_without_steady_samples():
    accumulator = CoordinationAccumulator.create(num_envs=1, action_dim=2)
    row = accumulator.episode_metrics(0, step_dt=0.02)
    assert row["协调性稳态样本数"] == 0
    assert row["协调性稳态时长_s"] == 0.0
    assert all(row[field] is None for field in accumulator.SUMMARY_FIELDS)


def test_coordination_accumulator_reset_is_per_environment():
    accumulator = CoordinationAccumulator.create(num_envs=2, action_dim=2)
    accumulator.update_state(
        lateral_velocity=torch.tensor([1.0, 2.0]),
        vertical_velocity=torch.tensor([1.0, 2.0]),
        roll_pitch_angular_velocity=torch.ones(2, 2),
        foot_contact=torch.ones(2, 4, dtype=torch.bool),
        foot_speed=torch.ones(2, 4),
        touchdown=torch.ones(2, 4, dtype=torch.bool),
        steady_mask=torch.ones(2, dtype=torch.bool),
    )
    accumulator.update_action(torch.ones(2, 2), torch.ones(2, dtype=torch.bool))
    accumulator.reset([0])

    assert accumulator.episode_metrics(0, step_dt=0.02)["协调性稳态样本数"] == 0
    assert accumulator.episode_metrics(1, step_dt=0.02)["协调性稳态样本数"] == 1
    assert not accumulator.has_previous_action[0]
    assert accumulator.has_previous_action[1]
    assert torch.count_nonzero(accumulator.foot_speed_history[:, 0]) == 0
    assert torch.count_nonzero(accumulator.foot_speed_history[:, 1]) > 0


def test_coordination_accumulator_rejects_wrong_shapes():
    accumulator = CoordinationAccumulator.create(num_envs=1, action_dim=2)
    with pytest.raises(ValueError, match="形状"):
        accumulator.update_state(
            lateral_velocity=torch.zeros(1),
            vertical_velocity=torch.zeros(1),
            roll_pitch_angular_velocity=torch.zeros(1, 3),
            foot_contact=torch.zeros(1, 4, dtype=torch.bool),
            foot_speed=torch.zeros(1, 4),
            touchdown=torch.zeros(1, 4, dtype=torch.bool),
            steady_mask=torch.ones(1, dtype=torch.bool),
        )
