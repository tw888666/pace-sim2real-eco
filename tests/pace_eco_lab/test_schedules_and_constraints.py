from __future__ import annotations

import pytest
import torch

from pace_eco_lab.rl.constraint_storage import ConstraintRollout
from pace_eco_lab.rl.schedules import entropy_coefficient, penalty_coefficient


def test_penalty_half_life():
    assert penalty_coefficient(0) == 0.0
    assert penalty_coefficient(500) == pytest.approx(0.5)


def test_entropy_schedule_endpoints_and_turnover():
    assert entropy_coefficient(0) == pytest.approx(0.002, rel=1.0e-4)
    assert entropy_coefficient(2_000) == pytest.approx((0.002 + 0.0005) / 2)
    assert entropy_coefficient(3_000) == pytest.approx(0.0005100392763864273)


def test_cost_advantage_is_normalized_over_entire_rollout():
    storage = ConstraintRollout.create(3, 2)
    storage.add(torch.tensor([1.0, 2.0]), torch.tensor([False, False]))
    storage.add(torch.tensor([2.0, 4.0]), torch.tensor([False, True]))
    storage.add(
        torch.tensor([3.0, 6.0]),
        torch.tensor([True, False]),
        completed_normalized_cost=torch.tensor([0.9, 1.1]),
        completed_mask=torch.tensor([True, False]),
    )
    advantages = storage.compute_advantages(gamma=0.998, lam=0.81)
    assert advantages.shape == (3, 2, 1)
    assert advantages.mean().item() == pytest.approx(0.0, abs=1.0e-6)
    assert advantages.std(unbiased=False).item() == pytest.approx(1.0, rel=1.0e-5)
    assert storage.mean_completed_normalized_cost().item() == pytest.approx(0.9)
