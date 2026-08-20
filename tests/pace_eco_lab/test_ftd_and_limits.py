from __future__ import annotations

import pytest
import torch

from pace_eco_lab.mdp.ftd import FootTouchdownHistory, touchdown_edges
from pace_eco_lab.mdp.joint_limits import joint_limit_collision_indicator, safe_joint_position_targets


def test_touchdown_edge_detection():
    previous = torch.tensor([[False, True, False, True]])
    current = torch.tensor([[True, True, False, False]])
    expected = torch.tensor([[True, False, False, False]])
    torch.testing.assert_close(touchdown_edges(previous, current), expected)


def test_three_step_ftd_history_uses_max_speed_only_on_touchdown():
    history = FootTouchdownHistory.create(num_envs=1)
    history.push(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    history.push(torch.tensor([[2.0, 1.0, 5.0, 2.0]]))
    history.push(torch.tensor([[0.5, 6.0, 1.0, 3.0]]))
    penalty = history.penalty(torch.tensor([[True, False, True, False]]))
    assert penalty.item() == pytest.approx(2.0 + 5.0)


def test_ftd_full_reset_clears_cursor_and_fill_state():
    history = FootTouchdownHistory.create(num_envs=1, num_feet=1, history_length=3)
    history.push(torch.tensor([[2.0]]))
    history.push(torch.tensor([[3.0]]))
    history.reset()
    assert history.cursor == 0
    assert history.filled == 0
    history.push(torch.tensor([[1.0]]))
    assert history.penalty(torch.tensor([[True]])).item() == pytest.approx(1.0)


def test_safe_limit_target_fades_to_zero_outward_error_at_hard_limit():
    target = torch.tensor([[2.0, -2.0]])
    current = torch.tensor([[1.0, -1.0]])
    soft = torch.tensor([[[-0.8, 0.8], [-0.8, 0.8]]])
    hard = torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]])
    safe = safe_joint_position_targets(target, current, soft, hard)
    torch.testing.assert_close(safe, current)


def test_safe_limit_preserves_motion_away_and_soft_boundary_target():
    soft = torch.tensor([[[-0.8, 0.8], [-0.8, 0.8]]])
    hard = torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]])
    target = torch.tensor([[-0.2, 2.0]])
    current = torch.tensor([[0.95, 0.8]])
    safe = safe_joint_position_targets(target, current, soft, hard)
    assert safe[0, 0].item() == pytest.approx(-0.2)
    assert safe[0, 1].item() == pytest.approx(2.0)


def test_joint_limit_collision_indicator():
    limits = torch.tensor([[[-1.0, 1.0], [-2.0, 2.0]]])
    positions = torch.tensor([[0.0, 1.9995]])
    assert joint_limit_collision_indicator(positions, limits).item()
