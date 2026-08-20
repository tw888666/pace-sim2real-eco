import torch

from pace_eco_lab.evaluation_snapshot import publish_eval_state


def test_eval_snapshot_is_disabled_by_default_path():
    extras = {}
    publish_eval_state(extras, torch.ones(1, 3), torch.ones(1, 3), enabled=False)
    assert extras == {}


def test_eval_snapshot_clones_pre_reset_state():
    position = torch.tensor([[1.0, 2.0, 3.0]])
    velocity = torch.tensor([[0.5, 0.0, 0.0]])
    extras = {}
    publish_eval_state(extras, position, velocity, enabled=True)
    position.zero_()
    velocity.zero_()
    assert torch.equal(extras["pace_eval_root_pos_w"], torch.tensor([[1.0, 2.0, 3.0]]))
    assert torch.equal(extras["pace_eval_root_lin_vel_b"], torch.tensor([[0.5, 0.0, 0.0]]))
