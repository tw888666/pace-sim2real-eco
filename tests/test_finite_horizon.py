import torch

from pace_sim2real.utils.finite_horizon import normalized_time_to_go


def test_normalized_time_to_go_has_no_transition_off_by_one() -> None:
    episode_step = torch.tensor([0, 1, 999, 1000, 1001])
    remaining = normalized_time_to_go(episode_step, 1000)
    torch.testing.assert_close(
        remaining,
        torch.tensor([1.0, 0.999, 0.001, 0.0, 0.0]),
    )


def test_normalized_time_to_go_rejects_non_positive_horizon() -> None:
    try:
        normalized_time_to_go(torch.tensor([0]), 0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero-length horizon must be rejected")
