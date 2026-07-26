import math

import torch

from pace_sim2real.dual import (
    evaluate_trajectory_predictions,
    regression_metrics,
    undiscounted_return_to_go,
    validate_trajectory_metrics,
)


def test_regression_bias_names_are_not_mae() -> None:
    prediction = torch.tensor([2.0, 0.0])
    target = torch.tensor([1.0, 1.0])
    metrics = regression_metrics(prediction, target)
    assert metrics.signed_bias == 0.0
    assert metrics.absolute_mean_bias == 0.0
    assert metrics.mae == 1.0
    assert metrics.rmse == 1.0


def test_undiscounted_return_to_go_masks_padding_and_keeps_terminal_barrier_once() -> None:
    costs = torch.tensor([[0.1, 0.2, 1.0, 99.0], [0.5, 0.5, 0.5, 0.5]])
    valid = torch.tensor([[True, True, True, False], [True, True, True, True]])
    target = undiscounted_return_to_go(costs, valid)
    torch.testing.assert_close(target[0], torch.tensor([1.3, 1.2, 1.0, 0.0]))
    torch.testing.assert_close(target[1], torch.tensor([2.0, 1.5, 1.0, 0.5]))


def test_complete_trajectory_metrics_are_exact_for_perfect_predictions() -> None:
    costs = torch.tensor([[0.2, 0.3, 0.0], [0.1, 1.15, 0.0]])
    valid = torch.tensor([[True, True, False], [True, True, False]])
    target = undiscounted_return_to_go(costs, valid)
    remaining_time = torch.tensor([[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]])
    success = torch.tensor([True, False])
    payload = evaluate_trajectory_predictions(
        target,
        costs,
        valid,
        remaining_time,
        success,
        num_time_bins=2,
    )
    validate_trajectory_metrics(payload)

    assert payload["initial"]["signed_bias"] == 0.0
    assert payload["initial"]["absolute_mean_bias"] == 0.0
    assert payload["initial"]["mae"] == 0.0
    assert payload["trajectory"]["episode_equal_rmse"] == 0.0
    assert math.isclose(payload["trajectory"]["pooled"]["explained_variance"], 1.0)
    assert payload["success"]["num_episodes"] == 1
    assert payload["failure"]["num_episodes"] == 1
