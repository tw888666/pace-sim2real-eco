"""Cost-value calibration metrics for complete finite-horizon trajectories."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class RegressionMetrics:
    """Scalar regression diagnostics with unambiguous bias definitions."""

    num_samples: int
    signed_bias: float
    absolute_mean_bias: float
    mae: float
    rmse: float
    explained_variance: float
    target_mean: float
    target_std: float
    prediction_mean: float
    prediction_std: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def regression_metrics(prediction: torch.Tensor, target: torch.Tensor) -> RegressionMetrics:
    """Compute finite regression metrics using ``prediction - target`` residuals."""
    prediction = prediction.detach().float().reshape(-1)
    target = target.detach().float().reshape(-1)
    if prediction.shape != target.shape:
        raise ValueError(f"prediction and target shapes differ: {prediction.shape} != {target.shape}")
    if prediction.numel() == 0:
        raise ValueError("regression metrics require at least one sample")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("regression metrics received NaN or infinity")

    residual = prediction - target
    target_variance = torch.var(target, unbiased=False)
    residual_variance = torch.var(residual, unbiased=False)
    explained_variance = 1.0 - residual_variance / (target_variance + 1.0e-8)
    signed_bias = residual.mean()
    return RegressionMetrics(
        num_samples=prediction.numel(),
        signed_bias=float(signed_bias.item()),
        absolute_mean_bias=float(torch.abs(signed_bias).item()),
        mae=float(torch.mean(torch.abs(residual)).item()),
        rmse=float(torch.sqrt(torch.mean(residual.square())).item()),
        explained_variance=float(explained_variance.item()),
        target_mean=float(target.mean().item()),
        target_std=float(torch.sqrt(target_variance).item()),
        prediction_mean=float(prediction.mean().item()),
        prediction_std=float(torch.std(prediction, unbiased=False).item()),
    )


def undiscounted_return_to_go(costs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Compute exact full-episode return-to-go, including a terminal barrier once."""
    if costs.ndim != 2 or valid.shape != costs.shape:
        raise ValueError("costs and valid must have the same [episode, time] shape")
    if valid.dtype != torch.bool:
        raise TypeError("valid must be a boolean tensor")
    masked_costs = torch.where(valid, costs, torch.zeros_like(costs))
    return torch.flip(torch.cumsum(torch.flip(masked_costs, dims=(1,)), dim=1), dims=(1,))


def _episode_equal_rmse(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> float:
    squared_error = torch.where(valid, (prediction - target).square(), torch.zeros_like(target))
    lengths = valid.sum(dim=1)
    if (lengths == 0).any():
        raise ValueError("every episode must contain at least one valid transition")
    episode_mse = squared_error.sum(dim=1) / lengths
    return float(torch.sqrt(episode_mse.mean()).item())


def _group_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    episode_selector: torch.Tensor,
) -> dict | None:
    if episode_selector.dtype != torch.bool or episode_selector.ndim != 1:
        raise TypeError("episode selector must be a one-dimensional boolean tensor")
    if not episode_selector.any():
        return None
    group_prediction = prediction[episode_selector]
    group_target = target[episode_selector]
    group_valid = valid[episode_selector]
    return {
        "num_episodes": int(episode_selector.sum().item()),
        "num_transitions": int(group_valid.sum().item()),
        "pooled": regression_metrics(group_prediction[group_valid], group_target[group_valid]).to_dict(),
        "episode_equal_rmse": _episode_equal_rmse(group_prediction, group_target, group_valid),
    }


def evaluate_trajectory_predictions(
    prediction: torch.Tensor,
    costs: torch.Tensor,
    valid: torch.Tensor,
    remaining_time: torch.Tensor,
    success: torch.Tensor,
    *,
    num_time_bins: int = 10,
) -> dict:
    """Evaluate initial and all-timestep cost values on complete trajectories.

    All tensors are episode-major. Padded transitions are excluded by ``valid``.
    The target is the exact undiscounted return-to-go rather than a GAE or
    bootstrapped rollout target.
    """
    if prediction.ndim != 2:
        raise ValueError("prediction must have [episode, time] shape")
    if costs.shape != prediction.shape or valid.shape != prediction.shape or remaining_time.shape != prediction.shape:
        raise ValueError("prediction, costs, valid, and remaining_time must share one shape")
    if success.shape != prediction.shape[:1] or success.dtype != torch.bool:
        raise ValueError("success must be a boolean vector with one value per episode")
    if num_time_bins <= 0:
        raise ValueError("num_time_bins must be positive")
    if not torch.isfinite(remaining_time[valid]).all():
        raise ValueError("remaining_time contains NaN or infinity")

    target = undiscounted_return_to_go(costs, valid)
    first_valid = valid[:, 0]
    if not first_valid.all():
        raise ValueError("every episode must start at time index zero")

    initial = regression_metrics(prediction[:, 0], target[:, 0])
    pooled = regression_metrics(prediction[valid], target[valid])
    time_bin_rmse: dict[str, float | None] = {}
    for index in range(num_time_bins):
        lower = index / num_time_bins
        upper = (index + 1) / num_time_bins
        if index == num_time_bins - 1:
            selected = valid & (remaining_time >= lower) & (remaining_time <= upper)
        else:
            selected = valid & (remaining_time >= lower) & (remaining_time < upper)
        label = f"{lower:.1f}-{upper:.1f}"
        if selected.any():
            error = prediction[selected] - target[selected]
            time_bin_rmse[label] = float(torch.sqrt(torch.mean(error.square())).item())
        else:
            time_bin_rmse[label] = None

    payload = {
        "target_definition": "undiscounted_full_episode_return_to_go",
        "initial": initial.to_dict(),
        "trajectory": {
            "num_episodes": int(prediction.shape[0]),
            "num_transitions": int(valid.sum().item()),
            "pooled": pooled.to_dict(),
            "episode_equal_rmse": _episode_equal_rmse(prediction, target, valid),
        },
        "time_bin_rmse": time_bin_rmse,
        "success": _group_metrics(prediction, target, valid, success),
        "failure": _group_metrics(prediction, target, valid, ~success),
    }
    _validate_finite_metrics(payload)
    return payload


def _validate_finite_metrics(value) -> None:
    """Reject non-finite metric leaves while allowing empty-bin ``None`` values."""
    if isinstance(value, dict):
        for nested in value.values():
            _validate_finite_metrics(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _validate_finite_metrics(nested)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("trajectory metrics contain NaN or infinity")


def validate_trajectory_metrics(payload: dict) -> None:
    """Validate the required public fields of a serialized v2 metric payload."""
    required_top_level = {
        "target_definition",
        "initial",
        "trajectory",
        "time_bin_rmse",
        "success",
        "failure",
    }
    if set(payload) != required_top_level:
        raise ValueError(f"invalid trajectory metric fields: {sorted(payload)}")
    if payload["target_definition"] != "undiscounted_full_episode_return_to_go":
        raise ValueError("unsupported cost-value target definition")
    regression_fields = {field.name for field in RegressionMetrics.__dataclass_fields__.values()}
    if set(payload["initial"]) != regression_fields:
        raise ValueError("invalid initial regression metric fields")
    trajectory = payload["trajectory"]
    if set(trajectory) != {"num_episodes", "num_transitions", "pooled", "episode_equal_rmse"}:
        raise ValueError("invalid trajectory metric fields")
    if set(trajectory["pooled"]) != regression_fields:
        raise ValueError("invalid pooled regression metric fields")
    if trajectory["num_episodes"] <= 0 or trajectory["num_transitions"] <= 0:
        raise ValueError("trajectory metric counts must be positive")
    for group_name in ("success", "failure"):
        group = payload[group_name]
        if group is None:
            continue
        if set(group) != {"num_episodes", "num_transitions", "pooled", "episode_equal_rmse"}:
            raise ValueError(f"invalid {group_name} metric fields")
        if set(group["pooled"]) != regression_fields:
            raise ValueError(f"invalid {group_name} pooled metric fields")
    _validate_finite_metrics(payload)
