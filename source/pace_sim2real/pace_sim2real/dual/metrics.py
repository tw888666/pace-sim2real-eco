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


@dataclass(frozen=True)
class EnergyFeasibilityMetrics:
    """Episode-level physical-energy diagnostics for one frozen policy.

    Relative excesses are signed: a negative value means the mean lies below
    the budget. Violation rates are episode proportions, not ratios of mean
    energy to the budget.
    """

    num_episodes: int
    num_successful_episodes: int
    mean_relative_budget_excess: float
    mean_success_energy_j: float | None
    mean_success_relative_budget_excess: float | None
    episode_energy_violation_rate: float
    success_conditional_energy_violation_rate: float | None
    joint_feasibility_rate: float
    mean_positive_energy_excess_j: float

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def energy_feasibility_metrics(
    physical_energy_j: torch.Tensor,
    success: torch.Tensor,
    budget_j: float,
) -> EnergyFeasibilityMetrics:
    """Summarize expected excess, episode violations, and joint feasibility."""
    energy = physical_energy_j.detach().float().reshape(-1)
    success = success.detach().reshape(-1)
    if energy.shape != success.shape:
        raise ValueError("physical_energy_j and success must contain one value per episode")
    if energy.numel() == 0:
        raise ValueError("energy feasibility metrics require at least one episode")
    if success.dtype != torch.bool:
        raise TypeError("success must be a boolean tensor")
    if not math.isfinite(budget_j) or budget_j <= 0.0:
        raise ValueError("budget_j must be finite and positive")
    if not torch.isfinite(energy).all() or (energy < 0.0).any():
        raise ValueError("physical energy must be finite and non-negative")

    over_budget = energy > budget_j
    jointly_feasible = success & ~over_budget
    successful_energy = energy[success]
    if successful_energy.numel() > 0:
        mean_success_energy_j = float(successful_energy.mean().item())
        mean_success_relative_budget_excess = mean_success_energy_j / budget_j - 1.0
        success_conditional_violation = float(over_budget[success].float().mean().item())
    else:
        mean_success_energy_j = None
        mean_success_relative_budget_excess = None
        success_conditional_violation = None

    metrics = EnergyFeasibilityMetrics(
        num_episodes=energy.numel(),
        num_successful_episodes=int(success.sum().item()),
        mean_relative_budget_excess=float(energy.mean().item() / budget_j - 1.0),
        mean_success_energy_j=mean_success_energy_j,
        mean_success_relative_budget_excess=mean_success_relative_budget_excess,
        episode_energy_violation_rate=float(over_budget.float().mean().item()),
        success_conditional_energy_violation_rate=success_conditional_violation,
        joint_feasibility_rate=float(jointly_feasible.float().mean().item()),
        mean_positive_energy_excess_j=float(torch.clamp_min(energy - budget_j, 0.0).mean().item()),
    )
    validate_energy_feasibility_metrics(metrics.to_dict(), budget_j=budget_j)
    return metrics


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


def validate_energy_feasibility_metrics(
    payload: dict,
    *,
    budget_j: float | None = None,
    num_episodes: int | None = None,
    success_rate: float | None = None,
    mean_physical_energy_j: float | None = None,
) -> None:
    """Validate a serialized episode-level energy feasibility payload."""
    required = {field.name for field in EnergyFeasibilityMetrics.__dataclass_fields__.values()}
    if set(payload) != required:
        raise ValueError(f"invalid energy feasibility metric fields: {sorted(payload)}")

    count = payload["num_episodes"]
    success_count = payload["num_successful_episodes"]
    if not isinstance(count, int) or count <= 0:
        raise ValueError("energy feasibility episode count must be positive")
    if not isinstance(success_count, int) or not 0 <= success_count <= count:
        raise ValueError("successful episode count is invalid")
    if num_episodes is not None and count != num_episodes:
        raise ValueError("energy feasibility episode count differs from the dual result")

    required_numeric = (
        "mean_relative_budget_excess",
        "episode_energy_violation_rate",
        "joint_feasibility_rate",
        "mean_positive_energy_excess_j",
    )
    if not all(isinstance(payload[name], (int, float)) and math.isfinite(payload[name]) for name in required_numeric):
        raise ValueError("energy feasibility metrics contain NaN, infinity, or a non-numeric value")
    for name in ("episode_energy_violation_rate", "joint_feasibility_rate"):
        if not 0.0 <= payload[name] <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
    if payload["mean_positive_energy_excess_j"] < 0.0:
        raise ValueError("mean positive energy excess cannot be negative")

    optional_success_fields = (
        "mean_success_energy_j",
        "mean_success_relative_budget_excess",
        "success_conditional_energy_violation_rate",
    )
    if success_count == 0:
        if any(payload[name] is not None for name in optional_success_fields):
            raise ValueError("success-conditional metrics must be null when there are no successful episodes")
    else:
        if not all(
            isinstance(payload[name], (int, float)) and math.isfinite(payload[name])
            for name in optional_success_fields
        ):
            raise ValueError("success-conditional metrics must be finite when successful episodes exist")
        if payload["mean_success_energy_j"] < 0.0:
            raise ValueError("mean successful energy cannot be negative")
        if not 0.0 <= payload["success_conditional_energy_violation_rate"] <= 1.0:
            raise ValueError("success-conditional violation rate must lie in [0, 1]")

    tolerance = 1.0e-6
    if success_rate is not None and not math.isclose(success_count / count, success_rate, abs_tol=tolerance):
        raise ValueError("successful episode count differs from success_rate")
    if budget_j is not None:
        if not math.isfinite(budget_j) or budget_j <= 0.0:
            raise ValueError("budget_j must be finite and positive")
        if success_count > 0:
            expected_success_excess = payload["mean_success_energy_j"] / budget_j - 1.0
            if not math.isclose(
                payload["mean_success_relative_budget_excess"],
                expected_success_excess,
                rel_tol=1.0e-6,
                abs_tol=1.0e-6,
            ):
                raise ValueError("mean successful relative excess is inconsistent with energy and budget")
        if mean_physical_energy_j is not None:
            expected_mean_excess = mean_physical_energy_j / budget_j - 1.0
            if not math.isclose(
                payload["mean_relative_budget_excess"],
                expected_mean_excess,
                rel_tol=1.0e-6,
                abs_tol=1.0e-6,
            ):
                raise ValueError("mean relative excess is inconsistent with the dual result")


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
