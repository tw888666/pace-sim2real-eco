"""Paired offline controls for testing the finite-horizon time hypothesis."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn


class OfflineCostCritic(nn.Module):
    """MLP with state-dict names compatible with the online RSL-RL model."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (256, 256, 128)):
        super().__init__()
        dims = [input_dim, *hidden_dims, 1]
        layers: list[nn.Module] = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            if index < len(dims) - 2:
                layers.append(nn.ELU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.mlp(observation).squeeze(-1)


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, metadata, and bytes in deterministic key order."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def module_sha256(module: nn.Module) -> str:
    return state_dict_sha256(module.state_dict())


def train_offline_cost_critic(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    seed: int,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_dims: Sequence[int] = (256, 256, 128),
) -> tuple[OfflineCostCritic, dict[str, float | int | str]]:
    """Fit one critic with deterministic initialization and batch ordering."""
    if features.ndim != 2 or targets.shape != features.shape[:1]:
        raise ValueError("features must be [sample, feature] and targets must be [sample]")
    if features.device.type != "cpu" or targets.device.type != "cpu":
        raise ValueError("offline source tensors must remain on CPU")
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0.0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")

    torch.manual_seed(seed)
    model = OfflineCostCritic(features.shape[-1], hidden_dims=hidden_dims).to(device)
    initial_hash = module_sha256(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 10_000)
    final_loss = float("nan")
    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(features.shape[0], generator=generator)
        squared_error_sum = 0.0
        sample_count = 0
        for start in range(0, features.shape[0], batch_size):
            selected = permutation[start : start + batch_size]
            batch_features = features[selected].to(device)
            batch_targets = targets[selected].to(device)
            prediction = model(batch_features)
            loss = torch.mean((prediction - batch_targets).square())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            squared_error_sum += float(loss.item()) * selected.numel()
            sample_count += selected.numel()
        final_loss = squared_error_sum / sample_count
    model.eval()
    return model, {
        "seed": seed,
        "initial_state_sha256": initial_hash,
        "final_state_sha256": module_sha256(model),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "final_training_mse": final_loss,
    }


def predict_in_batches(model: nn.Module, features: torch.Tensor, *, device: str, batch_size: int) -> torch.Tensor:
    prediction = torch.empty(features.shape[0], dtype=torch.float32)
    with torch.inference_mode():
        for start in range(0, features.shape[0], batch_size):
            stop = min(start + batch_size, features.shape[0])
            prediction[start:stop] = model(features[start:stop].to(device)).cpu()
    return prediction


def fit_time_only_linear_baseline(remaining_time: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Fit ``return = slope * remaining_time + intercept`` by least squares."""
    if remaining_time.ndim != 1 or target.shape != remaining_time.shape:
        raise ValueError("time-only baseline requires matching one-dimensional tensors")
    design = torch.stack((remaining_time.float(), torch.ones_like(remaining_time).float()), dim=1)
    return torch.linalg.lstsq(design, target.float().unsqueeze(1)).solution.squeeze(1)


def predict_time_only(remaining_time: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    if coefficients.shape != (2,):
        raise ValueError("time-only coefficients must contain slope and intercept")
    return coefficients[0] * remaining_time + coefficients[1]
