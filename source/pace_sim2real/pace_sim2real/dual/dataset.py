"""Reproducible frozen-policy datasets for paired cost-critic comparisons."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch


CRITIC_DATASET_SCHEMA_VERSION = 1


def atomic_torch_save(path: str | Path, payload: dict) -> None:
    """Atomically publish a torch dataset without exposing a partial file."""
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary_name)
        with open(temporary_name, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def validate_critic_dataset(payload: dict) -> None:
    """Validate the public episode-major trajectory dataset contract."""
    required = {
        "schema_version",
        "checkpoint_path",
        "checkpoint_sha256",
        "budget_j",
        "evaluation_seed",
        "policy_observation",
        "remaining_time",
        "cost",
        "valid",
        "success",
        "physical_energy_j",
    }
    if set(payload) != required:
        raise ValueError(f"invalid critic dataset fields: {sorted(payload)}")
    if payload["schema_version"] != CRITIC_DATASET_SCHEMA_VERSION:
        raise ValueError(f"unsupported critic dataset schema {payload['schema_version']}")
    if payload["budget_j"] <= 0.0 or payload["evaluation_seed"] < 0:
        raise ValueError("dataset budget and evaluation seed are invalid")

    observation = payload["policy_observation"]
    remaining_time = payload["remaining_time"]
    costs = payload["cost"]
    valid = payload["valid"]
    success = payload["success"]
    physical_energy = payload["physical_energy_j"]
    if not all(isinstance(value, torch.Tensor) for value in (observation, remaining_time, costs, valid, success, physical_energy)):
        raise TypeError("critic dataset arrays must be torch tensors")
    if observation.ndim != 3:
        raise ValueError("policy_observation must have [episode, time, feature] shape")
    episode_time_shape = observation.shape[:2]
    if remaining_time.shape != episode_time_shape or costs.shape != episode_time_shape or valid.shape != episode_time_shape:
        raise ValueError("remaining_time, cost, and valid must match observation episode/time dimensions")
    if success.shape != episode_time_shape[:1] or physical_energy.shape != episode_time_shape[:1]:
        raise ValueError("success and physical_energy_j must have one value per episode")
    if valid.dtype != torch.bool or success.dtype != torch.bool:
        raise TypeError("valid and success must be boolean")
    if not valid[:, 0].all():
        raise ValueError("every dataset episode must contain its initial transition")
    if (valid[:, 1:] & ~valid[:, :-1]).any():
        raise ValueError("valid transitions must form a contiguous episode prefix")
    if observation.device.type != "cpu":
        raise ValueError("published critic datasets must be device-independent CPU tensors")
    for value in (observation[valid], remaining_time[valid], costs[valid], physical_energy):
        if not torch.isfinite(value).all():
            raise ValueError("critic dataset contains NaN or infinity")
    if (remaining_time[valid] < 0.0).any() or (remaining_time[valid] > 1.0).any():
        raise ValueError("remaining_time must lie in [0, 1]")


def load_critic_dataset(path: str | Path) -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("critic dataset root must be a dictionary")
    validate_critic_dataset(payload)
    return payload
