"""Crash-safe protocol between segmented PPO training and dual evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .metrics import validate_trajectory_metrics


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: str | Path, payload: dict) -> None:
    """Atomically replace a JSON document and fsync both file and directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
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


@dataclass
class DualState:
    budget_j: float
    learning_rate: float
    multiplier: float = 0.0
    multiplier_max: float = 100.0
    last_cycle_id: int = -1
    applied_results: dict[str, str] = field(default_factory=dict)
    schema_version: int = 1
    updated_at_utc: str = ""

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported dual state schema {self.schema_version}")
        if self.budget_j <= 0.0 or self.learning_rate <= 0.0 or self.multiplier_max <= 0.0:
            raise ValueError("budget, learning rate, and multiplier maximum must be positive")
        if not 0.0 <= self.multiplier <= self.multiplier_max:
            raise ValueError("dual multiplier lies outside its configured projection interval")


@dataclass(frozen=True)
class DualEvaluationResult:
    cycle_id: int
    checkpoint_path: str
    checkpoint_sha256: str
    budget_j: float
    num_episodes: int
    mean_physical_energy_j: float
    mean_augmented_cost: float
    success_rate: float
    evaluation_seed: int | None = None
    cost_value_metrics: dict | None = None
    # Legacy v1 fields. They remain readable so already archived evaluations
    # can still be inspected and applied idempotently.
    cost_value_initial_bias: float | None = None
    cost_explained_variance: float | None = None
    schema_version: int = 2

    @classmethod
    def from_dict(cls, payload: dict) -> "DualEvaluationResult":
        expected = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = set(payload) - expected
        if unknown:
            raise ValueError(f"unknown dual evaluation fields: {sorted(unknown)}")
        return cls(**payload)

    def validate(self, *, require_checkpoint: bool = True) -> None:
        if self.schema_version not in (1, 2):
            raise ValueError(f"unsupported dual result schema {self.schema_version}")
        if self.cycle_id < 0 or self.budget_j <= 0.0 or self.num_episodes <= 0:
            raise ValueError("cycle id, budget, or episode count is invalid")
        if not 0.0 <= self.success_rate <= 1.0:
            raise ValueError("success_rate must lie in [0, 1]")
        numeric_fields = [
            self.budget_j,
            self.mean_physical_energy_j,
            self.mean_augmented_cost,
            self.success_rate,
        ]
        if self.schema_version == 1:
            if self.cost_value_initial_bias is None or self.cost_explained_variance is None:
                raise ValueError("v1 dual result is missing legacy cost-value metrics")
            numeric_fields.extend((self.cost_value_initial_bias, self.cost_explained_variance))
        else:
            if self.evaluation_seed is None or self.evaluation_seed < 0:
                raise ValueError("v2 dual result requires a non-negative evaluation seed")
            if self.cost_value_metrics is None:
                raise ValueError("v2 dual result requires complete trajectory cost-value metrics")
            validate_trajectory_metrics(self.cost_value_metrics)
        if not all(math.isfinite(value) for value in numeric_fields):
            raise ValueError("dual result contains NaN or infinity")
        if require_checkpoint:
            checkpoint = Path(self.checkpoint_path)
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            actual_hash = file_sha256(checkpoint)
            if actual_hash != self.checkpoint_sha256:
                raise ValueError("immutable checkpoint hash does not match the dual result")

    @property
    def initial_absolute_mean_bias(self) -> float:
        if self.schema_version == 1:
            assert self.cost_value_initial_bias is not None
            return self.cost_value_initial_bias
        assert self.cost_value_metrics is not None
        return float(self.cost_value_metrics["initial"]["absolute_mean_bias"])

    @property
    def trajectory_explained_variance(self) -> float:
        if self.schema_version == 1:
            assert self.cost_explained_variance is not None
            return self.cost_explained_variance
        assert self.cost_value_metrics is not None
        return float(self.cost_value_metrics["trajectory"]["pooled"]["explained_variance"])


def load_dual_state(path: str | Path) -> DualState:
    with Path(path).open(encoding="utf-8") as stream:
        state = DualState(**json.load(stream))
    state.validate()
    return state


def apply_evaluation_result(
    state_path: str | Path,
    result_path: str | Path,
    *,
    initial_state: DualState | None = None,
    minimum_episodes: int = 256,
) -> tuple[DualState, bool]:
    """Apply one dual result exactly once and atomically persist the state."""
    state_file = Path(state_path)
    state = load_dual_state(state_file) if state_file.exists() else initial_state
    if state is None:
        raise FileNotFoundError(f"dual state does not exist and no initial_state was supplied: {state_file}")
    state.validate()
    with Path(result_path).open(encoding="utf-8") as stream:
        result_payload = json.load(stream)
    result = DualEvaluationResult.from_dict(result_payload)
    result.validate()
    if result.num_episodes < minimum_episodes:
        raise ValueError(f"dual evaluation needs at least {minimum_episodes} episodes")
    if abs(result.budget_j - state.budget_j) > max(1.0e-6, 1.0e-9 * state.budget_j):
        raise ValueError("dual result budget differs from the active dual state")

    result_hash = _canonical_hash(result_payload)
    cycle_key = str(result.cycle_id)
    if cycle_key in state.applied_results:
        if state.applied_results[cycle_key] != result_hash:
            raise ValueError(f"cycle {result.cycle_id} was already applied with different contents")
        return state, False
    if result.cycle_id != state.last_cycle_id + 1:
        raise ValueError(
            f"expected cycle {state.last_cycle_id + 1}, received {result.cycle_id}; out-of-order updates are rejected"
        )

    violation = result.mean_augmented_cost - 1.0
    state.multiplier = min(max(state.multiplier + state.learning_rate * violation, 0.0), state.multiplier_max)
    state.last_cycle_id = result.cycle_id
    state.applied_results[cycle_key] = result_hash
    state.updated_at_utc = datetime.now(timezone.utc).isoformat()
    state.validate()
    atomic_write_json(state_file, asdict(state))
    return state, True


def freeze_checkpoint(
    checkpoint: str | Path,
    output_directory: str | Path,
    *,
    cycle_id: int,
    budget_j: float,
) -> Path:
    """Copy a training checkpoint into a cycle-addressed immutable location."""
    source = Path(checkpoint).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    cycle_directory = Path(output_directory).resolve() / f"cycle_{cycle_id:04d}"
    cycle_directory.mkdir(parents=True, exist_ok=True)
    frozen = cycle_directory / "checkpoint.pt"
    if frozen.exists():
        if file_sha256(frozen) != file_sha256(source):
            raise FileExistsError(f"cycle {cycle_id} already contains a different checkpoint")
    else:
        descriptor, temporary_name = tempfile.mkstemp(prefix=".checkpoint.", suffix=".tmp", dir=cycle_directory)
        os.close(descriptor)
        try:
            shutil.copyfile(source, temporary_name)
            with open(temporary_name, "rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary_name, frozen)
            frozen.chmod(0o444)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
    request = {
        "schema_version": 1,
        "cycle_id": cycle_id,
        "budget_j": budget_j,
        "checkpoint_path": str(frozen),
        "checkpoint_sha256": file_sha256(frozen),
    }
    atomic_write_json(cycle_directory / "request.json", request)
    return cycle_directory / "request.json"
