"""Single-model terrain evaluation with CPU-testable state accounting.

Isaac Lab imports are deliberately delayed until terrain configuration is
requested.  The episode accumulator and summary logic can therefore be tested
without launching Isaac Sim.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


TERRAIN_VARIANTS = {
    "flat": "plane",
    "slope": "hf_pyramid_slope_inv",
    "stairs": "pyramid_stairs_inv",
    "box": "boxes",
    "rough": "random_rough",
}

PAIRING_FIELDS = (
    "env_id",
    "terrain",
    "difficulty",
    "terrain_seed",
    "env_seed",
    "terrain_rows",
    "terrain_cols",
    "terrain_level",
    "terrain_type",
)


@dataclass(frozen=True)
class TerrainEvaluationProtocol:
    """Configuration shared by one model on one generated terrain."""

    terrain: str
    difficulty: float = 0.5
    num_envs: int = 200
    terrain_seed: int = 12_345
    env_seed: int = 24_680
    command_x_mps: float = 1.0
    goal_distance_m: float = 3.0
    max_time_s: float = 8.0
    terrain_rows: int = 10
    terrain_cols: int = 20

    def validate(self) -> None:
        if self.terrain not in TERRAIN_VARIANTS:
            raise ValueError(f"unknown terrain {self.terrain!r}; expected one of {sorted(TERRAIN_VARIANTS)}")
        if not math.isfinite(self.difficulty) or not 0.0 <= self.difficulty <= 1.0:
            raise ValueError("difficulty must be finite and within [0, 1]")
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        if self.terrain_rows < 1 or self.terrain_cols < 1:
            raise ValueError("terrain_rows and terrain_cols must be positive")
        for name, value in (
            ("command_x_mps", self.command_x_mps),
            ("goal_distance_m", self.goal_distance_m),
            ("max_time_s", self.max_time_s),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def terrain_variant(self) -> str:
        self.validate()
        return TERRAIN_VARIANTS[self.terrain]

    def max_steps(self, step_dt: float) -> int:
        self.validate()
        if not math.isfinite(step_dt) or step_dt <= 0.0:
            raise ValueError("step_dt must be finite and positive")
        steps = int(math.floor((self.max_time_s + 1.0e-9) / step_dt))
        if steps < 1:
            raise ValueError("max_time_s is shorter than one control step")
        return steps


def configure_eval_terrain(env_cfg, protocol: TerrainEvaluationProtocol, *, rough_terrains_cfg=None):
    """Replace only the physical plane while preserving the flat actor input."""
    protocol.validate()
    if protocol.terrain == "flat":
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None
        env_cfg.scene.terrain.max_init_terrain_level = None
        env_cfg.scene.height_scanner = None
        if hasattr(env_cfg.observations.policy, "height_scan"):
            env_cfg.observations.policy.height_scan = None
        env_cfg.curriculum.terrain_levels = None
        return None

    if rough_terrains_cfg is None:
        from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

        rough_terrains_cfg = ROUGH_TERRAINS_CFG

    terrain_cfg = deepcopy(rough_terrains_cfg)
    terrain_cfg.seed = protocol.terrain_seed
    terrain_cfg.curriculum = False
    terrain_cfg.num_rows = protocol.terrain_rows
    terrain_cfg.num_cols = protocol.terrain_cols
    terrain_cfg.difficulty_range = (protocol.difficulty, protocol.difficulty)

    terrain_key = protocol.terrain_variant
    if terrain_key not in terrain_cfg.sub_terrains:
        raise KeyError(f"Isaac Lab rough terrain config does not contain {terrain_key!r}")
    sub_terrain_cfg = deepcopy(terrain_cfg.sub_terrains[terrain_key])
    sub_terrain_cfg.proportion = 1.0
    terrain_cfg.sub_terrains = {terrain_key: sub_terrain_cfg}

    env_cfg.scene.terrain.terrain_type = "generator"
    env_cfg.scene.terrain.terrain_generator = terrain_cfg
    env_cfg.scene.terrain.max_init_terrain_level = None
    env_cfg.scene.height_scanner = None
    if hasattr(env_cfg.observations.policy, "height_scan"):
        env_cfg.observations.policy.height_scan = None
    env_cfg.curriculum.terrain_levels = None
    return terrain_cfg


def configure_eval_protocol(env_cfg, protocol: TerrainEvaluationProtocol) -> float:
    """Apply deterministic reset, command, and timeout settings to an env cfg."""
    protocol.validate()
    env_cfg.scene.num_envs = protocol.num_envs
    env_cfg.seed = protocol.env_seed

    command = env_cfg.commands.base_velocity
    command.heading_command = False
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.ranges.lin_vel_x = (protocol.command_x_mps, protocol.command_x_mps)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.ranges.heading = (0.0, 0.0)

    env_cfg.observations.policy.enable_corruption = False
    env_cfg.scene.height_scanner = None
    if hasattr(env_cfg.observations.policy, "height_scan"):
        env_cfg.observations.policy.height_scan = None

    for event_name in ("add_base_mass", "base_com", "base_external_force_torque", "push_robot"):
        if hasattr(env_cfg.events, event_name):
            setattr(env_cfg.events, event_name, None)

    reset_base = env_cfg.events.reset_base
    reset_base.params = deepcopy(reset_base.params)
    reset_base.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    reset_base.params["velocity_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }

    env_cfg.curriculum.terrain_levels = None
    env_cfg.pace_energy.publish_eval_state = True
    step_dt = float(env_cfg.sim.dt) * int(env_cfg.decimation)
    # Keep Isaac Lab's own truncation one control step beyond the protocol.
    # The evaluator can then classify every environment exactly once.
    env_cfg.episode_length_s = protocol.max_time_s + step_dt
    return step_dt


class TerrainEpisodeAccumulator:
    """Freeze each vectorized environment's metrics at its first outcome."""

    STATUS_ACTIVE = 0
    STATUS_SUCCESS = 1
    STATUS_FALL = 2
    STATUS_TIMEOUT = 3
    STATUS_NAMES = {
        STATUS_SUCCESS: "success",
        STATUS_FALL: "fall",
        STATUS_TIMEOUT: "timeout",
    }

    def __init__(
        self,
        num_envs: int,
        *,
        device: str | torch.device,
        step_dt: float,
        goal_distance_m: float,
        command_x_mps: float,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if step_dt <= 0.0 or goal_distance_m <= 0.0:
            raise ValueError("step_dt and goal_distance_m must be positive")
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.step_dt = float(step_dt)
        self.goal_distance_m = float(goal_distance_m)
        self.command_x_mps = float(command_x_mps)

        self.active = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        self.status = torch.zeros(num_envs, dtype=torch.int8, device=self.device)
        self.finish_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.current_progress_m = torch.zeros(num_envs, device=self.device)
        self.accumulated_energy_j = torch.zeros(num_envs, device=self.device)
        self.accumulated_electrical_j = torch.zeros(num_envs, device=self.device)
        self.accumulated_mechanical_j = torch.zeros(num_envs, device=self.device)
        self.accumulated_potential_j = torch.zeros(num_envs, device=self.device)
        self.tracking_squared_error = torch.zeros(num_envs, device=self.device)
        self.valid_steps = torch.zeros(num_envs, device=self.device)

        self.final_progress_m = torch.zeros(num_envs, device=self.device)
        self.final_energy_j = torch.zeros(num_envs, device=self.device)
        self.final_electrical_j = torch.zeros(num_envs, device=self.device)
        self.final_mechanical_j = torch.zeros(num_envs, device=self.device)
        self.final_potential_j = torch.zeros(num_envs, device=self.device)
        self.final_tracking_rmse = torch.zeros(num_envs, device=self.device)

    def _vector(self, name: str, value: torch.Tensor, *, boolean: bool = False) -> torch.Tensor:
        value = torch.as_tensor(value, device=self.device)
        if value.shape != (self.num_envs,):
            raise ValueError(f"{name} must have shape ({self.num_envs},), got {tuple(value.shape)}")
        if boolean:
            return value.bool()
        value = value.float()
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or infinity")
        return value

    def update(
        self,
        *,
        step_index: int,
        progress_m: torch.Tensor,
        forward_velocity_mps: torch.Tensor,
        step_energy_j: torch.Tensor,
        step_electrical_j: torch.Tensor,
        step_mechanical_j: torch.Tensor,
        step_potential_j: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        """Consume one pre-reset snapshot and freeze newly finished envs."""
        if step_index < 1:
            raise ValueError("step_index is one-based and must be positive")
        progress = self._vector("progress_m", progress_m)
        velocity = self._vector("forward_velocity_mps", forward_velocity_mps)
        total = self._vector("step_energy_j", step_energy_j)
        electrical = self._vector("step_electrical_j", step_electrical_j)
        mechanical = self._vector("step_mechanical_j", step_mechanical_j)
        potential = self._vector("step_potential_j", step_potential_j)
        terminated = self._vector("terminated", terminated, boolean=True)
        truncated = self._vector("truncated", truncated, boolean=True)

        if not torch.allclose(total, electrical + mechanical + potential, rtol=1.0e-4, atol=1.0e-5):
            raise ValueError("PACE step total does not equal its electrical, mechanical, and potential components")

        active_before = self.active.clone()
        self.current_progress_m[active_before] = progress[active_before]
        self.accumulated_energy_j[active_before] += total[active_before]
        self.accumulated_electrical_j[active_before] += electrical[active_before]
        self.accumulated_mechanical_j[active_before] += mechanical[active_before]
        self.accumulated_potential_j[active_before] += potential[active_before]
        self.tracking_squared_error[active_before] += (
            velocity[active_before] - self.command_x_mps
        ).square()
        self.valid_steps[active_before] += 1.0

        new_success = active_before & (progress >= self.goal_distance_m)
        new_fall = active_before & terminated & ~new_success
        new_timeout = active_before & truncated & ~terminated & ~new_success
        self._freeze(new_success, self.STATUS_SUCCESS, step_index)
        self._freeze(new_fall, self.STATUS_FALL, step_index)
        self._freeze(new_timeout, self.STATUS_TIMEOUT, step_index)

    def _freeze(self, mask: torch.Tensor, status: int, finish_step: int) -> None:
        if not mask.any():
            return
        self.status[mask] = status
        self.finish_steps[mask] = finish_step
        self.final_progress_m[mask] = self.current_progress_m[mask]
        self.final_energy_j[mask] = self.accumulated_energy_j[mask]
        self.final_electrical_j[mask] = self.accumulated_electrical_j[mask]
        self.final_mechanical_j[mask] = self.accumulated_mechanical_j[mask]
        self.final_potential_j[mask] = self.accumulated_potential_j[mask]
        self.final_tracking_rmse[mask] = torch.sqrt(
            self.tracking_squared_error[mask] / self.valid_steps[mask].clamp_min(1.0)
        )
        self.active[mask] = False

    def finalize_timeouts(self, max_steps: int) -> None:
        """Mark all environments still active at the protocol limit as timeout."""
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self._freeze(self.active.clone(), self.STATUS_TIMEOUT, max_steps)

    def episode_rows(self, metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
        if self.active.any() or (self.status == self.STATUS_ACTIVE).any():
            raise RuntimeError("all environments must be finalized before rows are exported")
        rows: list[dict[str, Any]] = []
        for env_id in range(self.num_envs):
            status_code = int(self.status[env_id].item())
            status_name = self.STATUS_NAMES[status_code]
            success = status_code == self.STATUS_SUCCESS
            progress = float(self.final_progress_m[env_id].item())
            energy = float(self.final_energy_j[env_id].item())
            duration = float(self.finish_steps[env_id].item()) * self.step_dt
            row = dict(metadata)
            row.update(
                {
                    "env_id": env_id,
                    "status": status_name,
                    "success": int(success),
                    "fallen": int(status_code == self.STATUS_FALL),
                    "timed_out": int(status_code == self.STATUS_TIMEOUT),
                    "finish_steps": int(self.finish_steps[env_id].item()),
                    "duration_s": duration,
                    "time_to_goal_s": duration if success else None,
                    "progress_m": progress,
                    "tracking_rmse": float(self.final_tracking_rmse[env_id].item()),
                    "energy_j": energy,
                    "electrical_energy_j": float(self.final_electrical_j[env_id].item()),
                    "mechanical_energy_j": float(self.final_mechanical_j[env_id].item()),
                    "potential_energy_j": float(self.final_potential_j[env_id].item()),
                    "energy_per_meter_j": energy / progress if success and progress > 0.0 else None,
                    "mean_power_w": energy / duration if duration > 0.0 else None,
                }
            )
            rows.append(row)
        return rows


def summarize_episode_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize outcomes globally and efficiency only on successful trials."""
    if not rows:
        raise ValueError("at least one episode row is required")
    valid_statuses = {"success", "fall", "timeout"}
    statuses = [str(row["status"]) for row in rows]
    if any(status not in valid_statuses for status in statuses):
        raise ValueError("episode rows contain an invalid status")

    total = len(rows)
    success_rows = [row for row in rows if row["status"] == "success"]
    fall_count = statuses.count("fall")
    timeout_count = statuses.count("timeout")
    summary: dict[str, Any] = {
        "num_episodes": total,
        "num_success": len(success_rows),
        "num_fall": fall_count,
        "num_timeout": timeout_count,
        "success_rate": len(success_rows) / total,
        "fall_rate": fall_count / total,
        "timeout_rate": timeout_count / total,
        "mean_progress_m": statistics.fmean(float(row["progress_m"]) for row in rows),
        "median_progress_m": statistics.median(float(row["progress_m"]) for row in rows),
        "efficiency_population": "successful_episodes_only",
    }
    successful_metrics = (
        "energy_j",
        "electrical_energy_j",
        "mechanical_energy_j",
        "potential_energy_j",
        "energy_per_meter_j",
        "mean_power_w",
        "time_to_goal_s",
        "tracking_rmse",
    )
    for metric in successful_metrics:
        values = [float(row[metric]) for row in success_rows]
        summary[f"successful_mean_{metric}"] = statistics.fmean(values) if values else None
        summary[f"successful_median_{metric}"] = statistics.median(values) if values else None
    return summary


def _pairing_value(field: str, value: Any) -> Any:
    if field in {"env_id", "terrain_seed", "env_seed", "terrain_rows", "terrain_cols", "terrain_level", "terrain_type"}:
        return int(value)
    if field == "difficulty":
        return round(float(value), 12)
    return str(value)


def pairing_signature(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash the algorithm-independent terrain assignment for later pairing."""
    normalized = []
    for row in rows:
        missing = [field for field in PAIRING_FIELDS if field not in row]
        if missing:
            raise ValueError(f"pairing row is missing fields: {missing}")
        normalized.append({field: _pairing_value(field, row[field]) for field in PAIRING_FIELDS})
    normalized.sort(key=lambda row: row["env_id"])
    if len({row["env_id"] for row in normalized}) != len(normalized):
        raise ValueError("pairing rows contain duplicate env_id values")
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_pairing_matches(
    current_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    *,
    position_tolerance: float = 1.0e-6,
) -> None:
    """Raise if two algorithm runs do not share the same terrain assignment."""
    if pairing_signature(current_rows) != pairing_signature(reference_rows):
        raise ValueError("paired evaluation mismatch in seeds, terrain config, level, or type")
    current_by_env = {int(row["env_id"]): row for row in current_rows}
    reference_by_env = {int(row["env_id"]): row for row in reference_rows}
    if current_by_env.keys() != reference_by_env.keys():
        raise ValueError("paired evaluation env_id sets differ")
    for env_id, current in current_by_env.items():
        reference = reference_by_env[env_id]
        for field in ("start_x_w", "start_y_w", "start_z_w"):
            if field in current and field in reference:
                if abs(float(current[field]) - float(reference[field])) > position_tolerance:
                    raise ValueError(f"paired evaluation start state differs for env {env_id}: {field}")


def atomic_write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a non-empty CSV through a sibling temporary file."""
    if not rows:
        raise ValueError("cannot write an empty CSV")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through a sibling temporary file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
