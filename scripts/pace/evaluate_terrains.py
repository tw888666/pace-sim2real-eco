"""Evaluate one frozen checkpoint on one deterministic generated terrain.

This script launches Isaac Sim and must be run by the user on an available GPU.
It intentionally does not implement the later 3-algorithm by 4-terrain batch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True, help="frozen RSL-RL checkpoint")
parser.add_argument(
    "--agent_task",
    required=True,
    help="registered task used only to construct the checkpoint-compatible RSL-RL runner",
)
parser.add_argument("--algorithm_name", required=True, help="label stored in result files")
parser.add_argument("--eval_task", default="Isaac-Pace-Eco-Anymal-D-Flat-Play-v0")
parser.add_argument("--terrain", choices=("flat", "slope", "stairs", "box", "rough"), required=True)
parser.add_argument("--difficulty", type=float, default=0.5)
parser.add_argument("--num_envs", type=int, default=200)
parser.add_argument("--env_seed", type=int, default=24_680)
parser.add_argument("--terrain_seed", type=int, default=12_345)
parser.add_argument("--goal_distance", type=float, default=3.0)
parser.add_argument("--max_time", type=float, default=8.0)
parser.add_argument("--command_x", type=float, default=1.0)
parser.add_argument("--terrain_rows", type=int, default=10)
parser.add_argument("--terrain_cols", type=int, default=20)
parser.add_argument("--expected_policy_obs_dim", type=int, default=48)
parser.add_argument("--output_dir", default=None)
parser.add_argument(
    "--paired_reference",
    default=None,
    help="earlier per-episode CSV whose terrain assignment must match this run",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import isaaclab_tasks  # noqa: F401
import pace_sim2real.tasks  # noqa: F401
from pace_sim2real.evaluation import (
    TerrainEpisodeAccumulator,
    TerrainEvaluationProtocol,
    assert_pairing_matches,
    atomic_write_csv,
    atomic_write_json,
    configure_eval_protocol,
    configure_eval_terrain,
    pairing_signature,
    summarize_episode_rows,
)


RESULT_FILENAMES = (
    "gpt-地形评估-逐回合.csv",
    "gpt-地形评估-汇总.csv",
    "gpt-地形评估-配对信息.json",
)
REQUIRED_STEP_EXTRAS = (
    "pace_step_energy_j",
    "pace_step_electrical_energy_j",
    "pace_step_mechanical_energy_j",
    "pace_step_potential_energy_j",
    "pace_eval_root_pos_w",
    "pace_eval_root_lin_vel_b",
)
ACTOR_ONLY_LOAD_CFG = {
    "actor": True,
    "critic": False,
    "optimizer": False,
    "iteration": False,
    "rnd": False,
    "cost_critic": False,
    "cost_optimizer": False,
    "lagrangian_multiplier": False,
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not label:
        raise ValueError("algorithm_name must contain at least one filename-safe character")
    return label


def _result_directory(protocol: TerrainEvaluationProtocol) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir).expanduser().resolve()
    difficulty_label = f"{protocol.difficulty:.3f}".rstrip("0").rstrip(".")
    run_name = (
        f"{_safe_label(args.algorithm_name)}-{protocol.terrain}-d{difficulty_label}"
        f"-ts{protocol.terrain_seed}-es{protocol.env_seed}"
    )
    return (Path.cwd() / "results" / "terrain_evaluation" / run_name).resolve()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"paired reference contains no episode rows: {path}")
    return rows


def _check_output_targets(output_dir: Path) -> tuple[Path, Path, Path]:
    targets = tuple(output_dir / name for name in RESULT_FILENAMES)
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing evaluation outputs: {existing}")
    return targets  # type: ignore[return-value]


def _terrain_assignment(raw_env, start_pos_w: torch.Tensor, protocol: TerrainEvaluationProtocol):
    terrain = raw_env.unwrapped.scene.terrain
    starts = start_pos_w.detach().cpu()
    if protocol.terrain == "flat":
        unavailable = torch.full((protocol.num_envs,), -1, dtype=torch.long)
        return unavailable.clone(), unavailable, starts
    if not hasattr(terrain, "terrain_levels") or not hasattr(terrain, "terrain_types"):
        raise RuntimeError("generated terrain did not expose terrain_levels and terrain_types")
    levels = terrain.terrain_levels.detach().cpu()
    types = terrain.terrain_types.detach().cpu()
    expected_shape = (protocol.num_envs,)
    if levels.shape != expected_shape or types.shape != expected_shape:
        raise RuntimeError(
            "terrain assignment shape mismatch: "
            f"levels={tuple(levels.shape)}, types={tuple(types.shape)}, expected={expected_shape}"
        )
    if starts.shape != (protocol.num_envs, 3):
        raise RuntimeError(f"initial root position must have shape ({protocol.num_envs}, 3)")
    return levels, types, starts


def main() -> None:
    protocol = TerrainEvaluationProtocol(
        terrain=args.terrain,
        difficulty=args.difficulty,
        num_envs=args.num_envs,
        terrain_seed=args.terrain_seed,
        env_seed=args.env_seed,
        command_x_mps=args.command_x,
        goal_distance_m=args.goal_distance,
        max_time_s=args.max_time,
        terrain_rows=args.terrain_rows,
        terrain_cols=args.terrain_cols,
    )
    protocol.validate()
    if args.expected_policy_obs_dim < 1:
        raise ValueError("expected_policy_obs_dim must be positive")

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    checkpoint_hash = _file_sha256(checkpoint)
    output_dir = _result_directory(protocol)
    episode_path, summary_path, pairing_path = _check_output_targets(output_dir)

    env_cfg = load_cfg_from_registry(args.eval_task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args.agent_task, "rsl_rl_cfg_entry_point")
    step_dt = configure_eval_protocol(env_cfg, protocol)
    configure_eval_terrain(env_cfg, protocol)
    env_cfg.sim.device = args.device
    agent_cfg.device = args.device
    agent_cfg.seed = protocol.env_seed

    random.seed(protocol.env_seed)
    np.random.seed(protocol.env_seed)
    torch.manual_seed(protocol.env_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(protocol.env_seed)

    env = None
    try:
        raw_env = gym.make(args.eval_task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        print(f"[INFO] Loading checkpoint: {checkpoint}", flush=True)
        # Terrain evaluation consumes only the deterministic actor. Loading
        # critics or optimizers would incorrectly couple inference to training-
        # only architecture changes such as the 48 -> 49 cost-time extension.
        runner.load(
            str(checkpoint),
            load_cfg=ACTOR_ONLY_LOAD_CFG,
            strict=True,
            map_location=args.device,
        )
        print("[INFO] Checkpoint loaded; starting deterministic evaluation.", flush=True)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        obs = env.get_observations().to(args.device)
        if "policy" not in obs.keys():
            raise RuntimeError("evaluation environment did not expose the policy observation group")
        policy_obs_dim = int(obs["policy"].shape[-1])
        if policy_obs_dim != args.expected_policy_obs_dim:
            raise RuntimeError(
                "flat actor observation contract changed: "
                f"expected {args.expected_policy_obs_dim}, got {policy_obs_dim}"
            )

        robot = env.unwrapped.scene["robot"]
        start_pos_w = robot.data.root_pos_w.clone()
        start_x_w = start_pos_w[:, 0].clone()
        terrain_levels, terrain_types, start_positions = _terrain_assignment(raw_env, start_pos_w, protocol)

        accumulator = TerrainEpisodeAccumulator(
            protocol.num_envs,
            device=env.unwrapped.device,
            step_dt=step_dt,
            goal_distance_m=protocol.goal_distance_m,
            command_x_mps=protocol.command_x_mps,
        )
        max_steps = protocol.max_steps(step_dt)
        for step_index in range(1, max_steps + 1):
            if not accumulator.active.any():
                break
            if not simulation_app.is_running():
                raise RuntimeError("simulation stopped before the terrain evaluation completed")
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, extras = env.step(actions)

            missing = [key for key in REQUIRED_STEP_EXTRAS if key not in extras]
            if missing:
                raise RuntimeError(f"evaluation environment did not publish required extras: {missing}")
            step_root_pos = extras["pace_eval_root_pos_w"].to(env.unwrapped.device)
            step_root_vel = extras["pace_eval_root_lin_vel_b"].to(env.unwrapped.device)
            time_outs = extras.get("time_outs", env.unwrapped.reset_time_outs).bool()
            done_mask = dones.bool()
            accumulator.update(
                step_index=step_index,
                progress_m=step_root_pos[:, 0] - start_x_w,
                forward_velocity_mps=step_root_vel[:, 0],
                step_energy_j=extras["pace_step_energy_j"],
                step_electrical_j=extras["pace_step_electrical_energy_j"],
                step_mechanical_j=extras["pace_step_mechanical_energy_j"],
                step_potential_j=extras["pace_step_potential_energy_j"],
                terminated=done_mask & ~time_outs,
                truncated=done_mask & time_outs,
            )

        if not simulation_app.is_running() and accumulator.active.any():
            raise RuntimeError("simulation stopped before the terrain evaluation completed")
        accumulator.finalize_timeouts(max_steps)

        if _file_sha256(checkpoint) != checkpoint_hash:
            raise RuntimeError("checkpoint changed during terrain evaluation")
        metadata = {
            "algorithm": args.algorithm_name,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "agent_task": args.agent_task,
            "eval_task": args.eval_task,
            "terrain": protocol.terrain,
            "terrain_variant": protocol.terrain_variant,
            "difficulty": protocol.difficulty,
            "terrain_seed": protocol.terrain_seed,
            "env_seed": protocol.env_seed,
            "terrain_rows": protocol.terrain_rows,
            "terrain_cols": protocol.terrain_cols,
            "command_x_mps": protocol.command_x_mps,
            "goal_distance_m": protocol.goal_distance_m,
            "max_time_s": protocol.max_time_s,
            "step_dt_s": step_dt,
            "policy_obs_dim": policy_obs_dim,
        }
        episode_rows = accumulator.episode_rows(metadata)
        for row in episode_rows:
            env_id = int(row["env_id"])
            row.update(
                {
                    "terrain_level": int(terrain_levels[env_id].item()),
                    "terrain_type": int(terrain_types[env_id].item()),
                    "start_x_w": float(start_positions[env_id, 0].item()),
                    "start_y_w": float(start_positions[env_id, 1].item()),
                    "start_z_w": float(start_positions[env_id, 2].item()),
                }
            )

        signature = pairing_signature(episode_rows)
        paired_reference_verified = False
        paired_reference = None
        if args.paired_reference is not None:
            reference_path = Path(args.paired_reference).expanduser().resolve()
            reference_rows = _read_csv(reference_path)
            assert_pairing_matches(episode_rows, reference_rows)
            paired_reference_verified = True
            paired_reference = str(reference_path)

        summary = dict(metadata)
        summary.update(summarize_episode_rows(episode_rows))
        summary.update(
            {
                "pairing_signature": signature,
                "paired_reference_verified": paired_reference_verified,
                "paired_reference": paired_reference,
            }
        )
        pairing_info = {
            "schema_version": 1,
            "pairing_signature": signature,
            "pairing_fields": [
                "env_id",
                "terrain",
                "difficulty",
                "terrain_seed",
                "env_seed",
                "terrain_rows",
                "terrain_cols",
                "terrain_level",
                "terrain_type",
            ],
            "num_envs": protocol.num_envs,
            "paired_reference_verified": paired_reference_verified,
            "paired_reference": paired_reference,
        }

        atomic_write_csv(episode_path, episode_rows)
        atomic_write_csv(summary_path, [summary])
        atomic_write_json(pairing_path, pairing_info)
        print(f"逐回合结果: {episode_path}")
        print(f"汇总结果: {summary_path}")
        print(f"配对信息: {pairing_path}")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
