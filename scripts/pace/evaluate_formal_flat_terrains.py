#!/usr/bin/env python3
"""在统一定距协议下评估三类正式平地 checkpoint 的零样本地形泛化。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import random
import traceback
from copy import deepcopy
from pathlib import Path

from isaaclab.app import AppLauncher


TASK_TO_METHOD = {
    "Isaac-PACE-TaskOnly-Flat-Anymal-D-v0": "task_only",
    "Isaac-PACE-FixedWeight-Flat-Anymal-D-v0": "fixed_weight",
    "Isaac-PACE-ECO-Flat-Anymal-D-v0": "eco",
}
TERRAIN_VARIANTS = {
    "flat": "plane",
    "slope": "hf_pyramid_slope_inv",
    "stairs": "pyramid_stairs_inv",
    "box": "boxes",
    "rough": "random_rough",
}
RESULT_FILENAMES = (
    "gpt-正式平地模型地形评估-逐回合.csv",
    "gpt-正式平地模型地形评估-汇总.csv",
    "gpt-正式平地模型地形评估-配对信息.json",
)
ACTOR_ONLY_LOAD_CFG = {
    "actor": True,
    "critic": False,
    "optimizer": False,
    "iteration": False,
    "rnd": False,
    "lagrange_multiplier": False,
    "lagrange_optimizer": False,
}


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--agent_task", required=True, choices=tuple(TASK_TO_METHOD))
parser.add_argument("--terrain", required=True, choices=tuple(TERRAIN_VARIANTS))
parser.add_argument("--difficulty", type=float, default=0.5)
parser.add_argument("--num_envs", type=int, default=200)
parser.add_argument("--env_seed", type=int, default=24_680)
parser.add_argument("--terrain_seed", type=int, default=12_345)
parser.add_argument("--goal_distance", type=float, default=3.0)
parser.add_argument("--max_time", type=float, default=8.0)
parser.add_argument("--command_x", type=float, default=1.0)
parser.add_argument("--terrain_rows", type=int, default=10)
parser.add_argument("--terrain_cols", type=int, default=20)
parser.add_argument("--state_set", default="holdout_v1")
parser.add_argument("--expected_policy_obs_dim", type=int, default=48)
parser.add_argument("--output_dir", default=None)
parser.add_argument("--paired_reference", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import load_cfg_from_registry

import pace_eco_lab  # noqa: F401
from pace_eco_lab.configs.env_cfg import configure_evaluation
from pace_eco_lab.constants import ECO_ID, TASK_ONLY_ID
from pace_eco_lab.mdp.resets import (
    EVALUATION_STATE_SETS,
    evaluation_state_count,
    evaluation_state_definition_sha256,
)
from pace_eco_lab.reproducibility import validate_evaluation_checkpoint, verify_dependency_versions
from pace_sim2real.evaluation import (
    TerrainEpisodeAccumulator,
    TerrainEvaluationProtocol,
    assert_pairing_matches,
    atomic_write_csv,
    atomic_write_json,
    pairing_signature,
    summarize_episode_rows,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"配对参考没有逐回合数据：{path}")
    return rows


def _checkpoint_metadata(checkpoint: Path) -> tuple[dict[str, object], float | None]:
    run_dir = checkpoint.parent
    record_path = run_dir / "gpt_复现信息.json"
    agent_path = run_dir / "gpt_算法配置.json"
    if not record_path.is_file() or not agent_path.is_file():
        raise FileNotFoundError("正式检查点目录缺少 gpt_复现信息.json 或 gpt_算法配置.json。")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    agent = json.loads(agent_path.read_text(encoding="utf-8"))
    saved_budget = agent.get("algorithm", {}).get("energy_budget_j")
    return record, None if saved_budget is None else float(saved_budget)


def _configure_protocol(env_cfg, protocol: TerrainEvaluationProtocol) -> float:
    protocol.validate()
    env_cfg.scene.num_envs = protocol.num_envs
    env_cfg.seed = protocol.env_seed
    env_cfg.pace_publish_eval_state = True
    command = env_cfg.commands.base_velocity
    command.heading_command = False
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.ranges.lin_vel_x = (protocol.command_x_mps, protocol.command_x_mps)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.ranges.heading = (0.0, 0.0)
    step_dt = float(env_cfg.sim.dt) * int(env_cfg.decimation)
    env_cfg.episode_length_s = protocol.max_time_s + step_dt
    return step_dt


def _configure_terrain(env_cfg, protocol: TerrainEvaluationProtocol) -> None:
    terrain = env_cfg.scene.terrain
    if protocol.terrain == "flat":
        terrain.terrain_type = "plane"
        terrain.terrain_generator = None
        terrain.max_init_terrain_level = None
        return
    generator = deepcopy(ROUGH_TERRAINS_CFG)
    generator.seed = protocol.terrain_seed
    generator.curriculum = False
    generator.num_rows = protocol.terrain_rows
    generator.num_cols = protocol.terrain_cols
    generator.difficulty_range = (protocol.difficulty, protocol.difficulty)
    key = protocol.terrain_variant
    sub_terrain = deepcopy(generator.sub_terrains[key])
    sub_terrain.proportion = 1.0
    generator.sub_terrains = {key: sub_terrain}
    terrain.terrain_type = "generator"
    terrain.terrain_generator = generator
    terrain.max_init_terrain_level = None


def _terrain_assignment(raw_env, start_pos_w: torch.Tensor, protocol: TerrainEvaluationProtocol):
    starts = start_pos_w.detach().cpu()
    if starts.shape != (protocol.num_envs, 3):
        raise RuntimeError(f"初始根位置形状应为 ({protocol.num_envs}, 3)，实际为 {tuple(starts.shape)}。")
    if protocol.terrain == "flat":
        unavailable = torch.full((protocol.num_envs,), -1, dtype=torch.long)
        return unavailable.clone(), unavailable, starts
    terrain = raw_env.unwrapped.scene.terrain
    levels = terrain.terrain_levels.detach().cpu()
    types = terrain.terrain_types.detach().cpu()
    expected = (protocol.num_envs,)
    if levels.shape != expected or types.shape != expected:
        raise RuntimeError(
            f"地形分配形状不一致：levels={tuple(levels.shape)}, types={tuple(types.shape)}。"
        )
    return levels, types, starts


def _result_directory(method: str, training_seed: int, protocol: TerrainEvaluationProtocol) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir).expanduser().resolve()
    difficulty = f"{protocol.difficulty:.3f}".rstrip("0").rstrip(".")
    return (
        Path.cwd()
        / "results"
        / "terrain_evaluation"
        / "formal_flat"
        / f"{method}-seed{training_seed}-{protocol.terrain}-d{difficulty}"
    ).resolve()


def _output_targets(output_dir: Path) -> tuple[Path, Path, Path]:
    targets = tuple(output_dir / name for name in RESULT_FILENAMES)
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"拒绝覆盖已有正式模型地形评估：{existing}")
    return targets  # type: ignore[return-value]


def main() -> None:
    verify_dependency_versions()
    if args.state_set not in EVALUATION_STATE_SETS:
        raise ValueError(f"未知固定初始状态集：{args.state_set}")
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
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.name != "model_2999.pt":
        raise FileNotFoundError(f"正式评估只接受存在的 model_2999.pt：{checkpoint}")
    record, saved_budget = _checkpoint_metadata(checkpoint)
    method = TASK_TO_METHOD[args.agent_task]
    training_seed = int(record.get("seed", -1))
    if record.get("task") != args.agent_task:
        raise ValueError(f"检查点任务 {record.get('task')} 与请求 {args.agent_task} 不一致。")
    validate_evaluation_checkpoint(
        checkpoint.parent,
        task=args.agent_task,
        energy_budget_j=saved_budget,
    )
    checkpoint_hash = _file_sha256(checkpoint)
    output_dir = _result_directory(method, training_seed, protocol)
    episode_path, summary_path, pairing_path = _output_targets(output_dir)

    env_cfg = load_cfg_from_registry(TASK_ONLY_ID, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args.agent_task, "rsl_rl_cfg_entry_point")
    env_cfg = configure_evaluation(env_cfg, state_set=args.state_set)
    step_dt = _configure_protocol(env_cfg, protocol)
    _configure_terrain(env_cfg, protocol)
    env_cfg.sim.device = args.device
    env_cfg.log_dir = str(checkpoint.parent)
    agent_cfg.device = args.device
    agent_cfg.seed = protocol.env_seed
    if args.agent_task == ECO_ID:
        if saved_budget is None or saved_budget <= 0.0:
            raise ValueError("PACE-ECO 检查点缺少正的训练预算。")
        agent_cfg.algorithm.energy_budget_j = saved_budget
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg,
        importlib.metadata.version("rsl-rl-lib"),
    )

    random.seed(protocol.env_seed)
    np.random.seed(protocol.env_seed)
    torch.manual_seed(protocol.env_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(protocol.env_seed)

    env = None
    try:
        raw_env = gym.make(TASK_ONLY_ID, cfg=env_cfg)
        env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        print(f"[PACE] 严格加载正式 actor：{checkpoint}", flush=True)
        runner.load(
            str(checkpoint),
            load_cfg=ACTOR_ONLY_LOAD_CFG,
            strict=True,
            map_location=args.device,
        )
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        observations = env.get_observations().to(args.device)
        if "policy" not in observations.keys():
            raise RuntimeError("正式评估环境没有 policy 观测组。")
        policy_obs_dim = int(observations["policy"].shape[-1])
        if policy_obs_dim != args.expected_policy_obs_dim:
            raise RuntimeError(
                f"正式 actor 输入应为 {args.expected_policy_obs_dim} 维，实际为 {policy_obs_dim} 维。"
            )

        raw = env.unwrapped
        robot = raw.scene["robot"]
        start_pos_w = robot.data.root_pos_w.clone()
        start_x_w = start_pos_w[:, 0].clone()
        terrain_levels, terrain_types, start_positions = _terrain_assignment(
            raw_env,
            start_pos_w,
            protocol,
        )
        accumulator = TerrainEpisodeAccumulator(
            protocol.num_envs,
            device=raw.device,
            step_dt=step_dt,
            goal_distance_m=protocol.goal_distance_m,
            command_x_mps=protocol.command_x_mps,
        )
        max_steps = protocol.max_steps(step_dt)
        print(
            f"[PACE] 开始零样本地形评估：方法={method}，训练seed={training_seed}，"
            f"地形={protocol.terrain}，环境数={protocol.num_envs}。",
            flush=True,
        )
        for step_index in range(1, max_steps + 1):
            if not accumulator.active.any():
                break
            if not simulation_app.is_running():
                raise RuntimeError("仿真在地形评估完成前停止。")
            with torch.inference_mode():
                actions = policy(observations)
                observations, _, dones, extras = env.step(actions)
            required = (
                "pace_energy_step",
                "pace_energy_components",
                "pace_eval_root_pos_w",
                "pace_eval_root_lin_vel_b",
            )
            missing = [name for name in required if name not in extras]
            if missing:
                raise RuntimeError(f"正式环境缺少评估 extras：{missing}")
            components = extras["pace_energy_components"]
            time_outs = extras.get("time_outs", raw.reset_time_outs).bool()
            done_mask = dones.bool()
            accumulator.update(
                step_index=step_index,
                progress_m=extras["pace_eval_root_pos_w"][:, 0] - start_x_w,
                forward_velocity_mps=extras["pace_eval_root_lin_vel_b"][:, 0],
                step_energy_j=extras["pace_energy_step"],
                step_electrical_j=components["electrical"],
                step_mechanical_j=components["mechanical"],
                step_potential_j=components["potential"],
                terminated=done_mask & ~time_outs,
                truncated=done_mask & time_outs,
            )
        accumulator.finalize_timeouts(max_steps)
        if _file_sha256(checkpoint) != checkpoint_hash:
            raise RuntimeError("评估期间检查点哈希发生变化。")

        state_count = evaluation_state_count(args.state_set)
        state_sha256 = evaluation_state_definition_sha256(args.state_set)
        metadata = {
            "algorithm": method,
            "training_seed": training_seed,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "agent_task": args.agent_task,
            "eval_task": TASK_ONLY_ID,
            "state_set": args.state_set,
            "state_set_sha256": state_sha256,
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
            "training_energy_budget_j": saved_budget,
        }
        episode_rows = accumulator.episode_rows(metadata)
        for row in episode_rows:
            env_id = int(row["env_id"])
            row.update(
                {
                    "initial_state_id": env_id % state_count,
                    "terrain_level": int(terrain_levels[env_id]),
                    "terrain_type": int(terrain_types[env_id]),
                    "start_x_w": float(start_positions[env_id, 0]),
                    "start_y_w": float(start_positions[env_id, 1]),
                    "start_z_w": float(start_positions[env_id, 2]),
                }
            )
        signature = pairing_signature(episode_rows)
        paired_reference_verified = False
        paired_reference = None
        if args.paired_reference is not None:
            reference_path = Path(args.paired_reference).expanduser().resolve()
            assert_pairing_matches(episode_rows, _read_csv(reference_path))
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
            "state_set": args.state_set,
            "state_set_sha256": state_sha256,
            "num_envs": protocol.num_envs,
            "paired_reference_verified": paired_reference_verified,
            "paired_reference": paired_reference,
        }
        atomic_write_csv(episode_path, episode_rows)
        atomic_write_csv(summary_path, [summary])
        atomic_write_json(pairing_path, pairing_info)
        print(f"[PACE] 逐回合结果：{episode_path}", flush=True)
        print(f"[PACE] 汇总结果：{summary_path}", flush=True)
        print(f"[PACE] 配对信息：{pairing_path}", flush=True)
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
