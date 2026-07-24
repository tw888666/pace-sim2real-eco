"""Run an independent full-episode dual evaluation.

This script launches Isaac Sim and therefore must be run by the user on an
available GPU. It reads an immutable request, never writes the checkpoint, and
atomically publishes one result JSON document.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--request", required=True, help="request.json created by freeze_dual_checkpoint.py")
parser.add_argument("--result", default=None)
parser.add_argument("--task", default="Isaac-Pace-Eco-Anymal-D-Flat-Play-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=12345)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import isaaclab_tasks  # noqa: F401
import pace_sim2real.tasks  # noqa: F401
from pace_sim2real.dual import DualEvaluationResult, atomic_write_json, file_sha256


def main() -> None:
    request_path = Path(args.request).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("schema_version") != 1:
        raise ValueError("unsupported dual request schema")
    checkpoint = Path(request["checkpoint_path"]).resolve()
    if file_sha256(checkpoint) != request["checkpoint_sha256"]:
        raise ValueError("checkpoint hash differs from the immutable dual request")
    if args.num_envs < 1:
        raise ValueError("num_envs must be positive")

    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.sim.device = args.device
    env_cfg.pace_energy.episode_budget_j = float(request["budget_j"])
    agent_cfg.device = args.device

    raw_env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint), map_location=args.device)
    runner.alg.eval_mode()

    obs = env.get_observations().to(args.device)
    with torch.inference_mode():
        initial_cost_values = runner.alg.cost_critic(obs).squeeze(-1).clone()
    active = torch.ones(args.num_envs, dtype=torch.bool, device=args.device)
    accumulated_cost = torch.zeros(args.num_envs, device=args.device)
    final_cost = torch.zeros(args.num_envs, device=args.device)
    final_energy = torch.zeros(args.num_envs, device=args.device)
    success = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)

    while active.any() and simulation_app.is_running():
        with torch.inference_mode():
            actions = runner.alg.actor(obs, stochastic_output=True)
            obs, _, dones, extras = env.step(actions)
            step_cost = extras["pace_cost"].to(args.device)
            accumulated_cost += step_cost * active
            newly_finished = active & dones.bool()
            if newly_finished.any():
                final_cost[newly_finished] = accumulated_cost[newly_finished]
                final_energy[newly_finished] = extras["pace_episode_energy_j"][newly_finished]
                time_outs = extras.get("time_outs", env.unwrapped.reset_time_outs).bool()
                success[newly_finished] = time_outs[newly_finished]
                active[newly_finished] = False

    if active.any():
        raise RuntimeError("simulation stopped before every evaluation environment completed one episode")

    residual = final_cost - initial_cost_values
    target_variance = torch.var(final_cost, unbiased=False)
    explained_variance = 1.0 - torch.var(residual, unbiased=False) / (target_variance + 1.0e-8)
    result = DualEvaluationResult(
        cycle_id=int(request["cycle_id"]),
        checkpoint_path=str(checkpoint),
        checkpoint_sha256=request["checkpoint_sha256"],
        budget_j=float(request["budget_j"]),
        num_episodes=args.num_envs,
        mean_physical_energy_j=float(final_energy.mean().item()),
        mean_augmented_cost=float(final_cost.mean().item()),
        success_rate=float(success.float().mean().item()),
        cost_value_initial_bias=float(torch.abs(residual.mean()).item()),
        cost_explained_variance=float(explained_variance.item()),
    )
    if file_sha256(checkpoint) != request["checkpoint_sha256"]:
        raise RuntimeError("checkpoint changed during dual evaluation")
    result_path = Path(args.result).resolve() if args.result else request_path.with_name("result.json")
    atomic_write_json(result_path, result.__dict__)
    print(result_path)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
