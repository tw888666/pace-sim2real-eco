"""Collect one frozen stochastic actor dataset for all cost-critic controls.

This script launches Isaac Sim and must be run by the user on an available GPU.
The archived policy observation is shared by the zero-time, real-time, and
time-only controls, so no simulator or action-sampling difference can confound
the comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--budget_j", type=float, required=True)
parser.add_argument("--task", default="Isaac-Pace-Eco-Anymal-D-Flat-Play-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=24680)
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
from pace_sim2real.dual import (
    CRITIC_DATASET_SCHEMA_VERSION,
    atomic_torch_save,
    file_sha256,
    validate_critic_dataset,
)


def main() -> None:
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.num_envs < 2 or args.budget_j <= 0.0:
        raise ValueError("num_envs must be at least two and budget_j must be positive")

    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.sim.device = args.device
    env_cfg.pace_energy.episode_budget_j = args.budget_j
    agent_cfg.device = args.device
    torch.manual_seed(args.seed)

    raw_env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    # model_3097 has the legacy 48-input cost critic. The actor and reward
    # critic remain shape-compatible after adding a separate time observation.
    runner.load(
        str(checkpoint),
        load_cfg={
            "actor": True,
            "critic": True,
            "optimizer": False,
            "iteration": True,
            "rnd": True,
            "cost_critic": False,
            "cost_optimizer": False,
            "lagrangian_multiplier": False,
        },
    )
    runner.alg.eval_mode()

    obs = env.get_observations().to(args.device)
    active = torch.ones(args.num_envs, dtype=torch.bool, device=args.device)
    success = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
    physical_energy = torch.zeros(args.num_envs, device=args.device)
    observations = []
    remaining_times = []
    costs = []
    valid_steps = []

    while active.any() and simulation_app.is_running():
        with torch.inference_mode():
            active_before_step = active.clone()
            observations.append(obs["policy"].clone())
            remaining_times.append(
                1.0 - env.unwrapped.episode_length_buf.float() / float(env.unwrapped.max_episode_length)
            )
            actions = runner.alg.actor(obs, stochastic_output=True)
            obs, _, dones, extras = env.step(actions)
            costs.append(extras["pace_cost"].to(args.device).clone())
            valid_steps.append(active_before_step)
            newly_finished = active_before_step & dones.bool()
            if newly_finished.any():
                physical_energy[newly_finished] = extras["pace_episode_energy_j"][newly_finished]
                time_outs = extras.get("time_outs", env.unwrapped.reset_time_outs).bool()
                success[newly_finished] = time_outs[newly_finished]
                active[newly_finished] = False

    if active.any():
        raise RuntimeError("simulation stopped before every dataset episode completed")

    payload = {
        "schema_version": CRITIC_DATASET_SCHEMA_VERSION,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "budget_j": float(args.budget_j),
        "evaluation_seed": args.seed,
        "policy_observation": torch.stack(observations, dim=1).cpu(),
        "remaining_time": torch.stack(remaining_times, dim=1).cpu(),
        "cost": torch.stack(costs, dim=1).cpu(),
        "valid": torch.stack(valid_steps, dim=1).cpu(),
        "success": success.cpu(),
        "physical_energy_j": physical_energy.cpu(),
    }
    validate_critic_dataset(payload)
    output = Path(args.output).resolve()
    atomic_torch_save(output, payload)
    print(f"dataset={output}")
    print(f"dataset_sha256={file_sha256(output)}")
    print(f"episodes={args.num_envs} transitions={int(payload['valid'].sum().item())}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
