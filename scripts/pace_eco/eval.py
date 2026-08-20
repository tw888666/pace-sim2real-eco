#!/usr/bin/env python3
"""在无观察噪声、指定确定性状态集下评估 PACE 策略。"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from time import monotonic

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="评估 PACE 策略的成功率和回合能耗。")
parser.add_argument("--task", default=None)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--episodes", type=int, default=200)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--energy_budget_j", type=float, default=None)
parser.add_argument("--warmup_s", type=float, default=5.0, help="稳态速度统计前忽略的秒数。")
parser.add_argument(
    "--state_set",
    default="calibration_v1",
    help="评估状态集：calibration_v1 用于预算标定，holdout_v1 只用于最终留出测试。",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.task is None or args_cli.checkpoint is None:
    parser.error("--task 和 --checkpoint 均为必填参数。")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import importlib.metadata
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import load_cfg_from_registry

import pace_eco_lab  # noqa: F401
from pace_eco_lab.configs.env_cfg import configure_evaluation
from pace_eco_lab.constants import ECO_ID, REGISTERED_TASKS
from pace_eco_lab.mdp.parameters import file_sha256
from pace_eco_lab.mdp.resets import (
    EVALUATION_STATE_SETS,
    evaluation_state_count,
    evaluation_state_definition_sha256,
)
from pace_eco_lab.reproducibility import validate_evaluation_checkpoint, verify_dependency_versions
from scripts.pace_eco.eval_metrics import (
    COORDINATION_METRIC_DEFINITIONS,
    COORDINATION_PROTOCOL_VERSION,
    FOOT_BODY_NAMES,
    CoordinationAccumulator,
)


def _format_metric(value: object, unit: str) -> str:
    if value is None:
        return "无稳态样本"
    return f"{float(value):.4f} {unit}"


def main() -> None:
    verify_dependency_versions()
    if args_cli.task not in REGISTERED_TASKS:
        raise ValueError(f"只允许 PACE 任务：{REGISTERED_TASKS}")
    if args_cli.state_set not in EVALUATION_STATE_SETS:
        raise ValueError(f"未知评估状态集：{args_cli.state_set}；允许值={EVALUATION_STATE_SETS}")
    if args_cli.episodes < 200:
        raise ValueError("正式校准评估至少需要 200 个完整回合。")
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"检查点不存在：{checkpoint}")

    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    device = args_cli.device or "cuda:0"
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = device
    if args_cli.task == ECO_ID:
        if args_cli.energy_budget_j is None or args_cli.energy_budget_j <= 0:
            raise ValueError("PACE-ECO 评估必须提供训练时相同的 --energy_budget_j。")
        agent_cfg.algorithm.energy_budget_j = args_cli.energy_budget_j
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
    validate_evaluation_checkpoint(
        checkpoint.parent,
        task=args_cli.task,
        energy_budget_j=args_cli.energy_budget_j,
    )
    env_cfg = configure_evaluation(env_cfg, state_set=args_cli.state_set)

    env_cfg.log_dir = str(checkpoint.parent)
    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint), map_location=agent_cfg.device)
    policy = runner.get_inference_policy(device=wrapped.device)
    observations = wrapped.get_observations()

    raw = wrapped.unwrapped
    robot = raw.scene["robot"]
    contact_sensor = raw.scene.sensors["contact_forces"]
    contact_threshold_n = float(contact_sensor.cfg.force_threshold)
    foot_asset_ids, foot_asset_names = robot.find_bodies(FOOT_BODY_NAMES, preserve_order=True)
    foot_sensor_ids, foot_sensor_names = contact_sensor.find_bodies(
        FOOT_BODY_NAMES,
        preserve_order=True,
    )
    if tuple(foot_asset_names) != FOOT_BODY_NAMES or tuple(foot_sensor_names) != FOOT_BODY_NAMES:
        raise RuntimeError(
            "ANYmal D 四足解析结果不符合协调性协议："
            f"机器人刚体={foot_asset_names}，接触传感器={foot_sensor_names}。"
        )
    warmup_steps = round(args_cli.warmup_s / raw.step_dt)
    velocity_sum = torch.zeros(raw.num_envs, device=raw.device)
    velocity_count = torch.zeros(raw.num_envs, device=raw.device)
    coordination = CoordinationAccumulator.create(
        raw.num_envs,
        raw.action_manager.total_action_dim,
        history_length=3,
        device=raw.device,
        dtype=robot.data.root_lin_vel_b.dtype,
    )
    component_names = ("electrical", "mechanical", "potential")
    component_labels = {
        "electrical": "电气能耗_J",
        "mechanical": "机械能耗_J",
        "potential": "势能能耗_J",
    }
    component_episode_running = {
        name: torch.zeros(raw.num_envs, device=raw.device) for name in component_names
    }
    state_count = evaluation_state_count(args_cli.state_set)
    state_definition_sha256 = evaluation_state_definition_sha256(args_cli.state_set)
    state_targets = [
        args_cli.episodes // state_count + int(index < args_cli.episodes % state_count)
        for index in range(state_count)
    ]
    state_completed = [0] * state_count
    results: list[dict[str, object]] = []
    evaluation_steps = 0
    evaluation_started = monotonic()
    last_progress_print = evaluation_started
    print(
        f"[PACE] 检查点加载完成，开始正式评估：{args_cli.episodes} 回合，"
        f"{raw.num_envs} 个并行环境，状态集={args_cli.state_set}，"
        f"{state_count} 个固定初始状态。",
        flush=True,
    )
    print(
        "[PACE] 评估期间每完成一个有效回合都会打印进度；"
        "若暂时没有回合结束，则至少每 30 秒打印一次运行状态。",
        flush=True,
    )
    print(
        f"[PACE] 协调性协议 {COORDINATION_PROTOCOL_VERSION} 已启用："
        f"四足顺序={','.join(FOOT_BODY_NAMES)}，只读统计，不改变策略动作或成功判定。",
        flush=True,
    )
    while len(results) < args_cli.episodes and simulation_app.is_running():
        evaluation_steps += 1
        current_speed = robot.data.root_lin_vel_b[:, 0].clone()
        steady_mask = raw.episode_length_buf >= warmup_steps
        velocity_sum.add_(current_speed * steady_mask)
        velocity_count.add_(steady_mask)
        foot_forces = contact_sensor.data.net_forces_w[:, foot_sensor_ids]
        coordination.update_state(
            lateral_velocity=robot.data.root_lin_vel_b[:, 1],
            vertical_velocity=robot.data.root_lin_vel_w[:, 2],
            roll_pitch_angular_velocity=robot.data.root_ang_vel_b[:, :2],
            foot_contact=torch.linalg.vector_norm(foot_forces, dim=-1) > contact_threshold_n,
            foot_speed=torch.linalg.vector_norm(
                robot.data.body_lin_vel_w[:, foot_asset_ids],
                dim=-1,
            ),
            touchdown=contact_sensor.compute_first_contact(raw.step_dt)[:, foot_sensor_ids],
            steady_mask=steady_mask,
        )
        with torch.inference_mode():
            actions = policy(observations)
            evaluated_actions = (
                torch.clamp(actions, -wrapped.clip_actions, wrapped.clip_actions)
                if wrapped.clip_actions is not None
                else actions
            )
            coordination.update_action(evaluated_actions, steady_mask)
            observations, _, dones, extras = wrapped.step(actions)
            policy.reset(dones)
        step_components = extras["pace_energy_components"]
        for name in component_names:
            component_episode_running[name].add_(step_components[name])
        completed_ids = extras["pace_energy_episode_mask"].nonzero(as_tuple=False).squeeze(-1)
        for env_id in completed_ids.tolist():
            if len(results) >= args_cli.episodes:
                break
            state_index = env_id % state_count
            # 异步环境中失败回合更短；按固定状态配额收集，避免失败状态因
            # 更快 reset 而在前 N 个完成回合中被过度采样。
            if state_completed[state_index] >= state_targets[state_index]:
                velocity_sum[env_id] = 0.0
                velocity_count[env_id] = 0.0
                for value in component_episode_running.values():
                    value[env_id] = 0.0
                coordination.reset([env_id])
                continue
            mean_speed = float((velocity_sum[env_id] / velocity_count[env_id].clamp_min(1)).item())
            timed_out = bool(raw.reset_time_outs[env_id].item())
            illegally_terminated = bool(raw.reset_terminated[env_id].item())
            energy_j = float(extras["pace_energy_episode"][env_id].item())
            component_values = {
                name: float(component_episode_running[name][env_id].item()) for name in component_names
            }
            component_sum_j = sum(component_values.values())
            row = {
                "回合": len(results),
                "环境编号": env_id,
                "固定初始状态编号": state_index,
                "完整20秒": timed_out,
                "非法终止": illegally_terminated,
                "稳态平均前进速度_m_s": mean_speed,
                "回合能耗_J": energy_j,
                **{component_labels[name]: component_values[name] for name in component_names},
                "能耗分项残差_J": energy_j - component_sum_j,
                "成功": timed_out and not illegally_terminated and 0.8 <= mean_speed <= 1.2,
                **coordination.episode_metrics(env_id, raw.step_dt),
            }
            results.append(row)
            state_completed[state_index] += 1
            elapsed_s = round(monotonic() - evaluation_started)
            print(
                f"[PACE] 评估进度 {len(results)}/{args_cli.episodes} | "
                f"固定状态 {state_index}: "
                f"{state_completed[state_index]}/{state_targets[state_index]} | "
                f"{'成功' if row['成功'] else '失败'} | "
                f"完整20秒={'是' if timed_out else '否'} | "
                f"非法终止={'是' if illegally_terminated else '否'} | "
                f"稳态速度={mean_speed:.4f} m/s | "
                f"回合能耗={energy_j:.2f} J "
                f"(电气={component_values['electrical']:.2f}, "
                f"机械={component_values['mechanical']:.2f}, "
                f"势能={component_values['potential']:.2f}) | "
                f"协调性: 横向速度RMS={_format_metric(row['机身横向速度RMS_m_s'], 'm/s')}, "
                "姿态角速度RMS="
                f"{_format_metric(row['机身横滚俯仰角速度RMS_rad_s'], 'rad/s')}, "
                f"触地足速={_format_metric(row['三步窗触地足速均值_m_s'], 'm/s')} | "
                f"已运行={elapsed_s} s",
                flush=True,
            )
            last_progress_print = monotonic()
            velocity_sum[env_id] = 0.0
            velocity_count[env_id] = 0.0
            for value in component_episode_running.values():
                value[env_id] = 0.0
            coordination.reset([env_id])
        now = monotonic()
        if now - last_progress_print >= 30.0:
            print(
                f"[PACE] 评估仍在运行 | 已完成 {len(results)}/{args_cli.episodes} 回合 | "
                f"策略步={evaluation_steps} | 已运行={round(now - evaluation_started)} s",
                flush=True,
            )
            last_progress_print = now

    if len(results) < args_cli.episodes:
        raise RuntimeError(f"仿真提前结束，只收集到 {len(results)} 个回合。")
    successful = [row for row in results if row["成功"]]
    successful_energy_mean = (
        sum(float(row["回合能耗_J"]) for row in successful) / len(successful) if successful else None
    )
    successful_component_means = {
        name: (
            sum(float(row[component_labels[name]]) for row in successful) / len(successful)
            if successful
            else None
        )
        for name in component_names
    }
    successful_component_fractions = {
        name: (
            successful_component_means[name] / successful_energy_mean
            if successful_energy_mean not in (None, 0.0)
            else None
        )
        for name in component_names
    }
    successful_coordination_means = {
        field: (
            sum(float(row[field]) for row in successful if row[field] is not None)
            / sum(row[field] is not None for row in successful)
            if any(row[field] is not None for row in successful)
            else None
        )
        for field in CoordinationAccumulator.SUMMARY_FIELDS
    }
    summary = {
        "任务": args_cli.task,
        "检查点": str(checkpoint),
        "检查点SHA256": file_sha256(checkpoint),
        "评估脚本SHA256": file_sha256(Path(__file__).resolve()),
        "协调性指标模块SHA256": file_sha256(Path(__file__).with_name("eval_metrics.py")),
        "协调性协议版本": COORDINATION_PROTOCOL_VERSION,
        "评估状态集": args_cli.state_set,
        "评估状态定义SHA256": state_definition_sha256,
        "回合数": len(results),
        "成功回合数": len(successful),
        "成功率": len(successful) / len(results),
        "成功回合平均能耗_J": successful_energy_mean,
        "成功回合平均电气能耗_J": successful_component_means["electrical"],
        "成功回合平均机械能耗_J": successful_component_means["mechanical"],
        "成功回合平均势能能耗_J": successful_component_means["potential"],
        "成功回合电气能耗占比": successful_component_fractions["electrical"],
        "成功回合机械能耗占比": successful_component_fractions["mechanical"],
        "成功回合势能能耗占比": successful_component_fractions["potential"],
        "成功回合平均能耗分项残差_J": (
            sum(float(row["能耗分项残差_J"]) for row in successful) / len(successful)
            if successful
            else None
        ),
        "成功回合协调性均值": successful_coordination_means,
        "协调性指标定义": COORDINATION_METRIC_DEFINITIONS,
        "协调性判定用途": "描述性比较，不单独设置通过或失败阈值。",
        "四足顺序": list(FOOT_BODY_NAMES),
        "足接触阈值_N": contact_threshold_n,
        "触地足速历史策略步数": 3,
        "评估动作裁剪": wrapped.clip_actions,
        "预算标定可用": len(successful) / len(results) >= 0.95,
        "成功定义": "完整20秒、无非法终止且稳态平均前进速度位于[0.8, 1.2] m/s",
        "稳态预热_s": args_cli.warmup_s,
        "固定初始状态数量": state_count,
        "各固定状态回合数": state_completed,
    }
    output_dir = checkpoint.parent / (
        f"gpt_评估_{args_cli.state_set}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "gpt_逐回合结果.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (output_dir / "gpt_评估摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[PACE] 评估结果：{output_dir}")
    wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
