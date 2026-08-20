#!/usr/bin/env python3
"""检查 PACE 环境维度、数值、执行器与开环回放。"""

from __future__ import annotations

import argparse
import faulthandler
import json
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

from check_env_orchestration import (
    OPEN_LOOP_REPLAY_KEY,
    build_phase_arguments,
    merge_phase_reports,
)

parser = argparse.ArgumentParser(description="运行 PACE 环境验收检查。")
parser.add_argument("--task", default="Isaac-PACE-TaskOnly-Flat-Anymal-D-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--open_loop_replay", action="store_true", help="额外运行 PACE 悬空轨迹诊断。")
parser.add_argument("--replay_steps", type=int, default=None, help="回放策略步数；默认使用完整轨迹。")
parser.add_argument(
    "--_pace_phase",
    choices=("standard", "replay"),
    default=None,
    help=argparse.SUPPRESS,
)
parser.add_argument("--_pace_result_path", type=Path, default=None, help=argparse.SUPPRESS)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


def _write_report(report: dict[str, object], output_path: Path | None = None) -> Path:
    if output_path is None:
        output_dir = Path("logs/checks").resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"gpt_环境检查_{datetime.now():%Y%m%d_%H%M%S}.json"
    else:
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def _load_phase_report(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"阶段报告不是 JSON 对象：{path}")
    return payload


def _run_isolated_full_acceptance() -> int:
    """分别启动两个 Isaac Sim 进程并合并验收结果。"""

    output_dir = Path("logs/checks").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()
    print(
        "[PACE] 完整验收将分为两个隔离进程，避免重复创建 SimulationContext："
        "常规环境检查 -> 固定机身开环回放。",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="gpt_完整环境验收_", dir=output_dir) as temporary_dir:
        temporary_path = Path(temporary_dir)
        phase_paths = {
            "standard": temporary_path / "常规环境检查.json",
            "replay": temporary_path / "开环回放.json",
        }
        phase_names = {
            "standard": "常规环境检查",
            "replay": "固定机身开环回放",
        }
        for phase in ("standard", "replay"):
            print(f"[PACE] 开始独立阶段：{phase_names[phase]}。", flush=True)
            child_arguments = build_phase_arguments(sys.argv[1:], phase, phase_paths[phase])
            completed = subprocess.run([sys.executable, str(script_path), *child_arguments], check=False)
            if completed.returncode != 0:
                print(
                    f"[PACE] 阶段失败：{phase_names[phase]}，退出码 {completed.returncode}。",
                    file=sys.stderr,
                    flush=True,
                )
                return completed.returncode
            print(f"[PACE] 阶段完成：{phase_names[phase]}。", flush=True)

        report = merge_phase_reports(
            _load_phase_report(phase_paths["standard"]),
            _load_phase_report(phase_paths["replay"]),
        )

    output_path = _write_report(report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"[PACE] 完整检查报告：{output_path}", flush=True)
    return 0


if args_cli.open_loop_replay and args_cli._pace_phase is None:
    raise SystemExit(_run_isolated_full_acceptance())
if args_cli._pace_phase is not None and args_cli._pace_result_path is None:
    parser.error("内部验收阶段必须提供 --_pace_result_path。")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

from isaaclab.utils.types import ArticulationActions
from isaaclab_tasks.utils import load_cfg_from_registry

import pace_eco_lab  # noqa: F401
from pace_eco_lab.constants import (
    ACTION_DIM,
    ACTION_SCALE,
    ACTOR_OBSERVATION_DIM,
    CRITIC_OBSERVATION_DIM,
    GLOBAL_DELAY_STEPS,
    PACE_JOINT_NAMES,
    PACE_REPLAY_PATH,
    PACE_REPLAY_SHA256,
    REGISTERED_TASKS,
)
from pace_eco_lab.mdp.parameters import file_sha256, load_pace_parameters, map_joint_values
from pace_eco_lab.reproducibility import verify_dependency_versions


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} 含 NaN 或 Inf。")


def _load_replay() -> dict:
    actual = file_sha256(PACE_REPLAY_PATH)
    if actual != PACE_REPLAY_SHA256:
        raise ValueError(f"回放数据哈希不匹配：{actual}")
    container = np.load(PACE_REPLAY_PATH, allow_pickle=True)
    if container.shape != () or container.dtype != object:
        raise ValueError("PACE data.npy 顶层格式错误。")
    payload = container.item()
    if not isinstance(payload, dict) or "real" not in payload:
        raise ValueError("PACE data.npy 缺少 real 轨迹。")
    return payload["real"]


def _check_bias_and_delay(actuator, joint_pos: torch.Tensor, joint_vel: torch.Tensor) -> None:
    """用解析 PD 结果验证偏置一次应用与三物理步力矩延迟。"""

    zeros = torch.zeros_like(joint_pos)
    actuator.reset()
    actuator.compute(
        ArticulationActions(
            joint_positions=joint_pos.clone(),
            joint_velocities=zeros.clone(),
            joint_efforts=zeros.clone(),
        ),
        joint_pos,
        joint_vel,
    )
    expected = actuator.stiffness * actuator.encoder_bias - actuator.damping * joint_vel
    if not torch.allclose(actuator.computed_effort, expected, atol=1.0e-5, rtol=1.0e-5):
        raise AssertionError("编码器偏置没有按 q_des-(q-bias) 恰好进入一次 PD 误差。")

    actuator.reset()
    first_applied = None
    for requested_torque in (1.0, 2.0, 3.0, 4.0):
        target = joint_pos - actuator.encoder_bias + requested_torque / actuator.stiffness
        action = actuator.compute(
            ArticulationActions(
                joint_positions=target,
                joint_velocities=joint_vel.clone(),
                joint_efforts=zeros.clone(),
            ),
            joint_pos,
            joint_vel,
        )
        if first_applied is None:
            first_applied = action.joint_efforts.clone()
    if not torch.allclose(action.joint_efforts, first_applied, atol=1.0e-5, rtol=0.0):
        raise AssertionError("力矩序列的第 4 个物理步没有输出第 1 个物理步力矩。")
    if not torch.allclose(actuator.applied_effort, action.joint_efforts, atol=0.0, rtol=0.0):
        raise AssertionError("applied_effort 未反映延迟后实际施加力矩。")
    actuator.reset()


def _run_replay(env) -> dict[str, float | int]:
    replay = _load_replay()
    desired = np.asarray(replay["des_dof_pos"], dtype=np.float32)
    measured = np.asarray(replay["dof_pos"], dtype=np.float32)
    if desired.shape != measured.shape or desired.ndim != 2 or desired.shape[1] != ACTION_DIM:
        raise ValueError(f"PACE 回放关节数据形状错误：{desired.shape}, {measured.shape}")
    stride = 8
    total = desired.shape[0] // stride
    if args_cli.replay_steps is not None:
        total = min(total, args_cli.replay_steps)
    print(f"[PACE] 开环回放开始：共 {total} 个策略步。", flush=True)
    env.reset()
    robot = env.unwrapped.scene["robot"]
    joint_ids, names = robot.find_joints(list(PACE_JOINT_NAMES), preserve_order=True)
    if tuple(names) != PACE_JOINT_NAMES:
        raise AssertionError(f"回放关节名称顺序错误：{names}")
    initial_position = torch.as_tensor(measured[0], device=env.unwrapped.device).unsqueeze(0)
    initial_velocity = torch.as_tensor(
        np.asarray(replay["dof_vel"], dtype=np.float32)[0],
        device=env.unwrapped.device,
    ).unsqueeze(0)
    robot.write_joint_state_to_sim(initial_position, initial_velocity, joint_ids=joint_ids)
    default = robot.data.default_joint_pos[0, joint_ids].detach().cpu().numpy()
    squared_error = 0.0
    samples = 0
    progress_interval = max(total // 10, 1)
    for policy_index in range(total):
        source_index = policy_index * stride
        action_np = (desired[source_index] - default) / ACTION_SCALE
        action = torch.as_tensor(action_np, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
        env.step(action)
        simulated = robot.data.joint_pos[0, joint_ids].detach().cpu().numpy()
        comparison_index = min(source_index + stride, measured.shape[0] - 1)
        squared_error += float(np.square(simulated - measured[comparison_index]).sum())
        samples += ACTION_DIM
        completed_steps = policy_index + 1
        if completed_steps % progress_interval == 0 or completed_steps == total:
            print(f"[PACE] 开环回放进度：{completed_steps}/{total}。", flush=True)
    return {
        "策略步数": total,
        "关节样本数": samples,
        "关节位置RMSE_rad": float(np.sqrt(squared_error / max(samples, 1))),
    }


def _run_replay_check() -> dict[str, object]:
    replay_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    replay_cfg.scene.num_envs = 1
    replay_cfg.sim.device = args_cli.device or "cuda:0"
    replay_cfg.seed = args_cli.seed
    replay_cfg.scene.robot.spawn.articulation_props.fix_root_link = True
    replay_cfg.scene.robot.init_state.pos = (0.0, 0.0, 1.0)
    replay_env = gym.make(args_cli.task, cfg=replay_cfg)
    try:
        return {OPEN_LOOP_REPLAY_KEY: _run_replay(replay_env)}
    finally:
        replay_env.close()


def main() -> None:
    verify_dependency_versions()
    if args_cli.task not in REGISTERED_TASKS:
        raise ValueError(f"只允许 PACE 任务：{REGISTERED_TASKS}")
    if args_cli.num_envs <= 0 or args_cli.steps <= 0:
        raise ValueError("num_envs 和 steps 必须为正。")
    if args_cli._pace_phase == "replay":
        report = _run_replay_check()
        output_path = _write_report(report, args_cli._pace_result_path)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        print(f"[PACE] 阶段检查报告：{output_path}", flush=True)
        return

    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device or "cuda:0"
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    observations, _ = env.reset()

    if observations["policy"].shape != (args_cli.num_envs, ACTOR_OBSERVATION_DIM):
        raise AssertionError(f"Actor 观察维度错误：{observations['policy'].shape}")
    if observations["critic"].shape != (args_cli.num_envs, CRITIC_OBSERVATION_DIM):
        raise AssertionError(f"Critic 观察维度错误：{observations['critic'].shape}")
    if env.unwrapped.action_manager.total_action_dim != ACTION_DIM:
        raise AssertionError(f"动作维度错误：{env.unwrapped.action_manager.total_action_dim}")

    action_term = env.unwrapped.action_manager._terms["joint_pos"]
    if tuple(action_term._joint_names) != PACE_JOINT_NAMES:
        raise AssertionError(f"动作关节顺序错误：{action_term._joint_names}")
    robot = env.unwrapped.scene["robot"]
    actuator = robot.actuators["pace_legs"]
    parameters = load_pace_parameters()
    expected_bias = torch.as_tensor(
        map_joint_values(parameters.joint_bias, actuator.joint_names),
        device=env.unwrapped.device,
        dtype=actuator.encoder_bias.dtype,
    )
    if not torch.allclose(actuator.encoder_bias[0], expected_bias, atol=1.0e-7, rtol=0.0):
        raise AssertionError("执行器编码器偏置与 PACE 参数不一致。")
    if not torch.all(actuator.torques_delay_buffer.time_lags == GLOBAL_DELAY_STEPS):
        raise AssertionError("电机力矩延迟不是固定 3 个物理步。")
    _check_bias_and_delay(
        actuator,
        robot.data.joint_pos[:, actuator.joint_indices].clone(),
        robot.data.joint_vel[:, actuator.joint_indices].clone(),
    )
    env.reset()

    max_abs_torque = 0.0
    max_abs_energy = 0.0
    terminated_count = 0
    generator = torch.Generator(device=env.unwrapped.device)
    generator.manual_seed(args_cli.seed)
    progress_interval = max(args_cli.steps // 4, 1)
    for step_index in range(args_cli.steps):
        action = 0.2 * torch.randn(
            args_cli.num_envs,
            ACTION_DIM,
            generator=generator,
            device=env.unwrapped.device,
        )
        observations, _, terminated, truncated, extras = env.step(action)
        _assert_finite("policy 观察", observations["policy"])
        _assert_finite("critic 观察", observations["critic"])
        _assert_finite("奖励", env.unwrapped.reward_buf)
        _assert_finite("策略步能耗", extras["pace_energy_step"])
        for name, component in extras["pace_energy_components"].items():
            _assert_finite(f"能耗分量 {name}", component)
        max_abs_torque = max(max_abs_torque, float(robot.data.applied_torque.abs().max().item()))
        max_abs_energy = max(max_abs_energy, float(extras["pace_energy_step"].abs().max().item()))
        terminated_count += int((terminated | truncated).sum().item())
        completed_steps = step_index + 1
        if completed_steps % progress_interval == 0 or completed_steps == args_cli.steps:
            print(f"[PACE] 常规环境检查进度：{completed_steps}/{args_cli.steps}。", flush=True)
    if max_abs_torque > 80.0 + 1.0e-4:
        raise AssertionError(f"实际力矩超过 80 N·m：{max_abs_torque}")

    report: dict[str, object] = {
        "任务": args_cli.task,
        "环境数": args_cli.num_envs,
        "检查步数": args_cli.steps,
        "Actor观察维度": ACTOR_OBSERVATION_DIM,
        "Critic观察维度": CRITIC_OBSERVATION_DIM,
        "动作维度": ACTION_DIM,
        "最大绝对力矩_Nm": max_abs_torque,
        "最大绝对策略步能耗_J": max_abs_energy,
        "终止总数": terminated_count,
        "偏置只进入执行器PD误差": True,
        "固定力矩物理步延迟": GLOBAL_DELAY_STEPS,
    }
    env.close()
    output_path = _write_report(report, args_cli._pace_result_path)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    report_name = "阶段检查报告" if args_cli._pace_phase is not None else "检查报告"
    print(f"[PACE] {report_name}：{output_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # 必须在 SimulationApp.close() 之前输出。非 RTX GPU 上的关闭流程可能卡住，
        # 否则主体异常会一直被 finally 中的关闭调用遮挡。
        print("[PACE] 环境检查主体异常，原始调用栈如下：", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise
    finally:
        print("[PACE] 正在关闭 Isaac Sim。", flush=True)
        # 正常关闭通常只需数秒；若原生关闭流程卡住，保留 Python 调用栈供诊断。
        faulthandler.dump_traceback_later(30.0, repeat=False, file=sys.stderr)
        try:
            simulation_app.close()
        finally:
            faulthandler.cancel_dump_traceback_later()
