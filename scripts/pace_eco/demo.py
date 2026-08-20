#!/usr/bin/env python3
"""录制 PACE ANYmal-D 的确定性脚本动作演示。"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="录制 PACE ANYmal-D 脚本动作演示；不加载训练策略。")
parser.add_argument("--task", default="Isaac-PACE-TaskOnly-Flat-Anymal-D-v0")
parser.add_argument("--duration_s", type=float, default=8.0, help="录像时长，单位秒。")
parser.add_argument("--video_fps", type=float, default=10.0, help="视频采样帧率；控制频率仍为 50 Hz。")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output_dir", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import math

import gymnasium as gym
import torch

from isaaclab_tasks.utils import load_cfg_from_registry

import pace_eco_lab  # noqa: F401
from pace_eco_lab.configs.env_cfg import configure_evaluation
from pace_eco_lab.constants import ACTION_DIM, POLICY_DT_S, REGISTERED_TASKS
from pace_eco_lab.reproducibility import verify_dependency_versions


def _scripted_action(step: int, device: str) -> torch.Tensor:
    """生成对称下蹲和小幅侧摆动作，零动作对应官方默认站姿。"""

    time_s = step * POLICY_DT_S
    action = torch.zeros((1, ACTION_DIM), device=device)
    if time_s < 1.0:
        return action

    phase = 2.0 * math.pi * 0.45 * (time_s - 1.0)
    crouch = 0.5 * (1.0 - math.cos(phase))
    sway = math.sin(phase)

    # PACE 顺序：LF、RF、LH、RH；每条腿依次为 HAA、HFE、KFE。
    # 前后腿关节符号相反，使四足同步屈伸，而不是把躯干向一个方向推倒。
    action[0, 1] = 0.35 * crouch
    action[0, 2] = -0.70 * crouch
    action[0, 4] = 0.35 * crouch
    action[0, 5] = -0.70 * crouch
    action[0, 7] = -0.35 * crouch
    action[0, 8] = 0.70 * crouch
    action[0, 10] = -0.35 * crouch
    action[0, 11] = 0.70 * crouch

    action[0, 0] = 0.08 * sway
    action[0, 3] = -0.08 * sway
    action[0, 6] = -0.08 * sway
    action[0, 9] = 0.08 * sway
    return action


def main() -> None:
    verify_dependency_versions()
    if args_cli.task not in REGISTERED_TASKS:
        raise ValueError(f"只允许 PACE 任务：{REGISTERED_TASKS}")
    if args_cli.duration_s <= 0.0:
        raise ValueError("duration_s 必须为正。")
    if not 0.0 < args_cli.video_fps <= 50.0:
        raise ValueError("video_fps 必须位于 (0, 50]。")

    output_dir = (
        Path(args_cli.output_dir).expanduser().resolve()
        if args_cli.output_dir
        else Path("logs/videos").resolve() / f"gpt_PACE脚本动作演示_{datetime.now():%Y%m%d_%H%M%S}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    steps = round(args_cli.duration_s / POLICY_DT_S)
    native_fps = round(1.0 / POLICY_DT_S)
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device or "cpu"
    # RecordVideo（录像封装器）每个环境步采集一帧，因此按原生控制频率渲染。
    env_cfg.sim.render_interval = env_cfg.decimation
    env_cfg.seed = args_cli.seed
    env_cfg = configure_evaluation(env_cfg)
    env_cfg.viewer.eye = (3.0, 2.6, 1.8)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.45)
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.resolution = (320, 180)

    raw_prefix = "gpt_PACE脚本动作演示_原始"
    video_path = output_dir / "gpt_PACE脚本动作演示.mp4"
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    # ManagerBasedRLEnv（管理器式强化学习环境）先完成确定性重置，再接录像封装器。
    # 这与 Isaac Lab 官方 play.py 一样，让 step_trigger 在首个 step 后启动录制。
    env.reset(seed=args_cli.seed)
    print("[PACE] 环境重置完成，准备挂载录像封装器。", flush=True)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(output_dir),
        step_trigger=lambda step: step == 0,
        video_length=steps,
        name_prefix=raw_prefix,
        fps=native_fps,
        disable_logger=True,
    )
    print("[PACE] 录像封装器已挂载，开始脚本动作。", flush=True)

    terminations = 0
    max_action = 0.0
    try:
        with torch.inference_mode():
            for step in range(steps):
                action = _scripted_action(step, env.unwrapped.device)
                max_action = max(max_action, float(action.abs().max().item()))
                if step == 0:
                    print("[PACE] 正在执行第一个仿真步。", flush=True)
                _, _, terminated, truncated, _ = env.step(action)
                if step == 0:
                    print("[PACE] 第一个仿真步及首帧录制完成。", flush=True)
                terminations += int((terminated | truncated).sum().item())
    finally:
        env.close()

    raw_videos = sorted(output_dir.glob(f"{raw_prefix}*.mp4"))
    if len(raw_videos) != 1 or raw_videos[0].stat().st_size == 0:
        raise RuntimeError(f"原始录像生成异常：{raw_videos}")
    raw_video_path = raw_videos[0]
    if math.isclose(args_cli.video_fps, native_fps):
        raw_video_path.replace(video_path)
    else:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(raw_video_path),
                "-vf",
                f"fps={args_cli.video_fps:g}",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(video_path),
            ],
            check=True,
        )
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise RuntimeError(f"录像文件没有生成：{video_path}")

    report = {
        "任务": args_cli.task,
        "演示类型": "确定性脚本关节动作，不是训练策略",
        "仿真设备": str(env_cfg.sim.device),
        "时长_s": args_cli.duration_s,
        "策略步数": steps,
        "原始视频帧率": native_fps,
        "输出视频帧率": args_cli.video_fps,
        "分辨率": list(env_cfg.viewer.resolution),
        "最大绝对动作": max_action,
        "终止或超时次数": terminations,
        "原始视频": str(raw_video_path),
        "视频": str(video_path),
    }
    report_path = output_dir / "gpt_演示说明.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[PACE] 演示视频：{video_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
