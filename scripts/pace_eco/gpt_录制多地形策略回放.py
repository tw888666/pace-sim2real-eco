#!/usr/bin/env python3
"""录制与 Terrain20sWide 正式留出评估一致的单回合策略回放。"""

from __future__ import annotations

import argparse
from datetime import datetime
from hashlib import sha256
from pathlib import Path
import subprocess

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="录制 PACE-ECO Terrain20sWide 训练策略的留出回放。")
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--ppo_seed", required=True, type=int)
parser.add_argument("--batch_index", type=int, default=0, choices=range(4))
parser.add_argument("--env_index", required=True, type=int, choices=range(50))
parser.add_argument("--energy_reference_json", default=None)
parser.add_argument("--duration_s", type=float, default=20.0)
parser.add_argument("--video_fps", type=float, default=15.0)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--resolution", type=int, nargs=2, default=(640, 360), metavar=("宽", "高"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata
import json
import math

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils import math as math_utils
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import load_cfg_from_registry

import pace_eco_lab  # noqa: F401
from pace_eco_lab.configs.multi_terrain_env_cfg import configure_terrain20s_evaluation
from pace_eco_lab.constants import POLICY_DT_S
from pace_eco_lab.envs.terrain20s_env import metadata_labels
from pace_eco_lab.evaluation_states import MULTI_TERRAIN_HOLDOUT_STATE_SET
from pace_eco_lab.multi_terrain_protocol import (
    EVAL_NUM_ENVS,
    EVAL_TERRAIN_ROWS,
    TASK_IDS,
    evaluation_batch_seed,
    terrain_seed,
)
from pace_eco_lab.reproducibility import validate_evaluation_checkpoint, verify_dependency_versions


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _task_parts() -> tuple[str, str]:
    inverse = {task_id: pair for pair, task_id in TASK_IDS.items()}
    try:
        return inverse[args_cli.task]
    except KeyError as error:
        raise ValueError("只允许 Terrain20sWide 多地形任务。") from error


def _load_energy_budget(method: str, terrain: str) -> tuple[float | None, str | None]:
    if method != "eco":
        if args_cli.energy_reference_json is not None:
            raise ValueError("非 PACE-ECO 回放不应提供能耗参考文件。")
        return None, None
    if args_cli.energy_reference_json is None:
        raise ValueError("PACE-ECO 回放必须提供训练时使用的 B_ref 冻结文件。")
    path = Path(args_cli.energy_reference_json).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    reference = float(data.get("B_ref_J", {}).get(terrain, 0.0))
    if reference <= 0.0:
        raise ValueError(f"B_ref 文件缺少 {terrain} 的正值。")
    return 0.8 * reference, _file_sha256(path)


def _unwrap_angle_series(values: list[float]) -> list[float]:
    if not values:
        return []
    result = [values[0]]
    for value in values[1:]:
        delta = (value - result[-1] + math.pi) % (2.0 * math.pi) - math.pi
        result.append(result[-1] + delta)
    return result


def main() -> None:
    verify_dependency_versions()
    method, terrain = _task_parts()
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.name != "model_2999.pt":
        raise FileNotFoundError("回放只接受正式终点 model_2999.pt。")
    if args_cli.duration_s <= 0.0 or args_cli.duration_s > 20.0:
        raise ValueError("duration_s 必须位于 (0, 20] 秒。")
    if not 0.0 < args_cli.video_fps <= round(1.0 / POLICY_DT_S):
        raise ValueError("video_fps 必须位于 (0, 50] 。")
    native_fps = round(1.0 / POLICY_DT_S)
    capture_stride = round(native_fps / args_cli.video_fps)
    if not math.isclose(native_fps / capture_stride, args_cli.video_fps):
        raise ValueError("为保持等间隔取帧，video_fps 必须整除 50 Hz，例如 10、12.5、25、50。")

    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    energy_budget, reference_sha256 = _load_energy_budget(method, terrain)

    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    device = args_cli.device or "cuda:0"
    env_cfg.sim.device = device
    env_cfg.seed = args_cli.ppo_seed
    agent_cfg.seed = args_cli.ppo_seed
    agent_cfg.device = device
    if method == "eco":
        agent_cfg.algorithm.energy_budget_j = energy_budget
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
    record = validate_evaluation_checkpoint(
        checkpoint.parent,
        task=args_cli.task,
        energy_budget_j=energy_budget,
        require_current_implementation=False,
    )
    if int(record.get("seed", -1)) != args_cli.ppo_seed:
        raise ValueError("检查点的 PPO seed 与回放参数不一致。")

    base_terrain_seed = terrain_seed("stage1_holdout", terrain)
    batch_terrain_seed = evaluation_batch_seed(base_terrain_seed, args_cli.batch_index)
    env_cfg = configure_terrain20s_evaluation(
        env_cfg,
        batch_terrain_seed,
        MULTI_TERRAIN_HOLDOUT_STATE_SET,
        args_cli.batch_index,
    )
    # 回放只建立正式评估中被选中的一台机器人，以降低渲染和仿真开销。
    # 地形生成 seed、5x10 网格、目标 tile 以及初始状态索引均与
    # 50 环境正式评估完全一致。
    formal_env_index = args_cli.env_index
    env_cfg.scene.num_envs = 1
    env_cfg.events.reset_base.params["state_index_offset"] += formal_env_index
    env_cfg.events.reset_joints.params["state_index_offset"] += formal_env_index
    env_cfg.log_dir = str(checkpoint.parent)
    env_cfg.sim.render_interval = env_cfg.decimation * capture_stride
    env_cfg.viewer.eye = (-3.0, -2.6, 1.8)
    env_cfg.viewer.lookat = (0.3, 0.0, 0.45)
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.resolution = tuple(args_cli.resolution)

    final_video = output_dir / "gpt_多地形策略回放.mp4"
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    raw = env.unwrapped
    terrain_manager = raw.scene.terrain
    env_ids = torch.arange(1, device=raw.device)
    levels = torch.tensor([formal_env_index % EVAL_TERRAIN_ROWS], device=raw.device)
    types = torch.tensor([formal_env_index // EVAL_TERRAIN_ROWS], device=raw.device)
    terrain_manager.terrain_levels.copy_(levels)
    terrain_manager.terrain_types.copy_(types)
    terrain_manager.env_origins.copy_(terrain_manager.terrain_origins[levels, types])
    raw.reset()

    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint), map_location=agent_cfg.device)
    policy = runner.get_inference_policy(device=wrapped.device)
    observations = wrapped.get_observations()

    robot = wrapped.unwrapped.scene["robot"]
    selected = 0
    origin = wrapped.unwrapped.scene.env_origins[selected, :2].clone()
    start_xy = robot.data.root_pos_w[selected, :2].clone()
    previous_xy = start_xy.clone()
    path_m = 0.0
    max_abs_lateral_m = 0.0
    yaw_values: list[float] = []
    selected_done = False
    completed_metrics: tuple[float, float, float, float] | None = None
    steps = round(args_cli.duration_s / POLICY_DT_S)
    width, height = args_cli.resolution
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            f"{args_cli.video_fps:g}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(final_video),
        ],
        stdin=subprocess.PIPE,
    )
    captured_frames = 0
    try:
        with torch.inference_mode():
            for step in range(steps):
                _, _, yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w[selected : selected + 1])
                yaw_values.append(float(yaw[0]))
                if step % capture_stride == 0:
                    frame = np.asarray(env.render(), dtype=np.uint8)
                    expected_shape = (height, width, 3)
                    if frame.shape != expected_shape:
                        raise RuntimeError(f"渲染帧形状异常：{frame.shape}，预期 {expected_shape}。")
                    if ffmpeg.stdin is None:
                        raise RuntimeError("ffmpeg 标准输入未建立。")
                    ffmpeg.stdin.write(frame.tobytes())
                    captured_frames += 1
                actions = policy(observations)
                observations, _, dones, _ = wrapped.step(actions)
                policy.reset(dones)
                xy = robot.data.root_pos_w[selected, :2].clone()
                path_m += float(torch.linalg.vector_norm(xy - previous_xy))
                previous_xy = xy
                displacement = xy - origin
                max_abs_lateral_m = max(max_abs_lateral_m, abs(float(displacement[1])))
                if bool(dones[selected]):
                    selected_done = True
                    completed_metrics = (
                        float(raw._completed_forward[selected]),
                        float(raw._completed_lateral[selected]),
                        float(raw._completed_path[selected]),
                        float(raw._completed_max_abs_lateral[selected]),
                    )
    finally:
        if ffmpeg.stdin is not None:
            ffmpeg.stdin.close()
        ffmpeg_return_code = ffmpeg.wait()
        if completed_metrics is None:
            end_xy = previous_xy.clone()
        else:
            completed_forward, completed_lateral, path_m, max_abs_lateral_m = completed_metrics
            end_xy = origin + torch.tensor(
                [completed_forward, completed_lateral], device=origin.device, dtype=origin.dtype
            )
        level = int(levels[selected])
        terrain_type = int(types[selected])
        category_code = int(raw._category_by_type[terrain_type])
        direction_code = int(raw._direction_by_type[terrain_type])
        wrapped.close()

    expected_frames = math.ceil(steps / capture_stride)
    if ffmpeg_return_code != 0 or captured_frames != expected_frames:
        raise RuntimeError(
            f"录像帧生成异常：ffmpeg={ffmpeg_return_code}，"
            f"实际帧数={captured_frames}，预期帧数={expected_frames}。"
        )
    if not final_video.is_file() or final_video.stat().st_size == 0:
        raise RuntimeError(f"录像文件没有生成：{final_video}")

    category, direction, difficulty = metadata_labels(
        category_code,
        direction_code,
        level,
        EVAL_TERRAIN_ROWS,
    )
    yaw_unwrapped = _unwrap_angle_series(yaw_values)
    displacement = end_xy - origin
    report = {
        "生成时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "任务": args_cli.task,
        "方法": method,
        "地形": category,
        "方向": direction,
        "难度": difficulty,
        "PPO_seed": args_cli.ppo_seed,
        "地形批次seed": batch_terrain_seed,
        "评估批次": args_cli.batch_index,
        "正式评估环境编号": formal_env_index,
        "回放场景环境数": 1,
        "全局回合编号": args_cli.batch_index * EVAL_NUM_ENVS + formal_env_index,
        "检查点": str(checkpoint),
        "检查点SHA256": _file_sha256(checkpoint),
        "B_ref文件SHA256": reference_sha256,
        "B80_J": energy_budget,
        "录像时长_s": args_cli.duration_s,
        "输出帧率": args_cli.video_fps,
        "录像帧数": captured_frames,
        "分辨率": list(args_cli.resolution),
        "完成时发生终止或超时": selected_done,
        "净前进位移_m": float(displacement[0]),
        "净横向位移_m": float(displacement[1]),
        "最大横向绝对位移_m": max_abs_lateral_m,
        "水平实际路径_m": path_m,
        "初始偏航角_deg": math.degrees(yaw_unwrapped[0]) if yaw_unwrapped else None,
        "最终偏航角_deg": math.degrees(yaw_unwrapped[-1]) if yaw_unwrapped else None,
        "偏航角变化_deg": (
            math.degrees(yaw_unwrapped[-1] - yaw_unwrapped[0]) if yaw_unwrapped else None
        ),
        "视频": str(final_video),
    }
    report_path = output_dir / "gpt_多地形策略回放说明.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"[PACE] 策略回放视频：{final_video}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
