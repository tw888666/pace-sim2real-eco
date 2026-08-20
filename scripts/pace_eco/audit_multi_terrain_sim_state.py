#!/usr/bin/env python3
"""用实际 Isaac Sim 碰撞网格审计 Terrain20sWide；不加载或生成策略权重。"""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

from pace_eco_lab.multi_terrain_protocol import (
    MAX_AUDITED_FORWARD_M,
    PROTOCOL_VERSION,
    TASK_IDS,
    TRAIN_TERRAIN_COLS,
    TRAIN_TERRAIN_ROWS,
    TERRAIN_NAMES,
    TERRAIN_LENGTH_M,
    TERRAIN_ORIGIN_X_M,
    TERRAIN_WIDTH_M,
    terrain_column_metadata,
    terrain_seed,
)


parser = argparse.ArgumentParser(description="PACE-ECO Terrain20sWide 仿真状态审计。")
parser.add_argument("--terrain", required=True, choices=TERRAIN_NAMES)
parser.add_argument("--terrain_seed", required=True, type=int)
parser.add_argument("--output", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

expected_seed = terrain_seed("stage1_smoke_train", args_cli.terrain, 900)
if args_cli.terrain_seed != expected_seed:
    parser.error(f"状态审计只接受冻结 smoke 地形 seed {expected_seed}。")
output_path = Path(args_cli.output).expanduser().resolve()
if output_path.exists():
    parser.error(f"拒绝覆盖状态审计结果：{output_path}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.sensors.ray_caster import RayCaster
from isaaclab.utils.warp import raycast_mesh
from isaaclab_tasks.utils import load_cfg_from_registry

import pace_eco_lab  # noqa: F401
from pace_eco_lab.multi_terrain_state_audit import audit_difficulty_table, audit_surface_profiles
from pace_eco_lab.reproducibility import verify_dependency_versions
from pace_eco_lab.terrain_boundary import (
    ANYMAL_D_FOOT_CENTER_CHAIN_BOUND_M,
    FOOT_BOUNDARY_AUDIT_VERSION,
    FOOT_CENTER_MAX_BASE_DISTANCE_M,
    base_near_tile_edge,
)
from pace_eco_lab.terrains import curriculum_difficulty_table


def main() -> None:
    verify_dependency_versions()
    task = TASK_IDS[("task_only", args_cli.terrain)]
    env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = TRAIN_TERRAIN_COLS
    env_cfg.scene.terrain.max_init_terrain_level = TRAIN_TERRAIN_ROWS - 1
    env_cfg.scene.terrain.terrain_generator.seed = args_cli.terrain_seed
    env_cfg.pace_terrain_seed = args_cli.terrain_seed
    env_cfg.seed = 900
    env_cfg.sim.device = args_cli.device or "cuda:0"
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.log_dir = str(output_path.parent)
    print(
        f"[PACE] 正在生成 {TRAIN_TERRAIN_ROWS}×{TRAIN_TERRAIN_COLS} "
        "Terrain20sWide 网格；本审计不加载模型。",
        flush=True,
    )
    env = gym.make(task, cfg=env_cfg)
    raw = env.unwrapped
    try:
        observations, _ = env.reset()
        actions = torch.zeros(raw.num_envs, raw.action_manager.total_action_dim, device=raw.device)
        observations, rewards, terminated, truncated, extras = env.step(actions)
        finite_observations = all(torch.isfinite(values).all().item() for values in observations.values())
        first_step_clean = bool(not torch.any(terminated | truncated).item())
        finite_rewards = bool(torch.isfinite(rewards).all().item())

        terrain = raw.scene.terrain
        expected_labels = terrain_column_metadata(args_cli.terrain, TRAIN_TERRAIN_COLS)
        origins_shape_ok = tuple(terrain.terrain_origins.shape) == (
            TRAIN_TERRAIN_ROWS,
            TRAIN_TERRAIN_COLS,
            3,
        )
        metadata_ok = tuple(raw.cfg.pace_terrain_column_metadata) == expected_labels
        foot_audit_shapes_ok = (
            tuple(extras["pace_terrain20s_contact_foot_crossed_by_foot"].shape)
            == (raw.num_envs, 4)
            and tuple(extras["pace_terrain20s_contact_foot_min_edge_margin_m"].shape)
            == (raw.num_envs, 4)
            and len(raw._foot_body_ids) == len(raw._foot_sensor_ids) == 4
        )
        contact_sensor = raw.scene.sensors["contact_forces"]
        sensor_body_positions = contact_sensor.body_physx_view.get_transforms().view(
            raw.num_envs, contact_sensor.num_bodies, 7
        )[:, raw._foot_sensor_ids, :3]
        robot = raw.scene["robot"]
        direct_link_positions = robot.root_physx_view.get_link_transforms()[:, raw._foot_body_ids, :3]
        articulation_foot_positions = robot.data.body_pos_w[:, raw._foot_body_ids]
        sensor_position_source_max_error_m = float(
            torch.max(torch.abs(sensor_body_positions - articulation_foot_positions)).item()
        )
        direct_link_position_source_max_error_m = float(
            torch.max(torch.abs(direct_link_positions - articulation_foot_positions)).item()
        )
        default_base_foot_distance_max_m = float(
            torch.linalg.vector_norm(
                direct_link_positions - robot.data.root_pos_w[:, None, :], dim=-1
            ).max().item()
        )
        foot_position_sources_equivalent = (
            sensor_position_source_max_error_m <= 1.0e-6
            and direct_link_position_source_max_error_m <= 1.0e-6
        )
        prefilter_probe = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 28.75, 0.0], [33.75, 0.0, 0.0]],
            device=raw.device,
        )
        prefilter_flags = base_near_tile_edge(
            prefilter_probe,
            terrain_length_m=TERRAIN_LENGTH_M,
            terrain_width_m=TERRAIN_WIDTH_M,
            terrain_origin_x_m=TERRAIN_ORIGIN_X_M,
        )["audit_required"].tolist()
        first_step_contact_foot_cross_clean = bool(not raw._contact_foot_crossed_running.any().item())

        x_values = torch.arange(-2.0, MAX_AUDITED_FORWARD_M + 0.0001, 0.25, device=raw.device)
        y_values = torch.arange(-3.0, 3.0001, 0.2, device=raw.device)
        xx, yy = torch.meshgrid(x_values, y_values, indexing="ij")
        local_points = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
        grid_origins = terrain.terrain_origins.reshape(-1, 3).clone()
        ray_starts = grid_origins[:, None, :].repeat(1, len(local_points), 1)
        ray_starts[:, :, 0] += local_points[None, :, 0]
        ray_starts[:, :, 1] += local_points[None, :, 1]
        ray_starts[:, :, 2] += 25.0
        ray_directions = torch.zeros_like(ray_starts)
        ray_directions[:, :, 2] = -1.0
        mesh = RayCaster.meshes.get("/World/ground")
        if mesh is None:
            raise RuntimeError("RayCaster 未注册 /World/ground 实际碰撞网格。")
        print(f"[PACE] 开始 {len(grid_origins) * len(local_points)} 条碰撞网格射线。", flush=True)
        ray_hits, _, _, _ = raycast_mesh(ray_starts, ray_directions, mesh, max_dist=50.0)
        relative_heights = (ray_hits[:, :, 2] - grid_origins[:, None, 2]).reshape(
            len(grid_origins), len(x_values), len(y_values)
        )
        directions_by_column = tuple(label.split(":", 1)[1] for label in expected_labels)
        directions = tuple(
            directions_by_column[column]
            for _level in range(TRAIN_TERRAIN_ROWS)
            for column in range(TRAIN_TERRAIN_COLS)
        )
        surface_audit = audit_surface_profiles(
            args_cli.terrain,
            directions,
            x_values.cpu().numpy(),
            y_values.cpu().numpy(),
            relative_heights.cpu().numpy(),
        )
        difficulties = curriculum_difficulty_table(
            args_cli.terrain_seed,
            TRAIN_TERRAIN_ROWS,
            TRAIN_TERRAIN_COLS,
            (0.0, 1.0),
        )
        difficulty_audit = audit_difficulty_table(difficulties, 0.0, 1.0)
        center_y = int(torch.argmin(torch.abs(y_values)).item())
        start_x = int(torch.argmin(torch.abs(x_values)).item())
        start_surface = relative_heights[:, start_x, center_y]
        start_origin_ok = bool(torch.all(torch.abs(start_surface) <= 0.01).item())
        checks = {
            "首次物理步无意外终止": first_step_clean,
            "首次物理步奖励有限": finite_rewards,
            "首次物理步观察有限": finite_observations,
            "地形origin形状与冻结5×10一致": origins_shape_ok,
            "列类别与方向元数据正确": metadata_ok,
            "四足边界审计张量为环境数×4": foot_audit_shapes_ok,
            "两种直接物理视图与关节体足端中心坐标一致": foot_position_sources_equivalent,
            "ANYmal_D运动学链上界小于冻结预筛上界": (
                ANYMAL_D_FOOT_CENTER_CHAIN_BOUND_M < FOOT_CENTER_MAX_BASE_DISTANCE_M
            ),
            "默认姿态足端距离小于冻结预筛上界": (
                default_base_foot_distance_max_m < FOOT_CENTER_MAX_BASE_DISTANCE_M
            ),
            "边缘预筛阈值含边界且中心安全": prefilter_flags == [False, True, True],
            "首步无接触足真实块越界": first_step_contact_foot_cross_clean,
            "出生中心碰撞面与origin高度一致": start_origin_ok,
            "实际碰撞表面审计通过": bool(surface_audit["全部通过"]),
            "几何难度逐行分层": bool(difficulty_audit["逐行难度带"]),
            "几何难度同列递增": bool(difficulty_audit["同列严格递增"]),
            "没有性能驱动课程项": len(raw.curriculum_manager.active_terms) == 0,
            "仅保留历史两个终止项": set(raw.termination_manager.active_terms)
            == {"time_out", "base_contact"},
        }
        origins_grid = terrain.terrain_origins.detach().cpu().numpy()
        payload = {
            "协议版本": PROTOCOL_VERSION,
            "边界审计版本": FOOT_BOUNDARY_AUDIT_VERSION,
            "审计状态": "全部通过" if all(checks.values()) else "失败",
            "创建时间": datetime.now().astimezone().isoformat(timespec="seconds"),
            "任务": task,
            "地形": args_cli.terrain,
            "地形seed": args_cli.terrain_seed,
            "PPO_seed": None,
            "是否加载模型权重": False,
            "是否生成模型权重": False,
            "是否正式评估": False,
            "环境数": raw.num_envs,
            "接触传感器视图足端中心最大绝对差_m": sensor_position_source_max_error_m,
            "关节体物理视图足端中心最大绝对差_m": direct_link_position_source_max_error_m,
            "ANYmal_D官方URDF运动学链长度上界_m": ANYMAL_D_FOOT_CENTER_CHAIN_BOUND_M,
            "冻结足端中心相对机身上界_m": FOOT_CENTER_MAX_BASE_DISTANCE_M,
            "默认姿态实测最大机身足端距离_m": default_base_foot_distance_max_m,
            "实际碰撞地形实例数": TRAIN_TERRAIN_ROWS * TRAIN_TERRAIN_COLS,
            "地形行列": [TRAIN_TERRAIN_ROWS, TRAIN_TERRAIN_COLS],
            "地形列计数": {
                label: expected_labels.count(label) for label in sorted(set(expected_labels))
            },
            "前向采样范围_m": [float(x_values.min()), float(x_values.max())],
            "横向采样范围_m": [float(y_values.min()), float(y_values.max())],
            "origin世界坐标范围": {
                "x": [float(origins_grid[..., 0].min()), float(origins_grid[..., 0].max())],
                "y": [float(origins_grid[..., 1].min()), float(origins_grid[..., 1].max())],
                "z": [float(origins_grid[..., 2].min()), float(origins_grid[..., 2].max())],
            },
            "总检查": checks,
            "碰撞网格表面审计": surface_audit,
            "几何难度审计": difficulty_audit,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"审计状态": payload["审计状态"], "总检查": checks}, ensure_ascii=False, indent=2))
        if not all(checks.values()):
            raise RuntimeError(f"仿真状态审计失败：{[name for name, ok in checks.items() if not ok]}")
    finally:
        env.close()


if __name__ == "__main__":
    failure: BaseException | None = None
    try:
        main()
    except BaseException as error:
        failure = error
        traceback.print_exc()
    finally:
        try:
            simulation_app.close()
        except BaseException:
            if failure is None:
                raise
    if failure is not None:
        raise failure
