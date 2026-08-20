"""为固定20秒多地形任务增加只读地形、位移和边界审计元数据。"""

from __future__ import annotations

import torch

from pace_eco_lab.envs.pace_env import PaceManagerBasedRLEnv
from pace_eco_lab.multi_terrain_protocol import (
    MAX_AUDITED_BACKWARD_M,
    MAX_AUDITED_FORWARD_M,
    MAX_AUDITED_LATERAL_M,
    TERRAIN_LENGTH_M,
    TERRAIN_ORIGIN_X_M,
    TERRAIN_WIDTH_M,
    difficulty_label,
)
from pace_eco_lab.terrain_boundary import (
    FOOT_CENTER_MAX_BASE_DISTANCE_M,
    base_near_tile_edge,
    contact_foot_tile_state,
)


_CATEGORY_CODES = {"flat": 0, "rough": 1, "stairs": 2, "boxes": 3, "slope": 4}
_DIRECTION_CODES = {"level": 0, "up": 1, "down": 2}
_FOOT_BODY_NAMES = ("LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT")


class PaceTerrain20sRLEnv(PaceManagerBasedRLEnv):
    """不改变奖励或终止，只记录固定20秒地形实验所需的逐回合证据。"""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        self._pace_terrain20s_in_step = False
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        self._configure_column_metadata()
        self._configure_foot_audit()
        self._runtime_spawn_grid_validated = False
        self._pre_xy = self.scene["robot"].data.root_pos_w[:, :2].clone()
        self._path_running = torch.zeros(self.num_envs, device=self.device)
        self._max_forward_running = torch.zeros(self.num_envs, device=self.device)
        self._min_forward_running = torch.zeros(self.num_envs, device=self.device)
        self._max_abs_lateral_running = torch.zeros(self.num_envs, device=self.device)
        self._allocate_running_foot_audit()
        self._start_z = self.scene["robot"].data.root_pos_w[:, 2].clone()
        self._allocate_completed_metadata()
        self._validate_runtime_spawn_and_grid(check_cuda_values=False)

    def _configure_foot_audit(self) -> None:
        robot = self.scene["robot"]
        contact_sensor = self.scene.sensors["contact_forces"]
        foot_body_ids, foot_body_names = robot.find_bodies(_FOOT_BODY_NAMES, preserve_order=True)
        foot_sensor_ids, foot_sensor_names = contact_sensor.find_bodies(_FOOT_BODY_NAMES, preserve_order=True)
        if tuple(foot_body_names) != _FOOT_BODY_NAMES or tuple(foot_sensor_names) != _FOOT_BODY_NAMES:
            raise RuntimeError(
                "Terrain20s 四足边界审计无法解析固定足序："
                f"robot={tuple(foot_body_names)}，sensor={tuple(foot_sensor_names)}。"
            )
        self._foot_body_ids = foot_body_ids
        self._foot_sensor_ids = foot_sensor_ids
        self._foot_contact_force_threshold_n = float(contact_sensor.cfg.force_threshold)

    def _allocate_running_foot_audit(self) -> None:
        shape = (self.num_envs, len(_FOOT_BODY_NAMES))
        self._contact_foot_crossed_running = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self._swing_foot_crossed_running = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self._contact_foot_min_edge_margin_running = torch.full(shape, torch.inf, device=self.device)
        self._contact_foot_max_forward_running = torch.full(shape, -torch.inf, device=self.device)
        self._contact_foot_min_forward_running = torch.full(shape, torch.inf, device=self.device)
        self._contact_foot_max_abs_lateral_running = torch.zeros(shape, device=self.device)
        self._foot_exact_audit_running = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _allocate_completed_metadata(self) -> None:
        self._completed_path = torch.zeros(self.num_envs, device=self.device)
        self._completed_forward = torch.zeros(self.num_envs, device=self.device)
        self._completed_lateral = torch.zeros(self.num_envs, device=self.device)
        self._completed_max_forward = torch.zeros(self.num_envs, device=self.device)
        self._completed_min_forward = torch.zeros(self.num_envs, device=self.device)
        self._completed_max_abs_lateral = torch.zeros(self.num_envs, device=self.device)
        self._completed_duration = torch.zeros(self.num_envs, device=self.device)
        self._completed_elevation = torch.zeros(self.num_envs, device=self.device)
        self._completed_level = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._completed_type = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._completed_category = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._completed_direction = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._completed_base_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._completed_timeout = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._completed_contact_foot_boundary_crossed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._completed_swing_foot_boundary_crossed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        foot_shape = (self.num_envs, len(_FOOT_BODY_NAMES))
        self._completed_contact_foot_crossed_by_foot = torch.zeros(
            foot_shape, dtype=torch.bool, device=self.device
        )
        self._completed_swing_foot_crossed_by_foot = torch.zeros(
            foot_shape, dtype=torch.bool, device=self.device
        )
        self._completed_contact_foot_min_edge_margin = torch.full(foot_shape, torch.nan, device=self.device)
        self._completed_contact_foot_max_forward = torch.full(foot_shape, torch.nan, device=self.device)
        self._completed_contact_foot_min_forward = torch.full(foot_shape, torch.nan, device=self.device)
        self._completed_contact_foot_max_abs_lateral = torch.full(foot_shape, torch.nan, device=self.device)
        self._completed_foot_exact_audit_sampled = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._completed_boundary_crossed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _zero_completed_metadata(self) -> None:
        for value in (
            self._completed_path,
            self._completed_forward,
            self._completed_lateral,
            self._completed_max_forward,
            self._completed_min_forward,
            self._completed_max_abs_lateral,
            self._completed_duration,
            self._completed_elevation,
        ):
            value.zero_()
        for value in (
            self._completed_level,
            self._completed_type,
            self._completed_category,
            self._completed_direction,
        ):
            value.fill_(-1)
        self._completed_base_contact.zero_()
        self._completed_timeout.zero_()
        self._completed_contact_foot_boundary_crossed.zero_()
        self._completed_swing_foot_boundary_crossed.zero_()
        self._completed_contact_foot_crossed_by_foot.zero_()
        self._completed_swing_foot_crossed_by_foot.zero_()
        self._completed_contact_foot_min_edge_margin.fill_(torch.nan)
        self._completed_contact_foot_max_forward.fill_(torch.nan)
        self._completed_contact_foot_min_forward.fill_(torch.nan)
        self._completed_contact_foot_max_abs_lateral.fill_(torch.nan)
        self._completed_foot_exact_audit_sampled.zero_()
        self._completed_boundary_crossed.zero_()

    def _configure_column_metadata(self) -> None:
        labels = tuple(self.cfg.pace_terrain_column_metadata)
        if not labels:
            raise ValueError("Terrain20s 环境缺少 pace_terrain_column_metadata。")
        categories: list[int] = []
        directions: list[int] = []
        for label in labels:
            category, direction = label.split(":", maxsplit=1)
            categories.append(_CATEGORY_CODES[category])
            directions.append(_DIRECTION_CODES[direction])
        self._category_by_type = torch.tensor(categories, dtype=torch.long, device=self.device)
        self._direction_by_type = torch.tensor(directions, dtype=torch.long, device=self.device)

    def _validate_runtime_spawn_and_grid(self, *, check_cuda_values: bool) -> None:
        terrain = self.scene.terrain
        if terrain.terrain_origins is None:
            raise RuntimeError("Terrain20s 任务没有生成式 terrain_origins。")
        if terrain.terrain_origins.shape[1] != len(self.cfg.pace_terrain_column_metadata):
            raise RuntimeError("地形列数与类别/方向元数据不一致。")
        if terrain.terrain_types.shape != (self.num_envs,):
            raise RuntimeError("每个环境必须恰好对应一个地形类型索引。")
        if not check_cuda_values:
            return
        displacement = self.scene["robot"].data.root_pos_w - self.scene.env_origins
        if torch.any(displacement[:, :2].abs() > 0.25):
            raise RuntimeError("机器人出生点偏离长地形 origin 超过训练重置范围。")
        if torch.any(displacement[:, 2] < 0.30):
            raise RuntimeError("机器人机身出生高度不足 0.30 m，疑似 origin 错误或地形穿透。")
        if torch.any(terrain.terrain_types < 0) or torch.any(
            terrain.terrain_types >= len(self.cfg.pace_terrain_column_metadata)
        ):
            raise RuntimeError("地形类型索引越界。")
        self._runtime_spawn_grid_validated = True

    def _update_running_motion(self, env_ids: torch.Tensor) -> torch.Tensor:
        robot = self.scene["robot"]
        self._path_running[env_ids] += torch.linalg.vector_norm(
            robot.data.root_pos_w[env_ids, :2] - self._pre_xy[env_ids], dim=-1
        )
        displacement = robot.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        forward = displacement[:, 0]
        lateral = displacement[:, 1]
        self._max_forward_running[env_ids] = torch.maximum(self._max_forward_running[env_ids], forward)
        self._min_forward_running[env_ids] = torch.minimum(self._min_forward_running[env_ids], forward)
        self._max_abs_lateral_running[env_ids] = torch.maximum(
            self._max_abs_lateral_running[env_ids], lateral.abs()
        )
        self._update_running_foot_audit(env_ids, displacement)
        return displacement

    def _update_running_foot_audit(
        self, env_ids: torch.Tensor, base_displacement: torch.Tensor
    ) -> None:
        if len(env_ids) == 0:
            return
        prefilter = base_near_tile_edge(
            base_displacement,
            terrain_length_m=TERRAIN_LENGTH_M,
            terrain_width_m=TERRAIN_WIDTH_M,
            terrain_origin_x_m=TERRAIN_ORIGIN_X_M,
            foot_center_max_base_distance_m=FOOT_CENTER_MAX_BASE_DISTANCE_M,
        )
        if not torch.any(prefilter["audit_required"]):
            return
        audit_ids = env_ids[prefilter["audit_required"]]
        robot = self.scene["robot"]
        contact_sensor = self.scene.sensors["contact_forces"]
        # 直接读取 articulation PhysX view（关节体物理视图）的连杆变换。使用
        # ``robot.data.body_pos_w`` 会在每个控制步强制更新整台机器人的
        # forward kinematics（前向运动学），对 200 环境正式评估造成数量级性能回退。
        # 这里直接读取 PhysX 已有的同一足部 link（连杆）刚体中心世界坐标，
        # 不调用额外运动学刷新，因此不改变审计语义。
        link_positions_w = robot.root_physx_view.get_link_transforms()[..., :3]
        state = contact_foot_tile_state(
            link_positions_w[audit_ids][:, self._foot_body_ids],
            contact_sensor.data.net_forces_w[audit_ids][:, self._foot_sensor_ids],
            self.scene.env_origins[audit_ids],
            terrain_length_m=TERRAIN_LENGTH_M,
            terrain_width_m=TERRAIN_WIDTH_M,
            terrain_origin_x_m=TERRAIN_ORIGIN_X_M,
            contact_force_threshold_n=self._foot_contact_force_threshold_n,
        )
        contact = state["contact"]
        forward = state["relative_xy"][..., 0]
        abs_lateral = state["relative_xy"][..., 1].abs()
        self._foot_exact_audit_running[audit_ids] = True
        self._contact_foot_crossed_running[audit_ids] |= state["contact_outside"]
        self._swing_foot_crossed_running[audit_ids] |= state["swing_outside"]
        self._contact_foot_min_edge_margin_running[audit_ids] = torch.minimum(
            self._contact_foot_min_edge_margin_running[audit_ids],
            torch.where(contact, state["edge_margin_m"], torch.inf),
        )
        self._contact_foot_max_forward_running[audit_ids] = torch.maximum(
            self._contact_foot_max_forward_running[audit_ids],
            torch.where(contact, forward, -torch.inf),
        )
        self._contact_foot_min_forward_running[audit_ids] = torch.minimum(
            self._contact_foot_min_forward_running[audit_ids],
            torch.where(contact, forward, torch.inf),
        )
        self._contact_foot_max_abs_lateral_running[audit_ids] = torch.maximum(
            self._contact_foot_max_abs_lateral_running[audit_ids],
            torch.where(contact, abs_lateral, 0.0),
        )

    def _reset_running_foot_audit(self, env_ids: torch.Tensor) -> None:
        self._contact_foot_crossed_running[env_ids] = False
        self._swing_foot_crossed_running[env_ids] = False
        self._contact_foot_min_edge_margin_running[env_ids] = torch.inf
        self._contact_foot_max_forward_running[env_ids] = -torch.inf
        self._contact_foot_min_forward_running[env_ids] = torch.inf
        self._contact_foot_max_abs_lateral_running[env_ids] = 0.0
        self._foot_exact_audit_running[env_ids] = False

    def _reset_idx(self, env_ids):
        if hasattr(self, "_path_running") and self._pace_terrain20s_in_step:
            displacement = self._update_running_motion(env_ids)
            terrain = self.scene.terrain
            levels = terrain.terrain_levels[env_ids]
            types = terrain.terrain_types[env_ids]
            self._completed_path[env_ids] = self._path_running[env_ids]
            self._completed_forward[env_ids] = displacement[:, 0]
            self._completed_lateral[env_ids] = displacement[:, 1]
            self._completed_max_forward[env_ids] = self._max_forward_running[env_ids]
            self._completed_min_forward[env_ids] = self._min_forward_running[env_ids]
            self._completed_max_abs_lateral[env_ids] = self._max_abs_lateral_running[env_ids]
            self._completed_duration[env_ids] = self.episode_length_buf[env_ids] * self.step_dt
            self._completed_elevation[env_ids] = self.scene["robot"].data.root_pos_w[env_ids, 2] - self._start_z[env_ids]
            self._completed_level[env_ids] = levels
            self._completed_type[env_ids] = types
            self._completed_category[env_ids] = self._category_by_type[types]
            self._completed_direction[env_ids] = self._direction_by_type[types]
            self._completed_base_contact[env_ids] = self.termination_manager.get_term("base_contact")[env_ids]
            self._completed_timeout[env_ids] = self.termination_manager.get_term("time_out")[env_ids]
            self._completed_contact_foot_crossed_by_foot[env_ids] = self._contact_foot_crossed_running[env_ids]
            self._completed_swing_foot_crossed_by_foot[env_ids] = self._swing_foot_crossed_running[env_ids]
            self._completed_contact_foot_boundary_crossed[env_ids] = self._contact_foot_crossed_running[
                env_ids
            ].any(dim=-1)
            self._completed_swing_foot_boundary_crossed[env_ids] = self._swing_foot_crossed_running[
                env_ids
            ].any(dim=-1)
            self._completed_contact_foot_min_edge_margin[env_ids] = self._contact_foot_min_edge_margin_running[
                env_ids
            ]
            self._completed_contact_foot_max_forward[env_ids] = self._contact_foot_max_forward_running[env_ids]
            self._completed_contact_foot_min_forward[env_ids] = self._contact_foot_min_forward_running[env_ids]
            self._completed_contact_foot_max_abs_lateral[env_ids] = (
                self._contact_foot_max_abs_lateral_running[env_ids]
            )
            self._completed_foot_exact_audit_sampled[env_ids] = self._foot_exact_audit_running[env_ids]
            self._completed_boundary_crossed[env_ids] = (
                (self._max_forward_running[env_ids] > MAX_AUDITED_FORWARD_M)
                | (self._min_forward_running[env_ids] < -MAX_AUDITED_BACKWARD_M)
                | (self._max_abs_lateral_running[env_ids] > MAX_AUDITED_LATERAL_M)
            )
        super()._reset_idx(env_ids)
        if hasattr(self, "_path_running"):
            self._path_running[env_ids] = 0.0
            self._max_forward_running[env_ids] = 0.0
            self._min_forward_running[env_ids] = 0.0
            self._max_abs_lateral_running[env_ids] = 0.0
            self._reset_running_foot_audit(env_ids)
            self._start_z[env_ids] = self.scene["robot"].data.root_pos_w[env_ids, 2]

    def step(self, action: torch.Tensor):
        self._zero_completed_metadata()
        self._pre_xy.copy_(self.scene["robot"].data.root_pos_w[:, :2])
        self._pace_terrain20s_in_step = True
        try:
            observations, rewards, terminated, time_outs, extras = super().step(action)
        finally:
            self._pace_terrain20s_in_step = False
        if not self._runtime_spawn_grid_validated:
            self._validate_runtime_spawn_and_grid(check_cuda_values=True)
        completed = extras["pace_energy_episode_mask"].bool()
        active_ids = (~completed).nonzero(as_tuple=False).squeeze(-1)
        if len(active_ids) > 0:
            self._update_running_motion(active_ids)
        extras.update(
            {
                "pace_terrain20s_path_episode_m": self._completed_path.clone(),
                "pace_terrain20s_forward_episode_m": self._completed_forward.clone(),
                "pace_terrain20s_lateral_episode_m": self._completed_lateral.clone(),
                "pace_terrain20s_max_forward_episode_m": self._completed_max_forward.clone(),
                "pace_terrain20s_min_forward_episode_m": self._completed_min_forward.clone(),
                "pace_terrain20s_max_abs_lateral_episode_m": self._completed_max_abs_lateral.clone(),
                "pace_terrain20s_duration_episode_s": self._completed_duration.clone(),
                "pace_terrain20s_elevation_episode_m": self._completed_elevation.clone(),
                "pace_terrain20s_terrain_level": self._completed_level.clone(),
                "pace_terrain20s_terrain_type": self._completed_type.clone(),
                "pace_terrain20s_category_code": self._completed_category.clone(),
                "pace_terrain20s_direction_code": self._completed_direction.clone(),
                "pace_terrain20s_base_contact": self._completed_base_contact.clone(),
                "pace_terrain20s_timeout": self._completed_timeout.clone(),
                "pace_terrain20s_contact_foot_boundary_crossed": (
                    self._completed_contact_foot_boundary_crossed.clone()
                ),
                "pace_terrain20s_swing_foot_boundary_crossed": (
                    self._completed_swing_foot_boundary_crossed.clone()
                ),
                "pace_terrain20s_contact_foot_crossed_by_foot": (
                    self._completed_contact_foot_crossed_by_foot.clone()
                ),
                "pace_terrain20s_swing_foot_crossed_by_foot": (
                    self._completed_swing_foot_crossed_by_foot.clone()
                ),
                "pace_terrain20s_contact_foot_min_edge_margin_m": (
                    self._completed_contact_foot_min_edge_margin.clone()
                ),
                "pace_terrain20s_contact_foot_max_forward_m": self._completed_contact_foot_max_forward.clone(),
                "pace_terrain20s_contact_foot_min_forward_m": self._completed_contact_foot_min_forward.clone(),
                "pace_terrain20s_contact_foot_max_abs_lateral_m": (
                    self._completed_contact_foot_max_abs_lateral.clone()
                ),
                "pace_terrain20s_foot_exact_audit_sampled": (
                    self._completed_foot_exact_audit_sampled.clone()
                ),
                "pace_terrain20s_base_corridor_warning": self._completed_boundary_crossed.clone(),
                "pace_terrain20s_boundary_crossed": self._completed_boundary_crossed.clone(),
            }
        )
        if torch.any(completed):
            log = extras.setdefault("log", {})
            log["PACE/base_corridor_warning_fraction"] = self._completed_boundary_crossed[completed].float().mean()
            log["PACE/contact_foot_tile_cross_fraction"] = self._completed_contact_foot_boundary_crossed[
                completed
            ].float().mean()
        return observations, rewards, terminated, time_outs, extras


def metadata_labels(category_code: int, direction_code: int, level: int, rows: int) -> tuple[str, str, str]:
    category_names = {value: key for key, value in _CATEGORY_CODES.items()}
    direction_names = {value: key for key, value in _DIRECTION_CODES.items()}
    return (
        category_names[category_code],
        direction_names[direction_code],
        difficulty_label(level, rows),
    )


__all__ = ["PaceTerrain20sRLEnv", "metadata_labels"]
