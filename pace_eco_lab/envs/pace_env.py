"""逐物理子步积分能耗的 PACE ManagerBasedRLEnv。"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv

from pace_eco_lab.evaluation_snapshot import publish_eval_state
from pace_eco_lab.mdp.energy import compute_energy_components


class PaceManagerBasedRLEnv(ManagerBasedRLEnv):
    """在官方 ManagerBasedRLEnv 步进流程中增加 PACE 能耗采样。"""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        self.pace_learning_iteration = 0
        self.pace_energy_step = torch.zeros(self.num_envs, device=self.device)
        self.pace_energy_episode_running = torch.zeros_like(self.pace_energy_step)
        self._pace_component_names = ("electrical", "mechanical", "potential")
        self.pace_energy_components = {
            name: torch.zeros_like(self.pace_energy_step) for name in self._pace_component_names
        }
        robot: Articulation = self.scene["robot"]
        # Isaac Sim 的 default_mass 是静态元数据，在部分 GPU/版本组合中仍位于 CPU。
        # 只在初始化时复制一次，避免每个物理子步发生 CPU 到 CUDA 的传输。
        self._pace_body_mass = robot.data.default_mass.to(
            device=robot.data.body_lin_vel_w.device,
            dtype=robot.data.body_lin_vel_w.dtype,
        ).contiguous()

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, "pace_energy_episode_running"):
            self.pace_energy_episode_running[env_ids] = 0.0

    def _reset_step_energy(self) -> None:
        self.pace_energy_step.zero_()
        for value in self.pace_energy_components.values():
            value.zero_()

    def _accumulate_substep_energy(self) -> None:
        robot: Articulation = self.scene["robot"]
        power = compute_energy_components(
            robot.data.applied_torque,
            robot.data.joint_vel,
            self._pace_body_mass,
            robot.data.body_lin_vel_w[..., 2],
            include_potential=bool(getattr(self.cfg, "pace_include_potential", True)),
        )
        dt = self.physics_dt
        self.pace_energy_components["electrical"].add_(power.electrical * dt)
        self.pace_energy_components["mechanical"].add_(power.mechanical * dt)
        self.pace_energy_components["potential"].add_(power.potential * dt)

    def _publish_eval_state(self) -> None:
        """在自动重置前发布只读根状态，供定距地形评估冻结终点。"""

        if not bool(getattr(self.cfg, "pace_publish_eval_state", False)):
            return
        robot: Articulation = self.scene["robot"]
        publish_eval_state(
            self.extras,
            robot.data.root_pos_w,
            robot.data.root_lin_vel_b,
            enabled=True,
        )

    def step(self, action: torch.Tensor):
        """复制官方 0.54.4 步进顺序，并在每个物理子步后积分实际量。"""

        self.extras["log"] = {}
        self._reset_step_energy()
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)
            self._accumulate_substep_energy()

        self.pace_energy_step.copy_(
            self.pace_energy_components["electrical"]
            + self.pace_energy_components["mechanical"]
            + self.pace_energy_components["potential"]
        )
        self.pace_energy_episode_running.add_(self.pace_energy_step)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        self._publish_eval_state()

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        completed_energy = torch.zeros_like(self.pace_energy_step)
        completed_mask = self.reset_buf.clone()
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            completed_energy[reset_env_ids] = self.pace_energy_episode_running[reset_env_ids]
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # RSL-RL 不修改这些键；算法据此只用完整回合更新乘子。
        self.extras["pace_energy_step"] = self.pace_energy_step.clone()
        self.extras["pace_energy_episode"] = completed_energy
        self.extras["pace_energy_episode_mask"] = completed_mask
        self.extras["pace_energy_components"] = {
            name: value.clone() for name, value in self.pace_energy_components.items()
        }
        self.extras.setdefault("log", {})
        self.extras["log"]["PACE/energy_step_j"] = self.pace_energy_step.mean()
        self.extras["log"]["PACE/base_forward_velocity_m_s"] = self.scene[
            "robot"
        ].data.root_lin_vel_b[:, 0].mean()
        self.extras["log"]["PACE/reset_fraction"] = self.reset_buf.float().mean()
        self.extras["log"]["PACE/base_contact_termination_fraction"] = self.reset_terminated.float().mean()
        if torch.any(completed_mask):
            self.extras["log"]["PACE/completed_episode_energy_j"] = completed_energy[completed_mask].mean()
        for name, value in self.pace_energy_components.items():
            self.extras["log"][f"PACE/energy_{name}_j"] = value.mean()

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
