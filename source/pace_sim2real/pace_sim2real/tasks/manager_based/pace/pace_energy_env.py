"""Manager-based RL environment with 400 Hz PACE energy accounting."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv

from pace_sim2real.energy import EnergyAccumulator, compute_power_components
from pace_sim2real.utils.identified_parameters import (
    ANYMAL_D_JOINT_ORDER,
    IdentifiedActuatorParameters,
    apply_identified_parameters,
)


class PaceEnergyRLEnv(ManagerBasedRLEnv):
    """PACE locomotion environment exposing physical and augmented costs."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        robot = self.scene["robot"]
        parameters = IdentifiedActuatorParameters.from_sequence(
            cfg.pace_identification.parameters,
            device=self.device,
            dtype=robot.data.joint_pos.dtype,
        )
        apply_identified_parameters(robot, parameters, cfg.pace_identification.joint_order)
        # PhysX exposes default masses on the host in Isaac Sim 5.1. Mass
        # randomization is disabled for this calibrated environment, so keep a
        # frozen device-local copy instead of transferring it at every 400 Hz
        # physics substep.
        self._pace_body_mass = robot.data.default_mass.to(
            device=self.device,
            dtype=robot.data.body_com_lin_vel_w.dtype,
        ).clone()
        self.energy_accumulator = EnergyAccumulator(
            self.num_envs,
            device=self.device,
            dtype=robot.data.joint_pos.dtype,
        )

    def _accumulate_pace_energy(self) -> None:
        robot = self.scene["robot"]
        power = compute_power_components(
            robot.data.applied_torque,
            robot.data.joint_vel,
            self._pace_body_mass,
            robot.data.body_com_lin_vel_w,
            electrical_coefficient=self.cfg.pace_energy.electrical_coefficient,
            regeneration_coefficient=self.cfg.pace_energy.regeneration_coefficient,
            gravity=self.cfg.pace_energy.gravity,
        )
        self.energy_accumulator.accumulate(power, self.physics_dt)

    def _publish_pace_cost(self) -> None:
        expected_substeps = int(self.cfg.decimation)
        if self.energy_accumulator.physics_substeps != expected_substeps:
            raise RuntimeError(
                "PACE energy accounting missed physics steps: "
                f"expected {expected_substeps}, accumulated {self.energy_accumulator.physics_substeps}"
            )
        budget = float(self.cfg.pace_energy.episode_budget_j)
        if budget <= 0.0:
            raise ValueError("pace_energy.episode_budget_j must be positive")

        control = self.energy_accumulator.control_step_snapshot()
        episode = self.energy_accumulator.episode_snapshot()
        physical_cost = control["total"] / budget
        episode_physical_cost = episode["total"] / budget
        barrier_addition = torch.where(
            self.reset_terminated,
            torch.clamp(float(self.cfg.pace_energy.failure_barrier) - episode_physical_cost, min=0.0),
            torch.zeros_like(physical_cost),
        )

        self.extras["pace_cost"] = physical_cost + barrier_addition
        self.extras["pace_physical_cost"] = physical_cost
        self.extras["pace_barrier_cost"] = barrier_addition
        self.extras["pace_step_energy_j"] = control["total"]
        self.extras["pace_episode_energy_j"] = episode["total"]
        for name in ("electrical", "mechanical", "potential"):
            self.extras[f"pace_step_{name}_energy_j"] = control[name]

    def step(self, action: torch.Tensor):
        """Execute one policy step and integrate PACE energy after every physics step."""
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        self.energy_accumulator.begin_control_step()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)
            self._accumulate_pace_energy()

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        self._publish_pace_cost()

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
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
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _reset_idx(self, env_ids: Sequence[int]):
        terminal_energy = None
        if hasattr(self, "energy_accumulator"):
            terminal_energy = self.energy_accumulator.episode["total"][env_ids].clone()
        super()._reset_idx(env_ids)
        if hasattr(self, "energy_accumulator"):
            if terminal_energy is not None and terminal_energy.numel() > 0:
                self.extras.setdefault("log", {})["Episode/pace_energy_j"] = terminal_energy.mean()
            self.energy_accumulator.reset(env_ids)
