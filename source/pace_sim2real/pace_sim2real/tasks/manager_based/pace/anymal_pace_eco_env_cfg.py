"""Simulation-only ANYmal-D environment for PACE + PPO-Lagrangian."""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.flat_env_cfg import AnymalDFlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg, RewardsCfg

from pace_sim2real.utils import PaceDCMotorCfg
from pace_sim2real.utils.identified_parameters import ANYMAL_D_JOINT_ORDER, ANYMAL_D_OFFICIAL_PARAMETERS

from . import mdp


ANYDRIVE_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    saturation_effort=140.0,
    effort_limit=89.0,
    velocity_limit=8.5,
    stiffness={".*": 85.0},
    damping={".*": 0.6},
    encoder_bias={".*": 0.0},
    friction={".*": 0.0},
    dynamic_friction={".*": 0.0},
    viscous_friction={".*": 0.0},
    max_delay=10,
)


@configclass
class PaceIdentificationCfg:
    parameters: tuple[float, ...] = ANYMAL_D_OFFICIAL_PARAMETERS
    joint_order: tuple[str, ...] = ANYMAL_D_JOINT_ORDER


@configclass
class PaceEnergyCfg:
    electrical_coefficient: float = 0.0192
    regeneration_coefficient: float = 0.0
    gravity: float = 9.81
    # Development placeholder. Formal 100/90/80% budgets are calibrated from
    # frozen-policy full episodes before multi-seed training.
    episode_budget_j: float = 10_000.0
    failure_barrier: float = 1.25


@configclass
class PaceRewardsCfg(RewardsCfg):
    ftd = RewTerm(
        func=mdp.foot_touchdown_velocity,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*FOOT"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "history_length": 3,
        },
    )


@configclass
class PaceObservationsCfg(ObservationsCfg):
    """Keep the actor observation unchanged and expose time only as cost context."""

    @configclass
    class CostTimeCfg(ObsGroup):
        remaining_time = ObsTerm(func=mdp.normalized_remaining_time)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    cost_time: CostTimeCfg = CostTimeCfg()


@configclass
class AnymalDPaceEcoEnvCfg(AnymalDFlatEnvCfg):
    observations: PaceObservationsCfg = PaceObservationsCfg()
    rewards: PaceRewardsCfg = PaceRewardsCfg()
    pace_identification: PaceIdentificationCfg = PaceIdentificationCfg()
    pace_energy: PaceEnergyCfg = PaceEnergyCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ANYMAL_D_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            actuators={"legs": ANYDRIVE_PACE_ACTUATOR_CFG},
        )

        # 400 Hz physics, 50 Hz policy/control.
        self.sim.dt = 0.0025
        self.decimation = 8
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt
        self.episode_length_s = 20.0
        self.observations.policy.joint_pos.func = mdp.encoder_joint_pos_rel

        # Fixed 1 m/s forward command on flat terrain for the first study.
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # PACE four-term design minus the energy reward: velocity, collision, FTD.
        self.rewards.track_lin_vel_xy_exp.weight = 0.2
        self.rewards.track_ang_vel_z_exp.weight = 0.2
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.func = mdp.pace_collision_indicator
        self.rewards.undesired_contacts.params = {
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*THIGH"]),
            "asset_cfg": SceneEntityCfg("robot"),
        }
        self.rewards.lin_vel_z_l2 = None
        self.rewards.ang_vel_xy_l2 = None
        self.rewards.dof_torques_l2 = None
        self.rewards.dof_acc_l2 = None
        self.rewards.action_rate_l2 = None
        self.rewards.feet_air_time = None
        self.rewards.flat_orientation_l2 = None
        self.rewards.dof_pos_limits = None

        # No dynamics randomization in the first calibrated simulation-only study.
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_scale_encoder,
            mode="reset",
            params={
                "position_range": (1.0, 1.0),
                "velocity_range": (0.0, 0.0),
            },
        )
        self.curriculum.terrain_levels = None


@configclass
class AnymalDPaceEcoEnvCfg_PLAY(AnymalDPaceEcoEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
