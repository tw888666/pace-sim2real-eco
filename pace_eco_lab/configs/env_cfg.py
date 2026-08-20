"""PACE ANYmal D 平地环境配置。"""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as UniformNoise
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG
from isaaclab_tasks.manager_based.locomotion.velocity import mdp as velocity_mdp

from pace_eco_lab.constants import (
    ACTION_SCALE,
    DECIMATION,
    EFFORT_LIMIT_NM,
    EPISODE_LENGTH_S,
    GLOBAL_DELAY_STEPS,
    GROUND_DYNAMIC_FRICTION,
    GROUND_STATIC_FRICTION,
    PACE_JOINT_NAMES,
    PD_DAMPING,
    PD_STIFFNESS,
    PHYSICS_DT_S,
    SATURATION_EFFORT_NM,
    VELOCITY_LIMIT_RAD_S,
)
from pace_eco_lab.mdp.actions import PaceJointPositionActionCfg
from pace_eco_lab.mdp.actuator import PaceDelayedPDActuatorCfg
from pace_eco_lab.mdp.observations import binary_foot_contacts, ground_friction, joint_observation_in_pace_order
from pace_eco_lab.mdp.parameters import load_pace_parameters, named_joint_values
from pace_eco_lab.mdp.rewards import (
    PaceFootTouchdownPenalty,
    pace_collision_indicator,
    pace_velocity_tracking,
    scheduled_energy_reward,
)
from pace_eco_lab.mdp.resets import (
    CALIBRATION_STATE_SET,
    reset_joints_from_evaluation_set,
    reset_joints_training,
    reset_root_state_from_evaluation_set,
)

_PACE_PARAMETERS = load_pace_parameters(applied_delay_steps=GLOBAL_DELAY_STEPS)


def make_pace_anymal_d_cfg() -> ArticulationCfg:
    robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.actuators = {
        "pace_legs": PaceDelayedPDActuatorCfg(
            joint_names_expr=list(PACE_JOINT_NAMES),
            effort_limit=EFFORT_LIMIT_NM,
            effort_limit_sim=SATURATION_EFFORT_NM,
            velocity_limit=VELOCITY_LIMIT_RAD_S,
            velocity_limit_sim=VELOCITY_LIMIT_RAD_S,
            stiffness=PD_STIFFNESS,
            damping=PD_DAMPING,
            armature=named_joint_values(_PACE_PARAMETERS.armature),
            friction=named_joint_values(_PACE_PARAMETERS.coulomb_friction),
            dynamic_friction=named_joint_values(_PACE_PARAMETERS.coulomb_friction),
            viscous_friction=named_joint_values(_PACE_PARAMETERS.viscous_damping),
            encoder_bias=named_joint_values(_PACE_PARAMETERS.joint_bias),
            min_delay=GLOBAL_DELAY_STEPS,
            max_delay=GLOBAL_DELAY_STEPS,
            saturation_effort=SATURATION_EFFORT_NM,
        )
    }
    robot.soft_joint_pos_limit_factor = 0.95
    return robot


@configclass
class PaceSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            # 机器人刚体在 startup 事件中固定为目标摩擦；地面取 1，避免
            # multiply 合并模式把目标系数再次平方。
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = MISSING
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.15, size=(3.0, 1.95)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )


@configclass
class CommandsCfg:
    base_velocity = velocity_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(EPISODE_LENGTH_S, EPISODE_LENGTH_S),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=velocity_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(1.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    joint_pos = PaceJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(PACE_JOINT_NAMES),
        scale=ACTION_SCALE,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel, noise=UniformNoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel, noise=UniformNoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=base_mdp.projected_gravity,
            noise=UniformNoise(n_min=-0.05, n_max=0.05),
        )
        velocity_command = ObsTerm(func=base_mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_position = ObsTerm(
            func=joint_observation_in_pace_order,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=list(PACE_JOINT_NAMES), preserve_order=True),
                "velocity": False,
            },
            noise=UniformNoise(n_min=-0.01, n_max=0.01),
        )
        joint_velocity = ObsTerm(
            func=joint_observation_in_pace_order,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=list(PACE_JOINT_NAMES), preserve_order=True),
                "velocity": True,
            },
            noise=UniformNoise(n_min=-1.5, n_max=1.5),
        )
        previous_action = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=base_mdp.projected_gravity)
        velocity_command = ObsTerm(func=base_mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_position = ObsTerm(
            func=joint_observation_in_pace_order,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=list(PACE_JOINT_NAMES), preserve_order=True),
                "velocity": False,
            },
        )
        joint_velocity = ObsTerm(
            func=joint_observation_in_pace_order,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=list(PACE_JOINT_NAMES), preserve_order=True),
                "velocity": True,
            },
        )
        previous_action = ObsTerm(func=base_mdp.last_action)
        base_wrench = ObsTerm(
            func=base_mdp.body_incoming_wrench,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
        )
        ground_friction = ObsTerm(func=ground_friction, params={"value": GROUND_DYNAMIC_FRICTION})
        foot_contact = ObsTerm(
            func=binary_foot_contacts,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT")},
        )
        height_scan = ObsTerm(
            func=base_mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    physics_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (GROUND_STATIC_FRICTION, GROUND_STATIC_FRICTION),
            "dynamic_friction_range": (GROUND_DYNAMIC_FRICTION, GROUND_DYNAMIC_FRICTION),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )
    reset_base = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.1, 0.1),
            },
            "velocity_range": {
                "x": (-0.25, 0.25),
                "y": (-0.25, 0.25),
                "z": (-0.1, 0.1),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    )
    reset_joints = EventTerm(
        func=reset_joints_training,
        mode="reset",
        params={"position_scale_range": (0.9, 1.1), "velocity_range": (-0.1, 0.1)},
    )


@configclass
class RewardsCfg:
    velocity = RewTerm(
        func=pace_velocity_tracking,
        weight=0.2,
        params={"command_name": "base_velocity", "sigma": 0.5},
    )
    collision = RewTerm(
        func=pace_collision_indicator,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(PACE_JOINT_NAMES), preserve_order=True),
            "thigh_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"),
        },
    )
    foot_touchdown = RewTerm(
        func=PaceFootTouchdownPenalty,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*FOOT"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "history_length": 3,
            "half_life_iterations": 500.0,
        },
    )
    energy = RewTerm(
        func=scheduled_energy_reward,
        weight=0.0,
        params={"half_life_iterations": 500.0},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=base_mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    pass


@configclass
class PaceTaskOnlyEnvCfg(ManagerBasedRLEnvCfg):
    scene: PaceSceneCfg = PaceSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    pace_include_potential: bool = True

    def __post_init__(self):
        self.scene.robot = make_pace_anymal_d_cfg()
        self.decimation = DECIMATION
        self.episode_length_s = EPISODE_LENGTH_S
        self.sim.dt = PHYSICS_DT_S
        self.sim.render_interval = DECIMATION
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.height_scanner.update_period = DECIMATION * PHYSICS_DT_S
        self.scene.contact_forces.update_period = PHYSICS_DT_S


@configclass
class PaceFixedWeightEnvCfg(PaceTaskOnlyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.energy.weight = -16.0e-5


@configclass
class PaceEcoEnvCfg(PaceTaskOnlyEnvCfg):
    pass


def configure_evaluation(
    env_cfg: PaceTaskOnlyEnvCfg,
    state_set: str = CALIBRATION_STATE_SET,
    state_index_offset: int = 0,
) -> PaceTaskOnlyEnvCfg:
    """关闭观察随机性并选择确定性状态集，供评估使用。"""

    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.reset_base.func = reset_root_state_from_evaluation_set
    env_cfg.events.reset_base.params = {
        "state_set": state_set,
        "state_index_offset": int(state_index_offset),
        "asset_cfg": SceneEntityCfg("robot"),
    }
    env_cfg.events.reset_joints.func = reset_joints_from_evaluation_set
    env_cfg.events.reset_joints.params = {
        "state_set": state_set,
        "state_index_offset": int(state_index_offset),
        "asset_cfg": SceneEntityCfg("robot"),
    }
    return env_cfg
