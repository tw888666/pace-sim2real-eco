"""与 v1 并存的方向条件多地形 v2.1 环境配置。"""

from __future__ import annotations

from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from pace_eco_lab.configs.env_cfg import CommandsCfg, ObservationsCfg, RewardsCfg
from pace_eco_lab.configs.multi_terrain_env_cfg import PaceTerrain20sTaskOnlyEnvCfg
from pace_eco_lab.constants import EPISODE_LENGTH_S
from pace_eco_lab.direction_conditioned_protocol import (
    DESIRED_DIRECTION_W,
    TARGET_SPEED_M_S_V2,
)
from pace_eco_lab.mdp.direction_command import DirectionCommandCfg
from pace_eco_lab.mdp.rewards import (
    pace_directional_velocity_tracking,
    pace_yaw_rate_tracking,
    scheduled_energy_reward,
)


@configclass
class DirectionCommandsCfg(CommandsCfg):
    """保留隐藏 v1 速度指令，并新增策略可见的方向指令。"""

    direction_command = DirectionCommandCfg(
        asset_name="robot",
        direction_w=DESIRED_DIRECTION_W,
        target_speed=TARGET_SPEED_M_S_V2,
        resampling_time_range=(EPISODE_LENGTH_S, EPISODE_LENGTH_S),
        debug_vis=False,
    )


@configclass
class DirectionPolicyCfg(ObservationsCfg.PolicyCfg):
    # 复用原三维 command 槽位，网络输入维度和顺序保持不变。
    velocity_command = ObsTerm(
        func=base_mdp.generated_commands,
        params={"command_name": "direction_command"},
    )


@configclass
class DirectionCriticCfg(ObservationsCfg.CriticCfg):
    velocity_command = ObsTerm(
        func=base_mdp.generated_commands,
        params={"command_name": "direction_command"},
    )


@configclass
class DirectionObservationsCfg(ObservationsCfg):
    policy: DirectionPolicyCfg = DirectionPolicyCfg()
    critic: DirectionCriticCfg = DirectionCriticCfg()


@configclass
class DirectionalRewardsCfg(RewardsCfg):
    """只替换线速度参考系，并把原组合核拆成等尺度的两项。"""

    velocity = None
    directional_velocity = RewTerm(
        func=pace_directional_velocity_tracking,
        weight=0.2,
        params={"command_name": "direction_command", "sigma": 0.5},
    )
    yaw_rate = RewTerm(
        func=pace_yaw_rate_tracking,
        weight=0.2,
        params={"target_yaw_rate": 0.0, "sigma": 0.5},
    )
    energy = RewTerm(
        func=scheduled_energy_reward,
        weight=0.0,
        params={"command_name": "direction_command", "half_life_iterations": 500.0},
    )


@configclass
class PaceDirectionObservationControlTerrain20sTaskOnlyEnvCfg(PaceTerrain20sTaskOnlyEnvCfg):
    """E1：策略看到方向，但训练奖励保持 v1 body-frame 定义。"""

    commands: DirectionCommandsCfg = DirectionCommandsCfg()
    observations: DirectionObservationsCfg = DirectionObservationsCfg()
    pace_direction_variant: str = "observation_control"


@configclass
class PaceDirectionConditionedTerrain20sTaskOnlyEnvCfg(PaceTerrain20sTaskOnlyEnvCfg):
    """E2：方向观测与世界方向速度奖励组成 v2.1 主要方法。"""

    commands: DirectionCommandsCfg = DirectionCommandsCfg()
    observations: DirectionObservationsCfg = DirectionObservationsCfg()
    rewards: DirectionalRewardsCfg = DirectionalRewardsCfg()
    pace_direction_variant: str = "directional"


def _make_category(base_cls, name: str, category: str):
    frozen_category = category

    @configclass
    class Category(base_cls):
        pace_terrain_category: str = frozen_category

    Category.__name__ = name
    Category.__qualname__ = name
    return Category


def _make_eco(base_cls, name: str):
    @configclass
    class Eco(base_cls):
        pass

    Eco.__name__ = name
    Eco.__qualname__ = name
    return Eco


_TERRAIN_TITLES = {
    "flat": "Flat",
    "rough": "Rough",
    "stairs": "Stairs",
    "boxes": "Boxes",
    "slope": "Slope",
    "mixed": "Mixed",
}
_VARIANT_BASES = {
    "DirectionObservationControl": PaceDirectionObservationControlTerrain20sTaskOnlyEnvCfg,
    "DirectionConditioned": PaceDirectionConditionedTerrain20sTaskOnlyEnvCfg,
}

for _variant_title, _base in _VARIANT_BASES.items():
    for _terrain, _terrain_title in _TERRAIN_TITLES.items():
        _task_name = f"Pace{_variant_title}{_terrain_title}Terrain20sTaskOnlyEnvCfg"
        _task_cls = _make_category(_base, _task_name, _terrain)
        globals()[_task_name] = _task_cls
        _eco_name = f"Pace{_variant_title}{_terrain_title}Terrain20sEcoEnvCfg"
        globals()[_eco_name] = _make_eco(_task_cls, _eco_name)


__all__ = [
    name
    for name in globals()
    if name.startswith("PaceDirection") or name.startswith("Direction")
]
