"""方向条件 v2.3 Mixed 三方法环境配置。"""

from __future__ import annotations

from isaaclab.utils import configclass

from pace_eco_lab.configs.direction_conditioned_env_cfg import (
    PaceDirectionConditionedMixedTerrain20sTaskOnlyEnvCfg,
)
from pace_eco_lab.direction_conditioned_v2_3_mixed_protocol import FIXED_ENERGY_REWARD_WEIGHT


@configclass
class PaceDirectionConditionedV23MixedTerrain20sTaskOnlyEnvCfg(
    PaceDirectionConditionedMixedTerrain20sTaskOnlyEnvCfg
):
    pass


@configclass
class PaceDirectionConditionedV23MixedTerrain20sFixedWeightEnvCfg(
    PaceDirectionConditionedV23MixedTerrain20sTaskOnlyEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.energy.weight = FIXED_ENERGY_REWARD_WEIGHT


@configclass
class PaceDirectionConditionedV23MixedTerrain20sEcoEnvCfg(
    PaceDirectionConditionedV23MixedTerrain20sTaskOnlyEnvCfg
):
    pass


__all__ = [name for name in globals() if name.startswith("PaceDirectionConditionedV23")]
