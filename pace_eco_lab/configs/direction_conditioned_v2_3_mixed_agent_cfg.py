"""方向条件 v2.3 Mixed 三方法独立输出配置。"""

from __future__ import annotations

from isaaclab.utils import configclass

from pace_eco_lab.configs.agent_cfg import (
    PaceEcoPPORunnerCfg,
    PaceFixedWeightPPORunnerCfg,
    PaceTaskOnlyPPORunnerCfg,
)


def _runner(base, name: str, method: str):
    frozen_name = f"pace_direction_conditioned_v2_3_mixed_{method}_terrain20s_wide_anymal_d"

    @configclass
    class Runner(base):
        experiment_name: str = frozen_name

    Runner.__name__ = name
    Runner.__qualname__ = name
    return Runner


PaceDirectionConditionedV23MixedTerrain20sTaskOnlyPPORunnerCfg = _runner(
    PaceTaskOnlyPPORunnerCfg,
    "PaceDirectionConditionedV23MixedTerrain20sTaskOnlyPPORunnerCfg",
    "task_only",
)
PaceDirectionConditionedV23MixedTerrain20sFixedWeightPPORunnerCfg = _runner(
    PaceFixedWeightPPORunnerCfg,
    "PaceDirectionConditionedV23MixedTerrain20sFixedWeightPPORunnerCfg",
    "fixed_weight",
)
PaceDirectionConditionedV23MixedTerrain20sEcoPPORunnerCfg = _runner(
    PaceEcoPPORunnerCfg,
    "PaceDirectionConditionedV23MixedTerrain20sEcoPPORunnerCfg",
    "eco",
)


__all__ = [name for name in globals() if name.endswith("PPORunnerCfg")]
