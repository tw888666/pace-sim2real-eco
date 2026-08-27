"""v2.2 方向条件固定权重 PPO 输出配置。"""

from __future__ import annotations

from isaaclab.utils import configclass

from pace_eco_lab.configs.agent_cfg import PaceFixedWeightPPORunnerCfg


def _make_runner(class_name: str, experiment_name: str):
    frozen_experiment_name = experiment_name

    @configclass
    class Runner(PaceFixedWeightPPORunnerCfg):
        experiment_name: str = frozen_experiment_name

    Runner.__name__ = class_name
    Runner.__qualname__ = class_name
    return Runner


_TERRAIN_TITLES = {
    "flat": "Flat",
    "rough": "Rough",
    "stairs": "Stairs",
    "boxes": "Boxes",
    "slope": "Slope",
}

for _terrain, _terrain_title in _TERRAIN_TITLES.items():
    _name = f"PaceDirectionConditioned{_terrain_title}Terrain20sFixedWeightPPORunnerCfg"
    globals()[_name] = _make_runner(
        _name,
        f"pace_direction_conditioned_fixed_weight_{_terrain}_terrain20s_wide_anymal_d",
    )


__all__ = [name for name in globals() if name.endswith("FixedWeightPPORunnerCfg")]
