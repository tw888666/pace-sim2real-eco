"""方向条件 v2.1 的独立 RSL-RL 输出配置。"""

from __future__ import annotations

from isaaclab.utils import configclass

from pace_eco_lab.configs.agent_cfg import PaceEcoPPORunnerCfg, PaceTaskOnlyPPORunnerCfg


def _make_runner(base_cls, class_name: str, experiment_name: str):
    frozen_experiment_name = experiment_name

    @configclass
    class Runner(base_cls):
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
    "mixed": "Mixed",
}
_VARIANTS = {
    "DirectionObservationControl": "direction_obs_control",
    "DirectionConditioned": "direction_conditioned",
}
_METHODS = {
    "TaskOnly": (PaceTaskOnlyPPORunnerCfg, "task_only"),
    "Eco": (PaceEcoPPORunnerCfg, "eco"),
}

for _variant_title, _variant_token in _VARIANTS.items():
    for _terrain, _terrain_title in _TERRAIN_TITLES.items():
        for _method_title, (_base, _method_token) in _METHODS.items():
            _name = (
                f"Pace{_variant_title}{_terrain_title}Terrain20s"
                f"{_method_title}PPORunnerCfg"
            )
            globals()[_name] = _make_runner(
                _base,
                _name,
                f"pace_{_variant_token}_{_method_token}_{_terrain}_terrain20s_wide_anymal_d",
            )


__all__ = [name for name in globals() if name.endswith("PPORunnerCfg")]
