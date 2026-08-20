"""多地形任务沿用同一网络和 3000 次 PPO 配置，只隔离输出名称。"""

from __future__ import annotations

from isaaclab.utils import configclass

from pace_eco_lab.configs.agent_cfg import (
    PaceEcoPPORunnerCfg,
    PaceFixedWeightPPORunnerCfg,
    PaceTaskOnlyPPORunnerCfg,
)


def _make_runner(base_cls, class_name: str, experiment_name: str):
    frozen_experiment_name = experiment_name

    @configclass
    class Runner(base_cls):
        experiment_name: str = frozen_experiment_name

    Runner.__name__ = class_name
    Runner.__qualname__ = class_name
    return Runner


for _terrain in ("flat", "rough", "stairs", "boxes", "slope", "mixed"):
    _title = {
        "flat": "Flat",
        "rough": "Rough",
        "stairs": "Stairs",
        "boxes": "Boxes",
        "slope": "Slope",
        "mixed": "Mixed",
    }[_terrain]
    globals()[f"Pace{_title}Terrain20sTaskOnlyPPORunnerCfg"] = _make_runner(
        PaceTaskOnlyPPORunnerCfg,
        f"Pace{_title}Terrain20sTaskOnlyPPORunnerCfg",
        f"pace_task_only_{_terrain}_terrain20s_wide_anymal_d",
    )
    globals()[f"Pace{_title}Terrain20sFixedWeightPPORunnerCfg"] = _make_runner(
        PaceFixedWeightPPORunnerCfg,
        f"Pace{_title}Terrain20sFixedWeightPPORunnerCfg",
        f"pace_fixed_weight_{_terrain}_terrain20s_wide_anymal_d",
    )
    globals()[f"Pace{_title}Terrain20sEcoPPORunnerCfg"] = _make_runner(
        PaceEcoPPORunnerCfg,
        f"Pace{_title}Terrain20sEcoPPORunnerCfg",
        f"pace_eco_{_terrain}_terrain20s_wide_anymal_d",
    )


__all__ = [name for name in globals() if name.endswith("PPORunnerCfg")]
