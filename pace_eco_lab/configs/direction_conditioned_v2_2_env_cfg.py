"""v2.2 方向条件固定权重环境配置包装；不改奖励函数或环境实现。"""

from __future__ import annotations

from isaaclab.utils import configclass

from pace_eco_lab.configs import direction_conditioned_env_cfg as v2_1_env_cfg
from pace_eco_lab.direction_conditioned_v2_2_protocol import FIXED_ENERGY_REWARD_WEIGHT


def _make_fixed(base_cls, name: str):
    @configclass
    class Fixed(base_cls):
        def __post_init__(self):
            super().__post_init__()
            self.rewards.energy.weight = FIXED_ENERGY_REWARD_WEIGHT

    Fixed.__name__ = name
    Fixed.__qualname__ = name
    return Fixed


_TERRAIN_TITLES = {
    "flat": "Flat",
    "rough": "Rough",
    "stairs": "Stairs",
    "boxes": "Boxes",
    "slope": "Slope",
}

for _terrain_title in _TERRAIN_TITLES.values():
    _base_name = f"PaceDirectionConditioned{_terrain_title}Terrain20sTaskOnlyEnvCfg"
    _name = f"PaceDirectionConditioned{_terrain_title}Terrain20sFixedWeightEnvCfg"
    globals()[_name] = _make_fixed(getattr(v2_1_env_cfg, _base_name), _name)


__all__ = [name for name in globals() if name.endswith("FixedWeightEnvCfg")]
