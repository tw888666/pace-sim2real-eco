"""Gymnasium 任务注册。

配置类与环境类使用字符串入口，因而本模块可在不启动 Isaac Sim 时安全导入。
"""

from __future__ import annotations

import gymnasium as gym

from pace_eco_lab.constants import ECO_ID, FIXED_WEIGHT_ID, TASK_ONLY_ID
from pace_eco_lab.multi_terrain_protocol import TASK_IDS


def _register(
    task_id: str,
    env_cfg: str,
    agent_cfg: str,
    env_entry_point: str = "pace_eco_lab.envs.pace_env:PaceManagerBasedRLEnv",
) -> None:
    if task_id in gym.registry:
        return
    gym.register(
        id=task_id,
        entry_point=env_entry_point,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "rsl_rl_cfg_entry_point": agent_cfg,
        },
    )


def register_tasks() -> None:
    """幂等注册三个原 Flat 任务和18个固定20秒宽场多地形任务。"""

    _register(
        TASK_ONLY_ID,
        "pace_eco_lab.configs.env_cfg:PaceTaskOnlyEnvCfg",
        "pace_eco_lab.configs.agent_cfg:PaceTaskOnlyPPORunnerCfg",
    )
    _register(
        FIXED_WEIGHT_ID,
        "pace_eco_lab.configs.env_cfg:PaceFixedWeightEnvCfg",
        "pace_eco_lab.configs.agent_cfg:PaceFixedWeightPPORunnerCfg",
    )
    _register(
        ECO_ID,
        "pace_eco_lab.configs.env_cfg:PaceEcoEnvCfg",
        "pace_eco_lab.configs.agent_cfg:PaceEcoPPORunnerCfg",
    )
    terrain_titles = {
        "flat": "Flat",
        "rough": "Rough",
        "stairs": "Stairs",
        "boxes": "Boxes",
        "slope": "Slope",
        "mixed": "Mixed",
    }
    method_titles = {
        "task_only": "TaskOnly",
        "fixed_weight": "FixedWeight",
        "eco": "Eco",
    }
    for (method, terrain), task_id in TASK_IDS.items():
        terrain_title = terrain_titles[terrain]
        method_title = method_titles[method]
        _register(
            task_id,
            (
                "pace_eco_lab.configs.multi_terrain_env_cfg:"
                f"Pace{terrain_title}Terrain20s{method_title}EnvCfg"
            ),
            (
                "pace_eco_lab.configs.multi_terrain_agent_cfg:"
                f"Pace{terrain_title}Terrain20s{method_title}PPORunnerCfg"
            ),
            "pace_eco_lab.envs.terrain20s_env:PaceTerrain20sRLEnv",
        )


__all__ = ["register_tasks"]
