# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""PACE: Precise Adaptation through Continuous Evolution.

Isaac Sim modules are imported lazily.  This keeps numerical utilities such as
the PACE energy model testable in a normal CPU Python process while preserving
the original public API after :class:`isaaclab.app.AppLauncher` starts Kit.
"""

__all__ = [
    "PaceSim2realEnvCfg",
    "PaceSim2realSceneCfg",
    "PaceCfg",
    "CMAESOptimizer",
]


def __getattr__(name: str):
    if name == "CMAESOptimizer":
        from .optim import CMAESOptimizer

        return CMAESOptimizer
    if name in {"PaceSim2realEnvCfg", "PaceSim2realSceneCfg", "PaceCfg"}:
        from .tasks.manager_based.pace.pace_sim2real_env_cfg import PaceCfg, PaceSim2realEnvCfg, PaceSim2realSceneCfg

        return {
            "PaceSim2realEnvCfg": PaceSim2realEnvCfg,
            "PaceSim2realSceneCfg": PaceSim2realSceneCfg,
            "PaceCfg": PaceCfg,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
