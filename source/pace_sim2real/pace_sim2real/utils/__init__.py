# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Utility functions and actuator models for PACE.

Actuator imports are lazy because they require a running Isaac Sim Python
environment, whereas parameter parsing is intentionally simulator-independent.
"""

__all__ = [
    "PaceDCMotorCfg",
    "PaceDCMotor",
    "project_root",
]


def __getattr__(name: str):
    if name == "PaceDCMotorCfg":
        from .pace_actuator_cfg import PaceDCMotorCfg

        return PaceDCMotorCfg
    if name == "PaceDCMotor":
        from .pace_actuator import PaceDCMotor

        return PaceDCMotor
    if name == "project_root":
        from .paths import project_root

        return project_root
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
