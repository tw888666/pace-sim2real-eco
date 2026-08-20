"""PACE-ECO 的独立 Isaac Lab 扩展包。

导入本包只执行 Gym 任务注册，不启动 Isaac Sim，也不修改官方 Isaac Lab。
"""

from __future__ import annotations

from .envs import register_tasks

register_tasks()

__all__ = ["register_tasks"]
