"""Terrain20s 的只读位移辅助；主协议不新增奖励、终止或课程项。"""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def terrain_displacement(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """返回机器人相对当前生成式地形出生 origin 的世界坐标位移。"""

    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


__all__ = ["terrain_displacement"]
