"""PACE 任务随机化；不随机化已辨识的动力学参数。"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def randomize_ground_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    static_friction_range: tuple[float, float],
    dynamic_friction_range: tuple[float, float],
    restitution: float = 0.0,
) -> None:
    """逐环境设置统一刚体摩擦，并记录供 critic（评论家网络）观察的动摩擦。

    Isaac Lab 自带材质事件会逐碰撞形状独立抽样，无法提供论文所需的单个
    ground-friction 特权状态；这里为同一环境的全部机器人形状使用同一组系数。
    """

    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    if not isinstance(asset, (RigidObject, Articulation)):
        raise TypeError(f"不支持为 {type(asset).__name__} 设置地面摩擦。")
    if static_friction_range[0] > static_friction_range[1]:
        raise ValueError("静摩擦随机化范围上下界颠倒。")
    if dynamic_friction_range[0] > dynamic_friction_range[1]:
        raise ValueError("动摩擦随机化范围上下界颠倒。")

    if env_ids is None:
        env_ids_cpu = torch.arange(env.num_envs, device="cpu")
    else:
        env_ids_cpu = env_ids.to(device="cpu", dtype=torch.long)
    count = len(env_ids_cpu)

    static = torch.empty(count, device="cpu").uniform_(*static_friction_range)
    if dynamic_friction_range == static_friction_range:
        # 训练采用论文描述的单一 ground-friction 系数。
        dynamic = static.clone()
    else:
        dynamic = torch.empty(count, device="cpu").uniform_(*dynamic_friction_range)
        dynamic = torch.minimum(dynamic, static)

    materials = asset.root_physx_view.get_material_properties().clone()
    materials[env_ids_cpu, :, 0] = static[:, None]
    materials[env_ids_cpu, :, 1] = dynamic[:, None]
    materials[env_ids_cpu, :, 2] = float(restitution)
    asset.root_physx_view.set_material_properties(materials, env_ids_cpu)

    if not hasattr(env, "pace_ground_friction"):
        env.pace_ground_friction = torch.zeros((env.num_envs, 1), device=env.device)
    env.pace_ground_friction[env_ids_cpu.to(env.device), 0] = dynamic.to(env.device)


__all__ = ["randomize_ground_friction"]
