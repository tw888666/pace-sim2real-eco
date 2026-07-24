"""PACE reset events."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from pace_sim2real.utils.identified_parameters import articulation_encoder_bias


def reset_joints_by_scale_encoder(
    env,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset encoder positions, then map them once into simulator coordinates."""
    asset: Articulation = env.scene[asset_cfg.name]
    iter_env_ids = env_ids[:, None] if asset_cfg.joint_ids != slice(None) else env_ids

    encoder_pos = asset.data.default_joint_pos[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_vel = asset.data.default_joint_vel[iter_env_ids, asset_cfg.joint_ids].clone()
    encoder_pos *= math_utils.sample_uniform(*position_range, encoder_pos.shape, encoder_pos.device)
    joint_vel *= math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    bias = articulation_encoder_bias(asset)[iter_env_ids, asset_cfg.joint_ids]
    simulator_pos = encoder_pos + bias
    joint_pos_limits = asset.data.soft_joint_pos_limits[iter_env_ids, asset_cfg.joint_ids]
    simulator_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel_limits = asset.data.soft_joint_vel_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)
    asset.write_joint_state_to_sim(
        simulator_pos,
        joint_vel,
        joint_ids=asset_cfg.joint_ids,
        env_ids=env_ids,
    )
