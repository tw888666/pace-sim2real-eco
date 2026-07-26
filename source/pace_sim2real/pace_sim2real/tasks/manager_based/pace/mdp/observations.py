"""PACE observations expressed in the robot encoder frame."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from pace_sim2real.utils.identified_parameters import articulation_encoder_bias


def encoder_joint_pos_rel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Joint position relative to the default pose in encoder coordinates.

    PACE defines ``q_enc = q_sim - bias``.  The default pose remains in the
    encoder coordinate system, so the bias is subtracted exactly once here.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    bias = articulation_encoder_bias(asset)
    return (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - bias[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )


def normalized_remaining_time(env) -> torch.Tensor:
    """Return the finite-horizon time-to-go in ``[0, 1]`` for the cost critic."""
    remaining = 1.0 - env.episode_length_buf.float() / float(env.max_episode_length)
    return remaining.clamp_(0.0, 1.0).unsqueeze(-1)
