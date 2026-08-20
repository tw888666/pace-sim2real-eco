"""无需启动 Isaac Sim 即可测试的评估前状态快照。"""

from __future__ import annotations

from collections.abc import MutableMapping

import torch


def publish_eval_state(
    extras: MutableMapping[str, object],
    root_pos_w: torch.Tensor,
    root_lin_vel_b: torch.Tensor,
    *,
    enabled: bool,
) -> None:
    """按需复制自动重置前的根位置和机身坐标系线速度。"""

    if not enabled:
        return
    extras["pace_eval_root_pos_w"] = root_pos_w.clone()
    extras["pace_eval_root_lin_vel_b"] = root_lin_vel_b.clone()


__all__ = ["publish_eval_state"]
