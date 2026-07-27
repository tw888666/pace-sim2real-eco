"""Simulator-independent finite-horizon time helpers."""

from __future__ import annotations

import torch


def normalized_time_to_go(episode_step: torch.Tensor, max_episode_steps: int) -> torch.Tensor:
    """Map a pre-action episode step to normalized remaining control time.

    Step zero is the initial observation and maps to one.  The observation
    before the final control transition has step ``max_episode_steps - 1`` and
    maps to ``1 / max_episode_steps``.  The terminal boundary maps to zero.
    """
    if max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive")
    remaining = 1.0 - episode_step.float() / float(max_episode_steps)
    return remaining.clamp(0.0, 1.0)
