"""Frozen-policy terrain evaluation protocol and metrics."""

from .terrain import (
    TERRAIN_VARIANTS,
    TerrainEpisodeAccumulator,
    TerrainEvaluationProtocol,
    assert_pairing_matches,
    atomic_write_csv,
    atomic_write_json,
    configure_eval_protocol,
    configure_eval_terrain,
    pairing_signature,
    summarize_episode_rows,
)

__all__ = [
    "TERRAIN_VARIANTS",
    "TerrainEpisodeAccumulator",
    "TerrainEvaluationProtocol",
    "assert_pairing_matches",
    "atomic_write_csv",
    "atomic_write_json",
    "configure_eval_protocol",
    "configure_eval_terrain",
    "pairing_signature",
    "summarize_episode_rows",
]
