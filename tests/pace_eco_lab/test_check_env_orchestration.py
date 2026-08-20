from __future__ import annotations

from pathlib import Path

import pytest

from scripts.pace_eco.check_env_orchestration import (
    OPEN_LOOP_REPLAY_KEY,
    build_phase_arguments,
    merge_phase_reports,
)


def test_phase_arguments_remove_parent_and_stale_internal_options(tmp_path: Path):
    result_path = tmp_path / "阶段报告.json"
    arguments = [
        "--task",
        "PACE",
        "--open_loop_replay",
        "--device",
        "cuda:0",
        "--_pace_phase=replay",
        "--_pace_result_path",
        "旧报告.json",
    ]

    actual = build_phase_arguments(arguments, "standard", result_path)

    assert "--open_loop_replay" not in actual
    assert "--_pace_phase=replay" not in actual
    assert "旧报告.json" not in actual
    assert actual[:4] == ["--task", "PACE", "--device", "cuda:0"]
    assert actual[-4:] == [
        "--_pace_phase",
        "standard",
        "--_pace_result_path",
        str(result_path.resolve()),
    ]


def test_phase_arguments_reject_unknown_phase(tmp_path: Path):
    with pytest.raises(ValueError, match="未知验收阶段"):
        build_phase_arguments([], "unknown", tmp_path / "报告.json")


def test_phase_reports_are_merged_only_with_valid_replay_result():
    standard = {"环境数": 16, "检查步数": 200}
    replay = {OPEN_LOOP_REPLAY_KEY: {"策略步数": 50}}

    assert merge_phase_reports(standard, replay) == {
        "环境数": 16,
        "检查步数": 200,
        OPEN_LOOP_REPLAY_KEY: {"策略步数": 50},
    }

    with pytest.raises(ValueError, match="有效结果"):
        merge_phase_reports(standard, {})
