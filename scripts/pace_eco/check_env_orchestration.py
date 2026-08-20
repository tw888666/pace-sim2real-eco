"""PACE 环境验收的多进程编排辅助函数。

这里只放不依赖 Isaac Lab 的纯函数，便于在无 GPU 条件下做回归测试。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

OPEN_LOOP_REPLAY_KEY = "PACE悬空轨迹开环回放"
_INTERNAL_OPTIONS_WITH_VALUE = ("--_pace_phase", "--_pace_result_path")


def build_phase_arguments(
    arguments: Sequence[str],
    phase: str,
    result_path: Path,
) -> list[str]:
    """移除父进程参数并构造一个隔离验收阶段的命令行参数。"""

    if phase not in {"standard", "replay"}:
        raise ValueError(f"未知验收阶段：{phase}")

    cleaned: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--open_loop_replay":
            index += 1
            continue
        if argument in _INTERNAL_OPTIONS_WITH_VALUE:
            index += 2
            continue
        if any(argument.startswith(f"{option}=") for option in _INTERNAL_OPTIONS_WITH_VALUE):
            index += 1
            continue
        cleaned.append(argument)
        index += 1

    return [
        *cleaned,
        "--_pace_phase",
        phase,
        "--_pace_result_path",
        str(result_path.resolve()),
    ]


def merge_phase_reports(
    standard_report: dict[str, Any],
    replay_report: dict[str, Any],
) -> dict[str, Any]:
    """校验并合并常规检查与开环回放的阶段报告。"""

    if OPEN_LOOP_REPLAY_KEY in standard_report:
        raise ValueError("常规检查报告不应包含开环回放结果。")
    replay_result = replay_report.get(OPEN_LOOP_REPLAY_KEY)
    if not isinstance(replay_result, dict):
        raise ValueError("开环回放阶段没有生成有效结果。")

    merged = dict(standard_report)
    merged[OPEN_LOOP_REPLAY_KEY] = replay_result
    return merged
