"""训练日志中的致命错误与非有限数值审计。"""

from __future__ import annotations

import re
from typing import Any


_ITERATION = re.compile(r"Learning iteration\s+(\d+)/3000")
_FATAL = re.compile(
    r"Traceback|CUDA out of memory|\bOOM\b|Segmentation fault|"
    r"illegal memory access|(?:^|\s)Killed(?:\s|$)|(?:^|\s)Aborted(?:\s|$)",
    re.IGNORECASE,
)
_NONFINITE = re.compile(r"(?<![A-Za-z0-9_])(?:nan|[+-]?inf(?:inity)?)(?![A-Za-z0-9_])", re.IGNORECASE)
_EMPTY_COMPLETED_EPISODE = re.compile(
    r"^\s*Mean (normalized_episode_cost|constraint_violation) loss:\s*nan\s*$",
    re.IGNORECASE,
)


def audit_training_log(text: str) -> dict[str, Any]:
    """区分优化数值异常与“本轮无完整回合”的空集合展示值。

    PPO-Lagrangian 在一轮没有结束回合时，明确用 ``nan`` 表示两个仅用于
    展示的 episode 汇总值；它们不进入策略、价值或拉格朗日乘子更新。
    除这两个精确字段外，任何非有限值仍然是硬失败。
    """

    iteration: int | None = None
    fatal_lines: list[dict[str, object]] = []
    unexpected_nonfinite: list[dict[str, object]] = []
    empty_episode_telemetry: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = _ITERATION.search(line)
        if match:
            iteration = int(match.group(1))
        if _FATAL.search(line):
            fatal_lines.append({"行号": line_number, "迭代": iteration, "内容": line.strip()})
        if not _NONFINITE.search(line):
            continue
        empty_match = _EMPTY_COMPLETED_EPISODE.fullmatch(line)
        item = {"行号": line_number, "迭代": iteration, "内容": line.strip()}
        if empty_match:
            item["字段"] = empty_match.group(1)
            empty_episode_telemetry.append(item)
        else:
            unexpected_nonfinite.append(item)

    iterations = [int(item["迭代"]) for item in empty_episode_telemetry if item["迭代"] is not None]
    return {
        "状态": "通过" if not fatal_lines and not unexpected_nonfinite else "失败",
        "致命错误": fatal_lines,
        "意外非有限值": unexpected_nonfinite,
        "空回合遥测非有限值": {
            "数量": len(empty_episode_telemetry),
            "首次迭代": min(iterations) if iterations else None,
            "末次迭代": max(iterations) if iterations else None,
            "字段": sorted({str(item["字段"]) for item in empty_episode_telemetry}),
            "解释": "该轮无已结束回合，两个episode汇总展示值为空；不参与优化更新。",
        },
    }
