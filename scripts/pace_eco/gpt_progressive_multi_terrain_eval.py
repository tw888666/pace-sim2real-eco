#!/usr/bin/env python3
"""保持原评估实现不变，仅允许经过冻结的渐进式 holdout 授权。"""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts/pace_eco/multi_terrain_eval.py"
EXPECTED_SHA256 = "145272a9b70932b0e01b62adb6511bcbd92cdc42b620a46b014df86d7ead5bb1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"渐进评估补丁锚点数量错误：{old[:60]!r}")
    return source.replace(old, new, 1)


actual = _sha256(SOURCE)
if actual != EXPECTED_SHA256:
    raise RuntimeError(
        f"原始多地形评估器哈希变化，拒绝运行渐进入口：{actual} != {EXPECTED_SHA256}"
    )

code = SOURCE.read_text(encoding="utf-8")
code = _replace_once(
    code,
    '        data.get("冻结状态") != "holdout已授权"\n',
    '        data.get("冻结状态") not in ("holdout已授权", "holdout渐进已授权")\n',
)
code = _replace_once(
    code,
    '    if len(models) != expected_count:\n'
    '        parser.error(f"holdout 授权应包含 {expected_count} 个模型，实际为 {len(models)}。")\n',
    '    progressive = data.get("冻结状态") == "holdout渐进已授权"\n'
    '    if progressive:\n'
    '        if data.get("授权模式") != "渐进式" or not 0 < len(models) < expected_count:\n'
    '            parser.error("渐进式 holdout 授权的状态、模式或模型数量无效。")\n'
    '        if int(data.get("计划模型数量", -1)) != expected_count:\n'
    '            parser.error("渐进式 holdout 授权的计划模型数量错误。")\n'
    '    elif len(models) != expected_count:\n'
    '        parser.error(f"holdout 授权应包含 {expected_count} 个模型，实际为 {len(models)}。")\n',
)

namespace = {"__name__": "__main__", "__file__": str(SOURCE), "__package__": None}
exec(compile(code, str(SOURCE), "exec"), namespace)
