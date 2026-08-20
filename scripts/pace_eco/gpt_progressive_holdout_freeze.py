#!/usr/bin/env python3
"""在不改动冻结器原文件的前提下生成渐进式 holdout 授权。"""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts/pace_eco/freeze_multi_terrain_holdout.py"
EXPECTED_SHA256 = "8ea515f1b7b60e4ce16217998583e77da7e0463df42483437916fc3997b263c2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"渐进授权补丁锚点数量错误：{old[:60]!r}")
    return source.replace(old, new, 1)


actual = _sha256(SOURCE)
if actual != EXPECTED_SHA256:
    raise RuntimeError(
        f"原始 holdout 冻结器哈希变化，拒绝运行渐进入口：{actual} != {EXPECTED_SHA256}"
    )

code = SOURCE.read_text(encoding="utf-8")
code = _replace_once(
    code,
    'parser.add_argument("--energy_reference_json", required=True)\n',
    'parser.add_argument("--energy_reference_json", required=True)\n'
    'parser.add_argument("--progressive", action="store_true")\n',
)
code = _replace_once(
    code,
    '        or "已冻结" not in str(protocol.get("冻结状态"))\n',
    '        or (not args.progressive and "已冻结" not in str(protocol.get("冻结状态")))\n',
)
code = _replace_once(
    code,
    '        if not checkpoint.is_file():\n'
    '            raise FileNotFoundError(f"正式运行缺少 model_2999.pt：{record_path.parent}")\n',
    '        if not checkpoint.is_file():\n'
    '            if args.progressive:\n'
    '                continue\n'
    '            raise FileNotFoundError(f"正式运行缺少 model_2999.pt：{record_path.parent}")\n',
)
code = _replace_once(
    code,
    '    if missing:\n'
    '        raise RuntimeError(f"正式终点未齐备，holdout 保持封存；缺少 {len(missing)} 项：{missing[:5]}")\n',
    '    if missing and not args.progressive:\n'
    '        raise RuntimeError(f"正式终点未齐备，holdout 保持封存；缺少 {len(missing)} 项：{missing[:5]}")\n'
    '    if args.progressive and not found:\n'
    '        raise RuntimeError("渐进授权至少需要一个已完成的正式终点。")\n',
)
code = _replace_once(
    code,
    '        "冻结状态": "holdout已授权",\n',
    '        "冻结状态": "holdout渐进已授权" if args.progressive else "holdout已授权",\n'
    '        "授权模式": "渐进式" if args.progressive else "完整",\n'
    '        "计划模型数量": len(expected),\n'
    '        "未完成模型数量": len(missing),\n',
)
code = _replace_once(
    code,
    '        "规则": "只按预注册任务、PPO seed、model_2999.pt 存在性和唯一性冻结，不读取性能。",\n',
    '        "规则": (\n'
    '            "渐进式授权：只按预注册任务、PPO seed、model_2999.pt 存在性和唯一性冻结，"\n'
    '            "不读取性能；后续授权必须写入新文件，不得覆盖。"\n'
    '            if args.progressive\n'
    '            else "只按预注册任务、PPO seed、model_2999.pt 存在性和唯一性冻结，不读取性能。"\n'
    '        ),\n',
)

namespace = {"__name__": "__main__", "__file__": str(SOURCE), "__package__": None}
exec(compile(code, str(SOURCE), "exec"), namespace)
