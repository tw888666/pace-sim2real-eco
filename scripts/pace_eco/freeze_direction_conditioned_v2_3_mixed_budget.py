#!/usr/bin/env python3
"""由 v2.3 Mixed task-only seed0 的四批 calibration 冻结 B_ref/B80。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from pace_eco_lab.direction_conditioned_v2_3_mixed_protocol import (
    BUDGET_SEED,
    EVAL_BATCHES,
    EVAL_EPISODES,
    FOOT_BOUNDARY_AUDIT_VERSION,
    PROTOCOL_VERSION,
    SUBTASKS,
    SUBTASK_SUCCESS_MINIMUM,
)


parser = argparse.ArgumentParser(description="冻结 v2.3 Mixed B_ref 和 B80。")
parser.add_argument("--calibration_root", required=True)
parser.add_argument("--manifest", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _true(value: object) -> bool:
    return str(value).lower() in ("true", "1", "是")


def _mean(rows: list[dict[str, str]], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows)


def main() -> None:
    root = Path(args.calibration_root).expanduser().resolve()
    manifest = Path(args.manifest).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖 v2.3 Mixed 预算冻结文件：{output}")
    if not manifest.is_file() or not checkpoint.is_file() or checkpoint.name != "model_2999.pt":
        raise FileNotFoundError("manifest 或最终 checkpoint 不存在。")
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        manifest_data.get("冻结状态") != "已冻结"
        or manifest_data.get("协议版本") != PROTOCOL_VERSION
        or manifest_data.get("数据拆分") != "calibration"
    ):
        raise ValueError("calibration manifest 状态或协议不匹配。")
    paths = sorted(root.rglob("gpt-v2.3-Mixed-calibration-batch-*.csv"))
    if len(paths) != EVAL_BATCHES:
        raise RuntimeError(f"应有4个 calibration 批次CSV，实际 {len(paths)}。")
    rows: list[dict[str, str]] = []
    evidence: list[dict[str, str]] = []
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            batch_rows = list(csv.DictReader(stream))
        if len(batch_rows) != EVAL_EPISODES // EVAL_BATCHES:
            raise ValueError(f"批次CSV应有50回合：{path}")
        rows.extend(batch_rows)
        evidence.append({"路径": str(path), "SHA256": _sha256(path)})
    if len(rows) != EVAL_EPISODES or len({row["episode_id"] for row in rows}) != EVAL_EPISODES:
        raise ValueError("calibration 回合不完整或重复。")
    manifest_rows = {item["episode_id"]: item for item in manifest_data["逐回合"]}
    for row in rows:
        expected = manifest_rows.get(row["episode_id"])
        if expected is None:
            raise ValueError(f"CSV 出现 manifest 外回合：{row['episode_id']}")
        checks = {
            "batch_index": int(row["评估批次"]),
            "batch_env_number": int(row["环境编号"]),
            "subtask": row["方向子任务"],
            "batch_seed": int(row["地形批次seed"]),
            "initial_state_number": int(row["固定初始状态编号"]),
        }
        if any(expected[key] != value for key, value in checks.items()):
            raise ValueError(f"CSV 与 manifest 不一致：{row['episode_id']}/{checks}")
        if (
            row["协议版本"] != PROTOCOL_VERSION
            or int(row["PPO_seed"]) != BUDGET_SEED
            or row["方法"] != "task_only"
        ):
            raise ValueError("calibration 协议、方法或 PPO seed 错误。")
        components = sum(float(row[key]) for key in ("电气能耗_J", "机械能耗_J", "势能能耗_J"))
        energy = float(row["回合能耗_J"])
        if not math.isfinite(energy) or abs(components - energy) > max(1.0e-3, abs(energy) * 1.0e-6):
            raise ValueError(f"能耗分量恒等式失败：{row['episode_id']}")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["方向子任务"]].append(row)
    if set(grouped) != set(SUBTASKS):
        raise ValueError(f"七方向子任务不完整：{sorted(grouped)}")
    subtask_stats: dict[str, dict[str, object]] = {}
    references: dict[str, float] = {}
    for subtask in SUBTASKS:
        items = grouped[subtask]
        successful = [row for row in items if _true(row["方向穿越成功"])]
        minimum = SUBTASK_SUCCESS_MINIMUM[subtask]
        if len(successful) < minimum:
            raise RuntimeError(
                f"{subtask} 成功 {len(successful)}/{len(items)}，低于独立门槛 {minimum}，禁止冻结。"
            )
        reference = _mean(successful, "回合能耗_J")
        if reference <= 0.0:
            raise RuntimeError(f"{subtask} 成功回合平均总能耗非正，禁止冻结。")
        references[subtask] = reference
        subtask_stats[subtask] = {
            "回合数": len(items),
            "成功数": len(successful),
            "成功率": len(successful) / len(items),
            "最低成功数": minimum,
            "方向成功回合平均能耗_J": reference,
            "方向成功回合平均电气能耗_J": _mean(successful, "电气能耗_J"),
            "方向成功回合平均机械能耗_J": _mean(successful, "机械能耗_J"),
            "方向成功回合平均potential能耗_J": _mean(successful, "势能能耗_J"),
            "方向成功回合总能耗最小值_J": min(float(row["回合能耗_J"]) for row in successful),
            "方向成功回合负总能耗数": sum(float(row["回合能耗_J"]) < 0.0 for row in successful),
        }
    main_terrains = {
        "flat": references["flat"],
        "rough": references["rough"],
        "boxes": references["boxes"],
        "stairs": (references["stairs-up"] + references["stairs-down"]) / 2.0,
        "slope": (references["slope-up"] + references["slope-down"]) / 2.0,
    }
    mixed_reference = statistics.fmean(main_terrains.values())
    payload = {
        "冻结状态": "已冻结",
        "协议版本": PROTOCOL_VERSION,
        "边界审计版本": FOOT_BOUNDARY_AUDIT_VERSION,
        "创建时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "来源实验": "E2 directional Mixed task_only seed0 calibration",
        "两层宏平均定义": "stairs/slope先对up/down等权，再对flat/rough/stairs/boxes/slope五类等权。",
        "七方向子任务": subtask_stats,
        "七方向参考能耗_J": references,
        "五类主地形参考能耗_J": main_terrains,
        "B_ref_mixed_J": mixed_reference,
        "B80_mixed_J": 0.8 * mixed_reference,
        "potential符号审计": {
            "stairs_up_minus_down_J": subtask_stats["stairs-up"]["方向成功回合平均potential能耗_J"] - subtask_stats["stairs-down"]["方向成功回合平均potential能耗_J"],
            "slope_up_minus_down_J": subtask_stats["slope-up"]["方向成功回合平均potential能耗_J"] - subtask_stats["slope-down"]["方向成功回合平均potential能耗_J"],
            "预期": "up potential为负，down potential为正；仅审计，不修改正式定义。",
        },
        "calibration_manifest_SHA256": _sha256(manifest),
        "四批CSV": evidence,
        "checkpoint": str(checkpoint),
        "checkpoint_SHA256": _sha256(checkpoint),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
