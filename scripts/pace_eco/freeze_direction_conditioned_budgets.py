#!/usr/bin/env python3
"""从 v2 E2 任务型 seed0 calibration 冻结 B_ref 和 B80。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from pace_eco_lab.direction_conditioned_protocol import (
    EVAL_EPISODES,
    FOOT_BOUNDARY_AUDIT_VERSION,
    PPO_SEEDS,
    PROTOCOL_VERSION,
    TERRAIN_NAMES,
    terrain_seed,
)
from pace_eco_lab.evaluation_states import MULTI_TERRAIN_CALIBRATION_STATE_SET


parser = argparse.ArgumentParser(description="冻结方向条件 v2.1 B_ref 与 B80。")
parser.add_argument("--stage", required=True, choices=("stage1", "stage2"))
parser.add_argument("--calibration_root", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_true(value: object) -> bool:
    return str(value).lower() in ("true", "1", "是")


def main() -> None:
    root = Path(args.calibration_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖 v2 B_ref 冻结文件：{output}")
    expected_seeds = set(PPO_SEEDS[f"{args.stage}_budget"])
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    evidence: list[dict[str, str]] = []
    for path in sorted(root.rglob("gpt-方向条件逐回合结果.csv")):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            continue
        if rows[0].get("实验变体") != "directional" or rows[0].get("方法") != "task_only":
            continue
        if any(
            row.get("协议版本") != PROTOCOL_VERSION
            or row.get("边界审计版本") != FOOT_BOUNDARY_AUDIT_VERSION
            or row.get("阶段") != args.stage
            or row.get("数据拆分") != "calibration"
            or row.get("评估初始状态集") != MULTI_TERRAIN_CALIBRATION_STATE_SET
            for row in rows
        ):
            raise ValueError(f"v2 calibration CSV 元数据错误：{path}")
        if any(_is_true(row.get("接触足越过真实地形边缘")) for row in rows):
            raise ValueError(f"v2 calibration 包含接触足越界回合：{path}")
        seed = int(rows[0]["PPO_seed"])
        if seed not in expected_seeds:
            continue
        if len(rows) != EVAL_EPISODES:
            raise ValueError(f"v2 calibration 应有 {EVAL_EPISODES} 回合：{path}")
        actual_seed = int(rows[0]["地形seed"])
        if args.stage == "stage2":
            expected_terrain_seed = terrain_seed("stage2_calibration", "mixed")
        else:
            categories = {row["地形类别"] for row in rows}
            if len(categories) != 1:
                raise ValueError(f"阶段一 calibration 混入多个地形类别：{path}")
            expected_terrain_seed = terrain_seed("stage1_calibration", next(iter(categories)))
        if actual_seed != expected_terrain_seed:
            raise ValueError(f"v2 calibration 地形 seed 错误：{path}")
        for row in rows:
            grouped[(row["地形类别"], seed)].append(row)
        evidence.append({"路径": str(path.resolve()), "SHA256": _sha256(path)})
    expected_keys = {(terrain, seed) for terrain in TERRAIN_NAMES for seed in expected_seeds}
    if set(grouped) != expected_keys:
        missing = sorted(expected_keys - set(grouped))
        extra = sorted(set(grouped) - expected_keys)
        raise RuntimeError(f"v2 B_ref calibration 不完整或重复；缺少={missing[:8]}，多余={extra[:8]}。")
    per_seed: list[dict[str, object]] = []
    references: dict[str, float] = {}
    for terrain in TERRAIN_NAMES:
        seed_means: list[float] = []
        for seed in sorted(expected_seeds):
            rows = grouped[(terrain, seed)]
            expected_count = EVAL_EPISODES if args.stage == "stage1" else EVAL_EPISODES // len(TERRAIN_NAMES)
            if len(rows) != expected_count:
                raise ValueError(f"{terrain}/seed{seed} 回合数应为 {expected_count}，实际 {len(rows)}。")
            successful = [row for row in rows if _is_true(row["方向穿越成功"])]
            success_rate = len(successful) / len(rows)
            if success_rate < 0.95:
                raise RuntimeError(
                    f"{terrain}/seed{seed} 方向穿越成功率 {success_rate:.3f} 低于 0.95，禁止冻结预算。"
                )
            mean_energy = statistics.fmean(float(row["回合能耗_J"]) for row in successful)
            seed_means.append(mean_energy)
            per_seed.append(
                {
                    "实验变体": "directional",
                    "方法": "task_only",
                    "地形": terrain,
                    "PPO_seed": seed,
                    "回合数": len(rows),
                    "方向穿越成功率": success_rate,
                    "方向成功回合平均能耗_J": mean_energy,
                }
            )
        references[terrain] = statistics.fmean(seed_means)
    mixed_reference = statistics.fmean(references.values()) if args.stage == "stage2" else None
    payload = {
        "冻结状态": "已冻结",
        "协议版本": PROTOCOL_VERSION,
        "边界审计版本": FOOT_BOUNDARY_AUDIT_VERSION,
        "阶段": args.stage,
        "创建时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "来源实验": "E2 directional/task_only seed0 calibration",
        "定义": (
            "仅对方向穿越成功回合求平均能耗；阶段一逐地形定义 B_ref,t，"
            "阶段二将五类 B_ref,t 等权宏平均为 B_ref,mixed。"
        ),
        "资格门槛": "每个地形方向穿越成功率均不低于0.95。",
        "共享规则": "同一阶段 E1-ECO 与 E2-ECO 使用同一份由 E2 任务型 seed0 导出的 B80。",
        "训练seed方差限制": "只用 seed0 标定；正式 seed1--5 禁止反向修改预算。",
        "B_ref_J": references,
        "B80_J": {terrain: 0.8 * value for terrain, value in references.items()},
        "B_ref_mixed_J": mixed_reference,
        "B80_mixed_J": 0.8 * mixed_reference if mixed_reference is not None else None,
        "逐seed证据": per_seed,
        "输入文件": evidence,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
