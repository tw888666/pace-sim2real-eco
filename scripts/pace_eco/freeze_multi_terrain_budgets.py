#!/usr/bin/env python3
"""从任务型 PPO 的冻结 calibration 逐回合结果计算并冻结各地形 B_ref。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from pace_eco_lab.evaluation_states import MULTI_TERRAIN_CALIBRATION_STATE_SET
from pace_eco_lab.multi_terrain_protocol import (
    EVAL_EPISODES,
    FOOT_BOUNDARY_AUDIT_VERSION,
    PPO_SEEDS,
    PROTOCOL_VERSION,
    TERRAIN_NAMES,
    terrain_seed,
)


parser = argparse.ArgumentParser(description="冻结多地形 B_ref 与 B80。")
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
        raise FileExistsError(f"拒绝覆盖 B_ref 冻结文件：{output}")
    expected_seeds = set(PPO_SEEDS[f"{args.stage}_budget"])
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    evidence: list[dict[str, str]] = []
    for path in sorted(root.rglob("gpt-多地形逐回合结果.csv")):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows or rows[0].get("方法") != "task_only":
            continue
        audit_versions = {row.get("边界审计版本") for row in rows}
        if audit_versions != {FOOT_BOUNDARY_AUDIT_VERSION}:
            raise ValueError(f"calibration CSV 边界审计版本错误：{path}，实际={audit_versions}")
        if any(
            row.get("协议版本") != PROTOCOL_VERSION
            or row.get("阶段") != args.stage
            or row.get("数据拆分") != "calibration"
            for row in rows
        ):
            raise ValueError(f"calibration CSV 的协议、阶段或拆分字段错误：{path}")
        if any(_is_true(row.get("接触足越过真实地形边缘")) for row in rows):
            raise ValueError(f"calibration CSV 包含接触足越界的无效回合：{path}")
        seed = int(rows[0]["PPO_seed"])
        if seed not in expected_seeds:
            continue
        if len(rows) != EVAL_EPISODES or any(
            row["评估初始状态集"] != MULTI_TERRAIN_CALIBRATION_STATE_SET for row in rows
        ):
            raise ValueError(f"任务型 calibration 文件回合数或状态集错误：{path}")
        actual_terrain_seed = int(rows[0]["地形seed"])
        if args.stage == "stage2":
            expected = terrain_seed("stage2_calibration", "mixed")
        else:
            categories = {row["地形类别"] for row in rows}
            if len(categories) != 1:
                raise ValueError(f"阶段一单地形文件混入多个类别：{path}")
            expected = terrain_seed("stage1_calibration", next(iter(categories)))
        if actual_terrain_seed != expected:
            raise ValueError(f"calibration 地形 seed 错误：{path}，期望 {expected}。")
        for row in rows:
            grouped[(row["地形类别"], seed)].append(row)
        evidence.append({"路径": str(path), "SHA256": _sha256(path)})
    expected_keys = {(terrain, seed) for terrain in TERRAIN_NAMES for seed in expected_seeds}
    if set(grouped) != expected_keys:
        missing = sorted(expected_keys - set(grouped))
        extra = sorted(set(grouped) - expected_keys)
        raise RuntimeError(f"B_ref calibration 不完整或重复；缺少={missing[:8]}，多余={extra[:8]}。")
    per_seed: list[dict[str, object]] = []
    references: dict[str, float] = {}
    for terrain in TERRAIN_NAMES:
        seed_means: list[float] = []
        for seed in sorted(expected_seeds):
            rows = grouped[(terrain, seed)]
            expected_count = EVAL_EPISODES if args.stage == "stage1" else EVAL_EPISODES // 5
            if len(rows) != expected_count:
                raise ValueError(f"{terrain}/seed{seed} 应有 {expected_count} 回合，实际 {len(rows)}。")
            successful = [row for row in rows if _is_true(row["成功"])]
            success_rate = len(successful) / len(rows)
            if success_rate < 0.95:
                raise RuntimeError(f"{terrain}/seed{seed} 成功率 {success_rate:.3f} 低于 0.95，协议不得冻结。")
            mean_energy = statistics.fmean(float(row["回合能耗_J"]) for row in successful)
            seed_means.append(mean_energy)
            per_seed.append(
                {
                    "地形": terrain,
                    "PPO_seed": seed,
                    "回合数": len(rows),
                    "成功率": success_rate,
                    "成功回合平均能耗_J": mean_energy,
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
        "定义": (
            "阶段一：每个地形使用任务型 PPO seed0 的 calibration 成功回合平均能耗作为 B_ref,t；"
            "阶段二：五类等权宏平均 B_ref,t 作为单一混合分布 B_ref,mixed。"
        ),
        "资格门槛": "每个地形、每个 seed 成功率均不低于 0.95。",
        "训练seed方差限制": "B_ref,t 仅由单个训练 seed0 标定，不估计训练 seed 间方差；正式 seed1--5 禁止反向修改预算。",
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
