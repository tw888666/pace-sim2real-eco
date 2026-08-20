"""多地形逐回合结果的 seed 级、宏平均和最差地形统计。"""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from pace_eco_lab.evaluation_states import MULTI_TERRAIN_HOLDOUT_STATE_SET
from pace_eco_lab.multi_terrain_protocol import (
    EVAL_EPISODES,
    FOOT_BOUNDARY_AUDIT_VERSION,
    METHOD_NAMES,
    PPO_SEEDS,
    PROTOCOL_VERSION,
    TERRAIN_NAMES,
)


TRUE_VALUES = {"True", "true", "1", "是"}
MEAN_FIELDS = (
    "相对1m_s平均绝对误差_m_s",
    "前进速度RMS_m_s",
    "速度跟踪RMSE_m_s",
    "回合能耗_J",
    "单位前进距离能耗_J_m",
    "单位实际路径能耗_J_m",
    "归一化能耗_E_t除以B_ref_t",
    "电气能耗_J",
    "机械能耗_J",
    "势能能耗_J",
    "机身横向速度RMS_m_s",
    "机身垂向速度RMS_m_s",
    "横滚俯仰角速度RMS_rad_s",
)
FAILURE_REASON_FIELDS = {
    "非法终止": "非法终止回合数",
    "越过长地形安全边界": "越过长地形安全边界回合数",
}


def _bool(value: object) -> bool:
    return str(value) in TRUE_VALUES


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return statistics.fmean(items) if items else None


def load_episode_csv(paths: Iterable[str | Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for value in paths:
        path = Path(value)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            current = list(csv.DictReader(stream))
        if len(current) != EVAL_EPISODES:
            raise ValueError(
                f"每份冻结多地形 CSV 必须为 {EVAL_EPISODES} 回合：{path} 实际 {len(current)}"
            )
        rows.extend(current)
    return rows


def validate_formal_rows(rows: list[dict[str, str]], stage: str) -> None:
    """在最终汇总前强制完整、配对的正式 seed×方法×地形设计。"""

    if stage not in ("stage1", "stage2"):
        raise ValueError(f"未知正式汇总阶段：{stage}")
    seeds = set(PPO_SEEDS[f"{stage}_formal"])
    expected_keys = {
        (method, terrain, seed)
        for method in METHOD_NAMES
        for terrain in TERRAIN_NAMES
        for seed in seeds
    }
    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if (
            row.get("协议版本") != PROTOCOL_VERSION
            or row.get("边界审计版本") != FOOT_BOUNDARY_AUDIT_VERSION
            or row.get("阶段") != stage
            or row.get("数据拆分") != "holdout"
        ):
            raise ValueError("最终汇总只接受当前协议、边界审计和当前阶段的 holdout 数据。")
        if _bool(row.get("接触足越过真实地形边缘", False)):
            raise ValueError("最终汇总拒绝任一接触足越过真实地形边缘的回合。")
        if row.get("评估初始状态集") != MULTI_TERRAIN_HOLDOUT_STATE_SET:
            raise ValueError("最终汇总只接受冻结的多地形 holdout 状态集。")
        key = (row["方法"], row["地形类别"], int(row["PPO_seed"]))
        grouped[key].append(row)
    if set(grouped) != expected_keys:
        missing = sorted(expected_keys - set(grouped))
        extra = sorted(set(grouped) - expected_keys)
        raise ValueError(
            f"正式汇总设计不完整或混入额外任务；缺少={missing[:5]}，额外={extra[:5]}。"
        )
    expected_count = EVAL_EPISODES if stage == "stage1" else EVAL_EPISODES // len(TERRAIN_NAMES)
    wrong_counts = {
        key: len(items)
        for key, items in grouped.items()
        if len(items) != expected_count
    }
    if wrong_counts:
        raise ValueError(f"正式 seed×方法×地形回合数错误：{list(wrong_counts.items())[:5]}")


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    """严格先回合→seed×地形，再做地形等权宏平均。"""

    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        method = row["方法"]
        terrain = row["地形类别"]
        if method not in METHOD_NAMES or terrain not in TERRAIN_NAMES:
            raise ValueError(f"未知方法/地形：{method}/{terrain}")
        grouped[(method, terrain, int(row["PPO_seed"]))].append(row)
    seed_terrain: list[dict[str, object]] = []
    for (method, terrain, seed), items in sorted(grouped.items()):
        success = [row for row in items if _bool(row["成功"])]
        failed = [row for row in items if not _bool(row["成功"])]
        failed_energy = [float(row["回合能耗_J"]) for row in failed]
        summary: dict[str, object] = {
            "方法": method,
            "地形": terrain,
            "PPO_seed": seed,
            "回合数": len(items),
            "成功回合数": len(success),
            "失败回合数": len(failed),
            "成功率": sum(_bool(row["成功"]) for row in items) / len(items),
            "联合合格率": sum(_bool(row["B80联合合格"]) for row in items) / len(items),
            "失败回合平均能耗_J": _mean(failed_energy),
            "失败回合能耗样本标准差_J": (
                statistics.stdev(failed_energy) if len(failed_energy) > 1 else None
            ),
            "失败回合最小能耗_J": min(failed_energy) if failed_energy else None,
            "失败回合最大能耗_J": max(failed_energy) if failed_energy else None,
            "未完整20秒回合数": sum(not _bool(row.get("完整20秒", False)) for row in items),
        }
        for field, output_name in FAILURE_REASON_FIELDS.items():
            summary[output_name] = sum(_bool(row.get(field, False)) for row in items)
        for field in MEAN_FIELDS:
            if field in items[0]:
                summary[field] = _mean(float(row[field]) for row in success)
        seed_terrain.append(summary)
    by_method_terrain: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for item in seed_terrain:
        by_method_terrain[(str(item["方法"]), str(item["地形"]))].append(item)
    terrain_summary: list[dict[str, object]] = []
    for (method, terrain), items in sorted(by_method_terrain.items()):
        success_values = [float(item["成功率"]) for item in items]
        joint_values = [float(item["联合合格率"]) for item in items]
        result: dict[str, object] = {
            "方法": method,
            "地形": terrain,
            "seed数": len(items),
            "总回合数": sum(int(item["回合数"]) for item in items),
            "成功回合数": sum(int(item["成功回合数"]) for item in items),
            "失败回合数": sum(int(item["失败回合数"]) for item in items),
            "成功率": statistics.fmean(success_values),
            "成功率_seed间样本标准差": statistics.stdev(success_values) if len(success_values) > 1 else None,
            "联合合格率": statistics.fmean(joint_values),
            "联合合格率_seed间样本标准差": statistics.stdev(joint_values) if len(joint_values) > 1 else None,
        }
        for output_name in FAILURE_REASON_FIELDS.values():
            result[output_name] = sum(int(item[output_name]) for item in items)
        for field in MEAN_FIELDS:
            values = [float(item[field]) for item in items if item.get(field) is not None]
            result[f"{field}_可计算seed数"] = len(values)
            result[field] = statistics.fmean(values) if len(values) == len(items) else None
            result[f"{field}_seed间样本标准差"] = (
                statistics.stdev(values) if len(values) == len(items) and len(values) > 1 else None
            )
        terrain_summary.append(result)
    macro: dict[str, dict[str, object]] = {}
    for method in METHOD_NAMES:
        terrain_items = [item for item in terrain_summary if item["方法"] == method]
        if not terrain_items:
            continue
        if {item["地形"] for item in terrain_items} != set(TERRAIN_NAMES):
            raise ValueError(f"方法 {method} 未覆盖全部五类地形，禁止宏平均。")
        joint_values = [float(item["联合合格率"]) for item in terrain_items]
        distance_energy_values = [
            float(item["单位前进距离能耗_J_m"])
            for item in terrain_items
            if item.get("单位前进距离能耗_J_m") is not None
        ]
        normalized_energy_values = [
            float(item["归一化能耗_E_t除以B_ref_t"])
            for item in terrain_items
            if item.get("归一化能耗_E_t除以B_ref_t") is not None
        ]
        macro[method] = {
            "地形等权宏平均成功率": _mean(float(item["成功率"]) for item in terrain_items),
            "地形等权宏平均联合合格率": statistics.fmean(joint_values),
            "最差地形联合合格率": min(joint_values),
            "最差地形": terrain_items[joint_values.index(min(joint_values))]["地形"],
            "单位前进距离能耗可计算地形数": len(distance_energy_values),
            "地形等权宏平均单位前进距离能耗_J_m": (
                statistics.fmean(distance_energy_values)
                if len(distance_energy_values) == len(TERRAIN_NAMES)
                else None
            ),
            "归一化能耗可计算地形数": len(normalized_energy_values),
            "地形等权宏平均归一化能耗": (
                statistics.fmean(normalized_energy_values)
                if len(normalized_energy_values) == len(TERRAIN_NAMES)
                else None
            ),
        }
    # 相对任务型节能先在同地形、同 seed 配对，绝不直接比较跨地形焦耳。
    lookup = {(str(item["方法"]), str(item["地形"]), int(item["PPO_seed"])): item for item in seed_terrain}
    savings: list[dict[str, object]] = []
    for item in seed_terrain:
        if item["方法"] == "task_only":
            continue
        baseline = lookup.get(("task_only", str(item["地形"]), int(item["PPO_seed"])))
        if baseline is None:
            raise ValueError("正式配对结果缺少同地形同 seed 的任务型 PPO。")
        base_value = baseline.get("回合能耗_J")
        method_value = item.get("回合能耗_J")
        computable = base_value is not None and method_value is not None and float(base_value) > 0.0
        saving = 1.0 - float(method_value) / float(base_value) if computable else None
        savings.append({
            "方法": item["方法"],
            "地形": item["地形"],
            "PPO_seed": item["PPO_seed"],
            "是否可计算": computable,
            "不可计算原因": None if computable else "任务型或当前方法该 seed 无成功回合",
            "相对任务型PPO节能比例": saving,
        })
    savings_summary: list[dict[str, object]] = []
    savings_groups: dict[tuple[str, str], list[float | None]] = defaultdict(list)
    for item in savings:
        value = item["相对任务型PPO节能比例"]
        savings_groups[(str(item["方法"]), str(item["地形"]))].append(
            float(value) if value is not None else None
        )
    for (method, terrain), values in sorted(savings_groups.items()):
        finite_values = [value for value in values if value is not None]
        savings_summary.append(
            {
                "方法": method,
                "地形": terrain,
                "seed数": len(values),
                "可计算seed数": len(finite_values),
                "相对任务型PPO节能比例": (
                    statistics.fmean(finite_values) if len(finite_values) == len(values) else None
                ),
                "seed间样本标准差": (
                    statistics.stdev(finite_values)
                    if len(finite_values) == len(values) and len(finite_values) > 1
                    else None
                ),
                "最小值": min(finite_values) if len(finite_values) == len(values) else None,
                "最大值": max(finite_values) if len(finite_values) == len(values) else None,
            }
        )

    # 楼梯/斜坡方向和所有地形难度均先在 seed 内汇总，再对 seed 等权。
    slice_groups: dict[tuple[str, str, str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        slice_groups[
            (
                row["方法"],
                row["地形类别"],
                row["方向"],
                row["难度"],
                int(row["PPO_seed"]),
            )
        ].append(row)
    slice_seed: list[dict[str, object]] = []
    for (method, terrain, direction, difficulty, seed), items in sorted(slice_groups.items()):
        successful = [item for item in items if _bool(item["成功"])]
        slice_seed.append(
            {
                "方法": method,
                "地形": terrain,
                "方向": direction,
                "难度": difficulty,
                "PPO_seed": seed,
                "回合数": len(items),
                "成功率": sum(_bool(item["成功"]) for item in items) / len(items),
                "联合合格率": sum(_bool(item["B80联合合格"]) for item in items) / len(items),
                "成功回合平均能耗_J": _mean(float(item["回合能耗_J"]) for item in successful),
                "成功回合平均单位前进距离能耗_J_m": _mean(
                    float(item["单位前进距离能耗_J_m"])
                    for item in successful
                    if item.get("单位前进距离能耗_J_m") not in (None, "")
                ),
            }
        )
    by_slice: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for item in slice_seed:
        by_slice[(str(item["方法"]), str(item["地形"]), str(item["方向"]), str(item["难度"]))].append(item)
    slice_summary: list[dict[str, object]] = []
    for (method, terrain, direction, difficulty), items in sorted(by_slice.items()):
        success_values = [float(item["成功率"]) for item in items]
        joint_values = [float(item["联合合格率"]) for item in items]
        energy_values = [float(item["成功回合平均能耗_J"]) for item in items if item["成功回合平均能耗_J"] is not None]
        slice_summary.append(
            {
                "方法": method,
                "地形": terrain,
                "方向": direction,
                "难度": difficulty,
                "seed数": len(items),
                "成功率": statistics.fmean(success_values),
                "成功率_seed间样本标准差": statistics.stdev(success_values) if len(success_values) > 1 else None,
                "联合合格率": statistics.fmean(joint_values),
                "联合合格率_seed间样本标准差": statistics.stdev(joint_values) if len(joint_values) > 1 else None,
                "能耗可计算seed数": len(energy_values),
                "成功回合平均能耗_J": (
                    statistics.fmean(energy_values) if len(energy_values) == len(items) else None
                ),
            }
        )
    return {
        "逐seed逐地形": seed_terrain,
        "逐地形seed等权": terrain_summary,
        "跨地形宏平均与最差地形": macro,
        "同地形同seed相对任务型节能": savings,
        "逐地形相对任务型节能汇总": savings_summary,
        "逐方向逐难度seed等权": slice_summary,
    }


def sample_standard_deviation(values: Iterable[float]) -> float:
    items = list(values)
    if len(items) < 2 or any(not math.isfinite(value) for value in items):
        raise ValueError("样本标准差至少需要两个有限值。")
    return statistics.stdev(items)


__all__ = [
    "load_episode_csv",
    "sample_standard_deviation",
    "summarize_rows",
    "validate_formal_rows",
]
