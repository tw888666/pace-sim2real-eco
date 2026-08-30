#!/usr/bin/env python3
"""审计 v2.3 Mixed 的1800回合 holdout，并生成阶段7统计和阶段8结论。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

from pace_eco_lab.direction_conditioned_v2_3_mixed_protocol import (
    EVAL_BATCHES,
    EVAL_EPISODES,
    EVAL_NUM_ENVS,
    FORMAL_SEEDS,
    MANIFEST_VERSION,
    METHOD_NAMES,
    PROTOCOL_VERSION,
    SUBTASK_COUNTS_PER_BATCH,
    SUBTASKS,
    TASK_IDS,
    evaluation_batch_seed,
)


PRIMARY_TERRAINS = ("flat", "rough", "boxes", "stairs", "slope")
METHOD_LABELS = {"task_only": "Task-only", "fixed_weight": "Fixed-weight W100", "eco": "ECO"}
FOOT_DUTY_FIELDS = (
    "左前足支撑相占比", "右前足支撑相占比", "左后足支撑相占比", "右后足支撑相占比",
)
TABLE1_FIELDS = (
    "方向穿越成功率", "生存成功率", "B80联合合格率", "平均方向净进度_m",
    "平均稳态前向速度_m_s", "平均路径效率", "平均最大横轨偏离_m",
    "方向成功回合平均能耗_J", "方向成功回合平均单位方向进度能耗_J_m",
    "超B80率", "方向成功回合归一化超预算幅度",
)
TABLE2_FIELDS = (
    "方向成功回合平均世界速度跟踪RMSE_m_s", "方向成功回合平均机身垂向速度RMS_m_s",
    "方向成功回合平均机身横滚俯仰角速度RMS_rad_s",
    "方向成功回合平均动作变化RMS_归一化动作", "方向成功回合平均足端滑移代理_m_s",
    "方向成功回合平均接触占空比", "方向成功回合平均接触占空比极差",
    "方向成功回合平均对角足接触不同步率", "腾空率", "绊倒率", "非法接触率",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="汇总 v2.3 Mixed 阶段6--8。")
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--budget", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stage7_output", required=True)
    parser.add_argument("--stage8_output", required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON不是对象：{path}")
    return value


def _bool(value: object) -> bool:
    return value is True or str(value).lower() in {"true", "1", "是"}


def _number(value: object) -> float | None:
    if value in (None, ""):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else None


def _std(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.stdev(finite) if len(finite) >= 2 else None


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _finite_tree(value: object, location: str = "root") -> list[str]:
    failures: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            failures.extend(_finite_tree(item, f"{location}/{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            failures.extend(_finite_tree(item, f"{location}/{index}"))
    elif isinstance(value, float) and not math.isfinite(value):
        failures.append(location)
    return failures


def _slice_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    success = [row for row in rows if _bool(row["方向穿越成功"])]
    budget = float(rows[0]["PACE主预算B80_J"])
    energies = [float(row["回合能耗_J"]) for row in success]
    output: dict[str, object] = {
        "回合数": len(rows),
        "方向成功回合数": len(success),
        "方向穿越成功率": len(success) / len(rows),
        "生存成功率": sum(_bool(row["生存成功"]) for row in rows) / len(rows),
        "非法终止率": sum(_bool(row["非法终止"]) for row in rows) / len(rows),
        "B80联合合格率": sum(_bool(row["B80联合合格"]) for row in rows) / len(rows),
        "平均方向净进度_m": _mean(float(row["方向净进度_m"]) for row in rows),
        "平均稳态前向速度_m_s": _mean(float(row["稳态平均机身前向速度_m_s"]) for row in rows),
        "平均路径效率": _mean(_number(row["路径效率"]) for row in rows),
        "平均最大横轨偏离_m": _mean(float(row["最大横轨偏离_m"]) for row in rows),
        "方向成功回合平均能耗_J": _mean(energies),
        "方向成功回合平均单位方向进度能耗_J_m": _mean(_number(row["单位方向进度能耗_J_m"]) for row in success),
        "超B80率": sum(float(row["回合能耗_J"]) > budget for row in rows) / len(rows),
        "方向成功回合归一化超预算幅度": _mean(max(energy / budget - 1.0, 0.0) for energy in energies),
        "方向成功回合平均电气能耗_J": _mean(float(row["电气能耗_J"]) for row in success),
        "方向成功回合平均机械能耗_J": _mean(float(row["机械能耗_J"]) for row in success),
        "方向成功回合平均potential能耗_J": _mean(float(row["势能能耗_J"]) for row in success),
        "方向成功回合平均去potential能耗_J": _mean(float(row["电气能耗_J"]) + float(row["机械能耗_J"]) for row in success),
        "方向成功回合平均E除以B_ref_t": _mean(_number(row["v2归一化能耗_E除以B_ref"]) for row in success),
        "方向成功回合最小总能耗_J": min(energies) if energies else None,
        "方向成功回合负总能耗数": sum(energy < 0.0 for energy in energies),
    }
    source_map = {
        "世界速度跟踪RMSE_m_s": "方向成功回合平均世界速度跟踪RMSE_m_s",
        "机身垂向速度RMS_m_s": "方向成功回合平均机身垂向速度RMS_m_s",
        "机身横滚俯仰角速度RMS_rad_s": "方向成功回合平均机身横滚俯仰角速度RMS_rad_s",
        "动作变化RMS_归一化动作": "方向成功回合平均动作变化RMS_归一化动作",
        "三步窗触地足速均值_m_s": "方向成功回合平均足端滑移代理_m_s",
        "支撑相占比极差": "方向成功回合平均接触占空比极差",
        "对角足接触不同步率": "方向成功回合平均对角足接触不同步率",
    }
    for source, target in source_map.items():
        output[target] = _mean(_number(row[source]) for row in success)
    output["方向成功回合平均接触占空比"] = _mean(
        statistics.fmean(float(row[field]) for field in FOOT_DUTY_FIELDS) for row in success
    )
    output.update({"腾空率": None, "绊倒率": None, "非法接触率": None})
    return output


def _aggregate_seed_rows(rows: list[dict[str, object]], dimensions: tuple[str, ...]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[name] for name in dimensions)].append(row)
    outputs: list[dict[str, object]] = []
    excluded = set(dimensions) | {"PPO_seed", "回合数", "方向成功回合数"}
    metric_fields = [field for field in rows[0] if field not in excluded]
    for key, items in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        items.sort(key=lambda item: int(item["PPO_seed"]))
        if [int(item["PPO_seed"]) for item in items] != list(FORMAL_SEEDS):
            raise ValueError(f"{key}缺少seed1--3。")
        output = {name: value for name, value in zip(dimensions, key, strict=True)}
        output.update({"seed数": len(items), "总回合数": sum(int(item["回合数"]) for item in items)})
        for field in metric_fields:
            values = [_number(item.get(field)) for item in items]
            output[field] = _mean(values)
            output[f"{field}_seed间样本标准差"] = _std(values)
            output[f"{field}_可计算seed数"] = sum(value is not None for value in values)
            for item, value in zip(items, values, strict=True):
                output[f"{field}_seed{item['PPO_seed']}"] = value
        outputs.append(output)
    return outputs


def _quality_check(
    result_root: Path,
    authorization_path: Path,
    budget_path: Path,
    manifest_path: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    authorization = _load_json(authorization_path)
    budget = _load_json(budget_path)
    manifest = _load_json(manifest_path)
    authorization_sha = _sha256(authorization_path)
    budget_sha = _sha256(budget_path)
    manifest_sha = _sha256(manifest_path)
    if authorization.get("冻结状态") != "holdout已授权" or authorization.get("模型数量") != 9:
        raise ValueError("阶段5授权状态或模型数错误。")
    if authorization.get("预算冻结SHA256") != budget_sha or authorization.get("holdout_manifest_SHA256") != manifest_sha:
        raise ValueError("阶段5授权未绑定当前预算或manifest。")
    if budget.get("冻结状态") != "已冻结" or budget.get("协议版本") != PROTOCOL_VERSION:
        raise ValueError("预算冻结状态错误。")
    if manifest.get("冻结状态") != "已冻结" or manifest.get("manifest版本") != MANIFEST_VERSION:
        raise ValueError("holdout manifest状态错误。")
    manifest_rows = {str(row["episode_id"]): row for row in manifest["逐回合"]}
    if len(manifest_rows) != EVAL_EPISODES:
        raise ValueError("holdout manifest不是200个唯一episode。")

    authorized = {(str(item["方法"]), int(item["PPO_seed"])): item for item in authorization["模型"]}
    expected_keys = {(method, seed) for method in METHOD_NAMES for seed in FORMAL_SEEDS}
    if set(authorized) != expected_keys:
        raise ValueError("阶段5授权矩阵不是三方法×三seed。")
    summary_paths: dict[tuple[str, int], Path] = {}
    for path in result_root.glob("gpt_方向评估_*/gpt-方向条件评估摘要.json"):
        summary = _load_json(path)
        key = (str(summary["方法"]), int(summary["PPO_seed"]))
        if key in summary_paths:
            raise ValueError(f"重复正式结果：{key}")
        summary_paths[key] = path
    failures: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    all_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    for method, seed in sorted(expected_keys):
        summary_path = summary_paths.get((method, seed))
        if summary_path is None:
            failures.append({"方法": method, "PPO_seed": seed, "失败项": "缺少正式摘要"})
            continue
        summary = _load_json(summary_path)
        result_json = summary_path.parent / "gpt-方向条件逐回合结果.json"
        result_csv = summary_path.parent / "gpt-方向条件逐回合结果.csv"
        payload = _load_json(result_json)
        rows = payload.get("逐回合", [])
        if not isinstance(rows, list):
            rows = []
        with result_csv.open(encoding="utf-8-sig", newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        staging = summary_path.parent.parent / summary_path.parent.name.replace("gpt_方向评估_", "gpt_方向批次暂存_")
        ids = [str(row.get("episode_id")) for row in rows]
        batch_counts = Counter(int(row["评估批次"]) for row in rows)
        state_counts = Counter(int(row["固定初始状态编号"]) for row in rows)
        subtask_counts = Counter(str(row["方向子任务"]) for row in rows)
        identity_relative = []
        reference_errors = []
        for row in rows:
            total = float(row["回合能耗_J"])
            components = float(row["电气能耗_J"]) + float(row["机械能耗_J"]) + float(row["势能能耗_J"])
            identity_relative.append(abs(total - components) / max(abs(total), 1.0))
            reference = float(budget["五类主地形参考能耗_J"][str(row["地形类别"])])
            reference_errors.append(abs(float(row["v2归一化能耗_E除以B_ref"]) - total / reference))
        checks = {
            "恰好4份批次JSON": len(list(staging.glob("gpt-方向评估批次-*.json"))) == EVAL_BATCHES,
            "恰好4份批次CSV": len(list(staging.glob("gpt-v2.3-Mixed-holdout-batch-*.csv"))) == EVAL_BATCHES,
            "JSON恰好200回合": len(rows) == EVAL_EPISODES,
            "CSV恰好200回合": len(csv_rows) == EVAL_EPISODES,
            "episode_id与manifest完全一致": len(ids) == len(set(ids)) == EVAL_EPISODES and set(ids) == set(manifest_rows),
            "每批恰好50回合": batch_counts == Counter({index: EVAL_NUM_ENVS for index in range(EVAL_BATCHES)}),
            "8初始状态各25回合": state_counts == Counter({index: 25 for index in range(8)}),
            "七方向子任务数量正确": subtask_counts == Counter({name: count * EVAL_BATCHES for name, count in SUBTASK_COUNTS_PER_BATCH.items()}),
            "任务方法seed一致": all(row.get("任务ID") == TASK_IDS[method] and row.get("方法") == method and int(row.get("PPO_seed", -1)) == seed for row in rows),
            "manifest逐字段匹配": all(
                str(row.get("episode_id")) in manifest_rows
                and int(row["评估批次"]) == int(manifest_rows[str(row["episode_id"])]["batch_index"])
                and int(row["环境编号"]) == int(manifest_rows[str(row["episode_id"])]["batch_env_number"])
                and row["方向子任务"] == manifest_rows[str(row["episode_id"])]["subtask"]
                and int(row["地形批次seed"]) == int(manifest_rows[str(row["episode_id"])]["batch_seed"])
                for row in rows
            ),
            "批次seed正确": {int(row["评估批次"]): int(row["地形批次seed"]) for row in rows} == {index: evaluation_batch_seed("holdout", index) for index in range(EVAL_BATCHES)},
            "checkpoint哈希一致": summary.get("检查点SHA256") == authorized[(method, seed)]["检查点SHA256"],
            "授权哈希一致": summary.get("holdout授权SHA256") == authorization_sha,
            "manifest哈希一致": summary.get("v2.3_manifest_SHA256") == manifest_sha,
            "预算哈希一致": summary.get("v2_B_ref文件SHA256") == budget_sha,
            "无接触足边界越界": not any(_bool(row.get("接触足越过真实地形边缘")) for row in rows),
            "JSON无非有限数值": not _finite_tree(payload),
            "能耗分量恒等式": max(identity_relative, default=math.inf) <= 5.0e-6,
            "E除以B_ref_t一致": max(reference_errors, default=math.inf) <= 1.0e-9,
        }
        for name, passed in checks.items():
            if not passed:
                failures.append({"方法": method, "PPO_seed": seed, "失败项": name})
        audits.append({"方法": method, "PPO_seed": seed, "状态": "通过" if all(checks.values()) else "失败", "结果目录": str(summary_path.parent), **checks})
        if all(checks.values()):
            all_rows.extend(rows)
            model_rows.append({"范围": "mixed", "方法": method, "PPO_seed": seed, **_slice_metrics(rows)})
    if set(summary_paths) != expected_keys:
        for method, seed in sorted(set(summary_paths) - expected_keys):
            failures.append({"方法": method, "PPO_seed": seed, "失败项": "授权外正式结果"})
    quality = {
        "质检状态": "通过" if not failures and len(all_rows) == 1800 else "失败",
        "授权SHA256": authorization_sha,
        "预算SHA256": budget_sha,
        "holdout_manifest_SHA256": manifest_sha,
        "授权模型数": len(authorized),
        "合格模型数": len(model_rows),
        "合格批次数": len(model_rows) * EVAL_BATCHES,
        "合格回合数": len(all_rows),
        "失败项": failures,
        "逐模型审计": audits,
    }
    return all_rows, model_rows, quality


def _dimension_model_rows(all_rows: list[dict[str, object]], dimension: str) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in all_rows:
        level = str(row[dimension])
        groups[(str(row["方法"]), int(row["PPO_seed"]), level)].append(row)
    return [{"范围": level, "方法": method, "PPO_seed": seed, **_slice_metrics(rows)} for (method, seed, level), rows in sorted(groups.items())]


def _sign_flip_pvalue(values: list[float]) -> float | None:
    if not values:
        return None
    observed = abs(statistics.fmean(values))
    permutations = [abs(statistics.fmean(sign * value for sign, value in zip(signs, values, strict=True))) for signs in itertools.product((-1.0, 1.0), repeat=len(values))]
    return sum(value >= observed - 1.0e-15 for value in permutations) / len(permutations)


def _paired_statistics(model_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    lookup = {(str(row["方法"]), int(row["PPO_seed"])): row for row in model_rows}
    metrics = ("方向穿越成功率", "B80联合合格率", "方向成功回合平均能耗_J", "方向成功回合归一化超预算幅度", "生存成功率", "平均方向净进度_m")
    outputs: list[dict[str, object]] = []
    for left, right in (("eco", "fixed_weight"), ("eco", "task_only"), ("fixed_weight", "task_only")):
        for metric in metrics:
            differences = [float(lookup[(left, seed)][metric]) - float(lookup[(right, seed)][metric]) for seed in FORMAL_SEEDS]
            outputs.append({
                "比较": f"{left}_vs_{right}", "指标": metric, "训练重复数": len(differences),
                "平均配对差": statistics.fmean(differences), "配对差样本标准差": statistics.stdev(differences),
                "双侧精确符号翻转P值": _sign_flip_pvalue(differences),
                **{f"seed{seed}配对差": value for seed, value in zip(FORMAL_SEEDS, differences, strict=True)},
            })
    return outputs


def _conclusion(
    overall: list[dict[str, object]],
    terrain: list[dict[str, object]],
    subtask: list[dict[str, object]],
) -> dict[str, object]:
    lookup = {str(row["方法"]): row for row in overall}
    eco, fixed = lookup["eco"], lookup["fixed_weight"]
    terrain_lookup = {(str(row["范围"]), str(row["方法"])): row for row in terrain}
    subtask_lookup = {(str(row["范围"]), str(row["方法"])): row for row in subtask}
    declines = []
    catastrophic = []
    for level in (*PRIMARY_TERRAINS, *SUBTASKS):
        source = terrain_lookup if level in PRIMARY_TERRAINS else subtask_lookup
        eco_value = float(source[(level, "eco")]["方向穿越成功率"])
        fixed_value = float(source[(level, "fixed_weight")]["方向穿越成功率"])
        declines.append({"范围": level, "ECO": eco_value, "Fixed_weight": fixed_value, "ECO减Fixed": eco_value - fixed_value})
        if fixed_value >= 0.75 and eco_value < 0.50:
            catastrophic.append(level)
    worst = min(declines, key=lambda item: float(item["ECO减Fixed"]))
    criteria = {
        "Mixed总体方向成功率下降不超过5个百分点": float(eco["方向穿越成功率"]) - float(fixed["方向穿越成功率"]) >= -0.05,
        "B80联合合格率提高": float(eco["B80联合合格率"]) > float(fixed["B80联合合格率"]),
        "成功回合能耗或超预算幅度下降": float(eco["方向成功回合平均能耗_J"]) < float(fixed["方向成功回合平均能耗_J"]) or float(eco["方向成功回合归一化超预算幅度"]) < float(fixed["方向成功回合归一化超预算幅度"]),
        "改善不来自提前终止": float(eco["生存成功率"]) >= float(fixed["生存成功率"]) - 0.01 and float(eco["非法终止率"]) <= float(fixed["非法终止率"]) + 0.01,
        "改善不来自方向净进展下降": float(eco["平均方向净进度_m"]) >= float(fixed["平均方向净进度_m"]),
        "无单一主地形或方向明显崩塌": float(worst["ECO减Fixed"]) >= -0.10 and not catastrophic,
    }
    return {
        "判定对象": "ECO相对Fixed-weight W100",
        "重复单位": "PPO seed（n=3）",
        "明显崩塌操作定义": "任一五类主地形或七方向子任务成功率下降超过10个百分点，或Fixed-weight>=75%而ECO<50%。",
        "提前终止操作定义": "ECO生存成功率最多下降1个百分点且非法终止率最多上升1个百分点。",
        "逐范围方向成功率差": declines,
        "最差范围差": worst,
        "灾难性崩塌范围": catastrophic,
        "六项冻结标准": criteria,
        "总体判断": "通过" if all(criteria.values()) else "未通过",
    }


def _fmt(value: object, digits: int = 4) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def _markdown_tables(stage7: Path, table1: list[dict[str, object]], table2: list[dict[str, object]]) -> None:
    lines = ["# 表1：任务与能效", "", "数值为先在每个模型内统计，再对PPO seed 1–3求均值 ± seed间样本标准差。", "", "| 范围 | 方法 | 方向成功率 | 生存成功率 | B80联合合格率 | 净进度/m | 稳态速度/(m/s) | 路径效率 | 最大横偏/m | 成功能耗/J | 单位进度能耗/(J/m) | 超B80率 | 超预算幅度 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in table1:
        values = [f"{_fmt(row[field])} ± {_fmt(row[field + '_seed间样本标准差'])}" for field in TABLE1_FIELDS]
        lines.append(f"| {row['范围']} | {METHOD_LABELS[str(row['方法'])]} | " + " | ".join(values) + " |")
    (stage7 / "gpt-表1任务与能效.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lines = ["# 表2：步态与稳定性", "", "仅在方向成功回合内统计；数值为PPO seed均值 ± seed间样本标准差。NA表示本次冻结采集没有独立记录该指标。", "", "| 范围 | 方法 | 速度RMSE | 垂向速度RMS | roll/pitch角速度RMS | 动作变化RMS | 足端滑移代理 | 接触占空比 | 占空比极差 | 对角不同步率 | 腾空率 | 绊倒率 | 非法接触率 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in table2:
        values = [f"{_fmt(row[field])} ± {_fmt(row[field + '_seed间样本标准差'])}" if row[field] is not None else "NA" for field in TABLE2_FIELDS]
        lines.append(f"| {row['范围']} | {METHOD_LABELS[str(row['方法'])]} | " + " | ".join(values) + " |")
    lines.extend(["", "采集限制：足端滑移使用三步窗触地足速作为预注册代理；腾空率、绊倒率和独立非法接触率未进入冻结逐回合字段，因此不做事后替代或推断。"])
    (stage7 / "gpt-表2步态与稳定性.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parser().parse_args()
    result_root = Path(args.result_root).expanduser().resolve()
    authorization = Path(args.authorization).expanduser().resolve()
    budget = Path(args.budget).expanduser().resolve()
    manifest = Path(args.manifest).expanduser().resolve()
    stage7 = Path(args.stage7_output).expanduser().resolve()
    stage8 = Path(args.stage8_output).expanduser().resolve()
    if stage7.exists() or stage8.exists():
        raise FileExistsError("拒绝覆盖阶段7或阶段8冻结输出目录。")
    all_rows, overall_models, quality = _quality_check(result_root, authorization, budget, manifest)
    if quality["质检状态"] != "通过":
        raise RuntimeError(f"阶段6质检失败，禁止统计：{quality['失败项'][:5]}")

    terrain_models = _dimension_model_rows(all_rows, "地形类别")
    subtask_models = _dimension_model_rows(all_rows, "方向子任务")
    difficulty_models = _dimension_model_rows(all_rows, "难度")
    overall = _aggregate_seed_rows(overall_models, ("范围", "方法"))
    terrain = _aggregate_seed_rows(terrain_models, ("范围", "方法"))
    subtask = _aggregate_seed_rows(subtask_models, ("范围", "方法"))
    difficulty = _aggregate_seed_rows(difficulty_models, ("范围", "方法"))
    table_source = overall + terrain
    table1 = [{key: row.get(key) for key in ("范围", "方法", *TABLE1_FIELDS, *(field + "_seed间样本标准差" for field in TABLE1_FIELDS))} for row in table_source]
    table2 = [{key: row.get(key) for key in ("范围", "方法", *TABLE2_FIELDS, *(field + "_seed间样本标准差" for field in TABLE2_FIELDS))} for row in table_source]
    paired = _paired_statistics(overall_models)
    conclusion = _conclusion(overall, terrain, subtask)

    potential_rows = []
    for row in subtask:
        if str(row["范围"]) not in SUBTASKS:
            continue
        potential_rows.append({key: row.get(key) for key in (
            "范围", "方法", "方向成功回合平均能耗_J", "方向成功回合平均电气能耗_J",
            "方向成功回合平均机械能耗_J", "方向成功回合平均potential能耗_J",
            "方向成功回合平均去potential能耗_J", "方向成功回合平均E除以B_ref_t",
            "B80联合合格率", "方向成功回合最小总能耗_J", "方向成功回合负总能耗数",
        )})
    potential_sign = {
        "预期": "stairs/slope的up平均potential<0，down平均potential>0。",
        "逐方法检查": [
            {
                "方法": method,
                "stairs_up负": next(float(row["方向成功回合平均potential能耗_J"]) for row in potential_rows if row["方法"] == method and row["范围"] == "stairs-up") < 0.0,
                "stairs_down正": next(float(row["方向成功回合平均potential能耗_J"]) for row in potential_rows if row["方法"] == method and row["范围"] == "stairs-down") > 0.0,
                "slope_up负": next(float(row["方向成功回合平均potential能耗_J"]) for row in potential_rows if row["方法"] == method and row["范围"] == "slope-up") < 0.0,
                "slope_down正": next(float(row["方向成功回合平均potential能耗_J"]) for row in potential_rows if row["方法"] == method and row["范围"] == "slope-down") > 0.0,
            }
            for method in METHOD_NAMES
        ],
    }
    potential_sign["总体状态"] = "通过" if all(all(value for key, value in item.items() if key != "方法") for item in potential_sign["逐方法检查"]) else "异常"

    stage7.mkdir(parents=True)
    (stage7 / "gpt-阶段6完整质检报告.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (stage7 / "gpt-1800回合正式明细.json").write_text(json.dumps({"协议版本": PROTOCOL_VERSION, "逐回合": all_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(stage7 / "gpt-1800回合正式明细.csv", all_rows)
    _write_csv(stage7 / "gpt-9模型Mixed总体摘要.csv", overall_models)
    _write_csv(stage7 / "gpt-五地形模型级摘要.csv", terrain_models)
    _write_csv(stage7 / "gpt-七方向子任务模型级摘要.csv", subtask_models)
    _write_csv(stage7 / "gpt-难度模型级摘要.csv", difficulty_models)
    _write_csv(stage7 / "gpt-表1任务与能效.csv", table1)
    _write_csv(stage7 / "gpt-表2步态与稳定性.csv", table2)
    _write_csv(stage7 / "gpt-七方向与potential专项分析.csv", potential_rows)
    _write_csv(stage7 / "gpt-PPO_seed配对比较.csv", paired)
    (stage7 / "gpt-potential符号审计.json").write_text(json.dumps(potential_sign, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (stage7 / "gpt-阶段7完整统计.json").write_text(json.dumps({
        "统计重复单位": "PPO seed（每方法n=3；1800回合不作为独立训练重复）",
        "Mixed总体": overall,
        "五类主地形": terrain,
        "七方向子任务": subtask,
        "难度": difficulty,
        "PPO_seed配对比较": paired,
        "potential专项": {"符号审计": potential_sign, "逐方向": potential_rows},
        "采集限制": {"腾空率": "未独立采集", "绊倒率": "未独立采集", "非法接触率": "未独立采集；仅有非法终止率和边界越界审计"},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _markdown_tables(stage7, table1, table2)

    stage8.mkdir(parents=True)
    (stage8 / "gpt-v2.3_Mixed最终结论.json").write_text(json.dumps({
        "冻结状态": "v2.3 Mixed结论已冻结",
        "创建时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "协议版本": PROTOCOL_VERSION,
        "阶段6质检状态": quality["质检状态"],
        "阶段7统计目录": str(stage7),
        "阶段7统计主文件SHA256": _sha256(stage7 / "gpt-阶段7完整统计.json"),
        "冻结结论": conclusion,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# v2.3 Mixed最终结论", "", f"总体判断：{conclusion['总体判断']}。", "", "## 六项冻结标准", ""]
    for name, passed in conclusion["六项冻结标准"].items():
        lines.append(f"- {'通过' if passed else '未通过'}：{name}")
    lines.extend(["", "统计重复单位为PPO seed（每方法n=3）；没有把1800回合当作独立训练重复。", "", f"明显崩塌定义：{conclusion['明显崩塌操作定义']}"])
    (stage8 / "gpt-v2.3_Mixed最终结论.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    evidence = [authorization, budget, manifest]
    evidence.extend(sorted(path for path in stage7.iterdir() if path.is_file()))
    evidence.extend(sorted(path for path in stage8.iterdir() if path.is_file()))
    archive = [{"文件": str(path), "字节数": path.stat().st_size, "SHA256": _sha256(path)} for path in evidence]
    (stage8 / "gpt-v2.3_Mixed归档清单.json").write_text(json.dumps({
        "协议版本": PROTOCOL_VERSION,
        "文件数（不含本清单）": len(archive),
        "文件": archive,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"阶段7": str(stage7), "阶段8": str(stage8), "总体判断": conclusion["总体判断"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
