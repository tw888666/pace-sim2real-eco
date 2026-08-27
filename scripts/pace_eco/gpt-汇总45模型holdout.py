#!/usr/bin/env python3
"""质检并汇总方向条件 v2.2 的45模型、9000回合正式 holdout。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from pace_eco_lab.direction_conditioned_protocol import (
    EVAL_BATCHES,
    EVAL_EPISODES,
    EVAL_NUM_ENVS,
    evaluation_batch_seed,
    terrain_seed,
)
from pace_eco_lab.direction_conditioned_v2_2_protocol import (
    AMENDED_METRIC_PROTOCOL_VERSION,
    EVALUATION_PROTOCOL_VERSION,
    METRIC_PROTOCOL_VERSION,
    METHOD_NAMES,
    TERRAIN_NAMES,
)
from pace_eco_lab.evaluation_states import (
    MULTI_TERRAIN_HOLDOUT_STATE_SET,
    evaluation_state_definition_sha256,
)


TABLE1_FIELDS = (
    "方向穿越成功率",
    "B80联合合格率",
    "成功条件B80合格率",
    "方向成功回合平均能耗_J",
    "方向成功回合平均单位方向进度能耗_J_m",
    "归一化超预算幅度",
    "方向成功回合能耗E80与B80比值",
    "方向成功回合平均预算余量率",
)
TABLE2_SOURCE_FIELDS = (
    "稳态平均机身前向速度_m_s",
    "世界速度跟踪RMSE_m_s",
    "机身横向速度RMS_m_s",
    "机身垂向速度RMS_m_s",
    "机身横滚俯仰角速度RMS_rad_s",
    "动作变化RMS_归一化动作",
    "三步窗触地足速均值_m_s",
)
TABLE2_FIELDS = tuple(f"方向成功回合平均{field}" for field in TABLE2_SOURCE_FIELDS)
METHOD_LABELS = {"task_only": "Task-only", "fixed_weight": "Fixed-weight", "eco": "ECO"}
TERRAIN_LABELS = {"flat": "Flat", "rough": "Rough", "stairs": "Stairs", "boxes": "Boxes", "slope": "Slope"}
DIFFICULTY_ORDER = ("低", "较低", "中", "较高", "高")
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRIC_AMENDMENT = ROOT / "gpt-方向条件v2.2_45模型评估指标修订v2.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="完成45模型holdout的阶段6--9质检、统计和绘图。")
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metric_amendment", default=str(DEFAULT_METRIC_AMENDMENT))
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON不是对象：{path}")
    return data


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


def _percentile_type7(values: Iterable[float], probability: float) -> float | None:
    """R/NumPy 默认的 type-7 线性插值经验分位数。"""

    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _mean_sd(item: dict[str, object], field: str, digits: int = 4) -> str:
    return f"{_fmt(item.get(field), digits)} ± {_fmt(item.get(field + '_seed间样本标准差'), digits)}"


def _model_summary(rows: list[dict[str, object]], terrain: str, method: str, seed: int) -> dict[str, object]:
    success = [row for row in rows if _bool(row["方向穿越成功"])]
    budget = float(rows[0]["PACE主预算B80_J"])
    energies = [float(row["回合能耗_J"]) for row in success]
    energy_e80 = _percentile_type7(energies, 0.8)
    mean_energy = _mean(energies)
    summary: dict[str, object] = {
        "地形": terrain,
        "方法": method,
        "PPO_seed": seed,
        "回合数": len(rows),
        "方向成功回合数": len(success),
        "完整20秒率": sum(_bool(row["完整20秒"]) for row in rows) / len(rows),
        "平均方向净进度_m": _mean(float(row["方向净进度_m"]) for row in rows),
        "方向穿越成功率": len(success) / len(rows),
        "B80联合合格率": sum(_bool(row["B80联合合格"]) for row in rows) / len(rows),
        "成功条件B80合格率": (
            sum(energy <= budget for energy in energies) / len(energies) if energies else None
        ),
        "方向成功回合平均能耗_J": mean_energy,
        "方向成功回合平均单位方向进度能耗_J_m": _mean(
            _number(row["单位方向进度能耗_J_m"]) for row in success
        ),
        "归一化超预算幅度": _mean(max(energy / budget - 1.0, 0.0) for energy in energies),
        "方向成功回合能耗E80与B80比值": energy_e80 / budget if energy_e80 is not None else None,
        "方向成功回合平均预算余量率": (
            (budget - mean_energy) / budget if mean_energy is not None else None
        ),
    }
    for source, output in zip(TABLE2_SOURCE_FIELDS, TABLE2_FIELDS, strict=True):
        summary[output] = _mean(_number(row[source]) for row in success)
    return summary


def _aggregate_models(model_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in model_rows:
        groups[(str(row["地形"]), str(row["方法"]))].append(row)
    result: list[dict[str, object]] = []
    all_fields = TABLE1_FIELDS + TABLE2_FIELDS + ("完整20秒率", "平均方向净进度_m")
    for terrain in TERRAIN_NAMES:
        for method in METHOD_NAMES:
            items = sorted(groups[(terrain, method)], key=lambda row: int(row["PPO_seed"]))
            if [int(item["PPO_seed"]) for item in items] != [1, 2, 3]:
                raise ValueError(f"{terrain}/{method}缺少seed1--3模型级摘要。")
            row: dict[str, object] = {
                "地形": terrain,
                "方法": method,
                "seed数": 3,
                "总回合数": 600,
                "总方向成功回合数": sum(int(item["方向成功回合数"]) for item in items),
            }
            for field in all_fields:
                values = [_number(item.get(field)) for item in items]
                row[f"{field}_可计算seed数"] = sum(value is not None for value in values)
                row[field] = _mean(values)
                row[f"{field}_seed间样本标准差"] = _std(values)
                for item, value in zip(items, values, strict=True):
                    row[f"{field}_seed{item['PPO_seed']}"] = value
            result.append(row)
    return result


def _quality_check(result_root: Path, authorization_path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    authorization = _load_json(authorization_path)
    authorization_sha256 = _sha256(authorization_path)
    models = authorization.get("模型", [])
    if not isinstance(models, list) or len(models) != 45:
        raise ValueError("授权模型数不是45。")
    expected = {(str(item["地形"]), str(item["方法"]), int(item["PPO_seed"])): item for item in models}
    summaries: dict[tuple[str, str, int], Path] = {}
    for path in result_root.glob("gpt_方向评估_*/gpt-方向条件评估摘要.json"):
        summary = _load_json(path)
        key = (str(summary["地形"]), str(summary["方法"]), int(summary["PPO_seed"]))
        if key in summaries:
            raise ValueError(f"重复正式结果：{key}")
        summaries[key] = path
    failures: list[dict[str, object]] = []
    model_audits: list[dict[str, object]] = []
    all_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    state_sha256 = evaluation_state_definition_sha256(MULTI_TERRAIN_HOLDOUT_STATE_SET)
    for key, authorized in expected.items():
        terrain, method, seed = key
        summary_path = summaries.get(key)
        checks: dict[str, bool] = {}
        if summary_path is None:
            failures.append({"地形": terrain, "方法": method, "PPO_seed": seed, "失败项": "缺少正式摘要"})
            model_audits.append({"地形": terrain, "方法": method, "PPO_seed": seed, "总体状态": "失败", "缺少正式摘要": False})
            continue
        summary = _load_json(summary_path)
        json_path = summary_path.parent / "gpt-方向条件逐回合结果.json"
        csv_path = summary_path.parent / "gpt-方向条件逐回合结果.csv"
        payload = _load_json(json_path)
        rows = payload.get("逐回合", [])
        if not isinstance(rows, list):
            rows = []
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        ids = [int(row["回合"]) for row in rows]
        state_counts = Counter(int(row["固定初始状态编号"]) for row in rows)
        batch_counts = Counter(int(row["评估批次"]) for row in rows)
        batch_seeds = {int(row["评估批次"]): int(row["地形批次seed"]) for row in rows}
        expected_base_seed = terrain_seed("stage1_holdout", terrain)
        checks.update({
            "恰好4份批次结果": len(list((summary_path.parent.parent / summary_path.parent.name.replace("gpt_方向评估_", "gpt_方向批次暂存_")).glob("gpt-方向评估批次-*.json"))) == EVAL_BATCHES,
            "JSON恰好200回合": len(rows) == EVAL_EPISODES,
            "CSV恰好200回合": len(csv_rows) == EVAL_EPISODES,
            "全局编号0到199且无重复": ids == list(range(EVAL_EPISODES)),
            "每批恰好50回合": batch_counts == Counter({index: EVAL_NUM_ENVS for index in range(EVAL_BATCHES)}),
            "8状态各25回合": state_counts == Counter({index: EVAL_EPISODES // 8 for index in range(8)}),
            "地形批次seed正确": batch_seeds == {index: evaluation_batch_seed(expected_base_seed, index) for index in range(EVAL_BATCHES)},
            "协议与指标版本正确": (
                all(row.get("协议版本") == EVALUATION_PROTOCOL_VERSION for row in rows)
                and summary.get("评估指标版本") == METRIC_PROTOCOL_VERSION
            ),
            "checkpoint哈希一致": summary.get("检查点SHA256") == authorized.get("检查点SHA256"),
            "授权哈希一致": summary.get("holdout授权SHA256") == authorization_sha256,
            "初始状态哈希一致": all(row.get("评估初始状态集SHA256") == state_sha256 for row in rows),
            "B80哈希一致": summary.get("v2_B_ref文件SHA256") == authorization.get("证据文件", {}).get("B_ref_B80_SHA256"),
            "对应地形类别": all(row.get("地形类别") == terrain for row in rows),
            "无接触足真实边界越界": not any(_bool(row.get("接触足越过真实地形边缘")) for row in rows),
            "正式摘要回合数200": int(summary.get("回合数", -1)) == EVAL_EPISODES,
        })
        for name, passed in checks.items():
            if not passed:
                failures.append({"地形": terrain, "方法": method, "PPO_seed": seed, "失败项": name})
        model_audits.append({
            "地形": terrain, "方法": method, "PPO_seed": seed,
            "总体状态": "通过" if all(checks.values()) else "失败",
            "结果目录": str(summary_path.parent),
            **checks,
        })
        if all(checks.values()):
            all_rows.extend(rows)
            model_rows.append(_model_summary(rows, terrain, method, seed))
    if set(summaries) != set(expected):
        for key in sorted(set(summaries) - set(expected)):
            failures.append({"地形": key[0], "方法": key[1], "PPO_seed": key[2], "失败项": "授权外正式结果"})
    report = {
        "质检状态": "通过" if not failures else "失败",
        "授权SHA256": authorization_sha256,
        "授权模型数": len(expected),
        "合格模型数": len(model_rows),
        "合格批次数": len(model_rows) * EVAL_BATCHES,
        "合格回合数": len(all_rows),
        "失败项": failures,
        "逐模型审计": model_audits,
        "规则": "任一检查失败的模型均不进入后续汇总。",
    }
    return all_rows, model_rows, report


def _slice_model_rows(all_rows: list[dict[str, object]], dimension: str) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in all_rows:
        terrain = str(row["地形类别"])
        if terrain not in ("stairs", "slope"):
            continue
        groups[(terrain, str(row["方法"]), int(row["PPO_seed"]), str(row[dimension]))].append(row)
    result: list[dict[str, object]] = []
    for (terrain, method, seed, level), rows in sorted(groups.items()):
        success = [row for row in rows if _bool(row["方向穿越成功"])]
        result.append({
            "地形": terrain, "方法": method, "PPO_seed": seed, dimension: level,
            "回合数": len(rows), "方向成功回合数": len(success),
            "方向穿越成功率": len(success) / len(rows),
            "B80联合合格率": sum(_bool(row["B80联合合格"]) for row in rows) / len(rows),
            "方向成功回合平均能耗_J": _mean(float(row["回合能耗_J"]) for row in success),
        })
    return result


def _aggregate_slices(rows: list[dict[str, object]], dimension: str) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["地形"]), str(row["方法"]), str(row[dimension]))].append(row)
    result: list[dict[str, object]] = []
    for (terrain, method, level), items in sorted(groups.items()):
        output: dict[str, object] = {"地形": terrain, "方法": method, dimension: level, "seed数": len(items)}
        for field in ("方向穿越成功率", "B80联合合格率", "方向成功回合平均能耗_J"):
            values = [_number(item[field]) for item in items]
            output[field] = _mean(values)
            output[f"{field}_seed间样本标准差"] = _std(values)
            output[f"{field}_可计算seed数"] = sum(value is not None for value in values)
        result.append(output)
    return result


def _sign_flip_pvalue(differences: list[float]) -> float | None:
    if not differences:
        return None
    observed = abs(statistics.fmean(differences))
    exceed = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        value = abs(statistics.fmean(sign * difference for sign, difference in zip(signs, differences, strict=True)))
        exceed += value >= observed - 1.0e-15
        total += 1
    return exceed / total


def _paired_statistics(model_rows: list[dict[str, object]], aggregates: list[dict[str, object]]) -> dict[str, object]:
    lookup = {(str(row["地形"]), str(row["方法"]), int(row["PPO_seed"])): row for row in model_rows}
    comparisons = (("eco", "fixed_weight"), ("eco", "task_only"), ("fixed_weight", "task_only"))
    metrics = ("方向穿越成功率", "B80联合合格率", "方向成功回合平均能耗_J", "归一化超预算幅度")
    paired_rows: list[dict[str, object]] = []
    comparison_summary: list[dict[str, object]] = []
    for method_a, method_b in comparisons:
        differences_by_metric: dict[str, list[float]] = {field: [] for field in metrics}
        for terrain in TERRAIN_NAMES:
            for seed in (1, 2, 3):
                a = lookup[(terrain, method_a, seed)]
                b = lookup[(terrain, method_b, seed)]
                row: dict[str, object] = {"比较": f"{method_a}_vs_{method_b}", "地形": terrain, "PPO_seed": seed}
                for field in metrics:
                    av, bv = _number(a[field]), _number(b[field])
                    difference = av - bv if av is not None and bv is not None else None
                    row[f"{method_a}_{field}"] = av
                    row[f"{method_b}_{field}"] = bv
                    row[f"差值_{field}"] = difference
                    if difference is not None:
                        differences_by_metric[field].append(difference)
                paired_rows.append(row)
        for field, differences in differences_by_metric.items():
            comparison_summary.append({
                "比较": f"{method_a}_vs_{method_b}", "指标": field,
                "配对单位数": len(differences), "平均配对差": _mean(differences),
                "配对差样本标准差": _std(differences),
                "正差单位数": sum(value > 0 for value in differences),
                "零差单位数": sum(value == 0 for value in differences),
                "负差单位数": sum(value < 0 for value in differences),
                "双侧精确符号翻转P值": _sign_flip_pvalue(differences),
            })

    aggregate_lookup = {(str(row["地形"]), str(row["方法"])): row for row in aggregates}
    macro: list[dict[str, object]] = []
    for method in METHOD_NAMES:
        items = [aggregate_lookup[(terrain, method)] for terrain in TERRAIN_NAMES]
        success = [float(item["方向穿越成功率"]) for item in items]
        joint = [float(item["B80联合合格率"]) for item in items]
        macro.append({
            "方法": method,
            "五地形等权宏平均方向成功率": statistics.fmean(success),
            "五地形等权宏平均B80联合合格率": statistics.fmean(joint),
            "最差地形方向成功率": min(success),
            "最差方向成功率地形": TERRAIN_NAMES[success.index(min(success))],
            "最差地形B80联合合格率": min(joint),
            "最差B80联合合格率地形": TERRAIN_NAMES[joint.index(min(joint))],
            "五地形等权宏平均成功回合能耗_J": _mean(item["方向成功回合平均能耗_J"] for item in items),
            "五地形等权宏平均归一化超预算幅度": _mean(item["归一化超预算幅度"] for item in items),
            "五地形等权宏平均完整20秒率": _mean(item["完整20秒率"] for item in items),
            "五地形等权宏平均方向净进度_m": _mean(item["平均方向净进度_m"] for item in items),
        })
    macro_lookup = {str(row["方法"]): row for row in macro}
    eco, fixed = macro_lookup["eco"], macro_lookup["fixed_weight"]
    criteria = {
        "方向成功率下降不超过5个百分点": float(eco["五地形等权宏平均方向成功率"]) - float(fixed["五地形等权宏平均方向成功率"]) >= -0.05,
        "B80联合合格率提高": float(eco["五地形等权宏平均B80联合合格率"]) > float(fixed["五地形等权宏平均B80联合合格率"]),
        "成功回合能耗或超预算幅度下降": (
            float(eco["五地形等权宏平均成功回合能耗_J"]) < float(fixed["五地形等权宏平均成功回合能耗_J"])
            or float(eco["五地形等权宏平均归一化超预算幅度"]) < float(fixed["五地形等权宏平均归一化超预算幅度"])
        ),
        "改善不来自提前终止": float(eco["五地形等权宏平均完整20秒率"]) >= float(fixed["五地形等权宏平均完整20秒率"]),
        "改善不来自方向进度降低": float(eco["五地形等权宏平均方向净进度_m"]) >= float(fixed["五地形等权宏平均方向净进度_m"]),
    }
    return {
        "配对单位": "地形+PPO_seed",
        "比较顺序": ["eco_vs_fixed_weight", "eco_vs_task_only", "fixed_weight_vs_task_only"],
        "逐配对原始值与差值": paired_rows,
        "配对比较汇总": comparison_summary,
        "五地形等权宏平均与最差地形": macro,
        "ECO相对Fixed-weight冻结标准": criteria,
        "ECO相对Fixed-weight总体判断": all(criteria.values()),
        "解释": "精确符号翻转检验以15个地形+PPO seed配对差为输入；回合不作为训练重复数。",
    }


def _plot_outputs(output: Path, aggregates: list[dict[str, object]], model_rows: list[dict[str, object]], direction: list[dict[str, object]], difficulty: list[dict[str, object]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    cjk_font = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(cjk_font)
    matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=cjk_font).get_name()
    matplotlib.rcParams["axes.unicode_minus"] = False
    colors = {"task_only": "#4C78A8", "fixed_weight": "#F58518", "eco": "#54A24B"}
    markers = {"flat": "o", "rough": "s", "stairs": "^", "boxes": "D", "slope": "v"}

    fig, ax = plt.subplots(figsize=(10, 7))
    points: list[tuple[float, float, str, str]] = []
    for row in aggregates:
        x = float(row["方向成功回合平均能耗_J"])
        y = float(row["方向穿越成功率"])
        method, terrain = str(row["方法"]), str(row["地形"])
        ax.errorbar(x, y, xerr=row["方向成功回合平均能耗_J_seed间样本标准差"], yerr=row["方向穿越成功率_seed间样本标准差"], fmt=markers[terrain], color=colors[method], capsize=3, alpha=0.85)
        points.append((x, y, method, terrain))
    frontier = sorted((point for point in points if not any(other[0] <= point[0] and other[1] >= point[1] and (other[0] < point[0] or other[1] > point[1]) for other in points)), key=lambda item: item[0])
    if frontier:
        ax.plot([p[0] for p in frontier], [p[1] for p in frontier], "k--", lw=1.2, label="Pareto前沿")
        for x, y, method, terrain in frontier:
            ax.annotate(f"{TERRAIN_LABELS[terrain]}-{METHOD_LABELS[method]}", (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("方向成功回合平均能耗 (J)")
    ax.set_ylabel("方向穿越成功率")
    ax.set_title("成功率—能耗 Pareto 图（点为seed均值，误差棒为样本标准差）")
    ax.grid(alpha=0.25)
    legend_items = [Line2D([0], [0], color=colors[method], marker="o", linestyle="none", label=METHOD_LABELS[method]) for method in METHOD_NAMES]
    legend_items += [Line2D([0], [0], color="black", marker=markers[terrain], linestyle="none", label=TERRAIN_LABELS[terrain]) for terrain in TERRAIN_NAMES]
    legend_items.append(Line2D([0], [0], color="black", linestyle="--", label="Pareto前沿"))
    ax.legend(handles=legend_items, ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "gpt-成功率-能耗Pareto图.png", dpi=220)
    plt.close(fig)

    lookup = {(str(row["地形"]), str(row["方法"]), int(row["PPO_seed"])): row for row in model_rows}
    fig, axes = plt.subplots(2, 5, figsize=(18, 8), sharex=True)
    x_methods = ("task_only", "fixed_weight", "eco")
    for column, terrain in enumerate(TERRAIN_NAMES):
        for row_index, field in enumerate(("方向穿越成功率", "B80联合合格率")):
            ax = axes[row_index, column]
            for seed in (1, 2, 3):
                values = [float(lookup[(terrain, method, seed)][field]) for method in x_methods]
                ax.plot(range(3), values, marker="o", lw=1.2, alpha=0.8, label=f"seed{seed}")
            ax.set_xticks(range(3), ["Task", "Fixed", "ECO"])
            ax.set_ylim(-0.03, 1.03)
            ax.grid(alpha=0.2)
            if row_index == 0:
                ax.set_title(TERRAIN_LABELS[terrain])
            if column == 0:
                ax.set_ylabel(field)
    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.suptitle("同地形同PPO seed配对图")
    fig.tight_layout()
    fig.savefig(output / "gpt-seed配对图.png", dpi=220)
    plt.close(fig)

    direction_lookup = {(str(row["地形"]), str(row["方法"]), str(row["方向"])): row for row in direction}
    difficulty_lookup = {(str(row["地形"]), str(row["方法"]), str(row["难度"])): row for row in difficulty}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for terrain_index, terrain in enumerate(("stairs", "slope")):
        ax = axes[terrain_index, 0]
        width = 0.12
        for method_index, method in enumerate(METHOD_NAMES):
            for direction_index, direction_name in enumerate(("up", "down")):
                center = direction_index + (method_index - 1) * 2.2 * width
                item = direction_lookup[(terrain, method, direction_name)]
                ax.bar(center - width / 2, float(item["方向穿越成功率"]), width, color=colors[method], alpha=0.9, label=METHOD_LABELS[method] if direction_index == 0 else None)
                ax.bar(center + width / 2, float(item["B80联合合格率"]), width, color=colors[method], alpha=0.45, hatch="//")
        ax.set_xticks((0, 1), ("上行", "下行"))
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"{TERRAIN_LABELS[terrain]}：方向分层\n实色=方向成功率，斜线=B80联合合格率")
        ax.grid(axis="y", alpha=0.2)
        ax = axes[terrain_index, 1]
        for method in METHOD_NAMES:
            success_values = [float(difficulty_lookup[(terrain, method, level)]["方向穿越成功率"]) for level in DIFFICULTY_ORDER]
            joint_values = [float(difficulty_lookup[(terrain, method, level)]["B80联合合格率"]) for level in DIFFICULTY_ORDER]
            ax.plot(DIFFICULTY_ORDER, success_values, marker="o", color=colors[method], label=f"{METHOD_LABELS[method]} 成功")
            ax.plot(DIFFICULTY_ORDER, joint_values, marker="x", ls="--", color=colors[method], alpha=0.8, label=f"{METHOD_LABELS[method]} 联合")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"{TERRAIN_LABELS[terrain]}：五难度层")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Stairs/Slope 方向与五难度层结果")
    fig.tight_layout()
    fig.savefig(output / "gpt-stairs-slope方向与难度分层图.png", dpi=220)
    plt.close(fig)


def _write_markdown(output: Path, aggregates: list[dict[str, object]], statistics_result: dict[str, object], quality: dict[str, object]) -> None:
    table1 = ["# 表1：任务性能与能效", "", "表中为先在每个模型内计算、再对PPO seed1–3求均值 ± 样本标准差。", "", "| 地形 | 方法 | 方向成功率 | B80联合合格率 | 成功条件B80合格率 | 成功能耗/J | 单位方向进度能耗/J·m⁻¹ | 归一化超预算幅度 | E₈₀/B80 | (B80−E)/B80 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in aggregates:
        table1.append(f"| {row['地形']} | {row['方法']} | {_mean_sd(row, '方向穿越成功率')} | {_mean_sd(row, 'B80联合合格率')} | {_mean_sd(row, '成功条件B80合格率')} | {_mean_sd(row, '方向成功回合平均能耗_J', 1)} | {_mean_sd(row, '方向成功回合平均单位方向进度能耗_J_m', 1)} | {_mean_sd(row, '归一化超预算幅度')} | {_mean_sd(row, '方向成功回合能耗E80与B80比值')} | {_mean_sd(row, '方向成功回合平均预算余量率')} |")
    table1.extend(["", "## 指标解读", "", "| 指标 | 方向 | 解读 |", "|---|:---:|---|", "| 方向穿越成功率 | ↑ | 越大越好；主要任务指标。 |", "| B80联合合格率 | ↑ | 方向成功且能耗不超B80的全部回合占比；越大越好，主要联合指标。 |", "| 成功条件B80合格率 | ↑ | 仅在方向成功回合内看守预算比例；越大越好。 |", "| 方向成功回合平均能耗 | ↓ | 完成任务的平均焦耳数；在成功率和进度相当时越小越好。 |", "| 方向成功回合平均单位方向进度能耗 | ↓ | 每获1 m有效方向进度的能耗；越小越好。 |", "| 归一化超预算幅度 | ↓ | 只计超出B80的部分；0最好，越大越差。 |", "| E₈₀/B80 | ↓ | E₈₀为方向成功回合能耗的经验第80百分位；≤1表示该尾部阈值不超预算，越小越好。 |", "| (B80−E)/B80 | ↑ | E为方向成功回合平均能耗；正值表示有预算余量，0表示恰好用完，负值表示平均超预算，越大越好。 |"])
    (output / "gpt-表1任务与能效主表.md").write_text("\n".join(table1) + "\n", encoding="utf-8")

    table2 = ["# 表2：步态与运动稳定性", "", "所有值均先在单模型方向成功回合内求平均，再对PPO seed1–3计算均值与样本标准差。", "", "| 地形 | 方法 | 稳态机身前向速度 | 世界速度RMSE | 横向速度RMS | 垂向速度RMS | 横滚俯仰角速度RMS | 动作变化RMS | 触地足速 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in aggregates:
        cells = [_mean_sd(row, field) for field in TABLE2_FIELDS]
        table2.append(f"| {row['地形']} | {row['方法']} | " + " | ".join(cells) + " |")
    table2.extend(["", "## 指标解读", "", "| 指标 | 方向 | 解读 |", "|---|:---:|---|", "| 稳态机身前向速度 | → 1.0 m/s | 去掉前5 s预热后的机身坐标系x轴线速度均值；越接近目标1.0 m/s越好，不是单纯越大越好。 |", "| 世界速度跟踪RMSE | ↓ | 世界坐标系速度相对(+1,0) m/s指令的均方根误差；0最好。 |", "| 机身横向速度RMS | ↓ | 机身侧向摇晃强度；越小越稳。 |", "| 机身垂向速度RMS | ↓ | 机身上下颠簸强度；越小越稳。 |", "| 机身横滚俯仰角速度RMS | ↓ | 横滚和俯仰旋转波动；越小越稳。 |", "| 动作变化RMS | ↓ | 相邻控制动作的变化强度；越小通常越平滑。 |", "| 三步窗触地足速均值 | ↓ | 落足附近足端速度；越小通常表示冲击和滑移风险越低。 |"])
    (output / "gpt-表2步态与稳定性主表.md").write_text("\n".join(table2) + "\n", encoding="utf-8")

    criteria = statistics_result["ECO相对Fixed-weight冻结标准"]
    report = ["# 45模型holdout质检与统计报告", "", f"- 质检状态：{quality['质检状态']}", f"- 合格模型：{quality['合格模型数']}/45", f"- 合格批次：{quality['合格批次数']}/180", f"- 合格回合：{quality['合格回合数']}/9000", "", "## ECO相对Fixed-weight冻结判断", ""]
    for name, passed in criteria.items():
        report.append(f"- {'通过' if passed else '未通过'}：{name}")
    report.extend(["", f"总体判断：{'通过' if statistics_result['ECO相对Fixed-weight总体判断'] else '未通过'}。", "", "统计重复单位为地形+PPO seed；未把回合数当作训练重复数。"])
    (output / "gpt-完整质检与统计报告.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    args = _parser().parse_args()
    result_root = Path(args.result_root).expanduser().resolve()
    authorization = Path(args.authorization).expanduser().resolve()
    metric_amendment_path = Path(args.metric_amendment).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖阶段6--9输出目录：{output}")
    metric_amendment = _load_json(metric_amendment_path)
    if metric_amendment.get("报告指标版本") != AMENDED_METRIC_PROTOCOL_VERSION:
        raise ValueError("指标修订文件版本不匹配。")
    if metric_amendment.get("holdout授权SHA256") != _sha256(authorization):
        raise ValueError("指标修订文件绑定的holdout授权SHA256不匹配。")
    source_manifest = ROOT / str(metric_amendment["原冻结配置"])
    if metric_amendment.get("原冻结配置SHA256") != _sha256(source_manifest):
        raise ValueError("指标修订文件绑定的原冻结配置SHA256不匹配。")
    amendment_evidence = {
        "报告指标版本": AMENDED_METRIC_PROTOCOL_VERSION,
        "指标修订文件": str(metric_amendment_path),
        "指标修订文件SHA256": _sha256(metric_amendment_path),
        "原始采集指标版本": METRIC_PROTOCOL_VERSION,
    }
    all_rows, model_rows, quality = _quality_check(result_root, authorization)
    if quality["质检状态"] != "通过":
        raise RuntimeError(f"质检失败，禁止汇总：{quality['失败项'][:5]}")
    aggregates = _aggregate_models(model_rows)
    direction_model = _slice_model_rows(all_rows, "方向")
    difficulty_model = _slice_model_rows(all_rows, "难度")
    direction_summary = _aggregate_slices(direction_model, "方向")
    difficulty_summary = _aggregate_slices(difficulty_model, "难度")
    statistics_result = _paired_statistics(model_rows, aggregates)
    output.mkdir(parents=True)

    quality["指标修订证据"] = amendment_evidence
    (output / "gpt-完整质检报告.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(output / "gpt-45模型逐项质检.csv", quality["逐模型审计"])
    (output / "gpt-9000回合明细.json").write_text(json.dumps({"协议版本": EVALUATION_PROTOCOL_VERSION, **amendment_evidence, "逐回合": all_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(output / "gpt-9000回合明细.csv", all_rows)
    _write_csv(output / "gpt-45行模型级摘要.csv", model_rows)
    _write_csv(output / "gpt-15行地形方法seed汇总.csv", aggregates)
    _write_csv(output / "gpt-表1任务与能效主表.csv", [{key: row.get(key) for key in ("地形", "方法") + TABLE1_FIELDS + tuple(field + "_seed间样本标准差" for field in TABLE1_FIELDS)} for row in aggregates])
    _write_csv(output / "gpt-表2步态与稳定性主表.csv", [{key: row.get(key) for key in ("地形", "方法") + TABLE2_FIELDS + tuple(field + "_seed间样本标准差" for field in TABLE2_FIELDS)} for row in aggregates])
    _write_csv(output / "gpt-stairs-slope方向分层seed汇总.csv", direction_summary)
    _write_csv(output / "gpt-stairs-slope难度分层seed汇总.csv", difficulty_summary)
    _write_csv(output / "gpt-三方法逐配对原始值与差值.csv", statistics_result["逐配对原始值与差值"])
    _write_csv(output / "gpt-三方法配对比较汇总.csv", statistics_result["配对比较汇总"])
    _write_csv(output / "gpt-五地形宏平均与最差地形.csv", statistics_result["五地形等权宏平均与最差地形"])
    (output / "gpt-阶段6到9最终统计.json").write_text(json.dumps({
        "指标修订证据": amendment_evidence,
        "质检": quality,
        "45行模型级摘要": model_rows,
        "15行地形方法seed汇总": aggregates,
        "stairs_slope方向分层": direction_summary,
        "stairs_slope难度分层": difficulty_summary,
        "统计比较": statistics_result,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(output, aggregates, statistics_result, quality)
    _plot_outputs(output, aggregates, model_rows, direction_summary, difficulty_summary)
    deliverables = []
    for path in sorted(output.iterdir()):
        if path.is_file():
            deliverables.append({"文件": path.name, "字节数": path.stat().st_size, "SHA256": _sha256(path)})
    (output / "gpt-阶段6到9交付清单.json").write_text(
        json.dumps({"文件数（不含本清单）": len(deliverables), "文件": deliverables}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
