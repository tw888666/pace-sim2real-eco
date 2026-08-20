#!/usr/bin/env python3
"""验证并汇总训练种子 1 至 5 的平地控制与正式地形评估。"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SUMMARY_NAME = "gpt-正式平地模型地形评估-汇总.csv"
EPISODE_NAME = "gpt-正式平地模型地形评估-逐回合.csv"
PAIRING_NAME = "gpt-正式平地模型地形评估-配对信息.json"
METHODS = ("task_only", "fixed_weight", "eco")
TERRAINS = ("flat", "slope", "stairs", "box", "rough")
SEEDS = (1, 2, 3, 4, 5)
METHOD_LABELS = {
    "task_only": "任务型",
    "fixed_weight": "固定能耗权重 PPO",
    "eco": "PACE-ECO",
}
TERRAIN_LABELS = {
    "flat": "平地控制",
    "slope": "坡地",
    "stairs": "上楼梯",
    "box": "离散障碍",
    "rough": "粗糙地形",
}
T_CRITICAL_95_DF4 = 2.7764451051977987


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_root",
        action="append",
        dest="input_roots",
        help="包含单次评估目录的根目录；可重复指定。",
    )
    parser.add_argument(
        "--output_dir",
        default="results/terrain_evaluation/formal_flat_seeds1_5_n200",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV 没有数据：{path}")
    return rows


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _mean_sd(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("不能汇总空序列。")
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _mean_sd_ci95(values: Sequence[float]) -> tuple[float, float, float, float]:
    mean, sd = _mean_sd(values)
    if len(values) == 1:
        return mean, sd, mean, mean
    half_width = T_CRITICAL_95_DF4 * sd / math.sqrt(len(values))
    return mean, sd, max(0.0, mean - half_width), min(1.0, mean + half_width)


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("拒绝写入空汇总。")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temp_path = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    temp_path.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temp_path = Path(stream.name)
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temp_path.replace(path)


def _discover_summaries(input_roots: Iterable[Path]) -> list[tuple[Path, dict[str, str]]]:
    discovered: list[tuple[Path, dict[str, str]]] = []
    for root in input_roots:
        if not root.is_dir():
            raise FileNotFoundError(f"评估根目录不存在：{root}")
        for path in sorted(root.glob(f"*/{SUMMARY_NAME}")):
            rows = _read_csv(path)
            if len(rows) != 1:
                raise ValueError(f"单次评估汇总应只有一行：{path}")
            discovered.append((path, rows[0]))
    return discovered


def _validate_and_index(
    discovered: Sequence[tuple[Path, dict[str, str]]],
) -> dict[tuple[int, str, str], dict[str, str]]:
    indexed: dict[tuple[int, str, str], dict[str, str]] = {}
    signatures_by_terrain: dict[str, set[str]] = defaultdict(set)
    for summary_path, row in discovered:
        seed = int(row["training_seed"])
        terrain = row["terrain"]
        method = row["algorithm"]
        key = (seed, terrain, method)
        if seed not in SEEDS or terrain not in TERRAINS or method not in METHODS:
            continue
        if key in indexed:
            raise ValueError(f"重复的正式评估组合：{key}")
        episode_path = summary_path.parent / EPISODE_NAME
        pairing_path = summary_path.parent / PAIRING_NAME
        if not episode_path.is_file() or not pairing_path.is_file():
            raise FileNotFoundError(f"正式评估结果不完整：{summary_path.parent}")
        episodes = _read_csv(episode_path)
        statuses = defaultdict(int)
        for episode in episodes:
            statuses[episode["status"]] += 1
        expected_counts = {
            "success": int(row["num_success"]),
            "fall": int(row["num_fall"]),
            "timeout": int(row["num_timeout"]),
        }
        if len(episodes) != 200 or int(row["num_episodes"]) != 200:
            raise ValueError(f"正式评估必须恰好包含 200 条轨迹：{summary_path.parent}")
        if any(statuses[name] != count for name, count in expected_counts.items()):
            raise ValueError(f"逐回合状态计数与汇总不一致：{summary_path.parent}")
        expected_paired = method != "task_only"
        if (row["paired_reference_verified"] == "True") != expected_paired:
            raise ValueError(f"配对验证标志错误：{summary_path.parent}")
        expected_protocol = {
            "difficulty": 0.5,
            "terrain_seed": 12345.0,
            "env_seed": 24680.0,
            "terrain_rows": 10.0,
            "terrain_cols": 20.0,
            "command_x_mps": 1.0,
            "goal_distance_m": 3.0,
            "max_time_s": 8.0,
            "policy_obs_dim": 48.0,
        }
        for field, expected in expected_protocol.items():
            if float(row[field]) != expected:
                raise ValueError(f"评估协议字段 {field} 不一致：{summary_path}")
        signatures_by_terrain[terrain].add(row["pairing_signature"])
        indexed[key] = row

    expected_keys = {
        (seed, terrain, method)
        for seed in SEEDS
        for terrain in TERRAINS
        for method in METHODS
    }
    missing = sorted(expected_keys - indexed.keys())
    unexpected = sorted(indexed.keys() - expected_keys)
    if missing or unexpected:
        raise ValueError(f"正式矩阵不完整：缺少={missing}，多余={unexpected}")
    for terrain, signatures in signatures_by_terrain.items():
        if len(signatures) != 1:
            raise ValueError(f"地形 {terrain} 在方法或训练种子间的配对签名不一致。")
    return indexed


def _per_checkpoint_rows(
    indexed: Mapping[tuple[int, str, str], Mapping[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for terrain in TERRAINS:
        for method in METHODS:
            for seed in SEEDS:
                source = indexed[(seed, terrain, method)]
                rows.append(
                    {
                        "地形": TERRAIN_LABELS[terrain],
                        "方法": METHOD_LABELS[method],
                        "训练种子": seed,
                        "回合数": int(source["num_episodes"]),
                        "成功数": int(source["num_success"]),
                        "摔倒数": int(source["num_fall"]),
                        "超时数": int(source["num_timeout"]),
                        "成功率": float(source["success_rate"]),
                        "摔倒率": float(source["fall_rate"]),
                        "超时率": float(source["timeout_rate"]),
                        "中位前进距离米": float(source["median_progress_m"]),
                        "成功回合中位能耗焦耳": _float_or_none(source["successful_median_energy_j"]),
                        "成功回合中位每米能耗焦耳每米": _float_or_none(
                            source["successful_median_energy_per_meter_j"]
                        ),
                        "成功回合中位到达时间秒": _float_or_none(
                            source["successful_median_time_to_goal_s"]
                        ),
                        "成功回合中位速度跟踪均方根误差米每秒": _float_or_none(
                            source["successful_median_tracking_rmse"]
                        ),
                        "检查点路径": source["checkpoint"],
                        "检查点SHA256": source["checkpoint_sha256"],
                        "配对签名": source["pairing_signature"],
                    }
                )
    return rows


def _optional_mean_sd(values: Sequence[float | None]) -> tuple[int, float | None, float | None]:
    available = [value for value in values if value is not None]
    if not available:
        return 0, None, None
    mean, sd = _mean_sd(available)
    return len(available), mean, sd


def _cross_seed_rows(
    indexed: Mapping[tuple[int, str, str], Mapping[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for terrain in TERRAINS:
        for method in METHODS:
            group = [indexed[(seed, terrain, method)] for seed in SEEDS]
            success_rates = [float(row["success_rate"]) for row in group]
            fall_rates = [float(row["fall_rate"]) for row in group]
            timeout_rates = [float(row["timeout_rate"]) for row in group]
            progress = [float(row["median_progress_m"]) for row in group]
            success_mean, success_sd, success_low, success_high = _mean_sd_ci95(success_rates)
            fall_mean, fall_sd = _mean_sd(fall_rates)
            timeout_mean, timeout_sd = _mean_sd(timeout_rates)
            progress_mean, progress_sd = _mean_sd(progress)
            energy_n, energy_mean, energy_sd = _optional_mean_sd(
                [_float_or_none(row["successful_median_energy_j"]) for row in group]
            )
            jpm_n, jpm_mean, jpm_sd = _optional_mean_sd(
                [_float_or_none(row["successful_median_energy_per_meter_j"]) for row in group]
            )
            time_n, time_mean, time_sd = _optional_mean_sd(
                [_float_or_none(row["successful_median_time_to_goal_s"]) for row in group]
            )
            rmse_n, rmse_mean, rmse_sd = _optional_mean_sd(
                [_float_or_none(row["successful_median_tracking_rmse"]) for row in group]
            )
            rows.append(
                {
                    "地形": TERRAIN_LABELS[terrain],
                    "方法": METHOD_LABELS[method],
                    "训练种子数": len(SEEDS),
                    "总回合数": sum(int(row["num_episodes"]) for row in group),
                    "总成功数": sum(int(row["num_success"]) for row in group),
                    "总摔倒数": sum(int(row["num_fall"]) for row in group),
                    "总超时数": sum(int(row["num_timeout"]) for row in group),
                    "成功率跨种子均值": success_mean,
                    "成功率跨种子样本标准差": success_sd,
                    "成功率95%置信区间下限": success_low,
                    "成功率95%置信区间上限": success_high,
                    "摔倒率跨种子均值": fall_mean,
                    "摔倒率跨种子样本标准差": fall_sd,
                    "超时率跨种子均值": timeout_mean,
                    "超时率跨种子样本标准差": timeout_sd,
                    "中位前进距离跨种子均值米": progress_mean,
                    "中位前进距离跨种子样本标准差米": progress_sd,
                    "能耗统计有效训练种子数": energy_n,
                    "成功回合中位能耗跨种子均值焦耳": energy_mean,
                    "成功回合中位能耗跨种子样本标准差焦耳": energy_sd,
                    "每米能耗统计有效训练种子数": jpm_n,
                    "成功回合中位每米能耗跨种子均值焦耳每米": jpm_mean,
                    "成功回合中位每米能耗跨种子样本标准差焦耳每米": jpm_sd,
                    "到达时间统计有效训练种子数": time_n,
                    "成功回合中位到达时间跨种子均值秒": time_mean,
                    "成功回合中位到达时间跨种子样本标准差秒": time_sd,
                    "跟踪误差统计有效训练种子数": rmse_n,
                    "成功回合中位速度跟踪均方根误差跨种子均值米每秒": rmse_mean,
                    "成功回合中位速度跟踪均方根误差跨种子样本标准差米每秒": rmse_sd,
                }
            )
    return rows


def _format_percent(mean: float, sd: float) -> str:
    return f"{100.0 * mean:.1f}% ± {100.0 * sd:.1f}%"


def _report(cross_rows: Sequence[Mapping[str, Any]]) -> str:
    indexed = {(row["地形"], row["方法"]): row for row in cross_rows}
    lines = [
        "# 三类正式平地模型跨训练种子地形评估统计",
        "",
        "本报告覆盖训练种子 1 至 5、三种方法、一个平地源域控制和四种非平地地形，每个检查点和地形包含 200 条配对轨迹，共 15000 条轨迹。成功率以训练种子为统计单位，表中为跨种子均值 ± 样本标准差；95% 置信区间使用自由度 4 的双侧 t 区间。",
        "",
        "## 成功率",
        "",
        "| 地形 | 任务型 | 固定能耗权重 PPO | PACE-ECO |",
        "| --- | ---: | ---: | ---: |",
    ]
    for terrain in TERRAIN_LABELS.values():
        cells = []
        for method in METHOD_LABELS.values():
            row = indexed[(terrain, method)]
            cells.append(_format_percent(row["成功率跨种子均值"], row["成功率跨种子样本标准差"]))
        lines.append(f"| {terrain} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.extend(
        [
            "",
            "## 成功回合的每米能耗",
            "",
            "下表为各训练种子成功回合中位每米能耗的跨种子均值 ± 样本标准差。括号内是存在至少一个成功回合、因而能进入能耗统计的训练种子数。",
            "",
            "| 地形 | 任务型 | 固定能耗权重 PPO | PACE-ECO |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for terrain in TERRAIN_LABELS.values():
        cells = []
        for method in METHOD_LABELS.values():
            row = indexed[(terrain, method)]
            n = int(row["每米能耗统计有效训练种子数"])
            mean = row["成功回合中位每米能耗跨种子均值焦耳每米"]
            sd = row["成功回合中位每米能耗跨种子样本标准差焦耳每米"]
            cells.append("—" if n == 0 else f"{mean:.1f} ± {sd:.1f}（{n}/5）")
        lines.append(f"| {terrain} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.extend(
        [
            "",
            "## 统计边界",
            "",
            "能耗、每米能耗、到达时间和速度跟踪误差只统计成功回合。不同方法的成功回合集合可能不同，因此能耗比较必须与成功率一起解释，不能视为无条件节能结论。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    cwd = Path.cwd()
    input_roots = (
        [Path(value).expanduser().resolve() for value in args.input_roots]
        if args.input_roots
        else [
            (cwd / "results/terrain_evaluation/formal_flat_seed1_n200").resolve(),
            (cwd / "results/terrain_evaluation/formal_flat_seeds2_5_n200").resolve(),
            (cwd / "results/terrain_evaluation/formal_flat_controls_seeds1_5_n200").resolve(),
        ]
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    discovered = _discover_summaries(input_roots)
    indexed = _validate_and_index(discovered)
    per_checkpoint = _per_checkpoint_rows(indexed)
    cross_seed = _cross_seed_rows(indexed)
    per_checkpoint_path = output_dir / "gpt-种子1至5-逐检查点汇总.csv"
    cross_seed_path = output_dir / "gpt-种子1至5-跨种子统计.csv"
    report_path = output_dir / "gpt-种子1至5-跨种子统计报告.md"
    _atomic_write_csv(per_checkpoint_path, per_checkpoint)
    _atomic_write_csv(cross_seed_path, cross_seed)
    _atomic_write_text(report_path, _report(cross_seed))
    print(f"[完成] 逐检查点汇总：{per_checkpoint_path}")
    print(f"[完成] 跨种子统计：{cross_seed_path}")
    print(f"[完成] 统计报告：{report_path}")


if __name__ == "__main__":
    main()
