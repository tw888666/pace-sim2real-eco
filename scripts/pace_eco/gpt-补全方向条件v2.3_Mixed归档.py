#!/usr/bin/env python3
"""补齐 v2.3 Mixed 的预注册宏平均、上下行差值和十训练任务归档。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
from pathlib import Path


PRIMARY_TERRAINS = ("flat", "rough", "boxes", "stairs", "slope")
METHODS = ("task_only", "fixed_weight", "eco")
METRICS = (
    "方向穿越成功率", "B80联合合格率", "生存成功率", "平均方向净进度_m",
    "方向成功回合平均能耗_J", "方向成功回合归一化超预算幅度",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="补齐 v2.3 Mixed 阶段7/8归档。")
    parser.add_argument("--root", required=True)
    return parser


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON不是对象：{path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖归档补充文件：{path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _record_file(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"文件": str(path.resolve()), "字节数": path.stat().st_size, "SHA256": _sha256(path)}


def _training_task(
    role: str,
    method: str,
    seed: int,
    terrain_seed: int,
    task: str,
    checkpoint: Path,
) -> dict[str, object]:
    run = checkpoint.parent
    files = {}
    for label, name in (
        ("检查点", "model_2999.pt"),
        ("复现记录", "gpt_复现信息.json"),
        ("环境配置", "gpt_环境配置.json"),
        ("算法配置", "gpt_算法配置.json"),
        ("配置指纹", "gpt_配置指纹.json"),
    ):
        files[label] = _record_file(run / name)
    return {
        "角色": role,
        "方法": method,
        "PPO_seed": seed,
        "训练地形seed": terrain_seed,
        "任务": task,
        "更新": "0--2999",
        "随机初始化": True,
        "运行目录": str(run.resolve()),
        "文件": files,
    }


def main() -> None:
    repository = Path(args.root).expanduser().resolve()
    result = repository / "results/supplementary/direction_conditioned_v2_3_mixed"
    stage7 = result / "stage7"
    stage8 = result / "stage8"
    statistics_path = stage7 / "gpt-阶段7完整统计.json"
    conclusion_path = stage8 / "gpt-v2.3_Mixed最终结论.json"
    authorization_path = result / "stage5/gpt-v2.3-Mixed九模型holdout授权.json"
    budget_path = result / "stage3/gpt-v2.3-Mixed-B_ref-B80冻结.json"
    task_output = stage8 / "gpt-v2.3_Mixed十训练任务清单.json"
    supplement_output = stage7 / "gpt-阶段7宏平均与上下行补充.json"
    archive_output = stage8 / "gpt-v2.3_Mixed完整归档清单.json"
    for path in (task_output, supplement_output, archive_output):
        if path.exists():
            raise FileExistsError(f"拒绝覆盖冻结归档：{path}")
    statistics_data = _json(statistics_path)
    conclusion = _json(conclusion_path)
    authorization = _json(authorization_path)
    budget = _json(budget_path)
    if conclusion.get("冻结状态") != "v2.3 Mixed结论已冻结":
        raise ValueError("阶段8结论没有冻结。")

    terrain = {(str(row["范围"]), str(row["方法"])): row for row in statistics_data["五类主地形"]}
    macro_rows = []
    for method in METHODS:
        items = [terrain[(name, method)] for name in PRIMARY_TERRAINS]
        row: dict[str, object] = {"方法": method}
        for metric in METRICS:
            values = [float(item[metric]) for item in items]
            row[f"五地形等权宏平均{metric}"] = statistics.fmean(values)
        success = [float(item["方向穿越成功率"]) for item in items]
        joint = [float(item["B80联合合格率"]) for item in items]
        row.update({
            "最差方向成功率地形": PRIMARY_TERRAINS[success.index(min(success))],
            "最差地形方向成功率": min(success),
            "最差B80联合合格率地形": PRIMARY_TERRAINS[joint.index(min(joint))],
            "最差地形B80联合合格率": min(joint),
        })
        macro_rows.append(row)

    subtask = {(str(row["范围"]), str(row["方法"])): row for row in statistics_data["七方向子任务"]}
    up_down_rows = []
    component_fields = (
        "方向成功回合平均能耗_J", "方向成功回合平均电气能耗_J",
        "方向成功回合平均机械能耗_J", "方向成功回合平均potential能耗_J",
        "方向成功回合平均去potential能耗_J",
    )
    for method in METHODS:
        for terrain_name in ("stairs", "slope"):
            up = subtask[(f"{terrain_name}-up", method)]
            down = subtask[(f"{terrain_name}-down", method)]
            row = {"方法": method, "地形": terrain_name, "差值定义": "up减down"}
            for field in component_fields:
                row[f"up_{field}"] = up[field]
                row[f"down_{field}"] = down[field]
                row[f"up减down_{field}"] = float(up[field]) - float(down[field])
            up_down_rows.append(row)

    _write_csv(stage7 / "gpt-五地形宏平均与最差地形.csv", macro_rows)
    _write_csv(stage7 / "gpt-上下行能耗差专项分析.csv", up_down_rows)
    supplement_output.write_text(json.dumps({
        "统计重复单位": "PPO seed（每方法n=3）",
        "五地形等权宏平均与最差地形": macro_rows,
        "stairs_slope上下行能耗差": up_down_rows,
        "结论冻结文件SHA256": _sha256(conclusion_path),
        "说明": "本补充只展开阶段7已有聚合值，不修改阶段8冻结判定。",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    reference_checkpoint = Path(str(budget["checkpoint"])).resolve()
    reference_record = _json(reference_checkpoint.parent / "gpt_复现信息.json")
    tasks = [_training_task(
        "stage2预算参考",
        "task_only",
        0,
        710000,
        str(reference_record["task"]),
        reference_checkpoint,
    )]
    for item in authorization["模型"]:
        tasks.append(_training_task(
            "stage4正式模型",
            str(item["方法"]),
            int(item["PPO_seed"]),
            int(item["训练地形seed"]),
            str(item["任务"]),
            Path(str(item["检查点"])).resolve(),
        ))
    if len(tasks) != 10 or len({(item["角色"], item["方法"], item["PPO_seed"]) for item in tasks}) != 10:
        raise RuntimeError("十训练任务矩阵不完整或重复。")
    task_output.write_text(json.dumps({
        "协议版本": conclusion["协议版本"],
        "训练任务数": 10,
        "预算参考模型数": 1,
        "正式模型数": 9,
        "任务": tasks,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    evidence = [
        repository / "gpt-方向条件v2.3_Mixed冻结配置.json",
        result / "stage1/gpt-v2.3-Mixed阶段1冻结审计.json",
        result / "manifests/calibration_manifest.json",
        result / "manifests/holdout_manifest.json",
        budget_path,
        authorization_path,
        task_output,
    ]
    evidence.extend(sorted(path for path in stage7.iterdir() if path.is_file()))
    evidence.extend(sorted(path for path in stage8.iterdir() if path.is_file() and path != archive_output))
    unique = {str(path.resolve()): path for path in evidence}
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True, capture_output=True, check=True
    ).stdout.strip()
    archive_output.write_text(json.dumps({
        "冻结状态": "v2.3 Mixed完整归档已建立",
        "协议版本": conclusion["协议版本"],
        "源码Git提交": git_commit,
        "结论": conclusion["冻结结论"]["总体判断"],
        "文件数（不含本清单）": len(unique),
        "文件": [_record_file(unique[key]) for key in sorted(unique)],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"训练任务": len(tasks), "归档文件": len(unique), "结论": conclusion["冻结结论"]["总体判断"]}, ensure_ascii=False))


if __name__ == "__main__":
    args = _parser().parse_args()
    main()
