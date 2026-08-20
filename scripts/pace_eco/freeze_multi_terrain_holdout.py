#!/usr/bin/env python3
"""仅按冻结任务/seed/终点完整性生成 holdout 模型授权，不读取性能。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from pace_eco_lab.multi_terrain_protocol import (
    FOOT_BOUNDARY_AUDIT_VERSION,
    PACE_PAPER_FIXED_WEIGHT,
    PACE_PAPER_FIXED_WEIGHT_LABEL,
    PPO_SEEDS,
    PROTOCOL_VERSION,
    TASK_IDS,
    TERRAIN_NAMES,
    terrain_seed,
)


parser = argparse.ArgumentParser(description="冻结多地形 holdout 模型清单。")
parser.add_argument("--stage", required=True, choices=("stage1", "stage2"))
parser.add_argument("--rsl_root", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--protocol_config", required=True)
parser.add_argument("--energy_reference_json", required=True)
args = parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path(args.rsl_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖授权文件：{output}")
    evidence_paths = {
        "协议配置": Path(args.protocol_config).expanduser().resolve(),
        "B_ref": Path(args.energy_reference_json).expanduser().resolve(),
    }
    for label, path in evidence_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} 文件不存在：{path}")
    references = json.loads(evidence_paths["B_ref"].read_text(encoding="utf-8"))
    if (
        references.get("冻结状态") != "已冻结"
        or references.get("协议版本") != PROTOCOL_VERSION
        or references.get("边界审计版本") != FOOT_BOUNDARY_AUDIT_VERSION
        or references.get("阶段") != args.stage
    ):
        raise ValueError("B_ref 文件未冻结、协议版本错误或阶段不匹配。")
    protocol = json.loads(evidence_paths["协议配置"].read_text(encoding="utf-8"))
    if (
        protocol.get("协议版本") != PROTOCOL_VERSION
        or protocol.get("边界审计修订", {}).get("版本") != FOOT_BOUNDARY_AUDIT_VERSION
        or "已冻结" not in str(protocol.get("冻结状态"))
    ):
        raise ValueError("协议配置版本或冻结状态错误。")
    terrains = TERRAIN_NAMES if args.stage == "stage1" else ("mixed",)
    seeds = PPO_SEEDS[f"{args.stage}_formal"]
    expected = {
        (TASK_IDS[(method, terrain)], seed)
        for terrain in terrains
        for method in ("task_only", "fixed_weight", "eco")
        for seed in seeds
    }
    found: dict[tuple[str, int], dict[str, object]] = {}
    for record_path in root.rglob("gpt_复现信息.json"):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        key = (str(record.get("task")), int(record.get("seed", -1)))
        if key not in expected:
            continue
        if "formal_train" not in str(record.get("run_name", "")):
            raise ValueError(f"正式根目录混入非 formal_train 运行：{record_path.parent}")
        checkpoint = record_path.parent / "model_2999.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"正式运行缺少 model_2999.pt：{record_path.parent}")
        if key in found:
            raise RuntimeError(f"发现重复正式任务/seed，禁止事后挑选：{key}")
        environment_path = record_path.parent / "gpt_环境配置.json"
        agent_path = record_path.parent / "gpt_算法配置.json"
        if not environment_path.is_file() or not agent_path.is_file():
            raise FileNotFoundError(f"正式运行缺少环境或算法配置：{record_path.parent}")
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        agent = json.loads(agent_path.read_text(encoding="utf-8"))
        inverse = {task_id: pair for pair, task_id in TASK_IDS.items()}
        method, terrain = inverse[key[0]]
        expected_terrain_seed = terrain_seed(f"{args.stage}_formal_train", terrain, key[1])
        if int(environment.get("pace_terrain_seed", -1)) != expected_terrain_seed:
            raise ValueError(f"正式模型地形 seed 不符合冻结映射：{record_path.parent}")
        if environment.get("pace_terrain_category") != terrain:
            raise ValueError(f"正式模型地形类别与任务 ID 不一致：{record_path.parent}")
        if method == "fixed_weight":
            trained = float(environment.get("rewards", {}).get("energy", {}).get("weight", 0.0))
            if abs(trained - PACE_PAPER_FIXED_WEIGHT) > 1.0e-12:
                raise ValueError(
                    f"正式固定权重模型未使用 PACE 论文冻结系数 "
                    f"{PACE_PAPER_FIXED_WEIGHT_LABEL}={PACE_PAPER_FIXED_WEIGHT}：{record_path.parent}"
                )
        if method == "eco":
            algorithm = agent.get("algorithm", {})
            if algorithm.get("constraint_normalization") not in (None, "absolute_j"):
                raise ValueError(f"正式 PACE-ECO 未保持历史绝对焦耳约束：{record_path.parent}")
            if args.stage == "stage2":
                expected_reference = float(references.get("B_ref_mixed_J", 0.0))
            else:
                expected_reference = float(references.get("B_ref_J", {}).get(terrain, 0.0))
            expected_budget = 0.8 * expected_reference
            trained_budget = float(algorithm.get("energy_budget_j", 0.0))
            if expected_budget <= 0.0 or abs(trained_budget - expected_budget) > 1.0e-9:
                raise ValueError(f"正式 PACE-ECO 的绝对焦耳 B80 不匹配：{record_path.parent}")
        found[key] = {
            "任务": key[0],
            "PPO_seed": key[1],
            "检查点": str(checkpoint.resolve()),
            "检查点SHA256": sha256(checkpoint),
            "复现记录": str(record_path.resolve()),
            "复现记录SHA256": sha256(record_path),
            "环境配置SHA256": sha256(environment_path),
            "算法配置SHA256": sha256(agent_path),
        }
    missing = sorted(expected - set(found))
    if missing:
        raise RuntimeError(f"正式终点未齐备，holdout 保持封存；缺少 {len(missing)} 项：{missing[:5]}")
    payload = {
        "冻结状态": "holdout已授权",
        "协议版本": PROTOCOL_VERSION,
        "边界审计版本": FOOT_BOUNDARY_AUDIT_VERSION,
        "阶段": args.stage,
        "创建时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "模型数量": len(found),
        "固定权重标签": PACE_PAPER_FIXED_WEIGHT_LABEL,
        "固定权重系数": PACE_PAPER_FIXED_WEIGHT,
        "模型": [found[key] for key in sorted(found)],
        "证据文件SHA256": {label: sha256(path) for label, path in evidence_paths.items()},
        "规则": "只按预注册任务、PPO seed、model_2999.pt 存在性和唯一性冻结，不读取性能。",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
