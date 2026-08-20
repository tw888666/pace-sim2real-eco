#!/usr/bin/env python3
"""按任务/seed/终点完整性冻结 v2 holdout 模型，不读取性能。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from pace_eco_lab.direction_conditioned_protocol import (
    FOOT_BOUNDARY_AUDIT_VERSION,
    PPO_SEEDS,
    PROTOCOL_VERSION,
    TASK_IDS,
    TERRAIN_NAMES,
    VARIANT_NAMES,
    terrain_seed,
)


parser = argparse.ArgumentParser(description="冻结方向条件 v2.1 holdout 模型清单。")
parser.add_argument("--stage", required=True, choices=("stage1", "stage2"))
parser.add_argument("--rsl_root", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--protocol_config", required=True)
parser.add_argument("--energy_reference_json", required=True)
args = parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reference_value(references: dict[str, object], stage: str, terrain: str) -> float:
    if stage == "stage2":
        return float(references.get("B_ref_mixed_J", 0.0))
    values = references.get("B_ref_J", {})
    return float(values.get(terrain, 0.0)) if isinstance(values, dict) else 0.0


def main() -> None:
    root = Path(args.rsl_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖 v2 holdout 授权：{output}")
    evidence_paths = {
        "协议配置": Path(args.protocol_config).expanduser().resolve(),
        "B_ref": Path(args.energy_reference_json).expanduser().resolve(),
    }
    for label, path in evidence_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label}不存在：{path}")
    references = json.loads(evidence_paths["B_ref"].read_text(encoding="utf-8"))
    if (
        references.get("冻结状态") != "已冻结"
        or references.get("协议版本") != PROTOCOL_VERSION
        or references.get("边界审计版本") != FOOT_BOUNDARY_AUDIT_VERSION
        or references.get("阶段") != args.stage
    ):
        raise ValueError("v2 B_ref 的冻结状态、协议、边界审计版本或阶段错误。")
    protocol = json.loads(evidence_paths["协议配置"].read_text(encoding="utf-8"))
    if (
        protocol.get("协议版本") != PROTOCOL_VERSION
        or protocol.get("边界审计版本") != FOOT_BOUNDARY_AUDIT_VERSION
        or "实现冻结" not in str(protocol.get("冻结状态"))
    ):
        raise ValueError("v2 机器可读协议配置的版本或冻结状态错误。")
    terrains = TERRAIN_NAMES if args.stage == "stage1" else ("mixed",)
    seeds = PPO_SEEDS[f"{args.stage}_formal"]
    expected = {
        (TASK_IDS[(variant, method, terrain)], seed)
        for variant in VARIANT_NAMES
        for method in ("task_only", "eco")
        for terrain in terrains
        for seed in seeds
    }
    inverse = {task_id: key for key, task_id in TASK_IDS.items()}
    found: dict[tuple[str, int], dict[str, object]] = {}
    for record_path in root.rglob("gpt_复现信息.json"):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        key = (str(record.get("task")), int(record.get("seed", -1)))
        if key not in expected:
            continue
        run_dir = record_path.parent
        if "formal_train" not in str(record.get("run_name", "")):
            raise ValueError(f"v2 正式根混入非 formal_train 运行：{run_dir}")
        if not bool(record.get("git_worktree_clean")):
            raise ValueError(f"v2 正式模型训练时工作树不干净：{run_dir}")
        checkpoint = run_dir / "model_2999.pt"
        environment_path = run_dir / "gpt_环境配置.json"
        agent_path = run_dir / "gpt_算法配置.json"
        for path in (checkpoint, environment_path, agent_path):
            if not path.is_file():
                raise FileNotFoundError(f"v2 正式运行缺少 {path.name}：{run_dir}")
        if key in found:
            raise RuntimeError(f"发现重复 v2 正式任务/seed，禁止事后挑选：{key}")
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        agent = json.loads(agent_path.read_text(encoding="utf-8"))
        variant, method, terrain = inverse[key[0]]
        expected_seed = terrain_seed(f"{args.stage}_formal_train", terrain, key[1])
        if int(environment.get("pace_terrain_seed", -1)) != expected_seed:
            raise ValueError(f"v2 正式模型地形 seed 不符合冻结映射：{run_dir}")
        if environment.get("pace_terrain_category") != terrain:
            raise ValueError(f"v2 正式模型地形类别与任务 ID 不一致：{run_dir}")
        if environment.get("pace_direction_variant") != variant:
            raise ValueError(f"v2 正式模型实验变体与任务 ID 不一致：{run_dir}")
        if method == "eco":
            expected_reference = _reference_value(references, args.stage, terrain)
            expected_budget = 0.8 * expected_reference
            algorithm = agent.get("algorithm", {})
            trained_budget = float(algorithm.get("energy_budget_j", 0.0))
            if expected_budget <= 0.0 or abs(trained_budget - expected_budget) > 1.0e-9:
                raise ValueError(f"v2 ECO 的共享绝对焦耳 B80 不匹配：{run_dir}")
        found[key] = {
            "任务": key[0],
            "实验变体": variant,
            "方法": method,
            "地形": terrain,
            "PPO_seed": key[1],
            "检查点": str(checkpoint.resolve()),
            "检查点SHA256": _sha256(checkpoint),
            "复现记录": str(record_path.resolve()),
            "复现记录SHA256": _sha256(record_path),
            "环境配置SHA256": _sha256(environment_path),
            "算法配置SHA256": _sha256(agent_path),
        }
    missing = sorted(expected - set(found))
    if missing:
        raise RuntimeError(f"v2 正式终点未齐，holdout 保持封存；缺少 {len(missing)} 项：{missing[:5]}")
    payload = {
        "冻结状态": "holdout已授权",
        "协议版本": PROTOCOL_VERSION,
        "边界审计版本": FOOT_BOUNDARY_AUDIT_VERSION,
        "阶段": args.stage,
        "创建时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "模型数量": len(found),
        "模型": [found[key] for key in sorted(found)],
        "证据文件SHA256": {label: _sha256(path) for label, path in evidence_paths.items()},
        "规则": "只按预注册任务、PPO seed、model_2999.pt、配置和唯一性冻结，不读取任何性能。",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
