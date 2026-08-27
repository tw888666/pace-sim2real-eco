#!/usr/bin/env python3
"""审计并冻结 v2.2 的 45 模型 holdout 清单；不读取性能或改写模型。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

from pace_eco_lab.direction_conditioned_v2_2_protocol import (
    EVALUATION_MATRIX, EVALUATION_PROTOCOL_VERSION, EVALUATION_REUSED_V2_1_MODELS,
    FIXED_ENERGY_REWARD_WEIGHT, FIXED_LAMBDA, FIXED_WEIGHT_LABEL,
    FOOT_BOUNDARY_AUDIT_VERSION, METHOD_NAMES, METRIC_PROTOCOL_VERSION,
    PROTOCOL_VERSION, TASK_IDS, TERRAIN_NAMES, terrain_seed,
)
from pace_eco_lab.evaluation_states import (
    MULTI_TERRAIN_HOLDOUT_STATE_SET, evaluation_state_count,
    evaluation_state_definition_sha256,
)

FAILURE_PATTERN = re.compile(
    r"Traceback|CUDA out of memory|\bOOM\b|异常退出|Segmentation fault|Aborted|退出码[^\n]*非零",
    re.IGNORECASE,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="审计并冻结方向条件 v2.2 的 45 模型。")
    parser.add_argument("--v2_1_rsl_root", required=True)
    parser.add_argument("--v2_2_rsl_root", required=True)
    parser.add_argument("--v2_1_launch_root", default=None)
    parser.add_argument("--v2_2_launch_root", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol_config", required=True)
    parser.add_argument("--evaluation_manifest", required=True)
    parser.add_argument("--energy_reference_json", required=True)
    return parser


def _load_json(path: Path, label: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"{label}不存在：{path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label}无法解析：{path}") from error
    if not isinstance(data, dict):
        raise ValueError(f"{label}不是 JSON 对象：{path}")
    return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_sha256(path: Path) -> tuple[str, int]:
    before = path.stat()
    if before.st_size <= 0:
        raise ValueError(f"文件为空：{path}")
    digest = _sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"哈希期间文件发生变化：{path}")
    return digest, after.st_size


def _candidate_records(root: Path, task: str, seed: int) -> list[Path]:
    result: list[Path] = []
    for path in root.rglob("gpt_复现信息.json"):
        record = _load_json(path, "复现记录")
        if record.get("task") == task and int(record.get("seed", -1)) == seed:
            result.append(path)
    return result


def _read_evaluation_matrix(manifest: dict[str, object]) -> tuple[tuple[str, str, int], ...]:
    if (
        manifest.get("冻结状态") != "45模型评估协议已冻结"
        or manifest.get("协议版本") != EVALUATION_PROTOCOL_VERSION
        or manifest.get("来源训练协议") != PROTOCOL_VERSION
        or manifest.get("指标版本") != METRIC_PROTOCOL_VERSION
    ):
        raise ValueError("45模型评估清单的冻结状态或版本错误。")
    matrix = manifest.get("评估矩阵")
    if not isinstance(matrix, dict):
        raise ValueError("45模型评估清单缺少评估矩阵。")
    terrains = tuple(str(value) for value in matrix.get("地形", []))
    methods = tuple(str(value) for value in matrix.get("方法", []))
    seeds = tuple(int(value) for value in matrix.get("PPO_seed", []))
    expanded = tuple((method, terrain, seed) for terrain in terrains for method in methods for seed in seeds)
    if terrains != TERRAIN_NAMES or methods != METHOD_NAMES or seeds != (1, 2, 3):
        raise ValueError("评估清单必须严格为5地形×3方法×seed1-3。")
    if expanded != EVALUATION_MATRIX or len(set(expanded)) != 45:
        raise ValueError("评估清单展开后不是45个唯一模型。")
    excluded = manifest.get("不进入本次评估", {})
    if not isinstance(excluded, dict) or excluded.get("PPO_seed") != [4, 5]:
        raise ValueError("评估清单必须明确标记 seed4、5 不进入评估。")
    return expanded


def _find_clean_completion_log(launch_root: Path, terrain: str, method: str, seed: int) -> tuple[Path, str]:
    token = f"stage1_formal_train_directional_{terrain}_{method}_seed{seed}_"
    candidates = sorted(path for path in launch_root.rglob("*.log") if token in path.name)
    valid: list[tuple[Path, str]] = []
    for path in candidates:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "Learning iteration 2999/3000" in text and FAILURE_PATTERN.search(text) is None:
            valid.append((path, _sha256(path)))
    if not valid:
        raise RuntimeError(f"缺少无 OOM/Traceback/异常退出且到达2999/3000的启动日志：{token}")
    return valid[-1]


def _nested(mapping: object, *keys: str) -> object:
    current = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"配置缺少字段：{'/'.join(keys)}")
        current = current[key]
    return current


def _validate_configs(*, method: str, terrain: str, seed: int, environment: dict[str, object], agent: dict[str, object], b80: dict[str, object]) -> None:
    if (
        int(environment.get("pace_terrain_seed", -1)) != terrain_seed("stage1_formal_train", terrain, seed)
        or environment.get("pace_terrain_category") != terrain
        or environment.get("pace_direction_variant") != "directional"
        or float(environment.get("episode_length_s", 0.0)) != 20.0
        or int(agent.get("seed", -1)) != seed
        or int(agent.get("max_iterations", -1)) != 3000
    ):
        raise ValueError(f"环境/算法基础配置错误：{terrain}/{method}/seed{seed}")
    energy_weight = float(_nested(environment, "rewards", "energy", "weight"))
    algorithm = _nested(agent, "algorithm")
    if not isinstance(algorithm, dict):
        raise ValueError("算法配置不是对象。")
    class_name = str(algorithm.get("class_name", "")).lower()
    if method == "fixed_weight":
        if abs(energy_weight - FIXED_ENERGY_REWARD_WEIGHT) > 1.0e-12 or "ppo_lagrangian" in class_name:
            raise ValueError("Fixed-weight 必须使用普通 PPO 和 -0.00016。")
    elif method == "task_only":
        if abs(energy_weight) > 1.0e-12 or "ppo_lagrangian" in class_name:
            raise ValueError("Task-only 必须使用零能耗奖励权重和普通 PPO。")
    else:
        trained_budget = float(algorithm.get("energy_budget_j", 0.0))
        if "ppo_lagrangian" not in class_name or abs(energy_weight) > 1.0e-12:
            raise ValueError("ECO 必须使用 PPO-Lagrangian 且不使用固定能耗奖励权重。")
        if abs(trained_budget - float(b80.get(terrain, 0.0))) > 1.0e-9:
            raise ValueError("ECO 训练预算不是对应地形的冻结 B80。")


def _assert_unique(items: Iterable[dict[str, object]]) -> None:
    items = list(items)
    combinations = [(item["地形"], item["方法"], item["PPO_seed"]) for item in items]
    checkpoints = [item["检查点"] for item in items]
    if len(combinations) != len(set(combinations)):
        raise RuntimeError("同一地形×方法×seed出现重复模型。")
    if len(checkpoints) != len(set(checkpoints)):
        raise RuntimeError("同一checkpoint被多个评估项重复引用。")


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    v2_1_root = Path(args.v2_1_rsl_root).expanduser().resolve()
    v2_2_root = Path(args.v2_2_rsl_root).expanduser().resolve()
    v2_1_launch = Path(args.v2_1_launch_root).expanduser().resolve() if args.v2_1_launch_root else v2_1_root.parent / "launch"
    v2_2_launch = Path(args.v2_2_launch_root).expanduser().resolve() if args.v2_2_launch_root else v2_2_root.parent / "launch"
    output = Path(args.output).expanduser().resolve()
    protocol_path = Path(args.protocol_config).expanduser().resolve()
    manifest_path = Path(args.evaluation_manifest).expanduser().resolve()
    reference_path = Path(args.energy_reference_json).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖45模型holdout授权：{output}")
    protocol = _load_json(protocol_path, "v2.2训练协议配置")
    manifest = _load_json(manifest_path, "45模型评估清单")
    references = _load_json(reference_path, "v2.1 B_ref/B80")
    matrix = _read_evaluation_matrix(manifest)
    if protocol.get("协议版本") != PROTOCOL_VERSION or "冻结" not in str(protocol.get("冻结状态")):
        raise ValueError("v2.2训练协议配置未冻结或版本错误。")
    if (
        references.get("冻结状态") != "已冻结"
        or references.get("协议版本") != "gpt-direction-conditioned-v2.1"
        or references.get("边界审计版本") != FOOT_BOUNDARY_AUDIT_VERSION
        or references.get("阶段") != "stage1"
    ):
        raise ValueError("B_ref/B80冻结状态、版本或阶段错误。")
    b80 = references.get("B80_J")
    if not isinstance(b80, dict) or any(float(b80.get(terrain, 0.0)) <= 0.0 for terrain in TERRAIN_NAMES):
        raise ValueError("B_ref/B80文件缺少五地形正B80。")

    models: list[dict[str, object]] = []
    for method, terrain, seed in matrix:
        reused = (method, terrain, seed) in EVALUATION_REUSED_V2_1_MODELS
        root = v2_1_root if reused else v2_2_root
        launch_root = v2_1_launch if reused else v2_2_launch
        task = TASK_IDS[("directional", method, terrain)]
        records = _candidate_records(root, task, seed)
        if len(records) != 1:
            raise RuntimeError(f"{method}/{terrain}/seed{seed}应有唯一复现记录，实际{len(records)}。")
        record_path = records[0]
        record = _load_json(record_path, "复现记录")
        run_name = str(record.get("run_name", ""))
        expected_marker = "gpt_direction_v2_1_" if reused else "gpt_direction_v2_2_"
        if expected_marker not in run_name or "formal_train" not in run_name or "smoke" in run_name.lower():
            raise ValueError(f"模型来源协议或角色错误：{record_path.parent}")
        run_dir = record_path.parent
        checkpoint = run_dir / "model_2999.pt"
        environment_path = run_dir / "gpt_环境配置.json"
        agent_path = run_dir / "gpt_算法配置.json"
        checkpoint_sha256, checkpoint_size = _stable_sha256(checkpoint)
        environment = _load_json(environment_path, "环境配置")
        agent = _load_json(agent_path, "算法配置")
        _validate_configs(method=method, terrain=terrain, seed=seed, environment=environment, agent=agent, b80=b80)
        launch_log, launch_log_sha256 = _find_clean_completion_log(launch_root, terrain, method, seed)
        models.append({
            "任务": task, "实验变体": "directional", "方法": method, "地形": terrain,
            "PPO_seed": seed, "训练地形seed": terrain_seed("stage1_formal_train", terrain, seed),
            "来源协议": "v2.1复用" if reused else "v2.2新增",
            "检查点": str(checkpoint), "检查点字节数": checkpoint_size, "检查点SHA256": checkpoint_sha256,
            "复现记录": str(record_path), "复现记录SHA256": _sha256(record_path),
            "环境配置": str(environment_path), "环境配置SHA256": _sha256(environment_path),
            "算法配置": str(agent_path), "算法配置SHA256": _sha256(agent_path),
            "启动日志": str(launch_log), "启动日志SHA256": launch_log_sha256,
        })
    _assert_unique(models)
    if len(models) != 45 or sum(item["来源协议"] == "v2.1复用" for item in models) != 14:
        raise RuntimeError("45模型或14/31来源计数不符合冻结清单。")

    state_sha256 = evaluation_state_definition_sha256(MULTI_TERRAIN_HOLDOUT_STATE_SET)
    payload = {
        "冻结状态": "holdout已授权", "协议版本": EVALUATION_PROTOCOL_VERSION,
        "来源训练协议": PROTOCOL_VERSION, "指标版本": METRIC_PROTOCOL_VERSION,
        "边界审计版本": FOOT_BOUNDARY_AUDIT_VERSION, "阶段": "stage1",
        "创建时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "模型数量": 45, "复用v2.1模型数": 14, "v2.2新增模型数": 31,
        "固定权重": {"标签": FIXED_WEIGHT_LABEL, "lambda_fixed": FIXED_LAMBDA, "能耗奖励系数": FIXED_ENERGY_REWARD_WEIGHT},
        "评估条件": manifest["holdout条件"],
        "证据文件": {
            "训练协议": str(protocol_path), "训练协议SHA256": _sha256(protocol_path),
            "45模型评估清单": str(manifest_path), "45模型评估清单SHA256": _sha256(manifest_path),
            "B_ref_B80": str(reference_path), "B_ref_B80_SHA256": _sha256(reference_path),
            "初始状态表": MULTI_TERRAIN_HOLDOUT_STATE_SET, "初始状态表SHA256": state_sha256,
            "初始状态数": evaluation_state_count(MULTI_TERRAIN_HOLDOUT_STATE_SET),
        },
        "模型": models,
        "规则": "只读审计45个预注册唯一model_2999.pt；不读取性能，不改写checkpoint，seed4/5不授权评估。",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
