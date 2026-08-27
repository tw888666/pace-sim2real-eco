#!/usr/bin/env python3
"""只读审计 v2.3 九个正式模型并生成不可覆盖的 holdout 授权。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

from pace_eco_lab.direction_conditioned_v2_3_mixed_protocol import (
    FIXED_ENERGY_REWARD_WEIGHT,
    FOOT_BOUNDARY_AUDIT_VERSION,
    FORMAL_SEEDS,
    METHOD_NAMES,
    PROTOCOL_VERSION,
    TASK_IDS,
    TRAINING_UPDATES,
    terrain_seed,
)


FAILURE = re.compile(r"Traceback|CUDA out of memory|\bOOM\b|NaN|Segmentation fault|Aborted", re.I)

parser = argparse.ArgumentParser(description="冻结 v2.3 Mixed 九模型 holdout 授权。")
parser.add_argument("--rsl_root", required=True)
parser.add_argument("--launch_root", required=True)
parser.add_argument("--budget", required=True)
parser.add_argument("--holdout_manifest", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON不是对象：{path}")
    return value


def _nested(value: object, *keys: str) -> object:
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"配置缺少字段：{'/'.join(keys)}")
        value = value[key]
    return value


def main() -> None:
    rsl_root = Path(args.rsl_root).expanduser().resolve()
    launch_root = Path(args.launch_root).expanduser().resolve()
    budget_path = Path(args.budget).expanduser().resolve()
    manifest_path = Path(args.holdout_manifest).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖holdout授权：{output}")
    budget = _json(budget_path)
    manifest = _json(manifest_path)
    if budget.get("冻结状态") != "已冻结" or budget.get("协议版本") != PROTOCOL_VERSION:
        raise ValueError("v2.3预算冻结文件错误。")
    if manifest.get("冻结状态") != "已冻结" or manifest.get("协议版本") != PROTOCOL_VERSION or manifest.get("数据拆分") != "holdout":
        raise ValueError("v2.3 holdout manifest错误。")
    b80 = float(budget.get("B80_mixed_J", 0.0))
    if b80 <= 0.0:
        raise ValueError("v2.3 B80必须为正。")
    record_paths = list(rsl_root.rglob("gpt_复现信息.json"))
    models: list[dict[str, object]] = []
    for method in METHOD_NAMES:
        for seed in FORMAL_SEEDS:
            task = TASK_IDS[method]
            candidates = [
                path for path in record_paths
                if (_json(path).get("task"), int(_json(path).get("seed", -1))) == (task, seed)
            ]
            if len(candidates) != 1:
                raise RuntimeError(f"{method}/seed{seed}应有唯一正式模型，实际{len(candidates)}。")
            record_path = candidates[0]
            run_dir = record_path.parent
            record = _json(record_path)
            if not bool(record.get("git_worktree_clean")) or "formal_train" not in str(record.get("run_name", "")):
                raise ValueError(f"训练工作树不干净或角色错误：{run_dir}")
            checkpoints = list(run_dir.glob("model_2999.pt"))
            if len(checkpoints) != 1:
                raise RuntimeError(f"{method}/seed{seed}最终checkpoint不唯一。")
            checkpoint = checkpoints[0]
            env_path = run_dir / "gpt_环境配置.json"
            agent_path = run_dir / "gpt_算法配置.json"
            fingerprint_path = run_dir / "gpt_配置指纹.json"
            env = _json(env_path)
            agent = _json(agent_path)
            if (
                env.get("pace_terrain_category") != "mixed"
                or env.get("pace_direction_variant") != "directional"
                or int(env.get("pace_terrain_seed", -1)) != terrain_seed("formal_train", seed)
                or int(env.get("scene", {}).get("num_envs", -1)) != 4096
                or int(agent.get("max_iterations", -1)) != TRAINING_UPDATES
                or int(agent.get("seed", -1)) != seed
            ):
                raise ValueError(f"基础配置错误：{method}/seed{seed}")
            weight = float(_nested(env, "rewards", "energy", "weight"))
            algorithm = _nested(agent, "algorithm")
            class_name = str(algorithm.get("class_name", "")).lower()
            if method == "task_only" and (abs(weight) > 1.0e-12 or "lagrangian" in class_name):
                raise ValueError("Task-only方法配置错误。")
            if method == "fixed_weight" and (abs(weight - FIXED_ENERGY_REWARD_WEIGHT) > 1.0e-12 or "lagrangian" in class_name):
                raise ValueError("Fixed-weight不是W100普通PPO。")
            if method == "eco" and (abs(weight) > 1.0e-12 or "lagrangian" not in class_name or abs(float(algorithm.get("energy_budget_j", 0.0)) - b80) > 1.0e-9):
                raise ValueError("ECO没有使用唯一冻结Mixed B80。")
            token = f"formal_train_{method}_seed{seed}_"
            valid_logs = []
            for path in launch_root.rglob("*.log"):
                if token not in path.name:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
                if "Learning iteration 2999/3000" in text and FAILURE.search(text) is None:
                    valid_logs.append(path)
            if not valid_logs:
                raise RuntimeError(f"缺少完成0--2999且无OOM/NaN/异常的日志：{method}/seed{seed}")
            log = sorted(valid_logs)[-1]
            models.append({
                "任务": task, "方法": method, "PPO_seed": seed,
                "训练地形seed": terrain_seed("formal_train", seed),
                "检查点": str(checkpoint), "检查点SHA256": _sha256(checkpoint),
                "复现记录": str(record_path), "复现记录SHA256": _sha256(record_path),
                "环境配置": str(env_path), "环境配置SHA256": _sha256(env_path),
                "算法配置": str(agent_path), "算法配置SHA256": _sha256(agent_path),
                "配置指纹": str(fingerprint_path), "配置指纹SHA256": _sha256(fingerprint_path),
                "训练日志": str(log), "训练日志SHA256": _sha256(log),
            })
    if len(models) != 9 or len({(item["方法"], item["PPO_seed"]) for item in models}) != 9:
        raise RuntimeError("九模型矩阵不完整或重复。")
    payload = {
        "冻结状态": "holdout已授权",
        "协议版本": PROTOCOL_VERSION,
        "边界审计版本": FOOT_BOUNDARY_AUDIT_VERSION,
        "阶段": "stage6",
        "创建时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "模型数量": 9,
        "预算冻结": str(budget_path),
        "预算冻结SHA256": _sha256(budget_path),
        "holdout_manifest": str(manifest_path),
        "holdout_manifest_SHA256": _sha256(manifest_path),
        "模型": models,
        "规则": "只审计预注册九个唯一model_2999.pt，不读取holdout性能，不允许候选模型选择。",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
