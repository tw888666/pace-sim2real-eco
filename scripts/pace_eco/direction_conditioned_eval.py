#!/usr/bin/env python3
"""方向条件 v2.1 calibration 与 v2.1/v2.2 holdout 正式评估。"""

from __future__ import annotations

import argparse
import csv
import faulthandler
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from time import monotonic

from isaaclab.app import AppLauncher

from pace_eco_lab.direction_conditioned_protocol import (
    DESIRED_DIRECTION_W,
    EVAL_BATCHES,
    EVAL_EPISODES,
    EVAL_NUM_ENVS,
    EVAL_TERRAIN_COLS,
    EVAL_TERRAIN_ROWS,
    FOOT_BOUNDARY_AUDIT_VERSION,
    MAX_CROSS_TRACK_DEVIATION_M,
    METHOD_LABELS,
    MIN_DIRECTIONAL_PROGRESS_M,
    PPO_SEEDS,
    PROTOCOL_VERSION,
    TASK_IDS,
    TARGET_SPEED_M_S_V2,
    TERRAIN_LABELS,
    VARIANT_LABELS,
    evaluation_batch_offset,
    evaluation_batch_seed,
    evaluation_global_id,
    terrain_seed,
)
from pace_eco_lab.direction_conditioned_v2_2_protocol import (
    EVALUATION_PROTOCOL_VERSION,
    FIXED_ENERGY_REWARD_WEIGHT,
    METHOD_LABELS as V2_2_METHOD_LABELS,
    METRIC_PROTOCOL_VERSION,
    TASK_IDS as V2_2_TASK_IDS,
    is_evaluation_target as is_v2_2_evaluation_target,
)
from pace_eco_lab.direction_conditioned_v2_3_mixed_protocol import (
    FORMAL_SEEDS as V2_3_FORMAL_SEEDS,
    MANIFEST_VERSION as V2_3_MANIFEST_VERSION,
    METHOD_LABELS as V2_3_METHOD_LABELS,
    PROTOCOL_VERSION as V2_3_PROTOCOL_VERSION,
    TASK_IDS as V2_3_TASK_IDS,
    evaluation_batch_seed as v2_3_evaluation_batch_seed,
    subtask_name as v2_3_subtask_name,
    terrain_seed as v2_3_terrain_seed,
)
from pace_eco_lab.multi_terrain_protocol import (
    PROTOCOL_VERSION as LEGACY_PROTOCOL_VERSION,
    SUCCESS_SPEED_RANGE_M_S,
    TASK_IDS as LEGACY_TASK_IDS,
)
from pace_eco_lab.evaluation_states import (
    MULTI_TERRAIN_CALIBRATION_STATE_SET,
    MULTI_TERRAIN_HOLDOUT_STATE_SET,
    evaluation_state_count,
    evaluation_state_definition_sha256,
)


parser = argparse.ArgumentParser(description="PACE-ECO 方向条件 v2.1/v2.2 冻结评估。")
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--ppo_seed", required=True, type=int)
parser.add_argument("--terrain_seed", required=True, type=int)
parser.add_argument("--batch_index", required=True, type=int)
parser.add_argument("--batch_group", required=True)
parser.add_argument("--stage", required=True, choices=("stage1", "stage2", "stage3", "stage6"))
parser.add_argument("--split", required=True, choices=("calibration", "holdout"))
parser.add_argument("--energy_reference_json", default=None)
parser.add_argument(
    "--training_energy_reference_json",
    default=None,
    help="仅 E0 使用：历史 holdout 授权及旧 ECO 训练预算对应的 v1 B_ref。",
)
parser.add_argument("--output_root", required=True)
parser.add_argument("--holdout_authorization", default=None)
parser.add_argument("--manifest", default=None, help="v2.3 必需的冻结逐回合 manifest。")
parser.add_argument("--warmup_s", type=float, default=5.0)
parser.add_argument(
    "--direction_protocol_version",
    choices=("v2.1", "v2.2", "v2.3"),
    default="v2.1",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
ACTIVE_PROTOCOL_VERSION = {
    "v2.1": PROTOCOL_VERSION,
    "v2.2": EVALUATION_PROTOCOL_VERSION,
    "v2.3": V2_3_PROTOCOL_VERSION,
}[args_cli.direction_protocol_version]
ACTIVE_METRIC_VERSION = {
    "v2.1": PROTOCOL_VERSION,
    "v2.2": METRIC_PROTOCOL_VERSION,
    "v2.3": V2_3_PROTOCOL_VERSION,
}[args_cli.direction_protocol_version]
ACTIVE_METHOD_LABELS = {
    "v2.1": METHOD_LABELS,
    "v2.2": V2_2_METHOD_LABELS,
    "v2.3": V2_3_METHOD_LABELS,
}[args_cli.direction_protocol_version]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _task_parts() -> tuple[str, str, str, bool]:
    if args_cli.direction_protocol_version == "v2.3":
        inverse = {task_id: method for method, task_id in V2_3_TASK_IDS.items()}
        if args_cli.task not in inverse:
            parser.error("v2.3 只允许 Mixed 三方法冻结任务 ID。")
        return "directional", inverse[args_cli.task], "mixed", False
    if args_cli.direction_protocol_version == "v2.2":
        current_v2_2 = {task_id: key for key, task_id in V2_2_TASK_IDS.items()}
        if args_cli.task not in current_v2_2:
            parser.error("v2.2 只允许冻结主矩阵中的方向条件任务 ID。")
        variant, method, terrain = current_v2_2[args_cli.task]
        return variant, method, terrain, False
    current = {task_id: key for key, task_id in TASK_IDS.items()}
    if args_cli.task in current:
        variant, method, terrain = current[args_cli.task]
        return variant, method, terrain, False
    legacy = {task_id: key for key, task_id in LEGACY_TASK_IDS.items()}
    if args_cli.task in legacy:
        method, terrain = legacy[args_cli.task]
        if method == "fixed_weight":
            parser.error("v2.1 E0 交叉评估不包含固定权重方法。")
        return "legacy_v1", method, terrain, True
    parser.error("只允许方向条件 v2.1 任务或 v1 E0 任务 ID。")
    raise AssertionError


def _reference_value(data: dict[str, object], stage: str, terrain: str) -> float:
    if stage == "stage2":
        value = float(data.get("B_ref_mixed_J", 0.0))
    else:
        values = data.get("B_ref_J", {})
        value = float(values.get(terrain, 0.0)) if isinstance(values, dict) else 0.0
    if value <= 0.0:
        parser.error("B_ref 文件缺少当前阶段/地形的正参考能耗。")
    return value


def _load_v2_reference(
    variant: str,
    method: str,
    terrain: str,
) -> tuple[dict[str, float], float | None, str | None]:
    calibration = args_cli.split == "calibration"
    if args_cli.direction_protocol_version == "v2.3":
        if calibration:
            if method != "task_only" or args_cli.energy_reference_json is not None:
                parser.error("v2.3 calibration 只允许 task_only 且禁止预读 B_ref。")
            return {}, None, None
        if args_cli.energy_reference_json is None:
            parser.error("v2.3 holdout 必须提供冻结的 Mixed B80 文件。")
        path = Path(args_cli.energy_reference_json).expanduser().resolve()
        if not path.is_file():
            parser.error(f"v2.3 Mixed B80 不存在：{path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("冻结状态") != "已冻结" or data.get("协议版本") != V2_3_PROTOCOL_VERSION:
            parser.error("v2.3 Mixed B80 状态或协议不匹配。")
        primary = float(data.get("B_ref_mixed_J", 0.0))
        source = data.get("五类主地形参考能耗_J", {})
        if primary <= 0.0 or not isinstance(source, dict):
            parser.error("v2.3 Mixed 冻结文件缺少正 B_ref 或五类地形参考。")
        references = {key: float(source.get(key, 0.0)) for key in ("flat", "rough", "stairs", "boxes", "slope")}
        if any(value <= 0.0 for value in references.values()):
            parser.error("v2.3 Mixed 冻结文件的五类地形参考不完整。")
        return references, primary, _sha256(path)
    if calibration:
        if variant != "directional" or method != "task_only":
            parser.error("v2 calibration 只允许 E2 directional/task_only seed0。")
        if args_cli.energy_reference_json is not None:
            parser.error("v2 calibration 禁止预先提供 B_ref，避免循环定义。")
        return {}, None, None
    if args_cli.energy_reference_json is None:
        parser.error("v2 holdout 和 E0 交叉评估必须提供 v2 B_ref。")
    path = Path(args_cli.energy_reference_json).expanduser().resolve()
    if not path.is_file():
        parser.error(f"v2 B_ref 不存在：{path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        data.get("冻结状态") != "已冻结"
        or data.get("协议版本") != PROTOCOL_VERSION
        or data.get("边界审计版本") != FOOT_BOUNDARY_AUDIT_VERSION
        or data.get("阶段") != args_cli.stage
    ):
        parser.error("v2 B_ref 的状态、协议、边界审计版本或阶段不匹配。")
    source = data.get("B_ref_J", {})
    if not isinstance(source, dict):
        parser.error("v2 B_ref 缺少 B_ref_J。")
    required = ("flat", "rough", "stairs", "boxes", "slope") if terrain == "mixed" else (terrain,)
    references = {name: float(source.get(name, 0.0)) for name in required}
    if any(value <= 0.0 for value in references.values()):
        parser.error("v2 B_ref 缺少当前评估涉及的地形参考值。")
    return references, _reference_value(data, args_cli.stage, terrain), _sha256(path)


def _load_legacy_training_budget(method: str, terrain: str) -> tuple[float | None, str | None]:
    if args_cli.training_energy_reference_json is None:
        parser.error("E0 必须提供历史 holdout 授权对应的 v1 B_ref。")
    path = Path(args_cli.training_energy_reference_json).expanduser().resolve()
    if not path.is_file():
        parser.error(f"E0 v1 B_ref 不存在：{path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        data.get("冻结状态") != "已冻结"
        or data.get("协议版本") != LEGACY_PROTOCOL_VERSION
        or data.get("阶段") != args_cli.stage
    ):
        parser.error("E0 的 v1 B_ref 状态、协议或阶段不匹配。")
    budget = 0.8 * _reference_value(data, args_cli.stage, terrain) if method == "eco" else None
    return budget, _sha256(path)


def _validate_authorization(legacy: bool) -> str | None:
    if args_cli.split == "calibration":
        if args_cli.holdout_authorization is not None:
            parser.error("calibration 禁止提供 holdout 授权。")
        return None
    if args_cli.holdout_authorization is None:
        parser.error("holdout 必须提供预先冻结的模型授权清单。")
    path = Path(args_cli.holdout_authorization).expanduser().resolve()
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    if not path.is_file() or not checkpoint.is_file():
        parser.error("holdout 授权或检查点不存在。")
    data = json.loads(path.read_text(encoding="utf-8"))
    expected_protocol = LEGACY_PROTOCOL_VERSION if legacy else ACTIVE_PROTOCOL_VERSION
    if (
        data.get("冻结状态") != "holdout已授权"
        or data.get("协议版本") != expected_protocol
        or data.get("边界审计版本") != FOOT_BOUNDARY_AUDIT_VERSION
        or data.get("阶段") != args_cli.stage
    ):
        parser.error("holdout 授权状态、协议、边界审计版本或阶段不匹配。")
    models = data.get("模型", [])
    matches = [
        item
        for item in models
        if Path(item.get("检查点", "")).resolve() == checkpoint
        and item.get("任务") == args_cli.task
        and int(item.get("PPO_seed", -1)) == args_cli.ppo_seed
    ]
    if len(matches) != 1:
        parser.error("当前模型不在冻结授权内，或任务/seed 记录不唯一。")
    if matches[0].get("检查点SHA256") != _sha256(checkpoint):
        parser.error("当前检查点哈希与冻结授权不一致。")
    if legacy:
        training_reference = Path(args_cli.training_energy_reference_json or "").expanduser().resolve()
        if (
            not training_reference.is_file()
            or data.get("证据文件SHA256", {}).get("B_ref") != _sha256(training_reference)
        ):
            parser.error("E0 当前 v1 B_ref 与历史 holdout 授权证据不一致。")
    elif args_cli.direction_protocol_version == "v2.1":
        reference = Path(args_cli.energy_reference_json or "").expanduser().resolve()
        if (
            not reference.is_file()
            or data.get("证据文件SHA256", {}).get("B_ref") != _sha256(reference)
        ):
            parser.error("当前 v2 B_ref 与 v2 holdout 授权证据不一致。")
    elif args_cli.direction_protocol_version == "v2.2":
        reference = Path(args_cli.energy_reference_json or "").expanduser().resolve()
        evidence = data.get("证据文件", {})
        state_sha256 = evaluation_state_definition_sha256(MULTI_TERRAIN_HOLDOUT_STATE_SET)
        if (
            data.get("指标版本") != METRIC_PROTOCOL_VERSION
            or not reference.is_file()
            or not isinstance(evidence, dict)
            or evidence.get("B_ref_B80_SHA256") != _sha256(reference)
            or evidence.get("初始状态表") != MULTI_TERRAIN_HOLDOUT_STATE_SET
            or evidence.get("初始状态表SHA256") != state_sha256
        ):
            parser.error("v2.2 授权中的指标、B80或初始状态表哈希与当前评估不一致。")
    else:
        reference = Path(args_cli.energy_reference_json or "").expanduser().resolve()
        if (
            data.get("协议版本") != V2_3_PROTOCOL_VERSION
            or not reference.is_file()
            or data.get("预算冻结SHA256") != _sha256(reference)
        ):
            parser.error("v2.3 holdout 授权与当前协议或预算冻结文件不一致。")
    return _sha256(path)


def _validate_protocol(variant: str, method: str, terrain: str, legacy: bool) -> None:
    if abs(args_cli.warmup_s - 5.0) > 1.0e-12:
        parser.error("v2.1 评估预热冻结为 5 秒。")
    if not 0 <= args_cli.batch_index < EVAL_BATCHES:
        parser.error(f"评估批次必须位于 [0, {EVAL_BATCHES - 1}]。")
    if re.fullmatch(r"[0-9]{8}_[0-9]{6}", args_cli.batch_group) is None:
        parser.error("评估批次组必须是 YYYYMMDD_HHMMSS。")
    if args_cli.direction_protocol_version == "v2.3":
        expected_stage = "stage3" if args_cli.split == "calibration" else "stage6"
        if args_cli.stage != expected_stage or terrain != "mixed":
            parser.error(f"v2.3 {args_cli.split} 必须使用 {expected_stage}/mixed。")
        expected_seed = v2_3_terrain_seed(args_cli.split)
        if args_cli.terrain_seed != expected_seed:
            parser.error(f"v2.3 {args_cli.split} 地形基准 seed 应为 {expected_seed}。")
        if args_cli.split == "calibration":
            if method != "task_only" or args_cli.ppo_seed != 0:
                parser.error("v2.3 calibration 只允许 task_only seed0。")
        elif args_cli.ppo_seed not in V2_3_FORMAL_SEEDS:
            parser.error("v2.3 holdout PPO seed 只允许1、2、3。")
        return
    if args_cli.stage == "stage1" and terrain == "mixed":
        parser.error("阶段一禁止 mixed。")
    if args_cli.stage == "stage2" and terrain != "mixed":
        parser.error("阶段二只允许 mixed。")
    expected_terrain = "mixed" if args_cli.stage == "stage2" else terrain
    expected_seed = terrain_seed(f"{args_cli.stage}_{args_cli.split}", expected_terrain)
    if args_cli.terrain_seed != expected_seed:
        parser.error(f"v2 评估地形 seed 应为 {expected_seed}。")
    if args_cli.split == "calibration":
        if args_cli.direction_protocol_version == "v2.2":
            parser.error("v2.2 复用已冻结的 v2.1 B_ref，禁止重新 calibration。")
        if legacy or variant != "directional" or method != "task_only":
            parser.error("calibration 只允许 v2 E2 任务型模型。")
        if args_cli.ppo_seed not in PPO_SEEDS[f"{args_cli.stage}_budget"]:
            parser.error("calibration PPO seed 不属于 v2 冻结集合。")
    elif args_cli.direction_protocol_version == "v2.2":
        if args_cli.stage != "stage1" or not is_v2_2_evaluation_target(method, terrain, args_cli.ppo_seed):
            parser.error("holdout 模型不属于 v2.2 冻结45模型评估矩阵。")
    elif args_cli.ppo_seed not in PPO_SEEDS[f"{args_cli.stage}_formal"]:
        parser.error("holdout PPO seed 不属于 v2.1 冻结集合。")


variant, method, terrain, legacy = _task_parts()
checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
if not checkpoint.is_file() or checkpoint.name != "model_2999.pt":
    parser.error("v2 正式 calibration/holdout 只接受最终 model_2999.pt。")
_validate_protocol(variant, method, terrain, legacy)
v2_references, v2_primary_reference, v2_reference_sha256 = _load_v2_reference(
    variant, method, terrain
)
legacy_training_budget, legacy_reference_sha256 = _load_legacy_training_budget(method, terrain) if legacy else (None, None)
authorization_sha256 = _validate_authorization(legacy)
output_root = Path(args_cli.output_root).expanduser().resolve()
batch_terrain_seed = (
    v2_3_evaluation_batch_seed(args_cli.split, args_cli.batch_index)
    if args_cli.direction_protocol_version == "v2.3"
    else evaluation_batch_seed(args_cli.terrain_seed, args_cli.batch_index)
)
checkpoint_source_protocol = (
    PROTOCOL_VERSION if "gpt_direction_v2_1_" in checkpoint.parent.name else ACTIVE_PROTOCOL_VERSION
)


def _load_v2_3_manifest() -> tuple[dict[str, dict[str, object]], str | None]:
    if args_cli.direction_protocol_version != "v2.3":
        if args_cli.manifest is not None:
            parser.error("--manifest 只用于 v2.3 Mixed。")
        return {}, None
    if args_cli.manifest is None:
        parser.error("v2.3 calibration/holdout 必须提供预先冻结的 manifest。")
    path = Path(args_cli.manifest).expanduser().resolve()
    if not path.is_file():
        parser.error(f"v2.3 manifest 不存在：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("冻结状态") != "已冻结"
        or payload.get("manifest版本") != V2_3_MANIFEST_VERSION
        or payload.get("协议版本") != V2_3_PROTOCOL_VERSION
        or payload.get("数据拆分") != args_cli.split
        or int(payload.get("回合数", -1)) != EVAL_EPISODES
    ):
        parser.error("v2.3 manifest 的状态、版本、拆分或回合数错误。")
    episodes = payload.get("逐回合", [])
    indexed = {str(item.get("episode_id")): item for item in episodes}
    if len(indexed) != EVAL_EPISODES:
        parser.error("v2.3 manifest episode ID 不唯一或不完整。")
    return indexed, _sha256(path)


v2_3_manifest_rows, v2_3_manifest_sha256 = _load_v2_3_manifest()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import importlib.metadata
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import load_cfg_from_registry

import pace_eco_lab  # noqa: F401
from pace_eco_lab.configs.multi_terrain_env_cfg import configure_terrain20s_evaluation
from pace_eco_lab.directional_metrics import directional_displacement_metrics, directional_success
from pace_eco_lab.envs.terrain20s_env import metadata_labels
from pace_eco_lab.reproducibility import validate_evaluation_checkpoint, verify_dependency_versions
from pace_eco_lab.terrains import curriculum_difficulty_table
from scripts.pace_eco.eval_metrics import FOOT_BODY_NAMES, CoordinationAccumulator


PROGRESS_INTERVAL_STEPS = 100
PROGRESS_INTERVAL_S = 30.0
STALL_TRACEBACK_S = 300.0


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _sample_std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def _mean_ci95(values: list[float]) -> list[float] | None:
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    half = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return [mean - half, mean + half]


def _staging_dir() -> Path:
    return (
        output_root
        / args_cli.stage
        / args_cli.split
        / f"gpt_方向批次暂存_{variant}_{terrain}_{method}_seed{args_cli.ppo_seed}_{args_cli.batch_group}"
    )


def _metadata() -> dict[str, object]:
    return {
        "协议版本": ACTIVE_PROTOCOL_VERSION,
        "评估指标版本": ACTIVE_METRIC_VERSION,
        "来源协议版本": LEGACY_PROTOCOL_VERSION if legacy else checkpoint_source_protocol,
        "边界审计版本": FOOT_BOUNDARY_AUDIT_VERSION,
        "阶段": args_cli.stage,
        "数据拆分": args_cli.split,
        "任务": args_cli.task,
        "实验变体": variant,
        "方法": method,
        "地形": terrain,
        "PPO_seed": args_cli.ppo_seed,
        "地形基准seed": args_cli.terrain_seed,
        "评估批次组": args_cli.batch_group,
        "检查点": str(checkpoint),
        "检查点SHA256": _sha256(checkpoint),
        "v2.3_manifest_SHA256": v2_3_manifest_sha256,
    }


def _write_staging(rows: list[dict[str, object]], record: dict[str, object]) -> None:
    directory = _staging_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"gpt-方向评估批次-{args_cli.batch_index}.json"
    if path.exists():
        raise FileExistsError(f"拒绝覆盖评估批次：{path}")
    payload = {
        **_metadata(),
        "地形批次seed": batch_terrain_seed,
        "评估批次": args_cli.batch_index,
        "训练复现记录": record,
        "逐回合": rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args_cli.direction_protocol_version == "v2.3":
        csv_path = directory / f"gpt-v2.3-Mixed-{args_cli.split}-batch-{args_cli.batch_index}.csv"
        if csv_path.exists():
            raise FileExistsError(f"拒绝覆盖 v2.3 批次CSV：{csv_path}")
        with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"[PACE-v2] 批次已写入：{path}", flush=True)


def _load_staged_rows() -> list[dict[str, object]]:
    combined: list[dict[str, object]] = []
    base_metadata = _metadata()
    for batch_index in range(EVAL_BATCHES):
        path = _staging_dir() / f"gpt-方向评估批次-{batch_index}.json"
        if not path.is_file():
            raise FileNotFoundError(f"评估批次未齐：{path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            **base_metadata,
            "地形批次seed": (
                v2_3_evaluation_batch_seed(args_cli.split, batch_index)
                if args_cli.direction_protocol_version == "v2.3"
                else evaluation_batch_seed(args_cli.terrain_seed, batch_index)
            ),
            "评估批次": batch_index,
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError(f"评估批次元数据不一致：{path}")
        rows = payload.get("逐回合", [])
        if len(rows) != EVAL_NUM_ENVS:
            raise ValueError(f"批次回合数错误：{path}")
        expected_ids = set(
            range(evaluation_batch_offset(batch_index), evaluation_batch_offset(batch_index) + EVAL_NUM_ENVS)
        )
        if {int(row["回合"]) for row in rows} != expected_ids:
            raise ValueError(f"批次全局回合编号错误：{path}")
        combined.extend(rows)
    combined.sort(key=lambda row: int(row["回合"]))
    if len(combined) != EVAL_EPISODES or {int(row["回合"]) for row in combined} != set(range(EVAL_EPISODES)):
        raise RuntimeError("四个批次没有形成完整且不重复的 200 回合。")
    return combined


def _group_summary(items: list[dict[str, object]], budget_j: float | None) -> dict[str, object]:
    energies = [float(row["回合能耗_J"]) for row in items]
    direction_ok = [row for row in items if bool(row["方向穿越成功"])]
    success_energies = [float(row["回合能耗_J"]) for row in direction_ok]
    table2_fields = (
        "稳态平均机身前向速度_m_s",
        "世界速度跟踪RMSE_m_s",
        "机身横向速度RMS_m_s",
        "机身垂向速度RMS_m_s",
        "机身横滚俯仰角速度RMS_rad_s",
        "动作变化RMS_归一化动作",
        "三步窗触地足速均值_m_s",
        "支撑相占比极差",
        "落足频率变异系数",
    )
    summary = {
        "回合数": len(items),
        "生存成功率": sum(bool(row["生存成功"]) for row in items) / len(items),
        "方向穿越成功率": len(direction_ok) / len(items),
        "B80联合合格率": (
            sum(bool(row["B80联合合格"]) for row in items) / len(items)
            if budget_j is not None
            else None
        ),
        "平均方向净进度_m": _mean([float(row["方向净进度_m"]) for row in items]),
        "平均最大横轨偏离_m": _mean([float(row["最大横轨偏离_m"]) for row in items]),
        "平均路径效率": _mean([float(row["路径效率"]) for row in items if row["路径效率"] is not None]),
        "平均世界速度跟踪RMSE_m_s": _mean([float(row["世界速度跟踪RMSE_m_s"]) for row in items]),
        "平均机身航向RMSE_rad": _mean([float(row["机身航向RMSE_rad"]) for row in items]),
        "全部回合平均能耗_J": _mean(energies),
        "全部回合能耗样本标准差_J": _sample_std(energies),
        "全部回合平均能耗95%置信区间_J": _mean_ci95(energies),
        "超B80率": (
            sum(float(row["回合能耗_J"]) > budget_j for row in items) / len(items)
            if budget_j is not None
            else None
        ),
        "方向成功回合平均能耗_J": _mean([float(row["回合能耗_J"]) for row in direction_ok]),
        "方向成功回合平均单位方向进度能耗_J_m": _mean(
            [float(row["单位方向进度能耗_J_m"]) for row in direction_ok]
        ),
        "成功条件B80合格率": (
            sum(energy <= budget_j for energy in success_energies) / len(success_energies)
            if budget_j is not None and success_energies else None
        ),
        "归一化超预算幅度": (
            _mean([max(energy / budget_j - 1.0, 0.0) for energy in success_energies])
            if budget_j is not None and success_energies else None
        ),
    }
    summary["表2方向成功回合数"] = len(direction_ok)
    for field in table2_fields:
        summary[f"方向成功回合平均{field}"] = _mean(
            [float(row[field]) for row in direction_ok if row.get(field) is not None]
        )
    return summary


def _write_outputs(rows: list[dict[str, object]], record: dict[str, object], budget_j: float | None) -> Path:
    result_dir = (
        output_root
        / args_cli.stage
        / args_cli.split
        / f"gpt_方向评估_{variant}_{terrain}_{method}_seed{args_cli.ppo_seed}_{args_cli.batch_group}"
    )
    if result_dir.exists():
        raise FileExistsError(f"拒绝覆盖结果目录：{result_dir}")
    result_dir.mkdir(parents=True)
    csv_path = result_dir / "gpt-方向条件逐回合结果.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (result_dir / "gpt-方向条件逐回合结果.json").write_text(
        json.dumps({"协议版本": ACTIVE_PROTOCOL_VERSION, "逐回合": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['地形类别']}|{row['方向']}|{row['难度']}"] .append(row)
    state_counts = Counter(int(row["固定初始状态编号"]) for row in rows)
    summary = {
        **_metadata(),
        "实验变体名称": "E0冻结v1交叉评估" if legacy else VARIANT_LABELS[variant],
        "方法名称": ACTIVE_METHOD_LABELS[method],
        "地形名称": TERRAIN_LABELS[terrain],
        "期望世界方向": list(DESIRED_DIRECTION_W),
        "目标速度_m_s": TARGET_SPEED_M_S_V2,
        "方向成功最低进度_m": MIN_DIRECTIONAL_PROGRESS_M,
        "方向成功最大横轨偏离_m": MAX_CROSS_TRACK_DEVIATION_M,
        "v2_B_ref文件SHA256": v2_reference_sha256,
        "E0训练B_ref文件SHA256": legacy_reference_sha256,
        "holdout授权SHA256": authorization_sha256,
        "PACE主预算B80_J": budget_j,
        "E0训练预算B80_J": legacy_training_budget,
        "回合数": len(rows),
        "各固定状态回合数": dict(sorted(state_counts.items())),
        **_group_summary(rows, budget_j),
        "方向穿越成功定义": (
            "完整20秒、无非法终止、相对回合起点沿指定方向净进度>=16m，且全程最大横轨偏离<=3m"
        ),
        "历史速度成功定义": "完整20秒、无非法终止且5秒预热后机身前向平均速度位于[0.8,1.2]m/s",
        "航向指标解释": "v2.1/v2.2 不约束机身朝向；航向误差仅为诊断，不属于方向穿越成功定义。",
        "E0预算解释": (
            "E0 的 B80联合合格仅为在 v2 B80 下的交叉协议诊断；E0 训练仍使用 E0训练预算B80_J。"
            if legacy
            else None
        ),
        "按地形方向难度": {
            key: _group_summary(items, budget_j) for key, items in sorted(grouped.items())
        },
        "训练复现记录": record,
    }
    (result_dir / "gpt-方向条件评估摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[PACE-v2] 正式结果已写入：{result_dir}", flush=True)
    return result_dir


def main() -> None:
    verify_dependency_versions()
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    device = args_cli.device or "cuda:0"
    env_cfg.sim.device = device
    env_cfg.seed = args_cli.ppo_seed
    env_cfg.pace_publish_eval_state = True
    agent_cfg.seed = args_cli.ppo_seed
    agent_cfg.device = device
    v2_budget = 0.8 * v2_primary_reference if v2_primary_reference is not None else None
    validation_budget = legacy_training_budget if legacy else (v2_budget if method == "eco" else None)
    if method == "eco":
        if validation_budget is None:
            raise RuntimeError("ECO 检查点缺少训练预算。")
        agent_cfg.algorithm.energy_budget_j = validation_budget
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
    record = validate_evaluation_checkpoint(
        checkpoint.parent,
        task=args_cli.task,
        energy_budget_j=validation_budget if method == "eco" else None,
        require_current_implementation=False,
    )
    if int(record.get("seed", -1)) != args_cli.ppo_seed:
        raise ValueError("检查点 PPO seed 与评估参数不一致。")
    if args_cli.direction_protocol_version == "v2.3" and not bool(record.get("git_worktree_clean")):
        raise ValueError("v2.3 正式 calibration/holdout 拒绝训练时工作树不干净的检查点。")
    if method == "fixed_weight":
        environment_path = checkpoint.parent / "gpt_环境配置.json"
        if not environment_path.is_file():
            raise FileNotFoundError("v2.2 固定权重检查点缺少 gpt_环境配置.json。")
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        trained_weight = float(environment.get("rewards", {}).get("energy", {}).get("weight", 0.0))
        if abs(trained_weight - FIXED_ENERGY_REWARD_WEIGHT) > 1.0e-12:
            raise ValueError("v2.2 固定权重检查点没有使用冻结的 W100 系数。")
    if args_cli.direction_protocol_version == "v2.1" and not bool(record.get("git_worktree_clean")):
        raise ValueError("v2.1 正式评估拒绝训练时工作树不干净的检查点。")
    state_set = (
        MULTI_TERRAIN_CALIBRATION_STATE_SET
        if args_cli.split == "calibration"
        else MULTI_TERRAIN_HOLDOUT_STATE_SET
    )
    state_set_sha256 = evaluation_state_definition_sha256(state_set)
    state_count = evaluation_state_count(state_set)
    env_cfg = configure_terrain20s_evaluation(
        env_cfg, batch_terrain_seed, state_set, args_cli.batch_index
    )
    env_cfg.pace_publish_eval_state = True
    env_cfg.log_dir = str(checkpoint.parent)
    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw = wrapped.unwrapped
    terrain_manager = raw.scene.terrain
    env_ids = torch.arange(EVAL_NUM_ENVS, device=raw.device)
    levels = torch.remainder(env_ids, EVAL_TERRAIN_ROWS)
    types = torch.div(env_ids, EVAL_TERRAIN_ROWS, rounding_mode="floor")
    terrain_manager.terrain_levels.copy_(levels)
    terrain_manager.terrain_types.copy_(types)
    terrain_manager.env_origins.copy_(terrain_manager.terrain_origins[levels, types])
    raw.reset()
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint), map_location=agent_cfg.device)
    policy = runner.get_inference_policy(device=wrapped.device)
    observations = wrapped.get_observations()
    robot = raw.scene["robot"]
    contact_sensor = raw.scene.sensors["contact_forces"]
    foot_asset_ids, _ = robot.find_bodies(FOOT_BODY_NAMES, preserve_order=True)
    foot_sensor_ids, _ = contact_sensor.find_bodies(FOOT_BODY_NAMES, preserve_order=True)
    coordination = CoordinationAccumulator.create(
        raw.num_envs,
        raw.action_manager.total_action_dim,
        device=raw.device,
        dtype=robot.data.root_lin_vel_b.dtype,
    )
    components = {
        name: torch.zeros(raw.num_envs, device=raw.device)
        for name in ("electrical", "mechanical", "potential")
    }
    scalar = lambda: torch.zeros(raw.num_envs, device=raw.device)
    body_forward_sum = scalar()
    world_velocity_error_sq_sum = scalar()
    heading_error_sq_sum = scalar()
    heading_error_abs_sum = scalar()
    sample_count = scalar()
    max_cross_track = scalar()
    recorded = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
    start_xy = robot.data.root_pos_w[:, :2].clone()
    direction_w = torch.tensor(DESIRED_DIRECTION_W, device=raw.device, dtype=start_xy.dtype)
    perpendicular_w = torch.stack((-direction_w[1], direction_w[0]))
    target_velocity_w = TARGET_SPEED_M_S_V2 * direction_w
    target_heading = math.atan2(DESIRED_DIRECTION_W[1], DESIRED_DIRECTION_W[0])
    rows: list[dict[str, object]] = []
    difficulties = curriculum_difficulty_table(
        batch_terrain_seed, EVAL_TERRAIN_ROWS, EVAL_TERRAIN_COLS, (0.10, 0.90)
    )
    warmup_steps = round(args_cli.warmup_s / raw.step_dt)
    started = monotonic()
    last_progress = started
    steps = 0
    faulthandler.dump_traceback_later(STALL_TRACEBACK_S, repeat=False)
    try:
        while len(rows) < EVAL_NUM_ENVS and simulation_app.is_running():
            steps += 1
            active = ~recorded
            steady = (raw.episode_length_buf >= warmup_steps) & active
            mask = steady.to(start_xy.dtype)
            displacement = robot.data.root_pos_w[:, :2] - start_xy
            cross_track = torch.abs(torch.sum(displacement * perpendicular_w, dim=-1))
            max_cross_track.copy_(torch.maximum(max_cross_track, cross_track))
            velocity_error = robot.data.root_lin_vel_w[:, :2] - target_velocity_w
            heading_error = torch.atan2(
                torch.sin(target_heading - robot.data.heading_w),
                torch.cos(target_heading - robot.data.heading_w),
            )
            body_forward_sum += robot.data.root_lin_vel_b[:, 0] * mask
            world_velocity_error_sq_sum += velocity_error.square().sum(dim=-1) * mask
            heading_error_sq_sum += heading_error.square() * mask
            heading_error_abs_sum += heading_error.abs() * mask
            sample_count += mask
            forces = contact_sensor.data.net_forces_w[:, foot_sensor_ids]
            coordination.update_state(
                lateral_velocity=robot.data.root_lin_vel_b[:, 1],
                vertical_velocity=robot.data.root_lin_vel_w[:, 2],
                roll_pitch_angular_velocity=robot.data.root_ang_vel_b[:, :2],
                foot_contact=torch.linalg.vector_norm(forces, dim=-1)
                > float(contact_sensor.cfg.force_threshold),
                foot_speed=torch.linalg.vector_norm(
                    robot.data.body_lin_vel_w[:, foot_asset_ids], dim=-1
                ),
                touchdown=contact_sensor.compute_first_contact(raw.step_dt)[:, foot_sensor_ids],
                steady_mask=steady,
            )
            with torch.inference_mode():
                actions = policy(observations)
                evaluated_actions = (
                    torch.clamp(actions, -wrapped.clip_actions, wrapped.clip_actions)
                    if wrapped.clip_actions is not None
                    else actions
                )
                coordination.update_action(evaluated_actions, steady)
                observations, _, dones, extras = wrapped.step(actions)
                policy.reset(dones)
            for name in components:
                components[name] += extras["pace_energy_components"][name] * active
            completed_ids = extras["pace_energy_episode_mask"].nonzero(as_tuple=False).squeeze(-1)
            final_xy_snapshot = extras.get("pace_eval_root_pos_w")
            if final_xy_snapshot is None:
                raise RuntimeError("评估环境没有发布自动 reset 前根位置快照。")
            for env_id in completed_ids.tolist():
                if recorded[env_id]:
                    continue
                recorded[env_id] = True
                final_displacement = final_xy_snapshot[env_id, :2] - start_xy[env_id]
                progress_tensor, signed_cross_tensor = directional_displacement_metrics(
                    final_displacement,
                    direction_w,
                )
                final_cross = abs(float(torch.sum(final_displacement * perpendicular_w)))
                max_cross = max(float(max_cross_track[env_id]), final_cross)
                category, terrain_direction, level_label = metadata_labels(
                    int(extras["pace_terrain20s_category_code"][env_id]),
                    int(extras["pace_terrain20s_direction_code"][env_id]),
                    int(extras["pace_terrain20s_terrain_level"][env_id]),
                    EVAL_TERRAIN_ROWS,
                )
                level = int(extras["pace_terrain20s_terrain_level"][env_id])
                terrain_type = int(extras["pace_terrain20s_terrain_type"][env_id])
                timed_out = bool(extras["pace_terrain20s_timeout"][env_id])
                illegal = bool(extras["pace_terrain20s_base_contact"][env_id])
                survival = timed_out and not illegal
                direction_ok = bool(directional_success(
                    completed_20s=torch.tensor(timed_out, device=raw.device),
                    illegal_termination=torch.tensor(illegal, device=raw.device),
                    directional_progress_m=progress_tensor,
                    max_cross_track_m=torch.tensor(max_cross, device=raw.device),
                    minimum_progress_m=MIN_DIRECTIONAL_PROGRESS_M,
                    maximum_cross_track_m=MAX_CROSS_TRACK_DEVIATION_M,
                ))
                count = max(float(sample_count[env_id]), 1.0)
                mean_body_speed = float(body_forward_sum[env_id]) / count
                legacy_success = (
                    survival
                    and SUCCESS_SPEED_RANGE_M_S[0] <= mean_body_speed <= SUCCESS_SPEED_RANGE_M_S[1]
                )
                path_m = float(extras["pace_terrain20s_path_episode_m"][env_id])
                progress_m = float(progress_tensor)
                energy = float(extras["pace_energy_episode"][env_id])
                category_reference = v2_references.get(category)
                global_episode_id = evaluation_global_id(args_cli.batch_index, env_id)
                local_instance_id = level * EVAL_TERRAIN_COLS + terrain_type
                row: dict[str, object] = {
                    "协议版本": ACTIVE_PROTOCOL_VERSION,
                    "评估指标版本": ACTIVE_METRIC_VERSION,
                    "来源协议版本": LEGACY_PROTOCOL_VERSION if legacy else checkpoint_source_protocol,
                    "边界审计版本": FOOT_BOUNDARY_AUDIT_VERSION,
                    "阶段": args_cli.stage,
                    "数据拆分": args_cli.split,
                    "任务ID": args_cli.task,
                    "实验变体": variant,
                    "方法": method,
                    "回合": global_episode_id,
                    "episode_id": f"{args_cli.split}-{global_episode_id:04d}",
                    "batch_id": f"{args_cli.split}-batch-{args_cli.batch_index}",
                    "环境编号": env_id,
                    "评估批次": args_cli.batch_index,
                    "评估批次组": args_cli.batch_group,
                    "固定初始状态编号": global_episode_id % state_count,
                    "PPO_seed": args_cli.ppo_seed,
                    "地形seed": args_cli.terrain_seed,
                    "地形批次seed": batch_terrain_seed,
                    "评估初始状态集": state_set,
                    "评估初始状态集SHA256": state_set_sha256,
                    "地形实例编号": evaluation_global_id(args_cli.batch_index, local_instance_id),
                    "地形类别": category,
                    "方向": terrain_direction,
                    "方向子任务": v2_3_subtask_name(category, terrain_direction),
                    "难度": level_label,
                    "难度层编号": level,
                    "难度连续值": float(difficulties[level, terrain_type]),
                    "方向指令世界坐标": list(DESIRED_DIRECTION_W),
                    "目标速度_m_s": TARGET_SPEED_M_S_V2,
                    "完整20秒": timed_out,
                    "非法终止": illegal,
                    "生存成功": survival,
                    "v1历史机身速度成功": legacy_success,
                    "方向穿越成功": direction_ok,
                    "方向净进度_m": progress_m,
                    "最终横轨偏离_m": abs(float(signed_cross_tensor)),
                    "最大横轨偏离_m": max_cross,
                    "水平实际路径_m": path_m,
                    "路径效率": progress_m / path_m if path_m > 0.0 else None,
                    "稳态平均机身前向速度_m_s": mean_body_speed,
                    "世界速度跟踪RMSE_m_s": math.sqrt(
                        float(world_velocity_error_sq_sum[env_id]) / count
                    ),
                    "机身航向平均绝对误差_rad": float(heading_error_abs_sum[env_id]) / count,
                    "机身航向RMSE_rad": math.sqrt(float(heading_error_sq_sum[env_id]) / count),
                    "回合能耗_J": energy,
                    "电气能耗_J": float(components["electrical"][env_id]),
                    "机械能耗_J": float(components["mechanical"][env_id]),
                    "势能能耗_J": float(components["potential"][env_id]),
                    "单位方向进度能耗_J_m": energy / progress_m if progress_m > 0.0 else None,
                    "单位实际路径能耗_J_m": energy / path_m if path_m > 0.0 else None,
                    "v2归一化能耗_E除以B_ref": energy / category_reference if category_reference else None,
                    "PACE主预算B80_J": v2_budget,
                    "超过B80": energy > v2_budget if v2_budget is not None else None,
                    "B80联合合格": direction_ok and energy <= v2_budget if v2_budget is not None else None,
                    "接触足越过真实地形边缘": bool(
                        extras["pace_terrain20s_contact_foot_boundary_crossed"][env_id]
                    ),
                    "摆动足越过真实地形边缘": bool(
                        extras["pace_terrain20s_swing_foot_boundary_crossed"][env_id]
                    ),
                    "机身超过保守路线预警线": bool(
                        extras["pace_terrain20s_base_corridor_warning"][env_id]
                    ),
                    "进入四足真实位置邻边审计带": bool(
                        extras["pace_terrain20s_foot_exact_audit_sampled"][env_id]
                    ),
                    **coordination.episode_metrics(env_id, raw.step_dt),
                }
                rows.append(row)
                if len(rows) % 10 == 0 or len(rows) == EVAL_NUM_ENVS:
                    print(f"[PACE-v2] 本批进度 {len(rows)}/{EVAL_NUM_ENVS}", flush=True)
            now = monotonic()
            if steps % PROGRESS_INTERVAL_STEPS == 0 or now - last_progress >= PROGRESS_INTERVAL_S:
                elapsed = now - started
                print(
                    f"[PACE-v2] 心跳：控制步={steps}，已记录={len(rows)}/{EVAL_NUM_ENVS}，"
                    f"耗时={elapsed:.1f}s，步率={steps / elapsed:.2f}步/s。",
                    flush=True,
                )
                last_progress = now
                faulthandler.cancel_dump_traceback_later()
                faulthandler.dump_traceback_later(STALL_TRACEBACK_S, repeat=False)
    finally:
        faulthandler.cancel_dump_traceback_later()
    if len(rows) != EVAL_NUM_ENVS:
        wrapped.close()
        raise RuntimeError(f"本批评估未收齐：{len(rows)}/{EVAL_NUM_ENVS}")
    if args_cli.direction_protocol_version == "v2.3":
        for row in rows:
            expected = v2_3_manifest_rows.get(str(row["episode_id"]))
            actual = {
                "batch_index": int(row["评估批次"]),
                "batch_env_number": int(row["环境编号"]),
                "batch_id": row["batch_id"],
                "terrain_category": row["地形类别"],
                "direction": row["方向"],
                "subtask": row["方向子任务"],
                "difficulty_level": int(row["难度层编号"]),
                "terrain_seed": int(row["地形seed"]),
                "batch_seed": int(row["地形批次seed"]),
                "initial_state_number": int(row["固定初始状态编号"]),
                "protocol_version": row["协议版本"],
                "initial_state_set": row["评估初始状态集"],
                "initial_state_set_sha256": row["评估初始状态集SHA256"],
                "direction_command_w": row["方向指令世界坐标"],
                "target_speed_m_s": float(row["目标速度_m_s"]),
            }
            if expected is None or any(expected[key] != value for key, value in actual.items()):
                wrapped.close()
                raise RuntimeError(f"评估结果未严格匹配 manifest：{row['episode_id']}/{actual}")
            if abs(float(expected["difficulty"]) - float(row["难度连续值"])) > 1.0e-12:
                wrapped.close()
                raise RuntimeError(f"评估难度未匹配 manifest：{row['episode_id']}")
    if args_cli.stage == "stage1" and any(row["地形类别"] != terrain for row in rows):
        wrapped.close()
        raise RuntimeError("评估结果包含模型对应类别以外的地形，拒绝写出正式批次。")
    if any(bool(row["接触足越过真实地形边缘"]) for row in rows):
        wrapped.close()
        raise RuntimeError("检测到接触足越过真实地形边缘，拒绝写出 v2 正式结果。")
    _write_staging(rows, record)
    if args_cli.batch_index == EVAL_BATCHES - 1:
        all_rows = _load_staged_rows()
        counts = Counter(int(row["固定初始状态编号"]) for row in all_rows)
        if set(counts) != set(range(state_count)) or set(counts.values()) != {
            EVAL_EPISODES // state_count
        }:
            wrapped.close()
            raise RuntimeError(f"固定初始状态未等量采样：{dict(counts)}")
        _write_outputs(all_rows, record, v2_budget)
    wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
