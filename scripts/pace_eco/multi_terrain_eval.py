#!/usr/bin/env python3
"""按历史 Flat 的固定20秒成功定义分批评估 Terrain20sWide 策略。"""

from __future__ import annotations

import argparse
import csv
import faulthandler
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from time import monotonic

from isaaclab.app import AppLauncher

from pace_eco_lab.multi_terrain_protocol import (
    EVALUATION_WARMUP_S,
    EVAL_BATCHES,
    EVAL_EPISODES,
    EVAL_NUM_ENVS,
    EVAL_TERRAIN_COLS,
    EVAL_TERRAIN_ROWS,
    FOOT_BOUNDARY_AUDIT_VERSION,
    METHOD_LABELS,
    PACE_PAPER_FIXED_WEIGHT,
    PACE_PAPER_FIXED_WEIGHT_LABEL,
    PPO_SEEDS,
    PROTOCOL_VERSION,
    SUCCESS_SPEED_RANGE_M_S,
    TASK_IDS,
    TERRAIN_LABELS,
    TERRAIN_LENGTH_M,
    TERRAIN_ORIGIN_X_M,
    TERRAIN_WIDTH_M,
    evaluation_batch_offset,
    evaluation_batch_seed,
    evaluation_global_id,
    terrain_seed,
)


parser = argparse.ArgumentParser(description="PACE-ECO Terrain20s 多地形冻结评估。")
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--ppo_seed", required=True, type=int)
parser.add_argument("--terrain_seed", required=True, type=int)
parser.add_argument("--batch_index", required=True, type=int)
parser.add_argument("--batch_group", required=True)
parser.add_argument("--stage", required=True, choices=("stage1", "stage2"))
parser.add_argument("--split", required=True, choices=("calibration", "holdout"))
parser.add_argument("--energy_reference_json", default=None)
parser.add_argument("--output_root", required=True)
parser.add_argument("--holdout_authorization", default=None)
parser.add_argument("--warmup_s", type=float, default=EVALUATION_WARMUP_S)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _task_parts() -> tuple[str, str]:
    inverse = {task_id: pair for pair, task_id in TASK_IDS.items()}
    if args_cli.task not in inverse:
        parser.error("只允许独立 Terrain20sWide 多地形任务 ID。")
    return inverse[args_cli.task]


def _validate_protocol_before_simulation(method: str, terrain: str) -> str | None:
    if abs(args_cli.warmup_s - EVALUATION_WARMUP_S) > 1.0e-12:
        parser.error(f"Terrain20sWide 评估预热时间冻结为 {EVALUATION_WARMUP_S} s，与历史 Flat 相同。")
    if not 0 <= args_cli.batch_index < EVAL_BATCHES:
        parser.error(f"评估批次必须位于 [0, {EVAL_BATCHES - 1}]。")
    if re.fullmatch(r"[0-9]{8}_[0-9]{6}", args_cli.batch_group) is None:
        parser.error("评估批次组必须是 YYYYMMDD_HHMMSS 格式，拒绝不安全路径。")
    if args_cli.stage == "stage1" and terrain == "mixed":
        parser.error("阶段一禁止混合任务。")
    if args_cli.stage == "stage2" and terrain != "mixed":
        parser.error("阶段二只评估从头训练的混合任务。")
    expected_terrain = "mixed" if args_cli.stage == "stage2" else terrain
    expected_seed = terrain_seed(f"{args_cli.stage}_{args_cli.split}", expected_terrain)
    if args_cli.terrain_seed != expected_seed:
        parser.error(f"评估地形 seed 应为 {expected_seed}，实际为 {args_cli.terrain_seed}。")
    if args_cli.split == "calibration":
        if method != "task_only":
            parser.error("calibration 只用于任务型 PPO 的 B_ref 标定。")
        if args_cli.ppo_seed not in PPO_SEEDS[f"{args_cli.stage}_budget"]:
            parser.error("PPO seed 不属于 calibration 冻结集合。")
        if args_cli.holdout_authorization is not None:
            parser.error("calibration 禁止提供 holdout 授权。")
        return None
    if args_cli.ppo_seed not in PPO_SEEDS[f"{args_cli.stage}_formal"]:
        parser.error("holdout 只接受正式 PPO seed。")
    if args_cli.holdout_authorization is None:
        parser.error("holdout 必须提供预先冻结的模型授权清单。")
    authorization = Path(args_cli.holdout_authorization).expanduser().resolve()
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    if not authorization.is_file() or not checkpoint.is_file():
        parser.error("holdout 授权或检查点不存在。")
    data = json.loads(authorization.read_text(encoding="utf-8"))
    if (
        data.get("冻结状态") != "holdout已授权"
        or data.get("协议版本") != PROTOCOL_VERSION
        or data.get("边界审计版本") != FOOT_BOUNDARY_AUDIT_VERSION
        or data.get("阶段") != args_cli.stage
    ):
        parser.error("holdout 授权状态、协议版本或阶段不匹配。")
    expected_count = 75 if args_cli.stage == "stage1" else 15
    models = data.get("模型", [])
    if len(models) != expected_count:
        parser.error(f"holdout 授权应包含 {expected_count} 个模型，实际为 {len(models)}。")
    matches = [
        item
        for item in models
        if Path(item["检查点"]).resolve() == checkpoint
        and item["任务"] == args_cli.task
        and int(item["PPO_seed"]) == args_cli.ppo_seed
    ]
    if len(matches) != 1 or matches[0]["检查点SHA256"] != _sha256(checkpoint):
        parser.error("当前模型不在冻结授权内，或检查点哈希不一致。")
    reference_path = Path(args_cli.energy_reference_json or "").expanduser().resolve()
    if (
        not reference_path.is_file()
        or data.get("证据文件SHA256", {}).get("B_ref") != _sha256(reference_path)
    ):
        parser.error("当前 B_ref 文件与 holdout 授权证据不一致。")
    return _sha256(authorization)


def _load_references(
    method: str,
    terrain: str,
) -> tuple[dict[str, float], float | None, str | None]:
    if args_cli.split == "calibration" and method == "task_only":
        if args_cli.energy_reference_json is not None:
            parser.error("B_ref calibration 禁止预先提供能耗参考，避免循环定义。")
        return {}, None, None
    if args_cli.energy_reference_json is None:
        parser.error("除任务型 B_ref calibration 外，评估必须提供冻结 B_ref。")
    path = Path(args_cli.energy_reference_json).expanduser().resolve()
    if not path.is_file():
        parser.error("B_ref 冻结文件不存在。")
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        data.get("冻结状态") != "已冻结"
        or data.get("协议版本") != PROTOCOL_VERSION
        or data.get("边界审计版本") != FOOT_BOUNDARY_AUDIT_VERSION
        or data.get("阶段") != args_cli.stage
    ):
        parser.error("B_ref 文件未冻结、协议版本错误或阶段不匹配。")
    source = data.get("B_ref_J", {})
    required = ("flat", "rough", "stairs", "boxes", "slope") if terrain == "mixed" else (terrain,)
    references = {name: float(source.get(name, 0.0)) for name in required}
    if any(value <= 0.0 for value in references.values()):
        parser.error("B_ref 文件缺少当前评估所需的正值。")
    primary = (
        float(data.get("B_ref_mixed_J", 0.0))
        if args_cli.stage == "stage2"
        else references[terrain]
    )
    if primary <= 0.0:
        parser.error("B_ref 文件缺少主协议所需的正预算参考。")
    return references, primary, _sha256(path)


def _fixed_weight_metadata(method: str) -> tuple[str | None, float | None]:
    if method != "fixed_weight":
        return None, None
    path = Path(args_cli.checkpoint).expanduser().resolve().parent / "gpt_环境配置.json"
    if not path.is_file():
        parser.error("固定权重检查点目录缺少 gpt_环境配置.json。")
    value = float(
        json.loads(path.read_text(encoding="utf-8"))
        .get("rewards", {})
        .get("energy", {})
        .get("weight", 0.0)
    )
    if abs(value - PACE_PAPER_FIXED_WEIGHT) > 1.0e-12:
        parser.error(f"训练固定权重 {value} 不是冻结的 PACE 论文系数 {PACE_PAPER_FIXED_WEIGHT}。")
    return PACE_PAPER_FIXED_WEIGHT_LABEL, value


method, terrain = _task_parts()
checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
if not checkpoint.is_file() or checkpoint.name != "model_2999.pt":
    parser.error("Terrain20sWide calibration/holdout 只接受最终 model_2999.pt。")
authorization_sha256 = _validate_protocol_before_simulation(method, terrain)
energy_references, primary_reference, energy_reference_sha256 = _load_references(method, terrain)
fixed_weight_label, fixed_weight_value = _fixed_weight_metadata(method)
output_root = Path(args_cli.output_root).expanduser().resolve()
batch_terrain_seed = evaluation_batch_seed(args_cli.terrain_seed, args_cli.batch_index)

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
from pace_eco_lab.envs.terrain20s_env import metadata_labels
from pace_eco_lab.evaluation_states import (
    MULTI_TERRAIN_CALIBRATION_STATE_SET,
    MULTI_TERRAIN_HOLDOUT_STATE_SET,
    evaluation_state_count,
    evaluation_state_definition_sha256,
)
from pace_eco_lab.reproducibility import validate_evaluation_checkpoint, verify_dependency_versions
from pace_eco_lab.terrains import LongTerrainCfg, curriculum_difficulty_table, resolved_parameters
from pace_eco_lab.terrain_boundary import FOOT_CENTER_MAX_BASE_DISTANCE_M
from scripts.pace_eco.eval_metrics import FOOT_BODY_NAMES, CoordinationAccumulator


EVALUATION_PROGRESS_INTERVAL_STEPS = 100
EVALUATION_PROGRESS_INTERVAL_S = 30.0
EVALUATION_STALL_TRACEBACK_S = 300.0


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _finite_float(value: torch.Tensor) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def _batch_staging_dir() -> Path:
    return (
        output_root
        / args_cli.stage
        / args_cli.split
        / (
            f"gpt_批次暂存_{terrain}_{method}_seed{args_cli.ppo_seed}_"
            f"{args_cli.batch_group}"
        )
    )


def _write_batch_staging(
    rows: list[dict[str, object]], evaluation_record: dict[str, object]
) -> Path:
    staging_dir = _batch_staging_dir()
    staging_dir.mkdir(parents=True, exist_ok=True)
    path = staging_dir / f"gpt-评估批次-{args_cli.batch_index}.json"
    if path.exists():
        raise FileExistsError(f"拒绝覆盖评估批次：{path}")
    payload = {
        "协议版本": PROTOCOL_VERSION,
        "边界审计版本": FOOT_BOUNDARY_AUDIT_VERSION,
        "阶段": args_cli.stage,
        "数据拆分": args_cli.split,
        "任务": args_cli.task,
        "方法": method,
        "地形": terrain,
        "PPO_seed": args_cli.ppo_seed,
        "地形基准seed": args_cli.terrain_seed,
        "地形批次seed": batch_terrain_seed,
        "评估批次": args_cli.batch_index,
        "评估批次组": args_cli.batch_group,
        "检查点": str(checkpoint),
        "检查点SHA256": _sha256(checkpoint),
        "训练复现记录": evaluation_record,
        "逐回合": rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[PACE] Terrain20sWide 评估批次已写入：{path}", flush=True)
    return path


def _load_all_staged_rows() -> list[dict[str, object]]:
    combined: list[dict[str, object]] = []
    checkpoint_sha256 = _sha256(checkpoint)
    for batch_index in range(EVAL_BATCHES):
        path = _batch_staging_dir() / f"gpt-评估批次-{batch_index}.json"
        if not path.is_file():
            raise FileNotFoundError(f"评估批次未齐，缺少：{path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "协议版本": PROTOCOL_VERSION,
            "边界审计版本": FOOT_BOUNDARY_AUDIT_VERSION,
            "阶段": args_cli.stage,
            "数据拆分": args_cli.split,
            "任务": args_cli.task,
            "方法": method,
            "地形": terrain,
            "PPO_seed": args_cli.ppo_seed,
            "地形基准seed": args_cli.terrain_seed,
            "地形批次seed": evaluation_batch_seed(args_cli.terrain_seed, batch_index),
            "评估批次": batch_index,
            "评估批次组": args_cli.batch_group,
            "检查点": str(checkpoint),
            "检查点SHA256": checkpoint_sha256,
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError(f"评估批次元数据不一致：{path}")
        batch_rows = payload.get("逐回合", [])
        if len(batch_rows) != EVAL_NUM_ENVS:
            raise ValueError(f"评估批次回合数错误：{path}")
        batch_seed = evaluation_batch_seed(args_cli.terrain_seed, batch_index)
        expected_ids = set(
            range(evaluation_batch_offset(batch_index), evaluation_batch_offset(batch_index) + EVAL_NUM_ENVS)
        )
        if any(
            row.get("协议版本") != PROTOCOL_VERSION
            or row.get("边界审计版本") != FOOT_BOUNDARY_AUDIT_VERSION
            or row.get("阶段") != args_cli.stage
            or row.get("数据拆分") != args_cli.split
            or row.get("任务ID") != args_cli.task
            or int(row.get("PPO_seed", -1)) != args_cli.ppo_seed
            or int(row.get("评估批次", -1)) != batch_index
            or row.get("评估批次组") != args_cli.batch_group
            or int(row.get("地形seed", -1)) != args_cli.terrain_seed
            or int(row.get("地形批次seed", -1)) != batch_seed
            for row in batch_rows
        ):
            raise ValueError(f"评估批次逐回合元数据不一致：{path}")
        if {int(row["回合"]) for row in batch_rows} != expected_ids:
            raise ValueError(f"评估批次全局回合编号缺失或重复：{path}")
        if {int(row["地形实例编号"]) for row in batch_rows} != expected_ids:
            raise ValueError(f"评估批次全局地形实例编号缺失或重复：{path}")
        combined.extend(batch_rows)
    combined.sort(key=lambda row: int(row["回合"]))
    if len(combined) != EVAL_EPISODES:
        raise RuntimeError(f"评估合并未收齐：{len(combined)}/{EVAL_EPISODES}")
    if {int(row["回合"]) for row in combined} != set(range(EVAL_EPISODES)):
        raise RuntimeError("评估合并的全局回合编号缺失或重复。")
    if {int(row["地形实例编号"]) for row in combined} != set(range(EVAL_EPISODES)):
        raise RuntimeError("评估合并的地形实例编号缺失或重复。")
    return combined


def _write_outputs(rows: list[dict[str, object]], evaluation_record: dict[str, object]) -> Path:
    if len(rows) != EVAL_EPISODES:
        raise ValueError(f"正式结果必须合并 {EVAL_EPISODES} 回合，实际为 {len(rows)}。")
    result_dir = (
        output_root
        / args_cli.stage
        / args_cli.split
        / f"gpt_评估_{terrain}_{method}_seed{args_cli.ppo_seed}_{args_cli.batch_group}"
    )
    if result_dir.exists():
        raise FileExistsError(f"拒绝覆盖结果目录：{result_dir}")
    result_dir.mkdir(parents=True)
    with (result_dir / "gpt-多地形逐回合结果.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (result_dir / "gpt-多地形逐回合结果.json").write_text(
        json.dumps(
            {
                "协议版本": PROTOCOL_VERSION,
                "边界审计版本": FOOT_BOUNDARY_AUDIT_VERSION,
                "逐回合": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    successful = [row for row in rows if row["成功"]]
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['地形类别']}|{row['方向']}|{row['难度']}"] .append(row)
    group_summary = {}
    for key, items in grouped.items():
        ok = [item for item in items if item["成功"]]
        group_summary[key] = {
            "回合数": len(items),
            "成功率": sum(bool(item["成功"]) for item in items) / len(items),
            "联合合格率": (
                sum(bool(item["B80联合合格"]) for item in items) / len(items)
                if primary_reference is not None
                else None
            ),
            "成功回合平均能耗_J": _mean([float(item["回合能耗_J"]) for item in ok]),
            "成功回合平均单位前进距离能耗_J_m": _mean(
                [float(item["单位前进距离能耗_J_m"]) for item in ok if item["单位前进距离能耗_J_m"] is not None]
            ),
        }
    state_counts = Counter(int(row["固定初始状态编号"]) for row in rows)
    summary = {
        "协议版本": PROTOCOL_VERSION,
        "边界审计版本": FOOT_BOUNDARY_AUDIT_VERSION,
        "阶段": args_cli.stage,
        "数据拆分": args_cli.split,
        "任务": args_cli.task,
        "方法": METHOD_LABELS[method],
        "地形": TERRAIN_LABELS[terrain],
        "PPO_seed": args_cli.ppo_seed,
        "固定权重标签": fixed_weight_label,
        "固定权重系数": fixed_weight_value,
        "地形seed": args_cli.terrain_seed,
        "评估批次组": args_cli.batch_group,
        "评估批次数": EVAL_BATCHES,
        "每批环境数": EVAL_NUM_ENVS,
        "评估初始状态集": rows[0]["评估初始状态集"],
        "评估初始状态集SHA256": rows[0]["评估初始状态集SHA256"],
        "各固定状态回合数": dict(sorted(state_counts.items())),
        "检查点": str(checkpoint),
        "检查点SHA256": _sha256(checkpoint),
        "B_ref文件SHA256": energy_reference_sha256,
        "holdout授权SHA256": authorization_sha256,
        "回合数": len(rows),
        "成功率": len(successful) / len(rows),
        "联合合格率": (
            sum(bool(row["B80联合合格"]) for row in rows) / len(rows)
            if primary_reference is not None
            else None
        ),
        "成功回合平均能耗_J": _mean([float(row["回合能耗_J"]) for row in successful]),
        "成功回合平均单位前进距离能耗_J_m": _mean(
            [float(row["单位前进距离能耗_J_m"]) for row in successful if row["单位前进距离能耗_J_m"] is not None]
        ),
        "成功定义": "完整20秒、无非法终止且5秒预热后平均前进速度位于[0.8, 1.2] m/s",
        "越界处置": "任一回合的接触足端中心越过当前65×60m真实地形块边缘，则整次评估无效",
        "四足位置审计预筛": (
            f"机身边缘余量>{FOOT_CENTER_MAX_BASE_DISTANCE_M} m时由ANYmal D运动学上界证明安全；"
            "进入邻边带后直接读取四足刚体中心"
        ),
        "进入四足真实位置邻边审计带回合数": sum(
            bool(row["进入四足真实位置邻边审计带"]) for row in rows
        ),
        "机身路线偏移预警回合数": sum(bool(row["机身超过保守路线预警线"]) for row in rows),
        "摆动足越过真实块边缘预警回合数": sum(
            bool(row["摆动足端中心越过真实地形边缘"]) for row in rows
        ),
        "接触足越过真实块边缘回合数": sum(bool(row["接触足越过真实地形边缘"]) for row in rows),
        "失败回合中接触足越界回合数": sum(
            not bool(row["成功"]) and bool(row["接触足越过真实地形边缘"])
            for row in rows
        ),
        "原成功回合中接触足越界回合数": sum(
            bool(row["成功"]) and bool(row["接触足越过真实地形边缘"])
            for row in rows
        ),
        "按地形方向难度": group_summary,
        "训练复现记录": evaluation_record,
    }
    (result_dir / "gpt-多地形评估摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[PACE] Terrain20sWide 多地形评估完成：{result_dir}", flush=True)
    return result_dir


def main() -> None:
    verify_dependency_versions()
    if EVAL_BATCHES * EVAL_NUM_ENVS != EVAL_EPISODES:
        raise RuntimeError("冻结评估批次数、每批环境数与总回合数不一致。")
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    device = args_cli.device or "cuda:0"
    env_cfg.sim.device = device
    env_cfg.seed = args_cli.ppo_seed
    agent_cfg.seed = args_cli.ppo_seed
    agent_cfg.device = device
    energy_budget = 0.8 * primary_reference if primary_reference is not None else None
    if method == "eco":
        if energy_budget is None:
            raise RuntimeError("PACE-ECO 缺少冻结的绝对焦耳预算。")
        agent_cfg.algorithm.energy_budget_j = energy_budget
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
    evaluation_record = validate_evaluation_checkpoint(
        checkpoint.parent,
        task=args_cli.task,
        energy_budget_j=energy_budget if method == "eco" else None,
        require_current_implementation=True,
    )
    if int(evaluation_record.get("seed", -1)) != args_cli.ppo_seed:
        raise ValueError("检查点 PPO seed 与评估参数不一致。")
    if not bool(evaluation_record.get("git_worktree_clean")):
        raise ValueError("正式多地形评估拒绝加载训练时工作树不干净的检查点。")
    state_set = (
        MULTI_TERRAIN_CALIBRATION_STATE_SET
        if args_cli.split == "calibration"
        else MULTI_TERRAIN_HOLDOUT_STATE_SET
    )
    state_set_sha256 = evaluation_state_definition_sha256(state_set)
    state_count = evaluation_state_count(state_set)
    env_cfg = configure_terrain20s_evaluation(
        env_cfg,
        batch_terrain_seed,
        state_set,
        args_cli.batch_index,
    )
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
        history_length=3,
        device=raw.device,
        dtype=robot.data.root_lin_vel_b.dtype,
    )
    components = {
        name: torch.zeros(raw.num_envs, device=raw.device)
        for name in ("electrical", "mechanical", "potential")
    }
    velocity_sum = torch.zeros(raw.num_envs, device=raw.device)
    velocity_sq = torch.zeros(raw.num_envs, device=raw.device)
    velocity_abs_error = torch.zeros(raw.num_envs, device=raw.device)
    velocity_error_sq = torch.zeros(raw.num_envs, device=raw.device)
    velocity_count = torch.zeros(raw.num_envs, device=raw.device)
    recorded = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
    rows: list[dict[str, object]] = []
    difficulties = curriculum_difficulty_table(
        batch_terrain_seed, EVAL_TERRAIN_ROWS, EVAL_TERRAIN_COLS, (0.10, 0.90)
    )
    warmup_steps = round(args_cli.warmup_s / raw.step_dt)
    physics_context = raw.sim.get_physics_context()
    print(
        "[PACE] 设备审计："
        f"环境={raw.device}，PhysX={physics_context.device}，"
        f"GPU仿真={physics_context.use_gpu_sim}，GPU数据管线={physics_context.use_gpu_pipeline}。",
        flush=True,
    )
    print(
        "[PACE] 开始 Terrain20sWide 评估："
        f"批次={args_cli.batch_index + 1}/{EVAL_BATCHES}，"
        f"本批地形seed={batch_terrain_seed}，本批={EVAL_NUM_ENVS} 个互异几何实例。",
        flush=True,
    )
    evaluation_steps = 0
    evaluation_started = monotonic()
    last_progress_print = evaluation_started
    faulthandler.dump_traceback_later(EVALUATION_STALL_TRACEBACK_S, repeat=False)
    try:
        while len(rows) < EVAL_NUM_ENVS and simulation_app.is_running():
            evaluation_steps += 1
            steady = raw.episode_length_buf >= warmup_steps
            forward_velocity = robot.data.root_lin_vel_b[:, 0]
            velocity_sum += forward_velocity * steady
            velocity_sq += forward_velocity.square() * steady
            velocity_abs_error += (forward_velocity - 1.0).abs() * steady
            velocity_error_sq += (forward_velocity - 1.0).square() * steady
            velocity_count += steady
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
                components[name] += extras["pace_energy_components"][name]
            completed_ids = extras["pace_energy_episode_mask"].nonzero(as_tuple=False).squeeze(-1)
            completed_id_list = completed_ids.tolist()
            for env_id in completed_id_list:
                if recorded[env_id]:
                    continue
                recorded[env_id] = True
                category, direction, level_label = metadata_labels(
                    int(extras["pace_terrain20s_category_code"][env_id]),
                    int(extras["pace_terrain20s_direction_code"][env_id]),
                    int(extras["pace_terrain20s_terrain_level"][env_id]),
                    EVAL_TERRAIN_ROWS,
                )
                level = int(extras["pace_terrain20s_terrain_level"][env_id])
                terrain_type = int(extras["pace_terrain20s_terrain_type"][env_id])
                difficulty = float(difficulties[level, terrain_type])
                instance_cfg = LongTerrainCfg(category=category, direction=direction)
                instance_cfg.size = (TERRAIN_LENGTH_M, TERRAIN_WIDTH_M)
                instance_cfg.seed = batch_terrain_seed
                generation_parameters = resolved_parameters(instance_cfg, difficulty)
                duration = float(extras["pace_terrain20s_duration_episode_s"][env_id])
                forward = float(extras["pace_terrain20s_forward_episode_m"][env_id])
                path_m = float(extras["pace_terrain20s_path_episode_m"][env_id])
                timed_out = bool(extras["pace_terrain20s_timeout"][env_id])
                illegally_terminated = bool(extras["pace_terrain20s_base_contact"][env_id])
                base_corridor_warning = bool(extras["pace_terrain20s_base_corridor_warning"][env_id])
                contact_foot_boundary_crossed = bool(
                    extras["pace_terrain20s_contact_foot_boundary_crossed"][env_id]
                )
                swing_foot_boundary_crossed = bool(
                    extras["pace_terrain20s_swing_foot_boundary_crossed"][env_id]
                )
                contact_crossed_by_foot = extras["pace_terrain20s_contact_foot_crossed_by_foot"][env_id]
                swing_crossed_by_foot = extras["pace_terrain20s_swing_foot_crossed_by_foot"][env_id]
                contact_min_margin = extras["pace_terrain20s_contact_foot_min_edge_margin_m"][env_id]
                contact_max_forward = extras["pace_terrain20s_contact_foot_max_forward_m"][env_id]
                contact_min_forward = extras["pace_terrain20s_contact_foot_min_forward_m"][env_id]
                contact_max_lateral = extras["pace_terrain20s_contact_foot_max_abs_lateral_m"][env_id]
                exact_foot_audit_sampled = bool(
                    extras["pace_terrain20s_foot_exact_audit_sampled"][env_id]
                )
                per_foot_boundary = {
                    name: {
                        "接触时越过真实块边缘": bool(contact_crossed_by_foot[index]),
                        "摆动时越过真实块边缘": bool(swing_crossed_by_foot[index]),
                        "邻边审计带内接触时最小边缘余量_m": _finite_float(contact_min_margin[index]),
                        "邻边审计带内接触时最大前向位移_m": _finite_float(contact_max_forward[index]),
                        "邻边审计带内接触时最小前向位移_m": _finite_float(contact_min_forward[index]),
                        "邻边审计带内接触时最大侧向绝对位移_m": _finite_float(contact_max_lateral[index]),
                    }
                    for index, name in enumerate(FOOT_BODY_NAMES)
                }
                crossed_contact_feet = [
                    name for index, name in enumerate(FOOT_BODY_NAMES) if bool(contact_crossed_by_foot[index])
                ]
                count = float(velocity_count[env_id].clamp_min(1.0))
                mean_speed = float(velocity_sum[env_id]) / count
                success = (
                    timed_out
                    and not illegally_terminated
                    and SUCCESS_SPEED_RANGE_M_S[0] <= mean_speed <= SUCCESS_SPEED_RANGE_M_S[1]
                )
                energy = float(extras["pace_energy_episode"][env_id])
                category_reference = energy_references.get(category)
                coordination_metrics = coordination.episode_metrics(env_id, raw.step_dt)
                global_episode_id = evaluation_global_id(args_cli.batch_index, env_id)
                local_instance_id = level * EVAL_TERRAIN_COLS + terrain_type
                global_instance_id = evaluation_global_id(args_cli.batch_index, local_instance_id)
                row: dict[str, object] = {
                    "协议版本": PROTOCOL_VERSION,
                    "边界审计版本": FOOT_BOUNDARY_AUDIT_VERSION,
                    "阶段": args_cli.stage,
                    "数据拆分": args_cli.split,
                    "任务ID": args_cli.task,
                    "回合": global_episode_id,
                    "环境编号": env_id,
                    "评估批次": args_cli.batch_index,
                    "评估批次组": args_cli.batch_group,
                    "固定初始状态编号": global_episode_id % state_count,
                    "PPO_seed": args_cli.ppo_seed,
                    "方法": method,
                    "固定权重标签": fixed_weight_label,
                    "固定权重系数": fixed_weight_value,
                    "地形seed": args_cli.terrain_seed,
                    "地形批次seed": batch_terrain_seed,
                    "评估初始状态集": state_set,
                    "评估初始状态集SHA256": state_set_sha256,
                    "地形实例编号": global_instance_id,
                    "地形类别": category,
                    "方向": direction,
                    "难度": level_label,
                    "难度连续值": difficulty,
                    "地形生成参数_JSON": json.dumps(generation_parameters, ensure_ascii=False, sort_keys=True),
                    "完整20秒": timed_out,
                    "非法终止": illegally_terminated,
                    "成功": success,
                    "越过长地形安全边界": contact_foot_boundary_crossed,
                    "接触足越过真实地形边缘": contact_foot_boundary_crossed,
                    "越界接触足": ",".join(crossed_contact_feet) if crossed_contact_feet else None,
                    "摆动足端中心越过真实地形边缘": swing_foot_boundary_crossed,
                    "机身超过保守路线预警线": base_corridor_warning,
                    "进入四足真实位置邻边审计带": exact_foot_audit_sampled,
                    "四足边界判定方式": (
                        "邻边带内直接四足刚体中心审计"
                        if exact_foot_audit_sampled
                        else f"机身边缘余量始终大于{FOOT_CENTER_MAX_BASE_DISTANCE_M}m，运动学上界证明四足安全"
                    ),
                    "足端中心相对机身运动学上界_m": FOOT_CENTER_MAX_BASE_DISTANCE_M,
                    "逐足边界统计_JSON": json.dumps(per_foot_boundary, ensure_ascii=False, sort_keys=True),
                    "真实地形相对边界_JSON": json.dumps(
                        {
                            "后向边缘_m": -TERRAIN_ORIGIN_X_M,
                            "前向边缘_m": TERRAIN_LENGTH_M - TERRAIN_ORIGIN_X_M,
                            "左右边缘_m": TERRAIN_WIDTH_M / 2.0,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "回合时间_s": duration,
                    "净前进距离_m": forward,
                    "水平实际路径_m": path_m,
                    "最大前进位移_m": float(extras["pace_terrain20s_max_forward_episode_m"][env_id]),
                    "最小前进位移_m": float(extras["pace_terrain20s_min_forward_episode_m"][env_id]),
                    "最大侧向绝对位移_m": float(extras["pace_terrain20s_max_abs_lateral_episode_m"][env_id]),
                    "净海拔变化_m": float(extras["pace_terrain20s_elevation_episode_m"][env_id]),
                    "稳态平均前进速度_m_s": mean_speed,
                    "相对1m_s平均绝对误差_m_s": float(velocity_abs_error[env_id]) / count,
                    "前进速度RMS_m_s": math.sqrt(float(velocity_sq[env_id]) / count),
                    "速度跟踪RMSE_m_s": math.sqrt(float(velocity_error_sq[env_id]) / count),
                    "回合能耗_J": energy,
                    "电气能耗_J": float(components["electrical"][env_id]),
                    "机械能耗_J": float(components["mechanical"][env_id]),
                    "势能能耗_J": float(components["potential"][env_id]),
                    "单位前进距离能耗_J_m": energy / forward if forward > 0.0 else None,
                    "单位实际路径能耗_J_m": energy / path_m if path_m > 0.0 else None,
                    "归一化能耗_E_t除以B_ref_t": energy / category_reference if category_reference else None,
                    "地形诊断B80_t_J": 0.8 * category_reference if category_reference else None,
                    "PACE主预算B80_J": energy_budget,
                    "B80联合合格": success and energy <= energy_budget if energy_budget else None,
                    **coordination_metrics,
                }
                rows.append(row)
                if len(rows) % 10 == 0 or len(rows) == EVAL_NUM_ENVS:
                    print(
                        f"[PACE] Terrain20sWide 本批进度 {len(rows)}/{EVAL_NUM_ENVS}",
                        flush=True,
                    )

            now = monotonic()
            if (
                evaluation_steps % EVALUATION_PROGRESS_INTERVAL_STEPS == 0
                or now - last_progress_print >= EVALUATION_PROGRESS_INTERVAL_S
            ):
                episode_min = int(raw.episode_length_buf.min().item())
                episode_max = int(raw.episode_length_buf.max().item())
                elapsed = now - evaluation_started
                print(
                    "[PACE] Terrain20sWide 运行心跳："
                    f"控制步={evaluation_steps}，已记录={len(rows)}/{EVAL_NUM_ENVS}，"
                    f"当前回合步范围=[{episode_min}, {episode_max}]，"
                    f"墙钟耗时={elapsed:.1f}s，平均控制步率={evaluation_steps / elapsed:.2f}步/s。",
                    flush=True,
                )
                last_progress_print = now
                faulthandler.cancel_dump_traceback_later()
                faulthandler.dump_traceback_later(EVALUATION_STALL_TRACEBACK_S, repeat=False)
    finally:
        faulthandler.cancel_dump_traceback_later()
    if len(rows) != EVAL_NUM_ENVS:
        raise RuntimeError(f"本批评估未收齐：{len(rows)}/{EVAL_NUM_ENVS}")
    boundary_crossings = [row for row in rows if row["接触足越过真实地形边缘"]]
    if boundary_crossings:
        first = boundary_crossings[0]
        print(
            "[PACE] 任一回合发生接触足真实越界，本批及整次正式评估被硬拒绝："
            f"回合={first['回合']}，越界足={first['越界接触足']}，"
            f"逐足统计={first['逐足边界统计_JSON']}",
            flush=True,
        )
        wrapped.close()
        raise RuntimeError(
            "检测到接触足端中心越过当前真实地形块边缘，拒绝写出批次和正式结果。"
        )
    _write_batch_staging(rows, evaluation_record)
    if args_cli.batch_index == EVAL_BATCHES - 1:
        all_rows = _load_all_staged_rows()
        counts = Counter(int(row["固定初始状态编号"]) for row in all_rows)
        if set(counts) != set(range(state_count)) or set(counts.values()) != {
            EVAL_EPISODES // state_count
        }:
            raise RuntimeError(f"八个历史初始状态未等量采样：{dict(counts)}")
        if any(bool(row["接触足越过真实地形边缘"]) for row in all_rows):
            raise RuntimeError("合并批次中存在接触足越界，拒绝正式结果。")
        _write_outputs(all_rows, evaluation_record)
    else:
        print(
            f"[PACE] 批次 {args_cli.batch_index + 1}/{EVAL_BATCHES} 完成；"
            "尚未生成正式 JSON/CSV。",
            flush=True,
        )
    wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
