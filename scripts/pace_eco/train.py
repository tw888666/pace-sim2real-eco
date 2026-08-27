#!/usr/bin/env python3
"""训练 PACE 任务型、固定权重或约束策略。"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="训练 PACE-ECO 的 RSL-RL 策略。")
parser.add_argument("--task", default=None, help="已注册的 PACE 任务名称。")
parser.add_argument("--num_envs", type=int, default=None, help="并行环境数量。")
parser.add_argument("--max_iterations", type=int, default=None, help="本次运行的目标总迭代数。")
parser.add_argument("--seed", type=int, default=0, help="随机种子。")
parser.add_argument("--run_name", type=str, default=None, help="运行名称；程序会自动增加 gpt_ 前缀。")
parser.add_argument("--energy_budget_j", type=float, default=None, help="PACE-ECO 每回合能耗预算，单位 J。")
parser.add_argument("--terrain_seed", type=int, default=None, help="多地形生成随机种子；与 PPO seed 独立。")
parser.add_argument("--protocol_role", type=str, default=None, help="多地形冻结 seed 角色。")
parser.add_argument(
    "--direction_protocol_version",
    choices=("v2.1", "v2.2"),
    default="v2.1",
    help="方向条件协议版本；旧入口默认保持 v2.1。",
)
parser.add_argument(
    "--fixed_energy_weight",
    type=float,
    default=None,
    help="保留参数；v1.4 主协议拒绝覆盖 PACE 论文固定系数。",
)
parser.add_argument(
    "--fixed_selection_json",
    type=str,
    default=None,
    help="保留参数；v1.4 主协议不读取网格选择文件。",
)
parser.add_argument("--energy_reference_json", type=str, default=None, help="冻结的各地形 B_ref JSON。")
parser.add_argument("--log_root", type=str, default=None, help="显式训练输出根；多地形任务必须提供。")
parser.add_argument("--resume_from", type=str, default=None, help="同一次运行内的检查点绝对或相对路径。")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.task is None or args_cli.max_iterations is None or args_cli.run_name is None:
    parser.error("--task、--max_iterations 和 --run_name 均为必填参数。")
if args_cli.max_iterations > 1_000 and not os.environ.get("TMUX") and not os.environ.get("STY"):
    parser.error("超过 1000 次迭代的训练必须从 tmux 或 GNU Screen 会话启动。")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import load_cfg_from_registry

import pace_eco_lab  # noqa: F401
from pace_eco_lab.constants import ECO_ID, REGISTERED_TASKS
from pace_eco_lab.direction_conditioned_protocol import (
    DIRECTION_CONDITIONED_TASK_IDS,
    FOOT_BOUNDARY_AUDIT_VERSION,
    PPO_SEEDS as DIRECTION_PPO_SEEDS,
    PROTOCOL_VERSION as DIRECTION_PROTOCOL_VERSION,
    TASK_IDS as DIRECTION_TASK_IDS,
    terrain_seed as direction_terrain_seed,
)
from pace_eco_lab.direction_conditioned_v2_2_protocol import (
    DIRECTION_CONDITIONED_V2_2_TASK_IDS,
    FIXED_ENERGY_REWARD_WEIGHT,
    FIXED_WEIGHT_LABEL as DIRECTION_V2_2_FIXED_WEIGHT_LABEL,
    PROTOCOL_VERSION as DIRECTION_V2_2_PROTOCOL_VERSION,
    SMOKE_SEED as DIRECTION_V2_2_SMOKE_SEED,
    SOURCE_PROTOCOL_VERSION as DIRECTION_V2_2_SOURCE_PROTOCOL_VERSION,
    TASK_IDS as DIRECTION_V2_2_TASK_IDS,
    is_new_training_target as is_direction_v2_2_new_training_target,
    terrain_seed as direction_v2_2_terrain_seed,
)
from pace_eco_lab.multi_terrain_protocol import (
    MULTI_TERRAIN_TASK_IDS,
    PACE_PAPER_FIXED_WEIGHT,
    PACE_PAPER_FIXED_WEIGHT_LABEL,
    PPO_SEEDS,
    PROTOCOL_VERSION,
    TASK_IDS,
    TERRAIN_NAMES,
    terrain_seed,
)
from pace_eco_lab.reproducibility import (
    validate_resume_records,
    verify_dependency_versions,
    write_run_records,
)


def _normalized_run_name(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_.-]+", "_", value.strip()).strip("._")
    if not cleaned:
        raise ValueError("run_name 清理后为空。")
    return cleaned if cleaned.startswith("gpt_") else f"gpt_{cleaned}"


def _resolve_checkpoint(value: str) -> Path:
    checkpoint = Path(value).expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.suffix != ".pt":
        raise FileNotFoundError(f"检查点不存在或不是 .pt 文件：{checkpoint}")
    return checkpoint


def _load_energy_budget(
    path_value: str,
    required_stage: str,
    required_terrain: str,
    protocol_version: str = PROTOCOL_VERSION,
    boundary_audit_version: str | None = None,
    source_experiment: str | None = None,
) -> float:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"B_ref 冻结文件不存在：{path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        data.get("冻结状态") != "已冻结"
        or data.get("协议版本") != protocol_version
        or data.get("阶段") != required_stage
    ):
        raise ValueError("B_ref 文件的冻结状态、协议版本或阶段不匹配。")
    if boundary_audit_version is not None and data.get("边界审计版本") != boundary_audit_version:
        raise ValueError("B_ref 文件的边界审计版本不匹配。")
    if source_experiment is not None and data.get("来源实验") != source_experiment:
        raise ValueError("B_ref 文件不是由冻结的 E2 任务型标定导出。")
    if required_stage == "stage2":
        reference = float(data.get("B_ref_mixed_J", 0.0))
        label = "B_ref_mixed_J"
    else:
        values = data.get("B_ref_J")
        if not isinstance(values, dict):
            raise ValueError("B_ref 文件缺少 B_ref_J 对象。")
        reference = float(values.get(required_terrain, 0.0))
        label = f"B_ref_J.{required_terrain}"
    if reference <= 0.0:
        raise ValueError(f"B_ref 文件缺少正参考能耗：{label}。")
    return 0.8 * reference


def _configure_multi_terrain_protocol(env_cfg, agent_cfg) -> str:
    inverse = {task_id: key for key, task_id in TASK_IDS.items()}
    method, terrain = inverse[args_cli.task]
    role_to_seed_group = {
        "stage1_smoke_train": "stage1_smoke",
        "stage1_capacity_smoke_train": "stage1_smoke",
        "stage1_budget_train": "stage1_budget",
        "stage1_formal_train": "stage1_formal",
        "stage2_budget_train": "stage2_budget",
        "stage2_formal_train": "stage2_formal",
        "stage2_smoke_train": "stage2_smoke",
        "stage2_capacity_smoke_train": "stage2_smoke",
    }
    if args_cli.protocol_role not in role_to_seed_group:
        raise ValueError(f"多地形训练缺少或使用了非法 --protocol_role：{args_cli.protocol_role}")
    stage = args_cli.protocol_role.split("_", maxsplit=1)[0]
    if args_cli.seed not in PPO_SEEDS[role_to_seed_group[args_cli.protocol_role]]:
        raise ValueError(f"PPO seed {args_cli.seed} 不属于角色 {args_cli.protocol_role} 的冻结集合。")
    smoke = args_cli.protocol_role.endswith("smoke_train")
    capacity_smoke = args_cli.protocol_role.endswith("capacity_smoke_train")
    expected_iterations = 2 if smoke else 3_000
    expected_envs = 4_096 if capacity_smoke or not smoke else 16
    if args_cli.max_iterations != expected_iterations or env_cfg.scene.num_envs != expected_envs:
        raise ValueError(
            f"角色 {args_cli.protocol_role} 冻结为 {expected_envs} 环境/{expected_iterations} 次更新，"
            f"实际为 {env_cfg.scene.num_envs}/{args_cli.max_iterations}。"
        )
    if smoke != ("smoke" in args_cli.run_name.lower()):
        raise ValueError("smoke 角色与 run_name 标记不一致。")
    if args_cli.protocol_role.startswith("stage1_") and terrain == "mixed":
        raise ValueError("阶段一禁止混合地形任务。")
    if args_cli.protocol_role.startswith("stage2_") and terrain != "mixed":
        raise ValueError("阶段二只允许从头训练混合地形任务。")
    required_method = None
    if args_cli.protocol_role.endswith("budget_train"):
        required_method = "task_only"
    if required_method is not None and method != required_method:
        raise ValueError(f"角色 {args_cli.protocol_role} 只允许方法 {required_method}。")
    expected_terrain_seed = terrain_seed(args_cli.protocol_role, terrain, args_cli.seed)
    if args_cli.terrain_seed != expected_terrain_seed:
        raise ValueError(f"地形种子不符合冻结映射：期望 {expected_terrain_seed}，实际 {args_cli.terrain_seed}。")
    env_cfg.pace_terrain_seed = int(args_cli.terrain_seed)
    env_cfg.scene.terrain.terrain_generator.seed = int(args_cli.terrain_seed)
    if method == "fixed_weight":
        if args_cli.fixed_energy_weight is not None or args_cli.fixed_selection_json is not None:
            raise ValueError(
                f"{PROTOCOL_VERSION} 固定权重已冻结为 PACE 论文系数 "
                f"{PACE_PAPER_FIXED_WEIGHT_LABEL}={PACE_PAPER_FIXED_WEIGHT}，禁止命令行覆盖或选择。"
            )
        env_cfg.rewards.energy.weight = PACE_PAPER_FIXED_WEIGHT
    elif args_cli.fixed_energy_weight is not None or args_cli.fixed_selection_json is not None:
        raise ValueError("固定权重专用参数只允许固定权重任务。")
    if method == "eco":
        if args_cli.energy_reference_json is None:
            raise ValueError("多地形 PACE-ECO 必须提供冻结的 --energy_reference_json。")
        agent_cfg.algorithm.energy_budget_j = _load_energy_budget(
            args_cli.energy_reference_json,
            stage,
            terrain,
        )
    elif args_cli.energy_reference_json is not None:
        raise ValueError("--energy_reference_json 只允许 PACE-ECO 任务。")
    if args_cli.energy_budget_j is not None:
        raise ValueError("多地形主实验从冻结 B_ref 文件得到绝对焦耳 B80，禁止命令行覆盖预算。")
    if args_cli.log_root is None:
        raise ValueError("多地形任务必须显式提供隔离的 --log_root。")
    return method


def _configure_direction_conditioned_protocol(env_cfg, agent_cfg) -> str:
    inverse = {task_id: key for key, task_id in DIRECTION_TASK_IDS.items()}
    variant, method, terrain = inverse[args_cli.task]
    role_to_seed_group = {
        "stage1_smoke_train": "stage1_smoke",
        "stage1_capacity_smoke_train": "stage1_smoke",
        "stage1_budget_train": "stage1_budget",
        "stage1_formal_train": "stage1_formal",
        "stage2_budget_train": "stage2_budget",
        "stage2_formal_train": "stage2_formal",
        "stage2_smoke_train": "stage2_smoke",
        "stage2_capacity_smoke_train": "stage2_smoke",
    }
    if args_cli.protocol_role not in role_to_seed_group:
        raise ValueError(f"方向条件训练缺少或使用了非法 --protocol_role：{args_cli.protocol_role}")
    stage = args_cli.protocol_role.split("_", maxsplit=1)[0]
    seed_group = role_to_seed_group[args_cli.protocol_role]
    if args_cli.seed not in DIRECTION_PPO_SEEDS[seed_group]:
        raise ValueError(f"PPO seed {args_cli.seed} 不属于 v2 角色 {args_cli.protocol_role}。")
    smoke = args_cli.protocol_role.endswith("smoke_train")
    capacity_smoke = args_cli.protocol_role.endswith("capacity_smoke_train")
    expected_iterations = 2 if smoke else 3_000
    expected_envs = 4_096 if capacity_smoke or not smoke else 16
    if args_cli.max_iterations != expected_iterations or env_cfg.scene.num_envs != expected_envs:
        raise ValueError(
            f"v2 角色 {args_cli.protocol_role} 冻结为 {expected_envs} 环境/"
            f"{expected_iterations} 次更新。"
        )
    if smoke != ("smoke" in args_cli.run_name.lower()):
        raise ValueError("v2 smoke 角色与 run_name 标记不一致。")
    if stage == "stage1" and terrain == "mixed":
        raise ValueError("v2 阶段一禁止 mixed。")
    if stage == "stage2" and terrain != "mixed":
        raise ValueError("v2 阶段二只允许 mixed。")
    if args_cli.protocol_role.endswith("budget_train") and (
        variant != "directional" or method != "task_only"
    ):
        raise ValueError("v2 B_ref 只允许 E2 directional/task_only seed0 标定模型。")
    expected_terrain_seed = direction_terrain_seed(
        args_cli.protocol_role,
        terrain,
        args_cli.seed,
    )
    if args_cli.terrain_seed != expected_terrain_seed:
        raise ValueError(
            f"v2 地形 seed 不符合冻结映射：期望 {expected_terrain_seed}，"
            f"实际 {args_cli.terrain_seed}。"
        )
    env_cfg.pace_terrain_seed = int(args_cli.terrain_seed)
    env_cfg.scene.terrain.terrain_generator.seed = int(args_cli.terrain_seed)
    if args_cli.fixed_energy_weight is not None or args_cli.fixed_selection_json is not None:
        raise ValueError("方向条件 v2.1 不包含固定权重方法。")
    if method == "eco":
        if args_cli.energy_reference_json is None:
            raise ValueError("方向条件 PACE-ECO 必须提供冻结的 v2 B_ref JSON。")
        agent_cfg.algorithm.energy_budget_j = _load_energy_budget(
            args_cli.energy_reference_json,
            stage,
            terrain,
            protocol_version=DIRECTION_PROTOCOL_VERSION,
            boundary_audit_version=FOOT_BOUNDARY_AUDIT_VERSION,
            source_experiment="E2 directional/task_only seed0 calibration",
        )
    elif args_cli.energy_reference_json is not None:
        raise ValueError("v2 --energy_reference_json 只允许 ECO 任务。")
    if args_cli.energy_budget_j is not None:
        raise ValueError("方向条件 v2 从冻结 B_ref 得到统一 B80，禁止命令行覆盖。")
    if args_cli.log_root is None:
        raise ValueError("方向条件 v2 任务必须提供独立 --log_root。")
    return method


def _configure_direction_conditioned_v2_2_protocol(env_cfg, agent_cfg) -> str:
    """配置只允许出现在 v2.2 新训练清单中的 E2 三方法任务。"""

    inverse = {task_id: key for key, task_id in DIRECTION_V2_2_TASK_IDS.items()}
    variant, method, terrain = inverse[args_cli.task]
    formal = args_cli.protocol_role == "stage1_formal_train"
    smoke = args_cli.protocol_role == "stage1_smoke_train"
    if not formal and not smoke:
        raise ValueError("方向条件 v2.2 只允许 stage1_smoke_train 或 stage1_formal_train。")
    if variant != "directional":
        raise ValueError("方向条件 v2.2 主矩阵只允许 E2 directional。")
    if smoke and (method != "fixed_weight" or terrain != "flat" or args_cli.seed != DIRECTION_V2_2_SMOKE_SEED):
        raise ValueError("v2.2 smoke 只允许 flat fixed_weight seed902。")
    if formal and not is_direction_v2_2_new_training_target(method, terrain, args_cli.seed):
        raise ValueError(
            f"{method}/{terrain}/seed{args_cli.seed} 不属于 v2.2 新训练清单；"
            "已迁移的 v2.1 模型禁止重复训练。"
        )
    expected_iterations = 3_000 if formal else 2
    expected_envs = 4_096 if formal else 16
    if args_cli.max_iterations != expected_iterations or env_cfg.scene.num_envs != expected_envs:
        raise ValueError(
            f"方向条件 v2.2 {args_cli.protocol_role} 冻结为 "
            f"{expected_envs} 环境/{expected_iterations} 次更新。"
        )
    if smoke != ("smoke" in args_cli.run_name.lower()):
        raise ValueError("方向条件 v2.2 smoke 角色与 run_name 标记不一致。")
    expected_terrain_seed = direction_v2_2_terrain_seed(
        args_cli.protocol_role,
        terrain,
        args_cli.seed,
    )
    if args_cli.terrain_seed != expected_terrain_seed:
        raise ValueError(
            f"v2.2 地形 seed 不符合配对映射：期望 {expected_terrain_seed}，"
            f"实际 {args_cli.terrain_seed}。"
        )
    env_cfg.pace_terrain_seed = int(args_cli.terrain_seed)
    env_cfg.scene.terrain.terrain_generator.seed = int(args_cli.terrain_seed)
    if args_cli.fixed_energy_weight is not None or args_cli.fixed_selection_json is not None:
        raise ValueError(
            f"{DIRECTION_V2_2_PROTOCOL_VERSION} 固定权重已冻结为 "
            f"{DIRECTION_V2_2_FIXED_WEIGHT_LABEL}={FIXED_ENERGY_REWARD_WEIGHT}，"
            "禁止命令行覆盖或重新选择。"
        )
    if method == "fixed_weight":
        trained_weight = float(env_cfg.rewards.energy.weight)
        if abs(trained_weight - FIXED_ENERGY_REWARD_WEIGHT) > 1.0e-12:
            raise ValueError(
                f"v2.2 固定权重配置错误：期望 {FIXED_ENERGY_REWARD_WEIGHT}，"
                f"实际 {trained_weight}。"
            )
        if args_cli.energy_reference_json is not None:
            raise ValueError("v2.2 fixed_weight 不读取 B_ref，也不运行拉格朗日乘子。")
    elif method == "eco":
        if args_cli.energy_reference_json is None:
            raise ValueError("v2.2 ECO 必须提供已冻结的 v2.1 E2 B_ref JSON。")
        agent_cfg.algorithm.energy_budget_j = _load_energy_budget(
            args_cli.energy_reference_json,
            "stage1",
            terrain,
            protocol_version=DIRECTION_V2_2_SOURCE_PROTOCOL_VERSION,
            boundary_audit_version=FOOT_BOUNDARY_AUDIT_VERSION,
            source_experiment="E2 directional/task_only seed0 calibration",
        )
    elif args_cli.energy_reference_json is not None:
        raise ValueError("v2.2 task_only 不接受 B_ref。")
    if args_cli.energy_budget_j is not None:
        raise ValueError("方向条件 v2.2 禁止命令行覆盖 ECO 预算。")
    if args_cli.log_root is None:
        raise ValueError("方向条件 v2.2 必须提供独立 --log_root。")
    return method


def _validate_run_role(run_name: str, max_iterations: int, resume: bool) -> None:
    lowered = run_name.lower()
    if "smoke" in lowered:
        if resume:
            raise ValueError("smoke 权重禁止恢复或继续训练。")
        if max_iterations != 2:
            raise ValueError("smoke 运行固定为 2 次迭代。")
    if "dev" in lowered and max_iterations > 300:
        raise ValueError("dev 权重最多训练 300 次迭代，禁止续成正式权重。")
    if "formal" in lowered and max_iterations != 3_000:
        raise ValueError(
            "formal 正式运行固定执行 3000 次更新；"
            "RSL-RL 从 0 编号，因此最终检查点为 model_2999.pt。"
        )


def main() -> None:
    verify_dependency_versions()
    if args_cli.task not in REGISTERED_TASKS:
        raise ValueError(f"只允许 PACE 正式任务：{REGISTERED_TASKS}")
    if args_cli.max_iterations <= 0:
        raise ValueError("max_iterations 必须为正。")
    if args_cli.num_envs is not None and args_cli.num_envs <= 0:
        raise ValueError("num_envs 必须为正。")

    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    run_name = _normalized_run_name(args_cli.run_name)
    _validate_run_role(run_name, args_cli.max_iterations, args_cli.resume_from is not None)
    device = args_cli.device or "cuda:0"

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = device
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = device
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.run_name = run_name
    v1_multi_terrain_task = args_cli.task in MULTI_TERRAIN_TASK_IDS
    direction_v2_2_task = (
        args_cli.direction_protocol_version == "v2.2"
        and args_cli.task in DIRECTION_CONDITIONED_V2_2_TASK_IDS
    )
    direction_conditioned_task = args_cli.task in DIRECTION_CONDITIONED_TASK_IDS
    if args_cli.direction_protocol_version == "v2.2" and not direction_v2_2_task:
        raise ValueError("--direction_protocol_version v2.2 只允许 v2.2 冻结任务。")
    if (
        args_cli.task in DIRECTION_CONDITIONED_V2_2_TASK_IDS
        and args_cli.task not in DIRECTION_CONDITIONED_TASK_IDS
        and not direction_v2_2_task
    ):
        raise ValueError("方向条件 fixed_weight 任务必须显式选择 v2.2 协议。")
    multi_terrain_task = v1_multi_terrain_task or direction_conditioned_task or direction_v2_2_task
    if v1_multi_terrain_task:
        multi_method = _configure_multi_terrain_protocol(env_cfg, agent_cfg)
    elif direction_v2_2_task:
        multi_method = _configure_direction_conditioned_v2_2_protocol(env_cfg, agent_cfg)
    elif direction_conditioned_task:
        multi_method = _configure_direction_conditioned_protocol(env_cfg, agent_cfg)
    else:
        multi_method = None
    if not multi_terrain_task and any(
        value is not None
        for value in (
            args_cli.terrain_seed,
            args_cli.protocol_role,
            args_cli.fixed_energy_weight,
            args_cli.fixed_selection_json,
            args_cli.energy_reference_json,
            args_cli.log_root,
        )
    ):
        raise ValueError("多地形专用参数禁止用于现有 Flat 正式任务。")
    if args_cli.task == ECO_ID:
        if args_cli.energy_budget_j is None or args_cli.energy_budget_j <= 0.0:
            raise ValueError("PACE-ECO 任务必须提供正的 --energy_budget_j。")
        agent_cfg.algorithm.energy_budget_j = float(args_cli.energy_budget_j)
    elif args_cli.energy_budget_j is not None and not multi_terrain_task:
        raise ValueError("--energy_budget_j 只用于 PACE-ECO 任务。")

    import importlib.metadata

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    log_root = Path(args_cli.log_root).expanduser().resolve() if args_cli.log_root else Path("logs/rsl_rl").resolve()
    root = log_root / agent_cfg.experiment_name
    checkpoint: Path | None = None
    if args_cli.resume_from:
        checkpoint = _resolve_checkpoint(args_cli.resume_from)
        run_dir = checkpoint.parent
        validate_resume_records(
            run_dir,
            task=args_cli.task,
            seed=args_cli.seed,
            run_name=run_name,
            env_cfg=env_cfg,
            agent_cfg=agent_cfg,
        )
        print(f"[PACE] 从同一次运行恢复：{checkpoint}")
    else:
        run_dir = root / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{run_name}"
        if run_dir.exists():
            raise FileExistsError(f"拒绝覆盖已有运行目录：{run_dir}")
        write_run_records(
            run_dir,
            task=args_cli.task,
            seed=args_cli.seed,
            run_name=run_name,
            env_cfg=env_cfg,
            agent_cfg=agent_cfg,
        )
        print(f"[PACE] 新运行目录：{run_dir}")

    env_cfg.log_dir = str(run_dir)
    print("[PACE] 正在构建环境。", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    print("[PACE] 环境构建完成，正在创建 RSL-RL 环境包装器。", flush=True)
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print("[PACE] 环境包装器创建完成，正在构建策略、价值网络和日志器。", flush=True)
    runner = OnPolicyRunner(
        wrapped_env,
        agent_cfg.to_dict(),
        log_dir=str(run_dir),
        device=agent_cfg.device,
    )
    print("[PACE] 策略、价值网络和日志器构建完成，正在记录代码状态。", flush=True)
    runner.add_git_repo_to_log(__file__)
    print("[PACE] 代码状态记录完成。", flush=True)

    completed_iterations = 0
    if checkpoint is not None:
        runner.load(str(checkpoint), map_location=agent_cfg.device)
        completed_iterations = runner.current_learning_iteration + 1
        runner.current_learning_iteration = completed_iterations
    remaining_iterations = args_cli.max_iterations - completed_iterations
    if remaining_iterations <= 0:
        raise ValueError(
            f"检查点已完成 {completed_iterations} 次迭代，不小于目标 {args_cli.max_iterations}；无需恢复。"
        )
    print(
        f"[PACE] 任务={args_cli.task} 环境数={env_cfg.scene.num_envs} "
        f"已完成={completed_iterations} 本次训练={remaining_iterations} 目标={args_cli.max_iterations}"
    )
    runner.learn(
        num_learning_iterations=remaining_iterations,
        # 约束乘子只能使用从真实 reset 开始的完整回合；ECO 首轮不得伪造随机回合年龄。
        init_at_random_ep_len=checkpoint is None and args_cli.task != ECO_ID and multi_method != "eco",
    )
    wrapped_env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # SimulationApp.close() 在部分异常路径会掩盖异常退出状态；先把
        # traceback 写入启动日志，Shell 再通过必需产物断言拒绝假成功。
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
