from __future__ import annotations

import json
import subprocess
from pathlib import Path

import gymnasium as gym

import pace_eco_lab  # noqa: F401
from pace_eco_lab.direction_conditioned_protocol import (
    DIRECTION_CONDITIONED_TASK_IDS as V2_1_TASK_IDS,
    TASK_IDS as V2_1_TASK_MAP,
)
from pace_eco_lab.direction_conditioned_v2_2_protocol import (
    DIRECTION_CONDITIONED_V2_2_TASK_IDS,
    EVALUATION_MATRIX,
    EVALUATION_PROTOCOL_VERSION,
    EVALUATION_REUSED_V2_1_MODELS,
    EXCLUDED_FROM_EVALUATION,
    FIXED_ENERGY_REWARD_WEIGHT,
    FIXED_LAMBDA,
    NEW_TRAINING_MATRIX,
    PROTOCOL_VERSION,
    REUSED_V2_1_MODELS,
    SMOKE_SEED,
    TARGET_MATRIX,
    TASK_IDS,
    is_new_training_target,
    is_evaluation_target,
    terrain_seed,
)


ROOT = Path(__file__).resolve().parents[2]
MANAGER_ROOT = ROOT.parent / "experiment-manager"


def test_v2_1_registry_is_preserved_and_v2_2_adds_only_five_fixed_tasks():
    assert len(V2_1_TASK_IDS) == 24
    assert len(DIRECTION_CONDITIONED_V2_2_TASK_IDS) == 15
    for terrain in ("flat", "rough", "stairs", "boxes", "slope"):
        for method in ("task_only", "eco"):
            assert TASK_IDS[("directional", method, terrain)] == V2_1_TASK_MAP[
                ("directional", method, terrain)
            ]
        fixed = gym.spec(TASK_IDS[("directional", "fixed_weight", terrain)])
        assert "direction_conditioned_v2_2_env_cfg" in fixed.kwargs["env_cfg_entry_point"]
        assert "direction_conditioned_v2_2_agent_cfg" in fixed.kwargs["rsl_rl_cfg_entry_point"]


def test_v2_2_frozen_counts_and_seed_pairing():
    assert PROTOCOL_VERSION == "gpt-direction-conditioned-v2.2"
    assert len(TARGET_MATRIX) == 53
    assert len(REUSED_V2_1_MODELS) == 22
    assert len(NEW_TRAINING_MATRIX) == 31
    assert len(EVALUATION_MATRIX) == len(set(EVALUATION_MATRIX)) == 45
    assert len(EVALUATION_REUSED_V2_1_MODELS) == 14
    assert len(EXCLUDED_FROM_EVALUATION) == 8
    assert EVALUATION_PROTOCOL_VERSION == "gpt-direction-conditioned-v2.2-45model-holdout-v1"
    assert FIXED_ENERGY_REWARD_WEIGHT == -1.6e-4
    assert FIXED_LAMBDA == 1.6e-4
    assert SMOKE_SEED == 902
    assert terrain_seed("stage1_formal_train", "stairs", 2) == 532_002
    assert not is_new_training_target("task_only", "flat", 1)
    assert not is_new_training_target("eco", "stairs", 1)
    assert is_new_training_target("fixed_weight", "flat", 1)
    assert is_new_training_target("task_only", "stairs", 2)
    assert is_evaluation_target("task_only", "flat", 3)
    assert not is_evaluation_target("task_only", "flat", 4)


def test_45_model_evaluation_manifest_freezes_metrics_and_excludes_seed4_5():
    path = ROOT / "gpt-方向条件v2.2_45模型评估冻结配置.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["冻结状态"] == "45模型评估协议已冻结"
    assert data["评估矩阵"]["唯一模型数"] == 45
    assert data["评估矩阵"]["PPO_seed"] == [1, 2, 3]
    assert data["不进入本次评估"]["PPO_seed"] == [4, 5]
    assert data["ECO相对Fixed-weight判断标准"]["方向成功率非劣界_百分点"] == 5.0
    assert len(data["表1字段"]) == 6
    assert len(data["表2字段"]) == 9
    assert data["holdout条件"]["总回合数"] == 9000


def test_v2_metric_amendment_preserves_authorization_and_updates_tables():
    path = ROOT / "gpt-方向条件v2.2_45模型评估指标修订v2.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["修订状态"] == "已冻结"
    assert len(data["表1新增指标"]) == 2
    assert data["表2字段数"] == 7
    assert data["表2重命名"]["新表头"] == "稳态机身前向速度"
    assert data["表2删除指标"] == ["支撑相占比极差", "落足频率变异系数"]
    source = (ROOT / "scripts/pace_eco/gpt-汇总45模型holdout.py").read_text(encoding="utf-8")
    for token in ("E80与B80比值", "平均预算余量率", "稳态机身前向速度", "指标解读"):
        assert token in source
    assert '"支撑相占比极差",' not in source.split("TABLE2_SOURCE_FIELDS", 1)[1].split(")", 1)[0]


def test_45_model_freezer_and_evaluator_have_read_only_hash_guards():
    freezer = (ROOT / "scripts/pace_eco/freeze_direction_conditioned_v2_2_holdout.py").read_text(encoding="utf-8")
    evaluator = (ROOT / "scripts/pace_eco/direction_conditioned_eval.py").read_text(encoding="utf-8")
    launcher = (ROOT / "scripts/pace_eco/gpt-运行方向条件v2.2评估.sh").read_text(encoding="utf-8")
    for token in ("--evaluation_manifest", "检查点SHA256", "B_ref_B80_SHA256", "初始状态表SHA256", "Learning iteration 2999/3000", "同一地形×方法×seed出现重复模型"):
        assert token in freezer
    for field in ("成功条件B80合格率", "归一化超预算幅度", "机身横向速度RMS_m_s", "评估指标版本"):
        assert field in evaluator
    assert "seed4/5禁止评估" in launcher


def test_v2_2_manager_plan_matches_protocol_and_archives_v2_1():
    plan = json.loads((MANAGER_ROOT / "gpt-方向条件v2.2训练计划.json").read_text(encoding="utf-8"))
    archive = json.loads((MANAGER_ROOT / "gpt-方向条件v2.1矩阵归档.json").read_text(encoding="utf-8"))
    planned = {
        (str(item["方法"]), str(item["地形"]), int(item["PPO_seed"]))
        for item in plan["训练任务"]
    }
    assert plan["协议版本"] == PROTOCOL_VERSION
    assert plan["启动许可"] == "只允许本文件训练任务中的v2.2任务"
    assert planned == set(NEW_TRAINING_MATRIX)
    assert archive["状态"] == "archived/frozen（已归档/冻结）"
    assert archive["调度规则"]["允许启动v2.1新任务"] is False


def test_v2_2_shell_scripts_parse_and_scheduler_has_hard_protocol_guard():
    for relative in (
        "scripts/pace_eco/gpt-方向条件v2.2公共.sh",
        "scripts/pace_eco/gpt-运行方向条件v2.2训练.sh",
        "scripts/pace_eco/gpt-运行方向条件v2.2评估.sh",
    ):
        subprocess.run(["bash", "-n", str(ROOT / relative)], check=True)
    scheduler = (MANAGER_ROOT / "overnight_scheduler.py").read_text(encoding="utf-8")
    assert 'CURRENT_RUN_PREFIX = "gpt_direction_v2_2_"' in scheduler
    assert 'ARCHIVED_RUN_PREFIX = "gpt_direction_v2_1_"' in scheduler
    assert 'TRAIN_LAUNCHER = PROJECT_ROOT / "scripts/pace_eco/gpt-运行方向条件v2.2训练.sh"' in scheduler
    assert "调度保护拒绝非v2.2任务" in scheduler


def test_v2_2_fixed_weight_uses_plain_ppo_and_frozen_reward_coefficient():
    agent_source = (ROOT / "pace_eco_lab/configs/direction_conditioned_v2_2_agent_cfg.py").read_text(
        encoding="utf-8"
    )
    env_source = (ROOT / "pace_eco_lab/configs/direction_conditioned_v2_2_env_cfg.py").read_text(
        encoding="utf-8"
    )
    train_source = (ROOT / "scripts/pace_eco/train.py").read_text(encoding="utf-8")
    assert "PaceFixedWeightPPORunnerCfg" in agent_source
    assert "PaceEcoPPORunnerCfg" not in agent_source
    assert "self.rewards.energy.weight = FIXED_ENERGY_REWARD_WEIGHT" in env_source
    assert "is_direction_v2_2_new_training_target" in train_source
    assert "v2.2 fixed_weight 不读取 B_ref" in train_source
