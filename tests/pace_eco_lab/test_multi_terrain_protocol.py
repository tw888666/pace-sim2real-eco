from __future__ import annotations

import ast
import json
import subprocess
from collections import Counter
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from pace_eco_lab.constants import ECO_ID, FIXED_WEIGHT_ID, TASK_ONLY_ID
from pace_eco_lab.evaluation_states import (
    CALIBRATION_STATE_SET,
    HOLDOUT_STATE_SET,
    MULTI_TERRAIN_CALIBRATION_STATE_SET,
    MULTI_TERRAIN_EVALUATION_STATE_SETS,
    MULTI_TERRAIN_HOLDOUT_STATE_SET,
    evaluation_state_definition,
    evaluation_state_definition_sha256,
)
from pace_eco_lab.multi_terrain_protocol import (
    DEFERRED_FIXED_WEIGHT_CANDIDATES,
    EVAL_BATCHES,
    EVAL_EPISODES,
    EVAL_NUM_ENVS,
    EVAL_TERRAIN_COLS,
    EVAL_TERRAIN_ROWS,
    EVALUATION_WARMUP_S,
    MAX_AUDITED_BACKWARD_M,
    MAX_AUDITED_FORWARD_M,
    MAX_AUDITED_LATERAL_M,
    MULTI_TERRAIN_TASK_IDS,
    PACE_PAPER_FIXED_WEIGHT,
    PACE_PAPER_FIXED_WEIGHT_LABEL,
    PPO_SEEDS,
    PROTOCOL_VERSION,
    TASK_IDS,
    TERRAIN_ACTIVE_END_X_M,
    TERRAIN_DISTRIBUTIONS,
    TERRAIN_LENGTH_M,
    TERRAIN_NAMES,
    TERRAIN_ORIGIN_X_M,
    TERRAIN_WIDTH_M,
    TRAIN_TERRAIN_COLS,
    evaluation_batch_offset,
    evaluation_batch_seed,
    evaluation_global_id,
    terrain_column_metadata,
    terrain_seed,
)
from pace_eco_lab.multi_terrain_state_audit import audit_difficulty_table, audit_surface_profiles
from pace_eco_lab.multi_terrain_statistics import summarize_rows, validate_formal_rows
from pace_eco_lab.terrain_geometry import (
    LongTerrainAuditCfg,
    geometry_audit,
    long_terrain,
    resolved_parameters,
)
from pace_eco_lab.terrain_boundary import (
    ANYMAL_D_FOOT_CENTER_CHAIN_BOUND_M,
    FOOT_BOUNDARY_AUDIT_VERSION,
    FOOT_CENTER_MAX_BASE_DISTANCE_M,
    base_near_tile_edge,
    contact_foot_tile_state,
)


ROOT = Path(__file__).resolve().parents[2]


def test_tasks_are_isolated_and_ppo_seed_roles_match_flat_protocol():
    assert len(MULTI_TERRAIN_TASK_IDS) == len(set(MULTI_TERRAIN_TASK_IDS)) == 18
    assert not set(MULTI_TERRAIN_TASK_IDS) & {TASK_ONLY_ID, FIXED_WEIGHT_ID, ECO_ID}
    assert set(TASK_IDS) == {
        (method, terrain)
        for method in ("task_only", "fixed_weight", "eco")
        for terrain in (*TERRAIN_NAMES, "mixed")
    }
    assert all("Terrain20sWide" in task_id for task_id in MULTI_TERRAIN_TASK_IDS)
    assert PPO_SEEDS["stage1_budget"] == PPO_SEEDS["stage2_budget"] == (0,)
    assert PPO_SEEDS["stage1_formal"] == PPO_SEEDS["stage2_formal"] == (1, 2, 3, 4, 5)
    for stage in ("stage1", "stage2"):
        assert set(PPO_SEEDS[f"{stage}_budget"]).isdisjoint(PPO_SEEDS[f"{stage}_formal"])


def test_machine_readable_frozen_config_matches_protocol_constants():
    frozen = json.loads((ROOT / "gpt-多地形补充实验冻结配置.json").read_text(encoding="utf-8"))
    assert frozen["协议版本"] == PROTOCOL_VERSION == "gpt-multi-terrain-v1.4"
    assert "B_ref待标定" in frozen["冻结状态"]
    assert frozen["唯一实验变量"] == "地形生成分布"
    assert frozen["长地形"] == {
        "块长度_m": TERRAIN_LENGTH_M,
        "块宽度_m": TERRAIN_WIDTH_M,
        "出生origin_x_m": TERRAIN_ORIGIN_X_M,
        "有效几何末端_x_m": TERRAIN_ACTIVE_END_X_M,
        "机身路线预警前向_m": MAX_AUDITED_FORWARD_M,
        "机身路线预警后向_m": MAX_AUDITED_BACKWARD_M,
        "机身路线预警侧向_m": MAX_AUDITED_LATERAL_M,
        "最大回合_s": 20.0,
        "评估预热_s": EVALUATION_WARMUP_S,
        "评估回合数": EVAL_EPISODES,
    }
    assert frozen["PPO_seeds"] == {key: list(value) for key, value in PPO_SEEDS.items()}
    assert all("Terrain20sWide-Anymal-D-v0" in value for value in frozen["任务ID模板"].values())
    assert frozen["训练"]["地形行"] == 5
    assert frozen["训练"]["地形列"] == TRAIN_TERRAIN_COLS
    assert frozen["训练"]["地形实例数"] == 5 * TRAIN_TERRAIN_COLS == 50
    assert frozen["评估"]["批次数"] == EVAL_BATCHES
    assert frozen["评估"]["每批环境数"] == EVAL_NUM_ENVS
    assert frozen["评估"]["总回合数"] == EVAL_EPISODES
    assert frozen["评估"]["calibration初始状态集"] == MULTI_TERRAIN_CALIBRATION_STATE_SET
    assert frozen["评估"]["holdout初始状态集"] == MULTI_TERRAIN_HOLDOUT_STATE_SET
    assert frozen["地形seed基数"] == {
        role: terrain_seed(role, "flat") for role in frozen["地形seed基数"]
    }
    assert frozen["输出根"] == {
        "训练": "logs/supplementary/multi_terrain_v1_4",
        "评估": "results/supplementary/multi_terrain_v1_4",
    }
    assert frozen["工作量"] == {
        "阶段一训练": 80,
        "阶段一评估": 80,
        "阶段二训练": 16,
        "阶段二评估": 16,
        "合计训练": 96,
        "合计评估": 96,
        "评估GPU进程启动次数": 384,
    }
    assert frozen["固定权重主基线"]["系数"] == PACE_PAPER_FIXED_WEIGHT
    assert frozen["固定权重主基线"]["标签"] == PACE_PAPER_FIXED_WEIGHT_LABEL
    assert frozen["边界审计修订"]["版本"] == FOOT_BOUNDARY_AUDIT_VERSION
    assert frozen["边界审计修订"]["硬拒绝"] == "任一回合发生接触足端中心越过真实地形块边缘"
    assert frozen["边界审计修订"]["失败越界处置"] == "同样硬拒绝整份评估，不写正式JSON或CSV"


def test_wide_cpu_geometry_audit_matches_frozen_protocol():
    audit = json.loads(
        (ROOT / "gpt-多地形Terrain20sWide几何静态审计.json").read_text(encoding="utf-8")
    )
    assert audit["协议版本"] == PROTOCOL_VERSION
    assert audit["边界审计版本"] == FOOT_BOUNDARY_AUDIT_VERSION
    assert audit["审计状态"] == "全部通过"
    assert audit["实例数"] == 63
    assert "后-30m、前35m、左右±30m" in audit["跨块结论"]
    assert all(all(item["检查"].values()) for item in audit["逐实例"])


def test_contact_foot_real_tile_boundary_distinguishes_contact_and_swing_feet():
    positions = torch.tensor(
        [
            [
                [34.99, 0.00, 0.0],
                [0.00, 30.01, 0.0],
                [0.00, -30.20, 0.0],
                [-30.01, 0.00, 0.0],
            ]
        ],
        dtype=torch.float64,
    )
    forces = torch.tensor(
        [
            [
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        ]
    )
    state = contact_foot_tile_state(
        positions,
        forces,
        torch.zeros((1, 3), dtype=torch.float64),
        terrain_length_m=TERRAIN_LENGTH_M,
        terrain_width_m=TERRAIN_WIDTH_M,
        terrain_origin_x_m=TERRAIN_ORIGIN_X_M,
        contact_force_threshold_n=1.0,
    )
    assert state["contact_outside"].tolist() == [[False, True, False, True]]
    assert state["swing_outside"].tolist() == [[False, False, True, False]]
    assert state["edge_margin_m"].tolist()[0] == pytest.approx([0.01, -0.01, -0.20, -0.01])


def test_contact_foot_boundary_uses_strict_force_and_edge_thresholds():
    positions = torch.tensor([[[35.0, 5.0, 0.0], [35.001, 0.0, 0.0]]])
    forces = torch.tensor([[[0.0, 0.0, 2.0], [0.0, 0.0, 1.0]]])
    state = contact_foot_tile_state(
        positions,
        forces,
        torch.zeros((1, 3)),
        terrain_length_m=TERRAIN_LENGTH_M,
        terrain_width_m=TERRAIN_WIDTH_M,
        terrain_origin_x_m=TERRAIN_ORIGIN_X_M,
        contact_force_threshold_n=1.0,
    )
    assert state["outside"].tolist() == [[False, True]]
    assert state["contact"].tolist() == [[True, False]]
    assert not state["contact_outside"].any()


def test_foot_position_prefilter_uses_conservative_anymal_d_chain_bound():
    assert ANYMAL_D_FOOT_CENTER_CHAIN_BOUND_M == pytest.approx(1.1351, abs=1.0e-4)
    assert ANYMAL_D_FOOT_CENTER_CHAIN_BOUND_M < FOOT_CENTER_MAX_BASE_DISTANCE_M == 1.25
    displacements = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 28.749, 0.0],
            [0.0, 28.750, 0.0],
            [33.749, 0.0, 0.0],
            [33.750, 0.0, 0.0],
            [-28.749, 0.0, 0.0],
            [-28.750, 0.0, 0.0],
        ]
    )
    state = base_near_tile_edge(
        displacements,
        terrain_length_m=TERRAIN_LENGTH_M,
        terrain_width_m=TERRAIN_WIDTH_M,
        terrain_origin_x_m=TERRAIN_ORIGIN_X_M,
    )
    assert state["audit_required"].tolist() == [False, False, True, False, True, False, True]
    assert torch.equal(state["kinematically_certified_safe"], ~state["audit_required"])


def test_all_multi_terrain_gym_entries_are_isolated_string_registrations():
    for task_id in TASK_IDS.values():
        spec = gym.spec(task_id)
        assert spec.entry_point == "pace_eco_lab.envs.terrain20s_env:PaceTerrain20sRLEnv"
        assert "multi_terrain_env_cfg:Pace" in spec.kwargs["env_cfg_entry_point"]
        assert "multi_terrain_agent_cfg:Pace" in spec.kwargs["rsl_rl_cfg_entry_point"]
    for task_id in (TASK_ONLY_ID, FIXED_WEIGHT_ID, ECO_ID):
        assert gym.spec(task_id).entry_point == "pace_eco_lab.envs.pace_env:PaceManagerBasedRLEnv"


def test_task_only_config_post_init_only_replaces_terrain_and_audit_metadata():
    path = ROOT / "pace_eco_lab/configs/multi_terrain_env_cfg.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PaceTerrain20sTaskOnlyEnvCfg"
    )
    assert [base.id for base in class_node.bases if isinstance(base, ast.Name)] == ["PaceTaskOnlyEnvCfg"]
    post_init = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "__post_init__"
    )
    source = ast.unparse(post_init)
    assert "super().__post_init__()" in source
    assert "self.scene.terrain = make_long_terrain_importer" in source
    assert "self.sim.physics_material = self.scene.terrain.physics_material" in source
    for forbidden in (
        "self.episode_length_s",
        "self.decimation",
        "self.rewards",
        "self.terminations",
        "self.curriculum",
        "self.commands",
        "self.observations",
        "self.actions",
        "self.events",
    ):
        assert forbidden not in source


def test_source_has_no_route_reward_termination_or_performance_curriculum():
    env_source = (ROOT / "pace_eco_lab/configs/multi_terrain_env_cfg.py").read_text(encoding="utf-8")
    runtime_source = (ROOT / "pace_eco_lab/envs/terrain20s_env.py").read_text(encoding="utf-8")
    boundary_source = (ROOT / "pace_eco_lab/terrain_boundary.py").read_text(encoding="utf-8")
    mdp_source = (ROOT / "pace_eco_lab/mdp/multi_terrain.py").read_text(encoding="utf-8")
    eval_source = (ROOT / "scripts/pace_eco/multi_terrain_eval.py").read_text(encoding="utf-8")
    assert "class PaceTerrain20sTaskOnlyEnvCfg(PaceTaskOnlyEnvCfg)" in env_source
    assert "self.rewards.energy.weight = PACE_PAPER_FIXED_WEIGHT" in env_source
    assert "route_completion_reward" not in env_source + mdp_source
    assert "route_complete" not in env_source + mdp_source
    assert "lateral_out_of_bounds" not in env_source + mdp_source
    assert "terrain_levels" not in mdp_source
    assert "完整20秒" in eval_source and "EVALUATION_WARMUP_S" in eval_source
    assert "terrain_reference" not in eval_source
    assert "agent_cfg.algorithm.energy_budget_j = energy_budget" in eval_source
    assert "contact_foot_tile_state" in runtime_source
    assert "base_near_tile_edge" in runtime_source
    assert "FOOT_CENTER_MAX_BASE_DISTANCE_M" in runtime_source
    assert "pace_terrain20s_contact_foot_boundary_crossed" in runtime_source
    assert "contact_outside" in boundary_source and "swing_outside" in boundary_source
    assert "RewardTerm" not in boundary_source and "TerminationTerm" not in boundary_source
    assert 'boundary_crossings = [row for row in rows if row["接触足越过真实地形边缘"]]' in eval_source
    assert "successful_crossings" not in eval_source


def test_wide_protocol_requires_exact_current_training_implementation():
    eval_source = (ROOT / "scripts/pace_eco/multi_terrain_eval.py").read_text(encoding="utf-8")
    assert "require_current_implementation=True" in eval_source
    assert "评估审计源码兼容" not in eval_source
    assert "_AMENDED_" not in eval_source


def test_multi_terrain_evaluation_has_progress_and_stall_guards():
    eval_source = (ROOT / "scripts/pace_eco/multi_terrain_eval.py").read_text(encoding="utf-8")
    launcher = (ROOT / "scripts/pace_eco/运行多地形评估.sh").read_text(encoding="utf-8")
    assert "EVALUATION_PROGRESS_INTERVAL_STEPS = 100" in eval_source
    assert "EVALUATION_PROGRESS_INTERVAL_S = 30.0" in eval_source
    assert "EVALUATION_STALL_TRACEBACK_S = 300.0" in eval_source
    assert "faulthandler.dump_traceback_later" in eval_source
    assert "Terrain20sWide 运行心跳" in eval_source
    assert "physics_context.use_gpu_sim" in eval_source
    assert "physics_context.use_gpu_pipeline" in eval_source
    assert "timeout --signal=TERM --kill-after=60s 30m" in launcher
    assert 'for batch_index in 0 1 2 3' in launcher
    assert '--batch_index "${batch_index}"' in launcher


def test_capacity_smoke_uses_formal_environment_count_but_only_two_updates():
    source = (ROOT / "scripts/pace_eco/运行多地形容量冒烟.sh").read_text(encoding="utf-8")
    assert '--num_envs 4096' in source
    assert '--max_iterations 2' in source
    assert 'smoke_capacity' in source
    assert 'capacity_smoke_train' in source
    assert 'model_1.pt' in source
    assert 'model_count_before + 1' in source
    assert 'formal_train' not in source
    assert 'PACE_MULTI_LOG_ROOT}/capacity_smoke' in source


def test_capacity_smoke_reuses_smoke_seed_sequence_without_changing_data_split():
    for stage, seed in (("stage1", 900), ("stage2", 901)):
        ordinary = terrain_seed(f"{stage}_smoke_train", "rough", seed)
        capacity = terrain_seed(f"{stage}_capacity_smoke_train", "rough", seed)
        assert capacity == ordinary

    train_source = (ROOT / "scripts/pace_eco/train.py").read_text(encoding="utf-8")
    assert 'expected_envs = 4_096 if capacity_smoke or not smoke else 16' in train_source
    assert 'traceback.print_exc()' in train_source


@pytest.mark.parametrize(
    ("relative_path", "function_name"),
    (
        ("scripts/pace_eco/_多地形公共.sh", "pace_multi_validate_gpu"),
        ("scripts/pace_eco/_训练命令公共.sh", "pace_validate_gpu_id"),
    ),
)
@pytest.mark.parametrize("gpu", range(6))
def test_gpu_validators_accept_all_six_devices(
    relative_path: str,
    function_name: str,
    gpu: int,
):
    script = ROOT / relative_path
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1" && "$2" "$3"',
            "gpu-validator",
            str(script),
            function_name,
            str(gpu),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("relative_path", "function_name"),
    (
        ("scripts/pace_eco/_多地形公共.sh", "pace_multi_validate_gpu"),
        ("scripts/pace_eco/_训练命令公共.sh", "pace_validate_gpu_id"),
    ),
)
def test_gpu_validators_still_reject_out_of_range_device(
    relative_path: str,
    function_name: str,
):
    script = ROOT / relative_path
    completed = subprocess.run(
        ["bash", "-c", 'source "$1" && "$2" 6', "gpu-validator", str(script), function_name],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "0、1、2、3、4 或 5" in completed.stderr


def test_multi_terrain_evaluation_imports_every_boundary_constant_it_records():
    eval_source = (ROOT / "scripts/pace_eco/multi_terrain_eval.py").read_text(encoding="utf-8")
    protocol_import = eval_source.split(
        "from pace_eco_lab.multi_terrain_protocol import (", 1
    )[1].split(")", 1)[0]
    assert "TERRAIN_LENGTH_M" in protocol_import
    assert "TERRAIN_ORIGIN_X_M" in protocol_import
    assert "TERRAIN_WIDTH_M" in protocol_import


def test_frozen_column_distribution_is_exact_for_train_and_200_episode_eval():
    assert set(TERRAIN_DISTRIBUTIONS) == {*TERRAIN_NAMES, "mixed"}
    training = terrain_column_metadata("mixed", TRAIN_TERRAIN_COLS)
    evaluation = terrain_column_metadata("mixed", EVAL_TERRAIN_COLS)
    assert Counter(training) == Counter(
        {
            "flat:level": 2,
            "rough:level": 2,
            "stairs:up": 1,
            "stairs:down": 1,
            "boxes:level": 2,
            "slope:up": 1,
            "slope:down": 1,
        }
    )
    assert Counter(evaluation) == Counter(training)
    assert EVAL_TERRAIN_ROWS * EVAL_TERRAIN_COLS == EVAL_NUM_ENVS == 50
    assert EVAL_BATCHES * EVAL_NUM_ENVS == EVAL_EPISODES == 200


def test_evaluation_batches_have_unique_seeds_ids_and_balanced_state_offsets():
    base_seed = terrain_seed("stage1_calibration", "flat")
    assert [evaluation_batch_seed(base_seed, index) for index in range(EVAL_BATCHES)] == [
        base_seed,
        base_seed + 1,
        base_seed + 2,
        base_seed + 3,
    ]
    global_ids = [
        evaluation_global_id(batch, env_id)
        for batch in range(EVAL_BATCHES)
        for env_id in range(EVAL_NUM_ENVS)
    ]
    assert global_ids == list(range(EVAL_EPISODES))
    assert Counter(value % 8 for value in global_ids) == Counter({index: 25 for index in range(8)})
    with pytest.raises(ValueError, match="评估批次"):
        evaluation_batch_seed(base_seed, EVAL_BATCHES)
    assert [evaluation_batch_offset(index) for index in range(EVAL_BATCHES)] == [0, 50, 100, 150]
    with pytest.raises(ValueError, match="批次内编号"):
        evaluation_global_id(0, EVAL_NUM_ENVS)


@pytest.mark.parametrize(
    ("category", "direction"),
    (
        ("flat", "level"),
        ("rough", "level"),
        ("stairs", "up"),
        ("stairs", "down"),
        ("boxes", "level"),
        ("slope", "up"),
        ("slope", "down"),
    ),
)
def test_long_geometry_is_finite_deterministic_and_inside_tile(category: str, direction: str):
    cfg = LongTerrainAuditCfg(category=category, direction=direction, seed=142_000)
    first = geometry_audit(cfg, 0.7)
    second = geometry_audit(cfg, 0.7)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["bounds_min"][0:2] == pytest.approx([0.0, 0.0])
    assert first["bounds_max"][0:2] == pytest.approx([TERRAIN_LENGTH_M, TERRAIN_WIDTH_M])
    assert first["origin"][0:2] == pytest.approx([TERRAIN_ORIGIN_X_M, TERRAIN_WIDTH_M / 2.0])
    assert TERRAIN_ORIGIN_X_M + MAX_AUDITED_FORWARD_M < TERRAIN_LENGTH_M
    if category in ("stairs", "slope"):
        elevation = first["parameters"]["total_elevation_change_m"]
        assert elevation > 0.0 if direction == "up" else elevation < 0.0


def test_box_instances_respect_frozen_spacing_and_do_not_block_width():
    specs = resolved_parameters(LongTerrainAuditCfg(category="boxes", seed=143_000), 0.9)["instances"]
    assert 5 <= len(specs) <= 9
    previous = None
    for item in specs:
        left = item["x_center_m"] - item["forward_size_m"] / 2.0
        right = item["x_center_m"] + item["forward_size_m"] / 2.0
        assert item["lateral_size_m"] <= 2.4
        if previous is not None:
            assert 0.45 <= left - previous <= 1.10
        previous = right


def test_slope_has_flat_start_and_long_end_platforms():
    cfg = LongTerrainAuditCfg(category="slope", direction="up", seed=144_000)
    meshes, origin = long_terrain(0.9, cfg)
    vertices = meshes[0].vertices
    start = vertices[vertices[:, 0] <= TERRAIN_ORIGIN_X_M + cfg.spawn_platform_m + 1.0e-9]
    end = vertices[vertices[:, 0] >= TERRAIN_ACTIVE_END_X_M - 1.5 - 1.0e-9]
    assert len(set(start[:, 2])) == 1
    assert len(set(end[:, 2])) == 1
    assert origin[2] == pytest.approx(start[0, 2])


def test_terrain20s_initial_states_are_exact_historical_flat_states():
    assert MULTI_TERRAIN_EVALUATION_STATE_SETS == (
        MULTI_TERRAIN_CALIBRATION_STATE_SET,
        MULTI_TERRAIN_HOLDOUT_STATE_SET,
    )
    assert evaluation_state_definition(MULTI_TERRAIN_CALIBRATION_STATE_SET) == evaluation_state_definition(
        CALIBRATION_STATE_SET
    )
    assert evaluation_state_definition(MULTI_TERRAIN_HOLDOUT_STATE_SET) == evaluation_state_definition(
        HOLDOUT_STATE_SET
    )
    assert evaluation_state_definition_sha256(MULTI_TERRAIN_CALIBRATION_STATE_SET) == (
        evaluation_state_definition_sha256(CALIBRATION_STATE_SET)
    )
    assert evaluation_state_definition_sha256(MULTI_TERRAIN_HOLDOUT_STATE_SET) == (
        evaluation_state_definition_sha256(HOLDOUT_STATE_SET)
    )


def _stat_row(method: str, terrain: str, seed: int, energy: float, success: bool = True):
    return {
        "方法": method,
        "地形类别": terrain,
        "方向": "up" if terrain in ("stairs", "slope") else "level",
        "难度": "中",
        "PPO_seed": str(seed),
        "完整20秒": str(success),
        "非法终止": str(not success),
        "越过长地形安全边界": "False",
        "成功": str(success),
        "B80联合合格": str(success),
        "回合能耗_J": str(energy),
        "单位前进距离能耗_J_m": str(energy / 20.0),
        "单位实际路径能耗_J_m": str(energy / 20.5),
        "归一化能耗_E_t除以B_ref_t": str(energy / 100.0),
    }


def test_macro_statistics_are_terrain_equal_and_energy_savings_are_paired():
    rows = [
        _stat_row(method, terrain, seed, energy)
        for method, energy in {"task_only": 100.0, "fixed_weight": 80.0, "eco": 70.0}.items()
        for terrain in TERRAIN_NAMES
        for seed in (1, 2)
    ]
    summary = summarize_rows(rows)
    assert summary["跨地形宏平均与最差地形"]["eco"]["地形等权宏平均联合合格率"] == 1.0
    savings = {
        (item["方法"], item["地形"]): item["相对任务型PPO节能比例"]
        for item in summary["逐地形相对任务型节能汇总"]
    }
    assert savings[("fixed_weight", "flat")] == pytest.approx(0.2)
    assert savings[("eco", "slope")] == pytest.approx(0.3)


def test_fixed_weight_and_absolute_j_budget_are_frozen_in_launchers():
    train = (ROOT / "scripts/pace_eco/train.py").read_text(encoding="utf-8")
    train_wrapper = (ROOT / "scripts/pace_eco/运行多地形训练.sh").read_text(encoding="utf-8")
    common = (ROOT / "scripts/pace_eco/_多地形公共.sh").read_text(encoding="utf-8")
    selector = (ROOT / "scripts/pace_eco/select_multi_terrain_fixed_weight.py").read_text(encoding="utf-8")
    assert "pace_paper_W100" in train_wrapper
    assert "normalized_budget_ratio" not in train_wrapper + train
    assert "terrain_reference" not in train
    assert "agent_cfg.algorithm.energy_budget_j = _load_energy_budget" in train
    assert "multi_terrain_v1_4" in common
    assert "不允许网格选择" in selector
    assert PACE_PAPER_FIXED_WEIGHT == pytest.approx(-1.6e-4)
    assert DEFERRED_FIXED_WEIGHT_CANDIDATES[PACE_PAPER_FIXED_WEIGHT_LABEL] == pytest.approx(
        PACE_PAPER_FIXED_WEIGHT
    )


def test_budget_freezer_requires_current_boundary_audit_and_rejects_any_crossing():
    source = (ROOT / "scripts/pace_eco/freeze_multi_terrain_budgets.py").read_text(
        encoding="utf-8"
    )
    holdout_source = (ROOT / "scripts/pace_eco/freeze_multi_terrain_holdout.py").read_text(
        encoding="utf-8"
    )
    assert "audit_versions != {FOOT_BOUNDARY_AUDIT_VERSION}" in source
    assert "忽略归档的旧边界审计 calibration" not in source
    assert 'row.get("接触足越过真实地形边缘")' in source
    assert 'references.get("边界审计版本") != FOOT_BOUNDARY_AUDIT_VERSION' in holdout_source
    assert '"边界审计版本": FOOT_BOUNDARY_AUDIT_VERSION' in holdout_source


def test_state_audit_accepts_geometry_bands_without_curriculum_transitions():
    table = np.asarray(((0.01, 0.19), (0.21, 0.39), (0.41, 0.59), (0.61, 0.79), (0.81, 0.99)))
    result = audit_difficulty_table(table, 0.0, 1.0)
    assert result["逐行难度带"] and result["同列严格递增"]


def test_state_audit_accepts_finite_up_and_down_profiles_and_rejects_infinite_tail():
    x = np.arange(-2.0, 30.0001, 0.25)
    y = np.arange(-3.0, 3.0001, 0.2)
    active = np.clip((x - 1.5) / 26.0, 0.0, 1.0)
    up = active[:, None] * np.ones((1, len(y)))
    down = -up
    result = audit_surface_profiles("slope", ("up", "down"), x, y, np.stack((up, down)))
    assert result["全部通过"]
    infinite = np.maximum(x - 1.5, 0.0)[:, None] * np.ones((1, len(y)))
    assert not audit_surface_profiles("slope", ("up",), x, y, infinite[None])["全部通过"]


def test_final_statistics_reject_missing_and_accept_complete_stage2_design():
    row = {
        "协议版本": PROTOCOL_VERSION,
        "边界审计版本": FOOT_BOUNDARY_AUDIT_VERSION,
        "阶段": "stage1",
        "数据拆分": "holdout",
        "方法": "task_only",
        "地形类别": "flat",
        "PPO_seed": "1",
        "评估初始状态集": MULTI_TERRAIN_CALIBRATION_STATE_SET,
        "成功": "False",
        "接触足越过真实地形边缘": "False",
    }
    with pytest.raises(ValueError, match="holdout"):
        validate_formal_rows([row], "stage1")
    rows = [
        {
            "协议版本": PROTOCOL_VERSION,
            "边界审计版本": FOOT_BOUNDARY_AUDIT_VERSION,
            "阶段": "stage2",
            "数据拆分": "holdout",
            "方法": method,
            "地形类别": terrain,
            "PPO_seed": str(seed),
            "评估初始状态集": MULTI_TERRAIN_HOLDOUT_STATE_SET,
            "成功": "False",
            "接触足越过真实地形边缘": "False",
        }
        for method in ("task_only", "fixed_weight", "eco")
        for terrain in TERRAIN_NAMES
        for seed in PPO_SEEDS["stage2_formal"]
        for _ in range(EVAL_EPISODES // len(TERRAIN_NAMES))
    ]
    validate_formal_rows(rows, "stage2")


def test_final_statistics_rejects_old_audit_and_any_boundary_crossing():
    base = {
        "协议版本": PROTOCOL_VERSION,
        "边界审计版本": FOOT_BOUNDARY_AUDIT_VERSION,
        "阶段": "stage1",
        "数据拆分": "holdout",
        "方法": "task_only",
        "地形类别": "flat",
        "PPO_seed": "1",
        "评估初始状态集": MULTI_TERRAIN_HOLDOUT_STATE_SET,
        "成功": "False",
        "接触足越过真实地形边缘": "True",
    }
    old = {**base, "边界审计版本": "gpt-四足真实地形边界-v2"}
    with pytest.raises(ValueError, match="边界审计"):
        validate_formal_rows([old], "stage1")
    with pytest.raises(ValueError, match="任一接触足"):
        validate_formal_rows([base], "stage1")


def test_statistics_preserve_zero_success_seed_as_not_computable():
    rows = [
        _stat_row(
            method,
            terrain,
            1,
            100.0,
            not (method == "task_only" and terrain == "flat"),
        )
        for method in ("task_only", "fixed_weight", "eco")
        for terrain in TERRAIN_NAMES
    ]
    summary = summarize_rows(rows)
    flat_fixed = next(
        item
        for item in summary["同地形同seed相对任务型节能"]
        if item["方法"] == "fixed_weight" and item["地形"] == "flat"
    )
    task_flat = next(
        item
        for item in summary["逐seed逐地形"]
        if item["方法"] == "task_only" and item["地形"] == "flat"
    )
    assert flat_fixed["相对任务型PPO节能比例"] is None
    assert task_flat["失败回合数"] == task_flat["未完整20秒回合数"] == 1
    assert summary["跨地形宏平均与最差地形"]["task_only"][
        "地形等权宏平均单位前进距离能耗_J_m"
    ] is None
