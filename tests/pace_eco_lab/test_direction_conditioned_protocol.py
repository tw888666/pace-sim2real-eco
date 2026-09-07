from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

import gymnasium as gym
import pytest
import torch

import pace_eco_lab  # noqa: F401
from pace_eco_lab.direction_conditioned_protocol import (
    DESIRED_DIRECTION_W,
    DIRECTION_CONDITIONED_TASK_IDS,
    MAX_CROSS_TRACK_DEVIATION_M,
    MIN_DIRECTIONAL_PROGRESS_M,
    PPO_SEEDS,
    PROTOCOL_VERSION,
    TASK_IDS,
    TERRAIN_NAMES,
    VARIANT_NAMES,
    terrain_seed,
)
from pace_eco_lab.directional_metrics import (
    directional_displacement_metrics,
    directional_success,
    normalized_direction,
)
from pace_eco_lab.multi_terrain_protocol import (
    MULTI_TERRAIN_TASK_IDS,
    terrain_seed as legacy_terrain_seed,
)


ROOT = Path(__file__).resolve().parents[2]


def test_v2_task_matrix_is_complete_unique_and_disjoint_from_v1():
    assert len(DIRECTION_CONDITIONED_TASK_IDS) == len(set(DIRECTION_CONDITIONED_TASK_IDS)) == 24
    assert not set(DIRECTION_CONDITIONED_TASK_IDS) & set(MULTI_TERRAIN_TASK_IDS)
    assert set(TASK_IDS) == {
        (variant, method, terrain)
        for variant in VARIANT_NAMES
        for method in ("task_only", "eco")
        for terrain in (*TERRAIN_NAMES, "mixed")
    }
    assert all("Terrain20sWide" in task_id for task_id in DIRECTION_CONDITIONED_TASK_IDS)


def test_v2_seed_spaces_are_frozen_paired_and_disjoint_from_v1():
    assert PPO_SEEDS["stage1_budget"] == PPO_SEEDS["stage2_budget"] == (0,)
    assert PPO_SEEDS["stage1_formal"] == PPO_SEEDS["stage2_formal"] == (1, 2, 3, 4, 5)
    for stage in ("stage1", "stage2"):
        assert set(PPO_SEEDS[f"{stage}_budget"]).isdisjoint(PPO_SEEDS[f"{stage}_formal"])
        for terrain in (*TERRAIN_NAMES, "mixed"):
            for seed in PPO_SEEDS[f"{stage}_formal"]:
                current = terrain_seed(f"{stage}_formal_train", terrain, seed)
                legacy = legacy_terrain_seed(f"{stage}_formal_train", terrain, seed)
                assert current != legacy
    # E1/E2 不进入 seed 函数，因此在相同 stage/terrain/PPO seed 下严格复用几何。
    assert terrain_seed("stage1_formal_train", "rough", 3) == 531_003


def test_directional_geometry_uses_arbitrary_normalized_direction():
    displacement = torch.tensor([[3.0, 4.0], [-3.0, -4.0]])
    direction = torch.tensor([0.0, 2.0])
    progress, cross_track = directional_displacement_metrics(displacement, direction)
    assert progress.tolist() == pytest.approx([4.0, -4.0])
    assert cross_track.tolist() == pytest.approx([-3.0, 3.0])
    assert normalized_direction(direction).tolist() == pytest.approx([0.0, 1.0])


def test_directional_success_keeps_survival_progress_and_cross_track_separate():
    result = directional_success(
        completed_20s=torch.tensor([True, True, True, False]),
        illegal_termination=torch.tensor([False, False, False, False]),
        directional_progress_m=torch.tensor([16.0, 15.999, 18.0, 20.0]),
        max_cross_track_m=torch.tensor([3.0, 0.0, 3.001, 0.0]),
        minimum_progress_m=MIN_DIRECTIONAL_PROGRESS_M,
        maximum_cross_track_m=MAX_CROSS_TRACK_DEVIATION_M,
    )
    assert result.tolist() == [True, False, False, False]


@pytest.mark.parametrize("bad", (torch.tensor([0.0, 0.0]), torch.tensor([1.0, float("nan")])) )
def test_direction_validation_rejects_invalid_vectors(bad: torch.Tensor):
    with pytest.raises(ValueError):
        normalized_direction(bad)


def test_v2_gym_entries_use_independent_configs_and_same_audited_runtime():
    for (variant, method, _terrain), task_id in TASK_IDS.items():
        spec = gym.spec(task_id)
        assert spec.entry_point == "pace_eco_lab.envs.terrain20s_env:PaceTerrain20sRLEnv"
        assert "direction_conditioned_env_cfg:Pace" in spec.kwargs["env_cfg_entry_point"]
        assert "direction_conditioned_agent_cfg:Pace" in spec.kwargs["rsl_rl_cfg_entry_point"]
        assert ("DirectionObservationControl" in spec.kwargs["env_cfg_entry_point"]) == (
            variant == "observation_control"
        )
        assert ("TaskOnly" in spec.kwargs["env_cfg_entry_point"]) == (method == "task_only")


def test_v2_reward_source_changes_only_world_xy_linear_reference():
    source = (ROOT / "pace_eco_lab/mdp/rewards.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: ast.unparse(node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    legacy = functions["pace_velocity_tracking"]
    directional = functions["pace_directional_velocity_tracking"]
    yaw = functions["pace_yaw_rate_tracking"]
    assert "root_lin_vel_b[:, :2]" in legacy
    assert "root_ang_vel_b[:, 2]" in legacy
    assert "root_lin_vel_w[:, :2]" in directional
    assert "root_lin_vel_w[:, 2]" not in directional
    assert "direction_w * command_term.target_speed" in directional
    assert "root_ang_vel_b[:, 2]" in yaw
    assert all("/ env.step_dt" in item for item in (legacy, directional, yaw))


def test_v2_environment_source_preserves_e1_reward_and_equal_e2_scale():
    source = (ROOT / "pace_eco_lab/configs/direction_conditioned_env_cfg.py").read_text(
        encoding="utf-8"
    )
    assert "class PaceDirectionObservationControlTerrain20sTaskOnlyEnvCfg" in source
    assert "rewards:" not in source.split(
        "class PaceDirectionObservationControlTerrain20sTaskOnlyEnvCfg", maxsplit=1
    )[1].split("class PaceDirectionConditionedTerrain20sTaskOnlyEnvCfg", maxsplit=1)[0]
    assert 'weight=0.2' in source
    assert source.count('weight=0.2') == 2
    assert 'params={"command_name": "direction_command", "sigma": 0.5}' in source
    assert 'params={"target_yaw_rate": 0.0, "sigma": 0.5}' in source
    assert "velocity = None" in source
    assert '"command_name": "direction_command"' in source


def test_paper_v2_energy_actuator_and_task_randomization_are_configured():
    reward_source = (ROOT / "pace_eco_lab/mdp/rewards.py").read_text(encoding="utf-8")
    env_source = (ROOT / "pace_eco_lab/configs/env_cfg.py").read_text(encoding="utf-8")
    constants_source = (ROOT / "pace_eco_lab/constants.py").read_text(encoding="utf-8")
    assert "gamma_v * env.pace_energy_step" in reward_source
    assert "randomize_ground_friction" in env_source
    assert "push_by_setting_velocity" in env_source
    assert "env_cfg.events.push_robot = None" in env_source
    assert "EFFORT_LIMIT_NM = 89.0" in constants_source
    assert "SATURATION_EFFORT_NM = 140.0" in constants_source
    assert "VELOCITY_LIMIT_RAD_S = 8.5" in constants_source


def test_direction_command_uses_yaw_only_body_transform_and_three_values():
    source = (ROOT / "pace_eco_lab/mdp/direction_command.py").read_text(encoding="utf-8")
    assert "quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w)" in source
    assert "torch.cat((self.direction_b, self.target_speed), dim=-1)" in source
    assert "goal_pos" not in source and "reached" not in source


def test_machine_readable_v2_frozen_config_matches_protocol():
    data = json.loads((ROOT / "gpt-方向条件v2.1冻结配置.json").read_text(encoding="utf-8"))
    assert data["协议版本"] == PROTOCOL_VERSION
    assert data["DirectionCommand"]["世界目标方向"] == list(DESIRED_DIRECTION_W)
    assert data["方向成功"]["最低方向净进度_m"] == MIN_DIRECTIONAL_PROGRESS_M
    assert data["方向成功"]["最大横轨偏离_m"] == MAX_CROSS_TRACK_DEVIATION_M
    assert data["B_ref"]["来源变体"] == "E2 directional"
    assert data["B_ref"]["共享方法"] == ["E1-ECO", "E2-ECO"]
    for role, base in data["地形seed基数"].items():
        assert terrain_seed(role, "flat") == base


def test_v2_evaluator_uses_pre_reset_snapshot_and_distinct_success_metrics():
    source = (ROOT / "scripts/pace_eco/direction_conditioned_eval.py").read_text(
        encoding="utf-8"
    )
    assert 'env_cfg.pace_publish_eval_state = True' in source
    assert 'extras.get("pace_eval_root_pos_w")' in source
    assert "final_xy_snapshot[env_id, :2] - start_xy[env_id]" in source
    assert '"v1历史机身速度成功"' in source
    assert '"生存成功"' in source
    assert '"方向穿越成功"' in source
    assert '"B80联合合格"' in source
    assert "direction_ok and energy <= v2_budget" in source
    assert "heading_error" in source and "航向指标解释" in source


def test_v2_budget_and_holdout_freezers_enforce_pre_registered_sources():
    budget = (ROOT / "scripts/pace_eco/freeze_direction_conditioned_budgets.py").read_text(
        encoding="utf-8"
    )
    holdout = (ROOT / "scripts/pace_eco/freeze_direction_conditioned_holdout.py").read_text(
        encoding="utf-8"
    )
    assert 'rows[0].get("实验变体") != "directional"' in budget
    assert 'rows[0].get("方法") != "task_only"' in budget
    assert 'row["方向穿越成功"]' in budget
    assert "success_rate < 0.95" in budget
    assert "for variant in VARIANT_NAMES" in holdout
    assert 'for method in ("task_only", "eco")' in holdout
    assert 'model_2999.pt' in holdout
    assert "禁止事后挑选" in holdout


@pytest.mark.parametrize(
    "relative_path",
    (
        "scripts/pace_eco/gpt-方向条件公共.sh",
        "scripts/pace_eco/gpt-运行方向条件训练.sh",
        "scripts/pace_eco/gpt-运行方向条件评估.sh",
    ),
)
def test_v2_shell_scripts_parse(relative_path: str):
    subprocess.run(["bash", "-n", str(ROOT / relative_path)], check=True)


def test_v2_shell_task_ids_match_python_protocol():
    common = ROOT / "scripts/pace_eco/gpt-方向条件公共.sh"
    for variant in VARIANT_NAMES:
        completed = subprocess.run(
            [
                "bash",
                "-c",
                'source "$1" && pace_direction_task_id "$2" "$3" "$4"',
                "v2-task",
                str(common),
                variant,
                "rough",
                "eco",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert completed.stdout == TASK_IDS[(variant, "eco", "rough")]
