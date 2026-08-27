from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

from pace_eco_lab.direction_conditioned_v2_2_protocol import (
    DIRECTION_CONDITIONED_V2_2_TASK_IDS,
    FIXED_ENERGY_REWARD_WEIGHT as V2_2_WEIGHT,
    TARGET_MATRIX as V2_2_TARGET_MATRIX,
)
from pace_eco_lab.direction_conditioned_v2_3_mixed_protocol import (
    DIRECTION_CONDITIONED_V2_3_MIXED_TASK_IDS,
    EVAL_EPISODES,
    FIXED_ENERGY_REWARD_WEIGHT,
    FORMAL_SEEDS,
    NUM_ENVS,
    PROTOCOL_VERSION,
    SUBTASK_COUNTS_PER_BATCH,
    TASK_IDS,
    TRAINING_UPDATES,
    evaluation_batch_seed,
    terrain_seed,
    training_target,
)
from pace_eco_lab.mdp.energy import compute_energy_components


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_SCRIPT = ROOT / "scripts/pace_eco/freeze_direction_conditioned_v2_3_mixed_manifests.py"
BUDGET_SCRIPT = ROOT / "scripts/pace_eco/freeze_direction_conditioned_v2_3_mixed_budget.py"
FROZEN_CONFIG = ROOT / "gpt-方向条件v2.3_Mixed冻结配置.json"


def _run(*args: object) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, *(str(value) for value in args)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_frozen_matrix_and_v2_2_backward_compatibility():
    assert len(DIRECTION_CONDITIONED_V2_3_MIXED_TASK_IDS) == 3
    assert set(DIRECTION_CONDITIONED_V2_3_MIXED_TASK_IDS).isdisjoint(
        DIRECTION_CONDITIONED_V2_2_TASK_IDS
    )
    assert len(V2_2_TARGET_MATRIX) == 53
    assert V2_2_WEIGHT == FIXED_ENERGY_REWARD_WEIGHT == -1.6e-4
    assert FORMAL_SEEDS == (1, 2, 3)
    assert NUM_ENVS == 4096
    assert TRAINING_UPDATES == 3000
    assert sum(SUBTASK_COUNTS_PER_BATCH.values()) == 50
    frozen = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
    assert frozen["协议版本"] == PROTOCOL_VERSION
    assert frozen["每50环境分布"] == SUBTASK_COUNTS_PER_BATCH
    assert frozen["固定权重"] == {"标签": "W100", "能耗奖励系数": -1.6e-4}


@pytest.mark.parametrize("seed", FORMAL_SEEDS)
def test_same_seed_three_methods_share_terrain_seed(seed):
    assert len({terrain_seed("formal_train", seed) for _ in TASK_IDS}) == 1
    assert all(training_target(method, seed, "formal_train") for method in TASK_IDS)


def test_roles_and_seed_segments_are_frozen():
    assert training_target("fixed_weight", 903, "smoke_train")
    assert training_target("task_only", 0, "budget_train")
    assert not training_target("eco", 0, "budget_train")
    ranges = {
        role: {terrain_seed(role, seed) for seed in range(1000)}
        for role in ("smoke_train", "budget_train", "formal_train", "calibration", "holdout")
    }
    for left, left_values in ranges.items():
        for right, right_values in ranges.items():
            if left < right:
                assert left_values.isdisjoint(right_values)


def test_potential_sign_is_up_negative_down_positive():
    common = dict(
        applied_torque=torch.zeros((1, 2)),
        joint_velocity=torch.zeros((1, 2)),
        body_mass=torch.tensor([[2.0, 3.0]]),
        electrical_coefficient=0.0,
    )
    up = compute_energy_components(
        body_vertical_velocity_world=torch.full((1, 2), 0.4), **common
    )
    down = compute_energy_components(
        body_vertical_velocity_world=torch.full((1, 2), -0.4), **common
    )
    assert up.potential.item() < 0.0
    assert down.potential.item() > 0.0


def test_manifests_are_frozen_before_results_and_have_exact_distribution(tmp_path):
    output = tmp_path / "manifests"
    completed = _run(MANIFEST_SCRIPT, "--output_dir", output)
    assert completed.returncode == 0, completed.stderr
    calibration = json.loads((output / "calibration_manifest.json").read_text(encoding="utf-8"))
    holdout = json.loads((output / "holdout_manifest.json").read_text(encoding="utf-8"))
    for split, payload in (("calibration", calibration), ("holdout", holdout)):
        assert payload["冻结状态"] == "已冻结"
        assert payload["协议版本"] == PROTOCOL_VERSION
        assert payload["数据拆分"] == split
        assert len(payload["逐回合"]) == EVAL_EPISODES
        by_batch = Counter((row["batch_index"], row["subtask"]) for row in payload["逐回合"])
        for batch in range(4):
            assert {subtask: by_batch[(batch, subtask)] for subtask in SUBTASK_COUNTS_PER_BATCH} == SUBTASK_COUNTS_PER_BATCH
            assert {row["batch_seed"] for row in payload["逐回合"] if row["batch_index"] == batch} == {evaluation_batch_seed(split, batch)}
    assert {row["batch_seed"] for row in calibration["逐回合"]}.isdisjoint(
        {row["batch_seed"] for row in holdout["逐回合"]}
    )
    refused = _run(MANIFEST_SCRIPT, "--output_dir", output)
    assert refused.returncode != 0
    assert "拒绝覆盖" in refused.stderr


def test_budget_freeze_uses_independent_thresholds_and_two_level_macro(tmp_path):
    manifests = tmp_path / "manifests"
    assert _run(MANIFEST_SCRIPT, "--output_dir", manifests).returncode == 0
    manifest_path = manifests / "calibration_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    fields = [
        "episode_id", "评估批次", "环境编号", "方向子任务", "地形批次seed",
        "固定初始状态编号", "协议版本", "PPO_seed", "方法", "方向穿越成功",
        "回合能耗_J", "电气能耗_J", "机械能耗_J", "势能能耗_J",
    ]
    for batch in range(4):
        path = calibration / f"gpt-v2.3-Mixed-calibration-batch-{batch}.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for item in manifest["逐回合"]:
                if item["batch_index"] != batch:
                    continue
                potential = -10.0 if item["direction"] == "up" else (10.0 if item["direction"] == "down" else 0.0)
                writer.writerow({
                    "episode_id": item["episode_id"],
                    "评估批次": batch,
                    "环境编号": item["batch_env_number"],
                    "方向子任务": item["subtask"],
                    "地形批次seed": item["batch_seed"],
                    "固定初始状态编号": item["initial_state_number"],
                    "协议版本": PROTOCOL_VERSION,
                    "PPO_seed": 0,
                    "方法": "task_only",
                    "方向穿越成功": True,
                    "回合能耗_J": 120.0 + potential,
                    "电气能耗_J": 100.0,
                    "机械能耗_J": 20.0,
                    "势能能耗_J": potential,
                })
    checkpoint = tmp_path / "model_2999.pt"
    checkpoint.write_bytes(b"frozen-checkpoint")
    output = tmp_path / "gpt-v2.3-Mixed-B_ref-B80冻结.json"
    completed = _run(
        BUDGET_SCRIPT,
        "--calibration_root", calibration,
        "--manifest", manifest_path,
        "--checkpoint", checkpoint,
        "--output", output,
    )
    assert completed.returncode == 0, completed.stderr
    frozen = json.loads(output.read_text(encoding="utf-8"))
    assert frozen["B_ref_mixed_J"] == pytest.approx(120.0)
    assert frozen["B80_mixed_J"] == pytest.approx(96.0)
    assert frozen["七方向子任务"]["stairs-up"]["方向成功回合平均potential能耗_J"] == -10.0
    assert frozen["七方向子任务"]["stairs-down"]["方向成功回合平均potential能耗_J"] == 10.0
