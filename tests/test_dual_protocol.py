import json
from dataclasses import asdict

import torch

from pace_sim2real.dual import (
    DualEvaluationResult,
    DualState,
    apply_evaluation_result,
    energy_feasibility_metrics,
    file_sha256,
    freeze_checkpoint,
)


def _write_result(tmp_path, checkpoint, *, cycle_id=0, augmented_cost=1.2):
    result = DualEvaluationResult(
        cycle_id=cycle_id,
        checkpoint_path=str(checkpoint),
        checkpoint_sha256=file_sha256(checkpoint),
        budget_j=100.0,
        num_episodes=256,
        mean_physical_energy_j=95.0,
        mean_augmented_cost=augmented_cost,
        success_rate=0.9,
        cost_value_initial_bias=0.01,
        cost_explained_variance=0.7,
        schema_version=1,
    )
    result_path = tmp_path / f"result-{cycle_id}.json"
    result_path.write_text(json.dumps(asdict(result)), encoding="utf-8")
    return result_path


def _complete_metrics() -> dict:
    return {
        "target_definition": "undiscounted_full_episode_return_to_go",
        "initial": {
            "num_samples": 2,
            "signed_bias": -0.01,
            "absolute_mean_bias": 0.01,
            "mae": 0.02,
            "rmse": 0.03,
            "explained_variance": 0.4,
            "target_mean": 1.0,
            "target_std": 0.1,
            "prediction_mean": 0.99,
            "prediction_std": 0.1,
        },
        "trajectory": {
            "num_episodes": 2,
            "num_transitions": 4,
            "pooled": {
                "num_samples": 4,
                "signed_bias": 0.0,
                "absolute_mean_bias": 0.0,
                "mae": 0.01,
                "rmse": 0.02,
                "explained_variance": 0.8,
                "target_mean": 0.5,
                "target_std": 0.3,
                "prediction_mean": 0.5,
                "prediction_std": 0.3,
            },
            "episode_equal_rmse": 0.02,
        },
        "time_bin_rmse": {"0.0-1.0": 0.02},
        "success": None,
        "failure": None,
    }


def test_dual_update_is_projected_atomic_and_idempotent(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"immutable")
    result_path = _write_result(tmp_path, checkpoint)
    state_path = tmp_path / "state.json"
    initial = DualState(budget_j=100.0, learning_rate=0.1)

    state, changed = apply_evaluation_result(state_path, result_path, initial_state=initial)
    assert changed
    assert abs(state.multiplier - 0.02) < 1.0e-8
    repeated, changed = apply_evaluation_result(state_path, result_path)
    assert not changed
    assert repeated.multiplier == state.multiplier


def test_out_of_order_cycle_is_rejected(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"immutable")
    result_path = _write_result(tmp_path, checkpoint, cycle_id=1)
    state_path = tmp_path / "state.json"
    try:
        apply_evaluation_result(
            state_path,
            result_path,
            initial_state=DualState(budget_j=100.0, learning_rate=0.1),
        )
    except ValueError as exc:
        assert "expected cycle 0" in str(exc)
    else:
        raise AssertionError("out-of-order cycle must fail")


def test_freeze_checkpoint_creates_hash_addressed_read_only_request(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint-v1")
    request_path = freeze_checkpoint(checkpoint, tmp_path / "dual", cycle_id=4, budget_j=123.0)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    frozen = request_path.with_name("checkpoint.pt")

    assert request["cycle_id"] == 4
    assert request["checkpoint_sha256"] == file_sha256(frozen)
    assert frozen.stat().st_mode & 0o222 == 0


def test_v2_result_exposes_unambiguous_gate_properties(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"immutable-v2")
    metrics = _complete_metrics()
    result = DualEvaluationResult(
        cycle_id=0,
        checkpoint_path=str(checkpoint),
        checkpoint_sha256=file_sha256(checkpoint),
        budget_j=100.0,
        num_episodes=2,
        mean_physical_energy_j=100.0,
        mean_augmented_cost=1.0,
        success_rate=1.0,
        evaluation_seed=123,
        cost_value_metrics=metrics,
    )
    result.validate()
    assert result.initial_absolute_mean_bias == 0.01
    assert result.trajectory_explained_variance == 0.8


def test_v3_result_requires_consistent_episode_energy_metrics(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"immutable-v3")
    energy_metrics = energy_feasibility_metrics(
        torch.tensor([80.0, 110.0]),
        torch.tensor([True, False]),
        budget_j=100.0,
    ).to_dict()
    result = DualEvaluationResult(
        cycle_id=0,
        checkpoint_path=str(checkpoint),
        checkpoint_sha256=file_sha256(checkpoint),
        budget_j=100.0,
        num_episodes=2,
        mean_physical_energy_j=95.0,
        mean_augmented_cost=1.0,
        success_rate=0.5,
        evaluation_seed=123,
        cost_value_metrics=_complete_metrics(),
        energy_feasibility_metrics=energy_metrics,
        schema_version=3,
    )
    result.validate()

    tampered = dict(energy_metrics)
    tampered["episode_energy_violation_rate"] = 1.5
    invalid = DualEvaluationResult(
        **{
            **asdict(result),
            "energy_feasibility_metrics": tampered,
        }
    )
    try:
        invalid.validate()
    except ValueError as exc:
        assert "episode_energy_violation_rate" in str(exc)
    else:
        raise AssertionError("an invalid episode violation rate must fail")
