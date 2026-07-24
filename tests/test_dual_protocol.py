import json
from dataclasses import asdict

from pace_sim2real.dual import (
    DualEvaluationResult,
    DualState,
    apply_evaluation_result,
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
    )
    result_path = tmp_path / f"result-{cycle_id}.json"
    result_path.write_text(json.dumps(asdict(result)), encoding="utf-8")
    return result_path


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
