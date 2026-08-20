import pytest

from pace_eco_lab.evaluation_states import (
    CALIBRATION_STATE_SET,
    EVALUATION_STATE_SETS,
    HOLDOUT_STATE_SET,
    evaluation_state_count,
    evaluation_state_definition,
    evaluation_state_definition_sha256,
)


def test_evaluation_state_sets_are_disjoint_and_inside_training_ranges():
    assert EVALUATION_STATE_SETS == (CALIBRATION_STATE_SET, HOLDOUT_STATE_SET)
    definitions = {
        state_set: evaluation_state_definition(state_set) for state_set in EVALUATION_STATE_SETS
    }
    for root_offsets, joint_scales in definitions.values():
        assert len(root_offsets) == 8
        assert len(joint_scales) == 8
        assert len(set(root_offsets)) == 8
        for row in root_offsets:
            assert len(row) == 12
            x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz = row
            assert -0.1 <= x <= 0.1
            assert -0.1 <= y <= 0.1
            assert z == 0.0
            assert -0.05 <= roll <= 0.05
            assert -0.05 <= pitch <= 0.05
            assert -0.1 <= yaw <= 0.1
            assert -0.25 <= vx <= 0.25
            assert -0.25 <= vy <= 0.25
            assert -0.1 <= vz <= 0.1
            assert -0.2 <= wx <= 0.2
            assert -0.2 <= wy <= 0.2
            assert -0.2 <= wz <= 0.2
        assert all(0.9 <= scale <= 1.1 for scale in joint_scales)

    calibration_roots, calibration_scales = definitions[CALIBRATION_STATE_SET]
    holdout_roots, holdout_scales = definitions[HOLDOUT_STATE_SET]
    assert set(calibration_roots).isdisjoint(holdout_roots)
    assert set(calibration_scales).isdisjoint(holdout_scales)


def test_holdout_states_are_paired_symmetric_perturbations():
    root_offsets, joint_scales = evaluation_state_definition(HOLDOUT_STATE_SET)
    for first, second in zip(root_offsets[::2], root_offsets[1::2], strict=True):
        assert second == pytest.approx(tuple(-value for value in first))
    for first, second in zip(joint_scales[::2], joint_scales[1::2], strict=True):
        assert first + second == pytest.approx(2.0)


def test_evaluation_state_protocol_has_stable_counts_and_hashes():
    assert evaluation_state_count(CALIBRATION_STATE_SET) == 8
    assert evaluation_state_count(HOLDOUT_STATE_SET) == 8
    assert evaluation_state_definition_sha256(CALIBRATION_STATE_SET) == (
        "5e85e721c96d4d8864b2147661a4545eec94659d99922cfcef5da89ae5f85ad7"
    )
    assert evaluation_state_definition_sha256(HOLDOUT_STATE_SET) == (
        "321fb72409a42f1aa67e58148d38a4691d945237c501ebb73676f95a91867a4e"
    )


def test_unknown_evaluation_state_set_is_rejected():
    with pytest.raises(ValueError, match="未知评估状态集"):
        evaluation_state_count("unknown")
