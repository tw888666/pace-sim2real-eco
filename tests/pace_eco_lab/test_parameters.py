from __future__ import annotations

import numpy as np
import pytest

from pace_eco_lab.constants import PACE_FITTING_SHA256, PACE_JOINT_NAMES
from pace_eco_lab.mdp.parameters import file_sha256, load_pace_parameters, map_joint_values


def test_fitting_hash_and_49_slices():
    params = load_pace_parameters()
    assert file_sha256("pace_data/1_in_air/anymal/fitting.npy") == PACE_FITTING_SHA256
    assert params.vector.shape == (49,)
    assert params.armature.shape == (12,)
    assert params.viscous_damping.shape == (12,)
    assert params.coulomb_friction.shape == (12,)
    assert params.joint_bias.shape == (12,)
    assert params.identified_delay == pytest.approx(3.2406, abs=5.0e-5)
    assert params.applied_delay_steps == 3


def test_parameter_ranges():
    params = load_pace_parameters()
    assert np.all(params.armature >= 0.0)
    assert np.all(params.viscous_damping >= 0.0)
    assert np.all(params.coulomb_friction >= 0.0)
    assert np.all(np.abs(params.joint_bias) <= 0.1)
    assert 0.0 <= params.identified_delay <= 7.0


def test_joint_name_mapping_never_uses_runtime_indices():
    source_values = np.arange(12, dtype=np.float64)
    reversed_names = tuple(reversed(PACE_JOINT_NAMES))
    mapped = map_joint_values(source_values, reversed_names)
    np.testing.assert_array_equal(mapped, source_values[::-1])


def test_joint_mapping_rejects_missing_name():
    with pytest.raises(ValueError, match="集合不一致"):
        map_joint_values(range(12), (*PACE_JOINT_NAMES[:-1], "UNKNOWN"))
