import torch

from pace_sim2real.utils.identified_parameters import (
    ANYMAL_D_OFFICIAL_PARAMETERS,
    IdentifiedActuatorParameters,
)


def test_anymal_parameter_partition_and_delay_truncation() -> None:
    parameters = IdentifiedActuatorParameters.from_sequence(ANYMAL_D_OFFICIAL_PARAMETERS)

    assert parameters.armature.shape == (12,)
    assert parameters.viscous_friction.shape == (12,)
    assert parameters.static_dynamic_friction.shape == (12,)
    assert parameters.encoder_bias.shape == (12,)
    assert parameters.delay_steps == 3
    torch.testing.assert_close(parameters.as_tensor()[48], torch.tensor(3.2406))


def test_parameter_vector_requires_exactly_49_values() -> None:
    try:
        IdentifiedActuatorParameters.from_sequence([0.0] * 48)
    except ValueError as exc:
        assert "49" in str(exc)
    else:
        raise AssertionError("expected invalid PACE parameter length to fail")
