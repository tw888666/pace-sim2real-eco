"""Frozen PACE parameters and helpers for applying them to an articulation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


ANYMAL_D_JOINT_ORDER = (
    "LF_HAA",
    "LF_HFE",
    "LF_KFE",
    "RF_HAA",
    "RF_HFE",
    "RF_KFE",
    "LH_HAA",
    "LH_HFE",
    "LH_KFE",
    "RH_HAA",
    "RH_HFE",
    "RH_KFE",
)

# Official PACE dataset: pace_data/1_in_air/anymal/fitting.npy, field ``params``.
# Source SHA-256: 4436941fa5e9a5e8e1ef93d55956fcffdb4c4c4526b8ef3e145e6ea619fbe1c8.
# The source tensor is float64; values are retained here with sufficient precision.
ANYMAL_D_OFFICIAL_PARAMETERS = (
    0.0043367671446232492, 0.02356833456849558, 0.027642794812147931,
    0.0031218328791020322, 0.017601453791252081, 0.034645468808248647,
    0.0035523434110258656, 0.026603725333277123, 0.036784320068959209,
    0.004672249715600707, 0.011683427500639471, 0.056901954367993945,
    4.9704211815664063, 4.4031685131226048, 5.1754473929119893,
    4.9590557218891389, 5.1028535129882666, 5.4652036818323459,
    4.8223544061454486, 4.3225585702053877, 5.2726738906112249,
    5.1068238395601622, 5.1949220119841879, 5.4336231797625132,
    0.077419141414778514, 0.074217066216105937, 0.063070389543511823,
    0.08183007667720224, 0.043439060576012878, 0.058589618620775025,
    0.078348881240944496, 0.07282275470405225, 0.060723801804437344,
    0.073717467957358507, 0.034290424821876359, 0.043388687438113749,
    0.022428540977364131, 0.0058054093216863001, -0.0034394926778502782,
    -0.012256469123324842, -0.0012743423965716899, -0.0092761431987473664,
    0.011688095025005604, -0.0081613497074311259, 0.0088607188010438459,
    -0.016243242318040751, 0.0043477076012446714, 0.0049138071251358523,
    3.2405974755163758,
)


@dataclass(frozen=True)
class IdentifiedActuatorParameters:
    """The 49-dimensional PACE parameter vector for a 12-DoF robot."""

    armature: torch.Tensor
    viscous_friction: torch.Tensor
    static_dynamic_friction: torch.Tensor
    encoder_bias: torch.Tensor
    delay_steps_continuous: torch.Tensor

    @property
    def delay_steps(self) -> int:
        """PACE uses integer truncation when applying the identified delay."""
        return int(torch.floor(self.delay_steps_continuous).item())

    @classmethod
    def from_sequence(
        cls,
        values: tuple[float, ...] | list[float] | torch.Tensor,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "IdentifiedActuatorParameters":
        tensor = torch.as_tensor(values, device=device, dtype=dtype).flatten()
        if tensor.numel() != 49:
            raise ValueError(f"PACE ANYmal-D parameter vector must contain 49 values, got {tensor.numel()}")
        return cls(
            armature=tensor[0:12],
            viscous_friction=tensor[12:24],
            static_dynamic_friction=tensor[24:36],
            encoder_bias=tensor[36:48],
            delay_steps_continuous=tensor[48],
        )

    def as_tensor(self) -> torch.Tensor:
        return torch.cat(
            (
                self.armature,
                self.viscous_friction,
                self.static_dynamic_friction,
                self.encoder_bias,
                self.delay_steps_continuous.reshape(1),
            )
        )


def actuator_joint_indices(actuator, num_joints: int, device: torch.device) -> torch.Tensor:
    """Normalize an Isaac Lab actuator's joint index representation."""
    indices = actuator.joint_indices
    if isinstance(indices, slice):
        return torch.arange(num_joints, device=device)[indices]
    return torch.as_tensor(indices, device=device, dtype=torch.long)


def articulation_encoder_bias(articulation) -> torch.Tensor:
    """Return encoder bias in articulation joint order for every environment."""
    bias = torch.zeros_like(articulation.data.joint_pos)
    for actuator in articulation.actuators.values():
        if not hasattr(actuator, "encoder_bias"):
            continue
        joint_ids = actuator_joint_indices(actuator, articulation.num_joints, articulation.device)
        bias[:, joint_ids] = actuator.encoder_bias
    return bias


def apply_identified_parameters(articulation, parameters: IdentifiedActuatorParameters, joint_order) -> None:
    """Apply one frozen PACE vector to every simulated environment.

    Encoder offsets are installed in the actuator only.  Reset positions are
    handled separately so actions remain in encoder coordinates and the bias is
    never applied twice.
    """
    if tuple(joint_order) != ANYMAL_D_JOINT_ORDER:
        raise ValueError("joint_order does not match the frozen ANYmal-D PACE parameter ordering")
    joint_ids = torch.tensor(
        [articulation.joint_names.index(name) for name in joint_order],
        device=articulation.device,
        dtype=torch.long,
    )
    env_ids = torch.arange(articulation.num_instances, device=articulation.device)

    def expand(values: torch.Tensor) -> torch.Tensor:
        return values.to(device=articulation.device, dtype=articulation.data.joint_pos.dtype).unsqueeze(0).expand(
            articulation.num_instances, -1
        )

    armature = expand(parameters.armature)
    damping = expand(parameters.viscous_friction)
    friction = expand(parameters.static_dynamic_friction)
    encoder_bias = expand(parameters.encoder_bias)

    articulation.write_joint_armature_to_sim(armature, joint_ids=joint_ids, env_ids=env_ids)
    articulation.data.default_joint_armature[:, joint_ids] = armature
    articulation.write_joint_viscous_friction_coefficient_to_sim(damping, joint_ids=joint_ids, env_ids=env_ids)
    articulation.data.default_joint_viscous_friction_coeff[:, joint_ids] = damping
    # PhysX requires static friction to be no lower than dynamic friction while values are changed.
    articulation.write_joint_dynamic_friction_coefficient_to_sim(0.0, joint_ids=joint_ids, env_ids=env_ids)
    articulation.write_joint_friction_coefficient_to_sim(friction, joint_ids=joint_ids, env_ids=env_ids)
    articulation.data.default_joint_friction_coeff[:, joint_ids] = friction
    articulation.write_joint_dynamic_friction_coefficient_to_sim(friction, joint_ids=joint_ids, env_ids=env_ids)
    articulation.data.default_joint_dynamic_friction_coeff[:, joint_ids] = friction

    for actuator in articulation.actuators.values():
        if not hasattr(actuator, "update_encoder_bias"):
            raise TypeError("PACE locomotion requires PaceDCMotor actuators on all controlled joints")
        actuator_ids = actuator_joint_indices(actuator, articulation.num_joints, articulation.device)
        parameter_columns = torch.tensor(
            [int((joint_ids == joint_id).nonzero(as_tuple=False).item()) for joint_id in actuator_ids],
            device=articulation.device,
        )
        actuator.update_encoder_bias(encoder_bias[:, parameter_columns])
        actuator.update_time_lags(parameters.delay_steps, env_ids)
        actuator.reset(env_ids)
