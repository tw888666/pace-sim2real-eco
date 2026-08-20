"""正式评估使用的只读运动协调性指标累计器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch


FOOT_BODY_NAMES = ("LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT")
FOOT_LABELS = ("左前足", "右前足", "左后足", "右后足")
COORDINATION_PROTOCOL_VERSION = "gpt-协调性-v1"


def _foot_metric_fields(suffix: str) -> tuple[str, ...]:
    return tuple(f"{label}{suffix}" for label in FOOT_LABELS)


COORDINATION_METRIC_DEFINITIONS = {
    "协调性稳态样本数": "回合前 5 秒预热后参与协调性统计的策略步数。",
    "协调性稳态时长_s": "协调性稳态样本数乘以策略步长。",
    "机身横向速度RMS_m_s": "机身坐标系 y 方向根线速度的均方根。",
    "机身垂向速度RMS_m_s": "世界坐标系 z 方向根线速度的均方根。",
    "机身横滚俯仰角速度RMS_rad_s": "机身坐标系横滚和俯仰角速度向量模长的均方根。",
    "动作变化RMS_归一化动作": "相邻策略步、经评估动作裁剪后的 12 维归一化动作差的逐元素均方根。",
    "三步窗触地足速均值_m_s": "每次首次触地时，该足最近三个策略步世界坐标速度模长最大值的事件均值。",
    "触地事件数": "稳态区间内四足首次触地事件总数。",
    **{
        f"{label}支撑相占比": f"{label}接触样本数除以协调性稳态样本数，接触阈值来自传感器配置。"
        for label in FOOT_LABELS
    },
    "支撑相占比极差": "四足支撑相占比的最大值减最小值。",
    **{
        f"{label}落足频率_Hz": f"{label}稳态首次触地事件数除以协调性稳态时长。"
        for label in FOOT_LABELS
    },
    "落足频率变异系数": "四足落足频率的总体标准差除以四足频率均值。",
    "对角足接触不同步率": "每步两组对角足接触异或值的平均，再对稳态区间取平均；仅作为小跑步态描述量。",
}


@dataclass
class CoordinationAccumulator:
    """在 GPU 或 CPU 张量上按并行环境累计只读协调性统计量。"""

    state_sample_count: torch.Tensor
    lateral_velocity_squared_sum: torch.Tensor
    vertical_velocity_squared_sum: torch.Tensor
    roll_pitch_angular_velocity_squared_sum: torch.Tensor
    contact_sample_count: torch.Tensor
    touchdown_count: torch.Tensor
    touchdown_speed_sum: torch.Tensor
    diagonal_contact_mismatch_sum: torch.Tensor
    action_delta_squared_sum: torch.Tensor
    action_delta_element_count: torch.Tensor
    previous_action: torch.Tensor
    has_previous_action: torch.Tensor
    foot_speed_history: torch.Tensor
    history_cursor: int = 0

    SUMMARY_FIELDS: ClassVar[tuple[str, ...]] = (
        "机身横向速度RMS_m_s",
        "机身垂向速度RMS_m_s",
        "机身横滚俯仰角速度RMS_rad_s",
        "动作变化RMS_归一化动作",
        "三步窗触地足速均值_m_s",
        "触地事件数",
        *_foot_metric_fields("支撑相占比"),
        "支撑相占比极差",
        *_foot_metric_fields("落足频率_Hz"),
        "落足频率变异系数",
        "对角足接触不同步率",
    )

    @classmethod
    def create(
        cls,
        num_envs: int,
        action_dim: int,
        *,
        history_length: int = 3,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "CoordinationAccumulator":
        if num_envs <= 0 or action_dim <= 0 or history_length <= 0:
            raise ValueError("环境数、动作维数和足速历史长度必须为正整数。")
        scalar = lambda: torch.zeros(num_envs, device=device, dtype=dtype)
        feet = lambda: torch.zeros(num_envs, len(FOOT_BODY_NAMES), device=device, dtype=dtype)
        return cls(
            state_sample_count=scalar(),
            lateral_velocity_squared_sum=scalar(),
            vertical_velocity_squared_sum=scalar(),
            roll_pitch_angular_velocity_squared_sum=scalar(),
            contact_sample_count=feet(),
            touchdown_count=feet(),
            touchdown_speed_sum=scalar(),
            diagonal_contact_mismatch_sum=scalar(),
            action_delta_squared_sum=scalar(),
            action_delta_element_count=scalar(),
            previous_action=torch.zeros(num_envs, action_dim, device=device, dtype=dtype),
            has_previous_action=torch.zeros(num_envs, device=device, dtype=torch.bool),
            foot_speed_history=torch.zeros(
                history_length,
                num_envs,
                len(FOOT_BODY_NAMES),
                device=device,
                dtype=dtype,
            ),
        )

    @property
    def num_envs(self) -> int:
        return self.state_sample_count.shape[0]

    def _check_state_shapes(
        self,
        lateral_velocity: torch.Tensor,
        vertical_velocity: torch.Tensor,
        roll_pitch_angular_velocity: torch.Tensor,
        foot_contact: torch.Tensor,
        foot_speed: torch.Tensor,
        touchdown: torch.Tensor,
        steady_mask: torch.Tensor,
    ) -> None:
        scalar_shape = (self.num_envs,)
        foot_shape = (self.num_envs, len(FOOT_BODY_NAMES))
        expected = {
            "横向速度": (lateral_velocity.shape, scalar_shape),
            "垂向速度": (vertical_velocity.shape, scalar_shape),
            "横滚俯仰角速度": (roll_pitch_angular_velocity.shape, (self.num_envs, 2)),
            "足接触": (foot_contact.shape, foot_shape),
            "足速": (foot_speed.shape, foot_shape),
            "触地事件": (touchdown.shape, foot_shape),
            "稳态掩码": (steady_mask.shape, scalar_shape),
        }
        mismatches = {
            name: {"实际": tuple(actual), "期望": wanted}
            for name, (actual, wanted) in expected.items()
            if tuple(actual) != wanted
        }
        if mismatches:
            raise ValueError(f"协调性状态张量形状不一致：{mismatches}")

    def update_state(
        self,
        *,
        lateral_velocity: torch.Tensor,
        vertical_velocity: torch.Tensor,
        roll_pitch_angular_velocity: torch.Tensor,
        foot_contact: torch.Tensor,
        foot_speed: torch.Tensor,
        touchdown: torch.Tensor,
        steady_mask: torch.Tensor,
    ) -> None:
        """记录当前策略步的机身、接触和触地状态。"""

        self._check_state_shapes(
            lateral_velocity,
            vertical_velocity,
            roll_pitch_angular_velocity,
            foot_contact,
            foot_speed,
            touchdown,
            steady_mask,
        )
        self.foot_speed_history[self.history_cursor].copy_(foot_speed)
        self.history_cursor = (self.history_cursor + 1) % self.foot_speed_history.shape[0]

        mask = steady_mask.to(dtype=self.state_sample_count.dtype)
        foot_mask = mask.unsqueeze(-1)
        self.state_sample_count.add_(mask)
        self.lateral_velocity_squared_sum.add_(lateral_velocity.square() * mask)
        self.vertical_velocity_squared_sum.add_(vertical_velocity.square() * mask)
        self.roll_pitch_angular_velocity_squared_sum.add_(
            roll_pitch_angular_velocity.square().sum(dim=-1) * mask
        )

        contact = foot_contact.bool()
        self.contact_sample_count.add_(contact.to(self.contact_sample_count.dtype) * foot_mask)
        diagonal_mismatch = 0.5 * (
            (contact[:, 0] != contact[:, 3]).to(mask.dtype)
            + (contact[:, 1] != contact[:, 2]).to(mask.dtype)
        )
        self.diagonal_contact_mismatch_sum.add_(diagonal_mismatch * mask)

        touchdown_in_steady_state = touchdown.bool() & steady_mask.bool().unsqueeze(-1)
        touchdown_float = touchdown_in_steady_state.to(self.touchdown_count.dtype)
        recent_max_speed = self.foot_speed_history.max(dim=0).values
        self.touchdown_count.add_(touchdown_float)
        self.touchdown_speed_sum.add_((recent_max_speed * touchdown_float).sum(dim=-1))

    def update_action(self, action: torch.Tensor, steady_mask: torch.Tensor) -> None:
        """记录相邻策略动作变化；动作应已按评估入口的裁剪范围裁剪。"""

        if action.shape != self.previous_action.shape:
            raise ValueError(
                f"动作形状与累计器不一致：实际 {tuple(action.shape)}，"
                f"期望 {tuple(self.previous_action.shape)}。"
            )
        if steady_mask.shape != (self.num_envs,):
            raise ValueError("动作稳态掩码形状不一致。")
        valid = steady_mask.bool() & self.has_previous_action
        action_delta_squared = (action - self.previous_action).square().sum(dim=-1)
        self.action_delta_squared_sum.add_(action_delta_squared * valid.to(action.dtype))
        self.action_delta_element_count.add_(
            valid.to(self.action_delta_element_count.dtype) * action.shape[-1]
        )
        self.previous_action.copy_(action)
        self.has_previous_action.fill_(True)

    def episode_metrics(self, env_id: int, step_dt: float) -> dict[str, int | float | None]:
        """返回一个已结束环境槽位的逐回合指标。"""

        if not 0 <= env_id < self.num_envs:
            raise IndexError(f"环境编号越界：{env_id}")
        if step_dt <= 0:
            raise ValueError("策略步长必须为正数。")
        sample_count = int(self.state_sample_count[env_id].item())
        duration_s = sample_count * step_dt
        row: dict[str, int | float | None] = {
            "协调性稳态样本数": sample_count,
            "协调性稳态时长_s": duration_s,
        }
        if sample_count == 0:
            for field in self.SUMMARY_FIELDS:
                row[field] = None
            return row

        denominator = self.state_sample_count[env_id]
        row.update(
            {
                "机身横向速度RMS_m_s": float(
                    torch.sqrt(self.lateral_velocity_squared_sum[env_id] / denominator).item()
                ),
                "机身垂向速度RMS_m_s": float(
                    torch.sqrt(self.vertical_velocity_squared_sum[env_id] / denominator).item()
                ),
                "机身横滚俯仰角速度RMS_rad_s": float(
                    torch.sqrt(self.roll_pitch_angular_velocity_squared_sum[env_id] / denominator).item()
                ),
            }
        )

        action_count = self.action_delta_element_count[env_id]
        row["动作变化RMS_归一化动作"] = (
            float(torch.sqrt(self.action_delta_squared_sum[env_id] / action_count).item())
            if action_count.item() > 0
            else None
        )
        total_touchdowns = self.touchdown_count[env_id].sum()
        row["三步窗触地足速均值_m_s"] = (
            float((self.touchdown_speed_sum[env_id] / total_touchdowns).item())
            if total_touchdowns.item() > 0
            else None
        )
        row["触地事件数"] = int(total_touchdowns.item())

        duty_factors = self.contact_sample_count[env_id] / denominator
        step_frequencies = self.touchdown_count[env_id] / duration_s
        for label, duty_factor, step_frequency in zip(
            FOOT_LABELS,
            duty_factors.tolist(),
            step_frequencies.tolist(),
            strict=True,
        ):
            row[f"{label}支撑相占比"] = float(duty_factor)
            row[f"{label}落足频率_Hz"] = float(step_frequency)
        row["支撑相占比极差"] = float((duty_factors.max() - duty_factors.min()).item())
        mean_frequency = step_frequencies.mean()
        row["落足频率变异系数"] = (
            float((step_frequencies.std(unbiased=False) / mean_frequency).item())
            if mean_frequency.item() > 0
            else None
        )
        row["对角足接触不同步率"] = float(
            (self.diagonal_contact_mismatch_sum[env_id] / denominator).item()
        )
        return row

    def reset(self, env_ids: torch.Tensor | list[int] | slice | None = None) -> None:
        """清空已结束环境的全部跨步状态。"""

        if env_ids is None:
            env_ids = slice(None)
        for value in (
            self.state_sample_count,
            self.lateral_velocity_squared_sum,
            self.vertical_velocity_squared_sum,
            self.roll_pitch_angular_velocity_squared_sum,
            self.contact_sample_count,
            self.touchdown_count,
            self.touchdown_speed_sum,
            self.diagonal_contact_mismatch_sum,
            self.action_delta_squared_sum,
            self.action_delta_element_count,
            self.previous_action,
        ):
            value[env_ids] = 0
        self.has_previous_action[env_ids] = False
        self.foot_speed_history[:, env_ids] = 0
        if isinstance(env_ids, slice) and env_ids == slice(None):
            self.history_cursor = 0


__all__ = [
    "COORDINATION_METRIC_DEFINITIONS",
    "COORDINATION_PROTOCOL_VERSION",
    "CoordinationAccumulator",
    "FOOT_BODY_NAMES",
    "FOOT_LABELS",
]
