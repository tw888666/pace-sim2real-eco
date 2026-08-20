"""PACE 辨识参数的加载、校验和按名称映射。"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from pace_eco_lab.constants import PACE_DATA_ROOT, PACE_FITTING_PATH, PACE_FITTING_SHA256, PACE_JOINT_NAMES


@dataclass(frozen=True)
class PaceParameters:
    """49 维 PACE 参数的结构化只读表示。"""

    armature: np.ndarray
    viscous_damping: np.ndarray
    coulomb_friction: np.ndarray
    joint_bias: np.ndarray
    identified_delay: float
    applied_delay_steps: int

    @property
    def vector(self) -> np.ndarray:
        """按原始切片顺序返回 49 维向量。"""

        return np.concatenate(
            (
                self.armature,
                self.viscous_damping,
                self.coulomb_friction,
                self.joint_bias,
                np.asarray([self.identified_delay], dtype=np.float64),
            )
        )


def file_sha256(path: str | Path) -> str:
    """流式计算文件 SHA256。"""

    resolved = Path(path)
    if not resolved.is_absolute() and not resolved.is_file() and resolved.parts[:1] == ("pace_data",):
        resolved = PACE_DATA_ROOT.joinpath(*resolved.parts[1:])
    digest = sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_numpy_vector(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (49,):
        raise ValueError(f"PACE params 必须是 49 维，实际为 {array.shape}。")
    if not np.isfinite(array).all():
        raise ValueError("PACE params 含 NaN 或 Inf。")
    return array


def load_pace_parameters(
    path: str | Path = PACE_FITTING_PATH,
    *,
    expected_sha256: str = PACE_FITTING_SHA256,
    applied_delay_steps: int = 3,
) -> PaceParameters:
    """加载可信的 PACE ``fitting.npy`` 并执行全部结构校验。

    文件包含由 PyTorch 保存的对象，因此 NumPy 必须启用 pickle。加载前先核对固定
    SHA256，避免对非预期对象执行反序列化。
    """

    resolved = Path(path).resolve()
    actual_sha256 = file_sha256(resolved)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"PACE 参数文件哈希不匹配：期望 {expected_sha256}，实际 {actual_sha256}，文件 {resolved}"
        )
    container = np.load(resolved, allow_pickle=True)
    if container.shape != () or container.dtype != object:
        raise ValueError("PACE fitting.npy 顶层必须是标量 object ndarray。")
    payload = container.item()
    if not isinstance(payload, dict) or "params" not in payload:
        raise ValueError("PACE fitting.npy 缺少字典键 params。")
    vector = _as_numpy_vector(payload["params"])

    if np.any(vector[0:12] < 0.0):
        raise ValueError("armature 不得为负。")
    if np.any(vector[12:24] < 0.0):
        raise ValueError("viscous damping 不得为负。")
    if np.any(vector[24:36] < 0.0):
        raise ValueError("Coulomb friction 不得为负。")
    if np.any(np.abs(vector[36:48]) > 0.1 + 1.0e-12):
        raise ValueError("joint bias 超出辨识边界 [-0.1, 0.1] rad。")
    if not 0.0 <= vector[48] <= 7.0:
        raise ValueError("identified delay 超出辨识边界 [0, 7]。")
    if applied_delay_steps != 3:
        raise ValueError("第一阶段全局延迟固定为 3 个 2.5 ms 物理步。")

    return PaceParameters(
        armature=vector[0:12].copy(),
        viscous_damping=vector[12:24].copy(),
        coulomb_friction=vector[24:36].copy(),
        joint_bias=vector[36:48].copy(),
        identified_delay=float(vector[48]),
        applied_delay_steps=applied_delay_steps,
    )


def map_joint_values(
    values: Iterable[float],
    runtime_joint_names: Iterable[str],
    *,
    source_joint_names: Iterable[str] = PACE_JOINT_NAMES,
) -> np.ndarray:
    """把 PACE 名称顺序的值重排到运行时关节顺序。

    显式验证名称集合，禁止依赖 Isaac Lab 内部关节索引。
    """

    source = tuple(source_joint_names)
    runtime = tuple(runtime_joint_names)
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.shape != (len(source),):
        raise ValueError(f"参数数量 {array.size} 与源关节数量 {len(source)} 不一致。")
    if len(set(source)) != len(source) or len(set(runtime)) != len(runtime):
        raise ValueError("关节名称不得重复。")
    missing = sorted(set(source) - set(runtime))
    unexpected = sorted(set(runtime) - set(source))
    if missing or unexpected:
        raise ValueError(f"关节名称集合不一致：缺少 {missing}，额外 {unexpected}。")
    by_name = dict(zip(source, array, strict=True))
    return np.asarray([by_name[name] for name in runtime], dtype=np.float64)


def named_joint_values(values: Iterable[float]) -> dict[str, float]:
    """生成供 Isaac Lab 配置使用的精确名称映射。"""

    array = np.asarray(tuple(values), dtype=np.float64)
    if array.shape != (len(PACE_JOINT_NAMES),):
        raise ValueError("必须提供 12 个关节参数。")
    return {name: float(value) for name, value in zip(PACE_JOINT_NAMES, array, strict=True)}
