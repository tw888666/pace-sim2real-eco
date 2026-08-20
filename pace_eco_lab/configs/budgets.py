"""能耗预算的显式表示和校准文件读写。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class EnergyBudget:
    name: str
    reference_energy_j: float
    fraction: float

    @property
    def joules(self) -> float:
        if self.reference_energy_j <= 0 or not 0 < self.fraction <= 1:
            raise ValueError("参考能耗必须为正，预算比例必须在 (0, 1]。")
        return self.reference_energy_j * self.fraction

    def to_dict(self) -> dict[str, float | str]:
        result = asdict(self)
        result["budget_j"] = self.joules
        return result


def standard_budgets(reference_energy_j: float) -> dict[str, EnergyBudget]:
    return {
        name: EnergyBudget(name=name, reference_energy_j=reference_energy_j, fraction=fraction)
        for name, fraction in (("B100", 1.0), ("B90", 0.9), ("B80", 0.8))
    }


def load_budget(path: str | Path, name: str) -> EnergyBudget:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if name not in payload:
        raise KeyError(f"预算文件中不存在 {name}。")
    item = payload[name]
    return EnergyBudget(
        name=name,
        reference_energy_j=float(item["reference_energy_j"]),
        fraction=float(item["fraction"]),
    )
