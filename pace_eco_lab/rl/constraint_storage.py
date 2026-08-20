"""无 cost critic（代价价值网络）的约束 rollout 存储。"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ConstraintRollout:
    """保存逐步代价、终止标记和完整回合归一化能耗。"""

    costs: torch.Tensor
    dones: torch.Tensor
    step: int = 0
    completed_normalized_costs: list[torch.Tensor] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        num_steps: int,
        num_envs: int,
        *,
        device: str | torch.device = "cpu",
    ) -> "ConstraintRollout":
        if num_steps <= 0 or num_envs <= 0:
            raise ValueError("num_steps 和 num_envs 必须为正。")
        return cls(
            costs=torch.zeros(num_steps, num_envs, 1, device=device),
            dones=torch.zeros(num_steps, num_envs, 1, dtype=torch.bool, device=device),
        )

    def add(
        self,
        normalized_step_cost: torch.Tensor,
        dones: torch.Tensor,
        completed_normalized_cost: torch.Tensor | None = None,
        completed_mask: torch.Tensor | None = None,
    ) -> None:
        if self.step >= self.costs.shape[0]:
            raise OverflowError("constraint rollout 已满。")
        self.costs[self.step].copy_(normalized_step_cost.reshape(-1, 1))
        self.dones[self.step].copy_(dones.bool().reshape(-1, 1))
        if completed_normalized_cost is not None:
            if completed_mask is None:
                raise ValueError("提供完整回合代价时必须同时提供 completed_mask。")
            selected = completed_normalized_cost.reshape(-1)[completed_mask.bool().reshape(-1)]
            if selected.numel():
                self.completed_normalized_costs.append(selected.detach().clone())
        self.step += 1

    def compute_advantages(self, *, gamma: float, lam: float, normalize: bool = True) -> torch.Tensor:
        """以零 cost critic 计算 GAE，并在整个 rollout 上统一归一化。"""

        if not 0.0 <= gamma <= 1.0 or not 0.0 <= lam <= 1.0:
            raise ValueError("gamma 和 lam 必须在 [0, 1]。")
        if self.step != self.costs.shape[0]:
            raise RuntimeError("必须填满 rollout 后才能计算代价优势。")
        advantages = torch.zeros_like(self.costs)
        running = torch.zeros_like(self.costs[0])
        for index in reversed(range(self.step)):
            not_terminal = 1.0 - self.dones[index].float()
            running = self.costs[index] + gamma * lam * not_terminal * running
            advantages[index] = running
        if normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1.0e-8)
        return advantages

    def mean_completed_normalized_cost(self) -> torch.Tensor | None:
        if not self.completed_normalized_costs:
            return None
        return torch.cat(self.completed_normalized_costs).mean()

    def clear(self) -> None:
        self.step = 0
        self.completed_normalized_costs.clear()
