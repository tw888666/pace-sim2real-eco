"""无 cost critic 的 PPO-Lagrangian。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from rsl_rl.storage import RolloutStorage

from .ppo import PacePPO


class ConstraintRolloutStorage(RolloutStorage):
    """在 RSL-RL rollout 上增加代价优势和完整回合能耗。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.costs = torch.zeros_like(self.rewards)
        self.cost_advantages = torch.zeros_like(self.rewards)
        self.completed_normalized_costs: list[torch.Tensor] = []

    def add_constraint(
        self,
        normalized_step_cost: torch.Tensor,
        completed_normalized_cost: torch.Tensor,
        completed_mask: torch.Tensor,
    ) -> None:
        index = self.step - 1
        if index < 0:
            raise RuntimeError("必须先添加 PPO transition。")
        self.costs[index].copy_(normalized_step_cost.reshape(-1, 1))
        selected = completed_normalized_cost.reshape(-1)[completed_mask.bool().reshape(-1)]
        if selected.numel():
            self.completed_normalized_costs.append(selected.detach().clone())

    def compute_cost_advantages(self, gamma: float, lam: float) -> None:
        running = torch.zeros_like(self.costs[0])
        for index in reversed(range(self.num_transitions_per_env)):
            not_terminal = 1.0 - self.dones[index].float()
            running = self.costs[index] + gamma * lam * not_terminal * running
            self.cost_advantages[index] = running
        self.cost_advantages = (
            self.cost_advantages - self.cost_advantages.mean()
        ) / (self.cost_advantages.std(unbiased=False) + 1.0e-8)

    def mean_completed_normalized_cost(self) -> torch.Tensor | None:
        if not self.completed_normalized_costs:
            return None
        return torch.cat(self.completed_normalized_costs).mean()

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, device=self.device)
        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        cost_advantages = self.cost_advantages.flatten(0, 1)
        old_distribution_params = tuple(item.flatten(0, 1) for item in self.distribution_params)
        for _ in range(num_epochs):
            for index in range(num_mini_batches):
                batch_idx = indices[index * mini_batch_size : (index + 1) * mini_batch_size]
                batch = RolloutStorage.Batch(
                    observations=observations[batch_idx],
                    actions=actions[batch_idx],
                    values=values[batch_idx],
                    advantages=advantages[batch_idx],
                    returns=returns[batch_idx],
                    old_actions_log_prob=old_log_prob[batch_idx],
                    old_distribution_params=tuple(item[batch_idx] for item in old_distribution_params),
                )
                batch.cost_advantages = cost_advantages[batch_idx]
                yield batch

    def clear(self) -> None:
        super().clear()
        self.completed_normalized_costs.clear()


class PPOLagrangian(PacePPO):
    """按 ECO 形式组合奖励与能耗代价策略损失。"""

    storage_class = ConstraintRolloutStorage

    def __init__(
        self,
        *args,
        cost_gamma: float = 0.998,
        cost_lam: float = 0.81,
        energy_budget_j: float = 0.0,
        lagrange_initial: float = 0.0,
        lagrange_learning_rate: float = 1.0e-3,
        **kwargs,
    ):
        if energy_budget_j <= 0.0:
            raise ValueError("PACE-ECO 训练前必须用 --energy_budget_j 提供正的已校准回合预算。")
        super().__init__(*args, **kwargs)
        if self.is_multi_gpu:
            raise ValueError("当前 PACE-ECO 实现只支持单 GPU，以保证乘子与代价统计语义明确。")
        self.cost_gamma = cost_gamma
        self.cost_lam = cost_lam
        self.energy_budget_j = energy_budget_j
        self.lagrange_multiplier = nn.Parameter(torch.tensor(float(lagrange_initial), device=self.device))
        self.lagrange_optimizer = optim.Adam([self.lagrange_multiplier], lr=lagrange_learning_rate)

    @property
    def constraint_storage(self) -> ConstraintRolloutStorage:
        return self.storage  # type: ignore[return-value]

    def process_env_step(self, obs, rewards, dones, extras):
        super().process_env_step(obs, rewards, dones, extras)
        required = ("pace_energy_step", "pace_energy_episode", "pace_energy_episode_mask")
        missing = [key for key in required if key not in extras]
        if missing:
            raise KeyError(f"PACE-ECO 环境 extras 缺少 {missing}。")
        self.constraint_storage.add_constraint(
            extras["pace_energy_step"].to(self.device) / self.energy_budget_j,
            extras["pace_energy_episode"].to(self.device) / self.energy_budget_j,
            extras["pace_energy_episode_mask"].to(self.device),
        )

    def compute_returns(self, obs):
        super().compute_returns(obs)
        self.constraint_storage.compute_cost_advantages(self.cost_gamma, self.cost_lam)

    def update(self) -> dict[str, float]:
        self.entropy_coef = self._scheduled_entropy()
        mean_value_loss = 0.0
        mean_reward_surrogate = 0.0
        mean_cost_surrogate = 0.0
        mean_entropy = 0.0
        generator = self.constraint_storage.mini_batch_generator(
            self.num_mini_batches,
            self.num_learning_epochs,
        )
        for batch in generator:
            self.actor(batch.observations, stochastic_output=True)
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations)
            distribution_params = tuple(item for item in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                    elif 0.0 < kl_mean < self.desired_kl / 2.0:
                        self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
                    for group in self.optimizer.param_groups:
                        group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            reward_advantage = torch.squeeze(batch.advantages)
            reward_surrogate = -reward_advantage * ratio
            reward_surrogate_clipped = -reward_advantage * torch.clamp(
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
            reward_loss = torch.max(reward_surrogate, reward_surrogate_clipped).mean()

            cost_advantage = torch.squeeze(batch.cost_advantages)
            cost_surrogate = cost_advantage * ratio
            cost_surrogate_clipped = cost_advantage * torch.clamp(
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
            cost_loss = torch.max(cost_surrogate, cost_surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.max(
                    (values - batch.returns).pow(2),
                    (value_clipped - batch.returns).pow(2),
                ).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            multiplier = self.lagrange_multiplier.detach()
            actor_loss = (reward_loss + multiplier * cost_loss) / (1.0 + multiplier)
            loss = actor_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
            self.optimizer.zero_grad()
            loss.backward()
            # 与 RSL-RL 5.0.1 PPO 保持一致：策略网络和价值网络分别裁剪。
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_reward_surrogate += reward_loss.item()
            mean_cost_surrogate += cost_loss.item()
            mean_entropy += entropy.mean().item()

        completed_cost = self.constraint_storage.mean_completed_normalized_cost()
        violation_value = float("nan")
        completed_value = float("nan")
        if completed_cost is not None:
            violation = completed_cost.detach() - 1.0
            self.lagrange_optimizer.zero_grad()
            multiplier_loss = -(self.lagrange_multiplier * violation)
            multiplier_loss.backward()
            self.lagrange_optimizer.step()
            with torch.no_grad():
                self.lagrange_multiplier.clamp_(min=0.0)
            violation_value = violation.item()
            completed_value = completed_cost.item()

        count = self.num_learning_epochs * self.num_mini_batches
        losses = {
            "value": mean_value_loss / count,
            "surrogate": mean_reward_surrogate / count,
            "cost_surrogate": mean_cost_surrogate / count,
            "entropy": mean_entropy / count,
            "entropy_coefficient": self.entropy_coef,
            "lagrange_multiplier": self.lagrange_multiplier.item(),
            "normalized_episode_cost": completed_value,
            "constraint_violation": violation_value,
        }
        self.constraint_storage.clear()
        self.pace_iteration += 1
        self._sync_environment_iteration()
        return losses

    def _scheduled_entropy(self) -> float:
        from .schedules import entropy_coefficient

        return entropy_coefficient(
            self.pace_iteration,
            initial=self.entropy_initial,
            final=self.entropy_final,
            turnover=self.entropy_turnover_iteration,
            slope=self.entropy_slope,
        )

    def save(self) -> dict:
        payload = super().save()
        payload["lagrange_multiplier"] = self.lagrange_multiplier.detach()
        payload["lagrange_optimizer_state_dict"] = self.lagrange_optimizer.state_dict()
        payload["energy_budget_j"] = self.energy_budget_j
        return payload

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        saved_budget = float(loaded_dict.get("energy_budget_j", self.energy_budget_j))
        if abs(saved_budget - self.energy_budget_j) > 1.0e-9:
            raise ValueError(f"恢复预算不一致：检查点 {saved_budget} J，当前 {self.energy_budget_j} J。")
        with torch.no_grad():
            self.lagrange_multiplier.copy_(loaded_dict["lagrange_multiplier"].to(self.device))
        self.lagrange_optimizer.load_state_dict(loaded_dict["lagrange_optimizer_state_dict"])
        return load_iteration
