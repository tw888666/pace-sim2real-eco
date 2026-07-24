"""RSL-RL 5.x compatible single-constraint PPO-Lagrangian.

Reward and cost returns are kept separate. Reward returns use RSL-RL's timeout
bootstrap convention; cost returns treat both a fall and the 20 second timeout
as true terminal states.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer


_DEPRECATED_MODEL_CFG_KEYS = (
    "stochastic",
    "init_noise_std",
    "noise_std_type",
    "state_dependent_std",
)


def _remove_deprecated_model_cfg(model_cfg: dict) -> None:
    """Remove Isaac Lab compatibility fields rejected by RSL-RL 5.x models."""
    for key in _DEPRECATED_MODEL_CFG_KEYS:
        model_cfg.pop(key, None)


def normalized_lagrangian_actor_loss(
    reward_surrogate_loss: torch.Tensor,
    cost_surrogate_loss: torch.Tensor,
    lagrangian_multiplier: float | torch.Tensor,
    entropy: torch.Tensor,
    entropy_coefficient: float,
) -> torch.Tensor:
    """Combine actor terms without letting a large multiplier scale gradients."""
    multiplier = torch.as_tensor(
        lagrangian_multiplier,
        device=reward_surrogate_loss.device,
        dtype=reward_surrogate_loss.dtype,
    )
    return (reward_surrogate_loss + multiplier * cost_surrogate_loss) / (1.0 + multiplier) - (
        entropy_coefficient * entropy.mean()
    )


class ConstrainedRolloutStorage(RolloutStorage):
    """RSL-RL rollout storage extended with one physical cost signal."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        shape = (self.num_transitions_per_env, self.num_envs, 1)
        self.costs = torch.zeros(*shape, device=self.device)
        self.cost_values = torch.zeros(*shape, device=self.device)
        self.cost_returns = torch.zeros(*shape, device=self.device)
        self.cost_advantages = torch.zeros(*shape, device=self.device)
        self.cost_dones = torch.zeros(*shape, device=self.device, dtype=torch.uint8)

    def add_transition(self, transition: RolloutStorage.Transition) -> None:
        index = self.step
        super().add_transition(transition)
        self.costs[index].copy_(transition.costs.view(-1, 1))
        self.cost_values[index].copy_(transition.cost_values)
        self.cost_dones[index].copy_(transition.cost_dones.view(-1, 1))

    def compute_cost_returns(self, last_values: torch.Tensor, gamma: float, lam: float) -> None:
        """Compute cost GAE; only ordinary rollout boundaries bootstrap."""
        advantage = torch.zeros_like(last_values)
        for step in reversed(range(self.num_transitions_per_env)):
            next_values = last_values if step == self.num_transitions_per_env - 1 else self.cost_values[step + 1]
            next_is_not_terminal = 1.0 - self.cost_dones[step].float()
            delta = self.costs[step] + next_is_not_terminal * gamma * next_values - self.cost_values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.cost_returns[step] = advantage + self.cost_values[step]
        self.cost_advantages = self.cost_returns - self.cost_values
        self.cost_advantages = (self.cost_advantages - self.cost_advantages.mean()) / (
            self.cost_advantages.std() + 1.0e-8
        )

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        if self.training_type != "rl":
            raise ValueError("ConstrainedRolloutStorage only supports reinforcement learning")
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, device=self.device)

        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_distribution_params = tuple(p.flatten(0, 1) for p in self.distribution_params)
        cost_values = self.cost_values.flatten(0, 1)
        cost_returns = self.cost_returns.flatten(0, 1)
        cost_advantages = self.cost_advantages.flatten(0, 1)

        for _ in range(num_epochs):
            for batch_index in range(num_mini_batches):
                start = batch_index * mini_batch_size
                stop = (batch_index + 1) * mini_batch_size
                selected = indices[start:stop]
                batch = RolloutStorage.Batch(
                    observations=observations[selected],
                    actions=actions[selected],
                    values=values[selected],
                    advantages=advantages[selected],
                    returns=returns[selected],
                    old_actions_log_prob=old_actions_log_prob[selected],
                    old_distribution_params=tuple(p[selected] for p in old_distribution_params),
                )
                batch.cost_values = cost_values[selected]
                batch.cost_returns = cost_returns[selected]
                batch.cost_advantages = cost_advantages[selected]
                yield batch


class PacePPOLagrangian(PPO):
    """PPO with a separately optimized cost critic and externally set dual multiplier."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        cost_critic: MLPModel,
        storage: ConstrainedRolloutStorage,
        *,
        cost_gamma: float = 1.0,
        cost_lam: float = 0.95,
        cost_value_loss_coef: float = 1.0,
        cost_critic_learning_rate: float = 1.0e-3,
        lagrangian_multiplier_init: float = 0.0,
        lagrangian_multiplier_max: float = 100.0,
        **kwargs,
    ):
        super().__init__(actor, critic, storage, **kwargs)
        if self.rnd is not None or self.symmetry is not None:
            raise NotImplementedError("The initial PACE constrained study does not combine RND or symmetry losses")
        self.cost_critic = cost_critic.to(self.device)
        self.cost_optimizer = resolve_optimizer(kwargs.get("optimizer", "adam"))(
            self.cost_critic.parameters(), lr=cost_critic_learning_rate
        )
        self.cost_gamma = cost_gamma
        self.cost_lam = cost_lam
        self.cost_value_loss_coef = cost_value_loss_coef
        self.cost_critic_learning_rate = cost_critic_learning_rate
        self.lagrangian_multiplier_max = lagrangian_multiplier_max
        self.lagrangian_multiplier = 0.0
        self.set_lagrangian_multiplier(lagrangian_multiplier_init)

    @property
    def constrained_storage(self) -> ConstrainedRolloutStorage:
        return self.storage

    def set_lagrangian_multiplier(self, value: float) -> None:
        self.lagrangian_multiplier = float(min(max(value, 0.0), self.lagrangian_multiplier_max))

    def act(self, obs: TensorDict) -> torch.Tensor:
        actions = super().act(obs)
        self.transition.cost_values = self.cost_critic(obs).detach()
        return actions

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        if "pace_cost" not in extras:
            raise KeyError("PacePPOLagrangian requires extras['pace_cost'] from PaceEnergyRLEnv")
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        self.cost_critic.update_normalization(obs)
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.costs = extras["pace_cost"].to(self.device).clone()
        self.transition.cost_dones = dones

        # Reward timeout bootstrap is intentionally not mirrored for cost.
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),
                1,
            )
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)
        self.cost_critic.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        super().compute_returns(obs)
        self.constrained_storage.compute_cost_returns(
            self.cost_critic(obs).detach(), self.cost_gamma, self.cost_lam
        )

    def update(self) -> dict[str, float]:
        if self.actor.is_recurrent or self.critic.is_recurrent or self.cost_critic.is_recurrent:
            raise NotImplementedError("The first constrained implementation supports feed-forward models only")
        storage = self.constrained_storage
        target_variance = torch.var(storage.cost_returns)
        cost_explained_variance = 1.0 - torch.var(storage.cost_returns - storage.cost_values) / (
            target_variance + 1.0e-8
        )
        totals = {"value": 0.0, "cost_value": 0.0, "surrogate": 0.0, "cost_surrogate": 0.0, "entropy": 0.0}

        for batch in storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                        batch.advantages.std() + 1.0e-8
                    )
                    batch.cost_advantages = (batch.cost_advantages - batch.cost_advantages.mean()) / (
                        batch.cost_advantages.std() + 1.0e-8
                    )

            self.actor(batch.observations, stochastic_output=True)
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations)
            distribution_params = tuple(self.actor.output_distribution_params)
            entropy = self.actor.output_entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl_mean = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params).mean()
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                    elif 0.0 < kl_mean < self.desired_kl / 2.0:
                        self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
                    for parameter_group in self.optimizer.param_groups:
                        parameter_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            reward_surrogate = -torch.squeeze(batch.advantages) * ratio
            reward_surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            reward_surrogate_loss = torch.max(reward_surrogate, reward_surrogate_clipped).mean()
            cost_surrogate = torch.squeeze(batch.cost_advantages) * ratio
            cost_surrogate_clipped = torch.squeeze(batch.cost_advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            cost_surrogate_loss = torch.max(cost_surrogate, cost_surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.max(
                    (values - batch.returns).square(),
                    (value_clipped - batch.returns).square(),
                ).mean()
            else:
                value_loss = (batch.returns - values).square().mean()

            actor_loss = normalized_lagrangian_actor_loss(
                reward_surrogate_loss,
                cost_surrogate_loss,
                self.lagrangian_multiplier,
                entropy,
                self.entropy_coef,
            )
            self.optimizer.zero_grad()
            (actor_loss + self.value_loss_coef * value_loss).backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            cost_values = self.cost_critic(batch.observations)
            if self.use_clipped_value_loss:
                cost_value_clipped = batch.cost_values + (cost_values - batch.cost_values).clamp(
                    -self.clip_param, self.clip_param
                )
                cost_value_loss = torch.max(
                    (cost_values - batch.cost_returns).square(),
                    (cost_value_clipped - batch.cost_returns).square(),
                ).mean()
            else:
                cost_value_loss = (batch.cost_returns - cost_values).square().mean()
            self.cost_optimizer.zero_grad()
            (self.cost_value_loss_coef * cost_value_loss).backward()
            nn.utils.clip_grad_norm_(self.cost_critic.parameters(), self.max_grad_norm)
            self.cost_optimizer.step()

            totals["value"] += value_loss.item()
            totals["cost_value"] += cost_value_loss.item()
            totals["surrogate"] += reward_surrogate_loss.item()
            totals["cost_surrogate"] += cost_surrogate_loss.item()
            totals["entropy"] += entropy.mean().item()

        update_count = self.num_learning_epochs * self.num_mini_batches
        for name in totals:
            totals[name] /= update_count
        totals["cost_explained_variance"] = cost_explained_variance.item()
        totals["lagrangian_multiplier"] = self.lagrangian_multiplier
        storage.clear()
        return totals

    def train_mode(self) -> None:
        super().train_mode()
        self.cost_critic.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.cost_critic.eval()

    def save(self) -> dict:
        saved = super().save()
        saved.update(
            {
                "cost_critic_state_dict": self.cost_critic.state_dict(),
                "cost_optimizer_state_dict": self.cost_optimizer.state_dict(),
                "lagrangian_multiplier": self.lagrangian_multiplier,
            }
        )
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        effective_cfg = load_cfg or {}
        if effective_cfg.get("cost_critic", True):
            self.cost_critic.load_state_dict(loaded_dict["cost_critic_state_dict"], strict=strict)
        if effective_cfg.get("cost_optimizer", True):
            self.cost_optimizer.load_state_dict(loaded_dict["cost_optimizer_state_dict"])
        if effective_cfg.get("lagrangian_multiplier", True):
            self.set_lagrangian_multiplier(float(loaded_dict["lagrangian_multiplier"]))
        return load_iteration

    @staticmethod
    def construct_algorithm(obs: TensorDict, env, cfg: dict, device: str) -> "PacePPOLagrangian":
        if cfg.get("multi_gpu") is not None:
            raise NotImplementedError("PACE PPO-Lagrangian is currently validated for one GPU per run")
        algorithm_cfg = cfg["algorithm"]
        alg_class = resolve_callable(algorithm_cfg.pop("class_name"))
        actor_class = resolve_callable(cfg["actor"].pop("class_name"))
        critic_class = resolve_callable(cfg["critic"].pop("class_name"))
        cost_critic_class = resolve_callable(cfg["cost_critic"].pop("class_name"))
        # Isaac Lab's compatibility helper migrates the built-in actor and
        # critic, but it does not know about our additional cost critic. Its
        # config therefore still contains deprecated placeholder fields that
        # RSL-RL 5.x model constructors reject. Sanitize all three model
        # dictionaries here so direct construction follows the same contract.
        for model_name in ("actor", "critic", "cost_critic"):
            _remove_deprecated_model_cfg(cfg[model_name])
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic", "cost_critic"])

        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        cost_critic = cost_critic_class(obs, cfg["obs_groups"], "cost_critic", 1, **cfg["cost_critic"]).to(device)
        storage = ConstrainedRolloutStorage(
            "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
        )
        return alg_class(
            actor,
            critic,
            cost_critic,
            storage,
            device=device,
            **algorithm_cfg,
            multi_gpu_cfg=None,
        )
