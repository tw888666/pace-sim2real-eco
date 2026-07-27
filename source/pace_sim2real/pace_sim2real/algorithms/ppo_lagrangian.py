"""RSL-RL 5.x compatible single-constraint PPO-Lagrangian.

Reward and cost returns are kept separate. Reward returns use RSL-RL's timeout
bootstrap convention; cost returns treat both a fall and the 20 second timeout
as true terminal states.
"""

from __future__ import annotations

import math

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


def terminal_cost_boundary_loss(
    cost_critic: nn.Module,
    observations: TensorDict,
    *,
    time_group: str = "cost_time",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize non-zero finite-horizon value predictions at zero time-to-go.

    The undiscounted finite-horizon cost satisfies ``V_C(s, t=0) = 0`` for
    every state because no control transition remains.  With ``gamma_C=1``,
    ordinary bootstrapped TD/GAE targets weakly constrain an additive value
    offset away from terminal samples.  This counterfactual boundary batch
    provides that missing absolute anchor without changing actor observations.

    Returns:
        A pair containing mean squared boundary prediction and mean absolute
        boundary prediction.  The input TensorDict is never modified.
    """
    if time_group not in observations.keys():
        raise KeyError(f"cost boundary supervision requires observation group {time_group!r}")
    remaining_time = observations.get(time_group)
    if not isinstance(remaining_time, torch.Tensor):
        raise TypeError(f"observation group {time_group!r} must be a tensor")
    if remaining_time.ndim < 1 or remaining_time.shape[-1] != 1:
        raise ValueError(f"observation group {time_group!r} must have final dimension 1")

    boundary_observations = observations.clone()
    boundary_observations.set(time_group, torch.zeros_like(remaining_time))
    boundary_values = cost_critic(boundary_observations)
    return boundary_values.square().mean(), boundary_values.abs().mean()


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

    def compute_cost_returns(
        self,
        last_values: torch.Tensor,
        gamma: float,
        lam: float,
        *,
        normalize_advantage: bool = True,
    ) -> None:
        """Compute cost GAE; only ordinary rollout boundaries bootstrap."""
        advantage = torch.zeros_like(last_values)
        for step in reversed(range(self.num_transitions_per_env)):
            next_values = last_values if step == self.num_transitions_per_env - 1 else self.cost_values[step + 1]
            next_is_not_terminal = 1.0 - self.cost_dones[step].float()
            delta = self.costs[step] + next_is_not_terminal * gamma * next_values - self.cost_values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.cost_returns[step] = advantage + self.cost_values[step]
        self.cost_advantages = self.cost_returns - self.cost_values
        if normalize_advantage:
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
        cost_terminal_boundary_coef: float = 0.0,
        cost_mc_replay_coef: float = 0.0,
        cost_mc_initial_coef: float = 0.0,
        cost_mc_batch_size: int = 4096,
        cost_mc_replay_seed: int = 13_579,
        lagrangian_multiplier_init: float = 0.0,
        lagrangian_multiplier_max: float = 100.0,
        normalize_cost_advantage: bool = True,
        critic_only: bool = False,
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
        if not math.isfinite(cost_terminal_boundary_coef) or cost_terminal_boundary_coef < 0.0:
            raise ValueError("cost_terminal_boundary_coef must be finite and non-negative")
        self.cost_terminal_boundary_coef = float(cost_terminal_boundary_coef)
        if not math.isfinite(cost_mc_replay_coef) or cost_mc_replay_coef < 0.0:
            raise ValueError("cost_mc_replay_coef must be finite and non-negative")
        if not math.isfinite(cost_mc_initial_coef) or cost_mc_initial_coef < 0.0:
            raise ValueError("cost_mc_initial_coef must be finite and non-negative")
        if cost_mc_batch_size <= 0 or cost_mc_replay_seed < 0:
            raise ValueError("cost_mc_batch_size must be positive and cost_mc_replay_seed non-negative")
        self.cost_mc_replay_coef = float(cost_mc_replay_coef)
        self.cost_mc_initial_coef = float(cost_mc_initial_coef)
        self.cost_mc_batch_size = int(cost_mc_batch_size)
        self.cost_mc_replay_seed = int(cost_mc_replay_seed)
        self._cost_mc_policy_observation: torch.Tensor | None = None
        self._cost_mc_remaining_time: torch.Tensor | None = None
        self._cost_mc_target: torch.Tensor | None = None
        self._cost_mc_initial_indices: torch.Tensor | None = None
        self.cost_mc_dataset_sha256: str | None = None
        self._cost_mc_generator = torch.Generator(device=self.device)
        self._cost_mc_generator.manual_seed(self.cost_mc_replay_seed)
        self.lagrangian_multiplier_max = lagrangian_multiplier_max
        self.normalize_cost_advantage = bool(normalize_cost_advantage)
        self.critic_only = False
        self.lagrangian_multiplier = 0.0
        self.set_lagrangian_multiplier(lagrangian_multiplier_init)
        self.set_critic_only(critic_only)

    @property
    def constrained_storage(self) -> ConstrainedRolloutStorage:
        return self.storage

    def set_lagrangian_multiplier(self, value: float) -> None:
        self.lagrangian_multiplier = float(min(max(value, 0.0), self.lagrangian_multiplier_max))

    def set_critic_only(self, enabled: bool) -> None:
        """Freeze actor/reward critic parameters while warming only the cost critic."""
        self.critic_only = bool(enabled)
        for module in (self.actor, self.critic):
            for parameter in module.parameters():
                parameter.requires_grad_(not self.critic_only)
        if self.critic_only:
            self.set_lagrangian_multiplier(0.0)

    def set_cost_mc_replay(
        self,
        policy_observation: torch.Tensor,
        remaining_time: torch.Tensor,
        target: torch.Tensor,
        *,
        dataset_sha256: str,
    ) -> None:
        """Install fresh complete-episode Monte Carlo targets for critic-only calibration."""
        if policy_observation.ndim != 2:
            raise ValueError("MC policy_observation must have [sample, feature] shape")
        if remaining_time.shape != (policy_observation.shape[0], 1):
            raise ValueError("MC remaining_time must have [sample, 1] shape")
        if target.shape not in ((policy_observation.shape[0],), (policy_observation.shape[0], 1)):
            raise ValueError("MC target must have [sample] or [sample, 1] shape")
        if not dataset_sha256:
            raise ValueError("MC replay requires a dataset SHA-256")
        tensors = (policy_observation, remaining_time, target)
        if not all(torch.isfinite(value).all() for value in tensors):
            raise ValueError("MC replay contains NaN or infinity")
        if (remaining_time < 0.0).any() or (remaining_time > 1.0).any():
            raise ValueError("MC remaining_time must lie in [0, 1]")

        self._cost_mc_policy_observation = policy_observation.detach().to(self.device).clone()
        self._cost_mc_remaining_time = remaining_time.detach().to(self.device).clone()
        self._cost_mc_target = target.detach().reshape(-1, 1).to(self.device).clone()
        self._cost_mc_initial_indices = torch.nonzero(
            torch.isclose(
                self._cost_mc_remaining_time.squeeze(-1),
                torch.ones((), device=self.device, dtype=self._cost_mc_remaining_time.dtype),
            ),
            as_tuple=False,
        ).squeeze(-1)
        if self.cost_mc_initial_coef > 0.0 and self._cost_mc_initial_indices.numel() == 0:
            raise ValueError("positive cost_mc_initial_coef requires MC samples at remaining_time=1")
        self.cost_mc_dataset_sha256 = dataset_sha256

    def _sample_cost_mc_replay(self) -> tuple[TensorDict, torch.Tensor]:
        if (
            self._cost_mc_policy_observation is None
            or self._cost_mc_remaining_time is None
            or self._cost_mc_target is None
        ):
            raise RuntimeError("cost_mc_replay_coef is positive but no fresh MC dataset was installed")
        num_samples = self._cost_mc_target.shape[0]
        indices = torch.randint(
            num_samples,
            (min(self.cost_mc_batch_size, num_samples),),
            device=self.device,
            generator=self._cost_mc_generator,
        )
        observations = TensorDict(
            {
                "policy": self._cost_mc_policy_observation[indices],
                "cost_time": self._cost_mc_remaining_time[indices],
            },
            batch_size=[indices.numel()],
            device=self.device,
        )
        return observations, self._cost_mc_target[indices]

    def _cost_mc_initial_batch(self) -> tuple[TensorDict, torch.Tensor]:
        if (
            self._cost_mc_policy_observation is None
            or self._cost_mc_remaining_time is None
            or self._cost_mc_target is None
            or self._cost_mc_initial_indices is None
            or self._cost_mc_initial_indices.numel() == 0
        ):
            raise RuntimeError("cost_mc_initial_coef is positive but no MC initial-state samples are installed")
        indices = self._cost_mc_initial_indices
        observations = TensorDict(
            {
                "policy": self._cost_mc_policy_observation[indices],
                "cost_time": self._cost_mc_remaining_time[indices],
            },
            batch_size=[indices.numel()],
            device=self.device,
        )
        return observations, self._cost_mc_target[indices]

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
        if not self.critic_only:
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
            self.cost_critic(obs).detach(),
            self.cost_gamma,
            self.cost_lam,
            normalize_advantage=self.normalize_cost_advantage,
        )

    def update(self) -> dict[str, float]:
        if self.actor.is_recurrent or self.critic.is_recurrent or self.cost_critic.is_recurrent:
            raise NotImplementedError("The first constrained implementation supports feed-forward models only")
        storage = self.constrained_storage
        target_variance = torch.var(storage.cost_returns)
        cost_explained_variance = 1.0 - torch.var(storage.cost_returns - storage.cost_values) / (
            target_variance + 1.0e-8
        )
        totals = {
            "value": 0.0,
            "cost_value": 0.0,
            "cost_boundary": 0.0,
            "cost_boundary_abs": 0.0,
            "cost_mc": 0.0,
            "cost_mc_initial": 0.0,
            "surrogate": 0.0,
            "cost_surrogate": 0.0,
            "entropy": 0.0,
        }

        for batch in storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                        batch.advantages.std() + 1.0e-8
                    )
                    if self.normalize_cost_advantage:
                        batch.cost_advantages = (batch.cost_advantages - batch.cost_advantages.mean()) / (
                            batch.cost_advantages.std() + 1.0e-8
                        )

            if not self.critic_only:
                self.actor(batch.observations, stochastic_output=True)
                actions_log_prob = self.actor.get_output_log_prob(batch.actions)
                values = self.critic(batch.observations)
                distribution_params = tuple(self.actor.output_distribution_params)
                entropy = self.actor.output_entropy

                if self.desired_kl is not None and self.schedule == "adaptive":
                    with torch.inference_mode():
                        kl_mean = self.actor.get_kl_divergence(
                            batch.old_distribution_params, distribution_params
                        ).mean()
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
                totals["value"] += value_loss.item()
                totals["surrogate"] += reward_surrogate_loss.item()
                totals["cost_surrogate"] += cost_surrogate_loss.item()
                totals["entropy"] += entropy.mean().item()

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
            if self.cost_terminal_boundary_coef > 0.0:
                cost_boundary_loss, cost_boundary_abs = terminal_cost_boundary_loss(
                    self.cost_critic,
                    batch.observations,
                )
            else:
                cost_boundary_loss = torch.zeros((), device=cost_value_loss.device, dtype=cost_value_loss.dtype)
                cost_boundary_abs = torch.zeros((), device=cost_value_loss.device, dtype=cost_value_loss.dtype)
            if self.cost_mc_replay_coef > 0.0:
                mc_observations, mc_target = self._sample_cost_mc_replay()
                cost_mc_loss = (self.cost_critic(mc_observations) - mc_target).square().mean()
            else:
                cost_mc_loss = torch.zeros((), device=cost_value_loss.device, dtype=cost_value_loss.dtype)
            if self.cost_mc_initial_coef > 0.0:
                mc_initial_observations, mc_initial_target = self._cost_mc_initial_batch()
                cost_mc_initial_loss = (
                    self.cost_critic(mc_initial_observations) - mc_initial_target
                ).square().mean()
            else:
                cost_mc_initial_loss = torch.zeros(
                    (), device=cost_value_loss.device, dtype=cost_value_loss.dtype
                )
            self.cost_optimizer.zero_grad()
            (
                self.cost_value_loss_coef * cost_value_loss
                + self.cost_terminal_boundary_coef * cost_boundary_loss
                + self.cost_mc_replay_coef * cost_mc_loss
                + self.cost_mc_initial_coef * cost_mc_initial_loss
            ).backward()
            nn.utils.clip_grad_norm_(self.cost_critic.parameters(), self.max_grad_norm)
            self.cost_optimizer.step()

            totals["cost_value"] += cost_value_loss.item()
            totals["cost_boundary"] += cost_boundary_loss.item()
            totals["cost_boundary_abs"] += cost_boundary_abs.item()
            totals["cost_mc"] += cost_mc_loss.item()
            totals["cost_mc_initial"] += cost_mc_initial_loss.item()

        update_count = self.num_learning_epochs * self.num_mini_batches
        for name in totals:
            totals[name] /= update_count
        totals["cost_explained_variance"] = cost_explained_variance.item()
        totals["lagrangian_multiplier"] = self.lagrangian_multiplier
        totals["critic_only"] = float(self.critic_only)
        totals["normalize_cost_advantage"] = float(self.normalize_cost_advantage)
        totals["cost_terminal_boundary_coef"] = self.cost_terminal_boundary_coef
        totals["cost_mc_replay_coef"] = self.cost_mc_replay_coef
        totals["cost_mc_initial_coef"] = self.cost_mc_initial_coef
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
                "cost_terminal_boundary_coef": self.cost_terminal_boundary_coef,
                "cost_mc_replay_coef": self.cost_mc_replay_coef,
                "cost_mc_initial_coef": self.cost_mc_initial_coef,
                "cost_mc_dataset_sha256": self.cost_mc_dataset_sha256,
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
        if algorithm_cfg.pop("share_cnn_encoders", False):
            raise NotImplementedError("PACE PPO-Lagrangian currently supports independent MLP encoders only")
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
