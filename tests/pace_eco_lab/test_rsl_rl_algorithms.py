from __future__ import annotations

import torch
import pytest
from tensordict import TensorDict

from pace_eco_lab.rl.ppo import PacePPO
from pace_eco_lab.rl.ppo_lagrangian import PPOLagrangian


class FakeEnvironment:
    num_envs = 2
    num_actions = 12
    pace_learning_iteration = 0

    @property
    def unwrapped(self):
        return self


def _observations() -> TensorDict:
    return TensorDict(
        {
            "policy": torch.randn(2, 48),
            "critic": torch.randn(2, 353),
        },
        batch_size=[2],
    )


def _config(class_name: str, constrained: bool) -> dict:
    algorithm = {
        "class_name": class_name,
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.002,
        "num_learning_epochs": 1,
        "num_mini_batches": 1,
        "learning_rate": 1.0e-3,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
        "normalize_advantage_per_mini_batch": False,
        "optimizer": "adam",
        "rnd_cfg": None,
        "symmetry_cfg": None,
        "share_cnn_encoders": False,
        "entropy_initial": 0.002,
        "entropy_final": 0.0005,
        "entropy_turnover_iteration": 2_000.0,
        "entropy_slope": 2.5e-3,
    }
    if constrained:
        algorithm.update(
            cost_gamma=0.998,
            cost_lam=0.81,
            energy_budget_j=10.0,
            lagrange_initial=0.0,
            lagrange_learning_rate=1.0e-3,
        )
    return {
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [8],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.5,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [8],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": None,
        },
        "algorithm": algorithm,
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "num_steps_per_env": 4,
        "multi_gpu": None,
    }


def _one_update(algorithm_class, class_name: str, constrained: bool):
    algorithm = algorithm_class.construct_algorithm(
        _observations(),
        FakeEnvironment(),
        _config(class_name, constrained),
        "cpu",
    )
    observations = _observations()
    for step in range(4):
        algorithm.act(observations)
        next_observations = _observations()
        dones = torch.tensor([step == 3, False])
        extras = {}
        if constrained:
            extras = {
                "pace_energy_step": torch.tensor([1.0, 2.0]),
                "pace_energy_episode": torch.tensor([12.0, 0.0]) if step == 3 else torch.zeros(2),
                "pace_energy_episode_mask": (
                    torch.tensor([True, False]) if step == 3 else torch.zeros(2, dtype=torch.bool)
                ),
            }
        algorithm.process_env_step(next_observations, torch.randn(2), dones, extras)
        observations = next_observations
    algorithm.compute_returns(observations)
    return algorithm, algorithm.update()


def test_pace_ppo_runs_one_rsl_rl_update():
    algorithm, losses = _one_update(PacePPO, "pace_eco_lab.rl.ppo:PacePPO", False)
    assert algorithm.pace_iteration == 1
    assert set(losses) >= {"value", "surrogate", "entropy", "entropy_coefficient"}


def test_lagrangian_runs_one_update_and_uses_complete_episode_cost():
    algorithm, losses = _one_update(
        PPOLagrangian,
        "pace_eco_lab.rl.ppo_lagrangian:PPOLagrangian",
        True,
    )
    assert algorithm.pace_iteration == 1
    assert losses["normalized_episode_cost"] == pytest.approx(1.2)
    assert losses["lagrange_multiplier"] > 0.0
    assert set(losses) >= {"cost_surrogate", "constraint_violation"}
