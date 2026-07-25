import torch
from tensordict import TensorDict

from pace_sim2real.algorithms import (
    ConstrainedRolloutStorage,
    PacePPOLagrangian,
    normalized_lagrangian_actor_loss,
)


def _storage(last_transition_done: bool) -> ConstrainedRolloutStorage:
    observations = TensorDict({"policy": torch.zeros(1, 2)}, batch_size=[1])
    storage = ConstrainedRolloutStorage("rl", 1, 2, observations, [1], "cpu")
    storage.costs[:, 0, 0] = 1.0
    storage.cost_values.zero_()
    storage.cost_dones[1, 0, 0] = int(last_transition_done)
    return storage


def test_cost_timeout_is_terminal_and_does_not_bootstrap() -> None:
    storage = _storage(last_transition_done=True)
    storage.compute_cost_returns(torch.tensor([[10.0]]), gamma=1.0, lam=1.0)
    torch.testing.assert_close(storage.cost_returns[:, 0, 0], torch.tensor([2.0, 1.0]))


def test_ordinary_rollout_boundary_bootstraps_cost_critic() -> None:
    storage = _storage(last_transition_done=False)
    storage.compute_cost_returns(torch.tensor([[10.0]]), gamma=1.0, lam=1.0)
    torch.testing.assert_close(storage.cost_returns[:, 0, 0], torch.tensor([12.0, 11.0]))


def test_actor_loss_normalizes_only_policy_surrogates() -> None:
    loss = normalized_lagrangian_actor_loss(
        torch.tensor(2.0),
        torch.tensor(4.0),
        lagrangian_multiplier=3.0,
        entropy=torch.tensor([5.0]),
        entropy_coefficient=0.1,
    )
    torch.testing.assert_close(loss, torch.tensor(3.0))


def test_rsl_rl_5_algorithm_constructs_and_updates_on_cpu() -> None:
    class DummyEnv:
        num_envs = 4
        num_actions = 2

    obs = TensorDict({"policy": torch.randn(4, 3)}, batch_size=[4])
    cfg = {
        "num_steps_per_env": 2,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"], "cost_critic": ["policy"]},
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [8],
            "activation": "elu",
            "obs_normalization": False,
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.5, "std_type": "scalar"},
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [8],
            "activation": "elu",
            "obs_normalization": False,
        },
        "cost_critic": {
            "class_name": "MLPModel",
            "hidden_dims": [8],
            "activation": "elu",
            "obs_normalization": False,
            # Isaac Lab leaves these compatibility fields on custom models;
            # RSL-RL 5.x no longer accepts them in MLPModel.__init__.
            "stochastic": False,
            "init_noise_std": 1.0,
            "noise_std_type": "scalar",
            "state_dependent_std": False,
        },
        "algorithm": {
            "class_name": "pace_sim2real.algorithms:PacePPOLagrangian",
            # RSL-RL consumes this construction-only option before PPO.__init__.
            "share_cnn_encoders": False,
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "schedule": "fixed",
            "desired_kl": 0.01,
        },
        "multi_gpu": None,
    }
    algorithm = PacePPOLagrangian.construct_algorithm(obs, DummyEnv(), cfg, "cpu")
    for step in range(2):
        algorithm.act(obs)
        dones = torch.zeros(4, dtype=torch.long)
        if step == 1:
            dones[0] = 1
        algorithm.process_env_step(
            obs,
            rewards=torch.ones(4),
            dones=dones,
            extras={"pace_cost": torch.full((4,), 0.1)},
        )
    algorithm.compute_returns(obs)
    losses = algorithm.update()
    assert "cost_value" in losses
    assert "cost_explained_variance" in losses
    assert algorithm.constrained_storage.step == 0

    algorithm.set_lagrangian_multiplier(2.5)
    saved = algorithm.save()
    algorithm.set_lagrangian_multiplier(0.0)
    assert algorithm.load(saved, load_cfg=None, strict=True)
    assert algorithm.lagrangian_multiplier == 2.5

    # A budget change invalidates the scale of the saved cost critic. Selective
    # resume must preserve its freshly initialized/reinitialized parameters and
    # multiplier while restoring the locomotion policy and iteration state.
    with torch.no_grad():
        for parameter in algorithm.cost_critic.parameters():
            parameter.add_(1.0)
    reinitialized_cost_state = {
        name: value.clone() for name, value in algorithm.cost_critic.state_dict().items()
    }
    algorithm.set_lagrangian_multiplier(0.0)
    assert algorithm.load(
        saved,
        load_cfg={
            "actor": True,
            "critic": True,
            "optimizer": True,
            "iteration": True,
            "rnd": True,
            "cost_critic": False,
            "cost_optimizer": False,
            "lagrangian_multiplier": False,
        },
        strict=True,
    )
    assert algorithm.lagrangian_multiplier == 0.0
    for name, value in algorithm.cost_critic.state_dict().items():
        torch.testing.assert_close(value, reinitialized_cost_state[name])
