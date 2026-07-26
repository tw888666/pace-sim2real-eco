import torch

from pace_sim2real.dual import (
    atomic_torch_save,
    fit_time_only_linear_baseline,
    load_critic_dataset,
    module_sha256,
    predict_time_only,
    train_offline_cost_critic,
    validate_critic_dataset,
)


def _payload() -> dict:
    return {
        "schema_version": 1,
        "checkpoint_path": "/tmp/model.pt",
        "checkpoint_sha256": "abc",
        "budget_j": 100.0,
        "evaluation_seed": 7,
        "policy_observation": torch.arange(24, dtype=torch.float32).reshape(2, 3, 4),
        "remaining_time": torch.tensor([[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]]),
        "cost": torch.tensor([[0.1, 0.2, 0.0], [0.3, 1.0, 0.0]]),
        "valid": torch.tensor([[True, True, False], [True, True, False]]),
        "success": torch.tensor([True, False]),
        "physical_energy_j": torch.tensor([30.0, 40.0]),
    }


def test_critic_dataset_round_trip(tmp_path) -> None:
    payload = _payload()
    validate_critic_dataset(payload)
    path = tmp_path / "dataset.pt"
    atomic_torch_save(path, payload)
    loaded = load_critic_dataset(path)
    torch.testing.assert_close(loaded["policy_observation"], payload["policy_observation"])


def test_paired_critics_have_identical_initial_parameters() -> None:
    features_zero = torch.randn(12, 5)
    features_real = features_zero.clone()
    features_zero[:, -1] = 0.0
    targets = torch.linspace(0.0, 1.0, 12)
    zero, zero_info = train_offline_cost_critic(
        features_zero,
        targets,
        seed=3,
        device="cpu",
        epochs=1,
        batch_size=4,
        learning_rate=1.0e-3,
        hidden_dims=(8,),
    )
    real, real_info = train_offline_cost_critic(
        features_real,
        targets,
        seed=3,
        device="cpu",
        epochs=1,
        batch_size=4,
        learning_rate=1.0e-3,
        hidden_dims=(8,),
    )
    assert zero_info["initial_state_sha256"] == real_info["initial_state_sha256"]
    assert module_sha256(zero) != module_sha256(real)


def test_time_only_linear_baseline_recovers_constant_power_law() -> None:
    remaining = torch.tensor([1.0, 0.75, 0.5, 0.25])
    target = 2.0 * remaining + 0.1
    coefficients = fit_time_only_linear_baseline(remaining, target)
    torch.testing.assert_close(coefficients, torch.tensor([2.0, 0.1]), atol=1.0e-5, rtol=1.0e-5)
    torch.testing.assert_close(predict_time_only(remaining, coefficients), target)
