"""Run paired zero-time, real-time, and time-only cost-value controls."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch

from pace_sim2real.dual import (
    atomic_torch_save,
    atomic_write_json,
    evaluate_trajectory_predictions,
    file_sha256,
    fit_time_only_linear_baseline,
    load_critic_dataset,
    predict_in_batches,
    predict_time_only,
    train_offline_cost_critic,
    undiscounted_return_to_go,
)


def _episode_split(num_episodes: int, validation_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(num_episodes, generator=generator)
    num_validation = max(1, round(num_episodes * validation_fraction))
    if num_validation >= num_episodes:
        raise ValueError("validation split leaves no training episodes")
    return permutation[num_validation:], permutation[:num_validation]


def _masked_flatten(value: torch.Tensor, episode_ids: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    selected = value[episode_ids]
    selected_valid = valid[episode_ids]
    return selected[selected_valid]


def _heldout_metrics(
    prediction: torch.Tensor,
    dataset: dict,
    heldout_ids: torch.Tensor,
) -> dict:
    return evaluate_trajectory_predictions(
        prediction[heldout_ids],
        dataset["cost"][heldout_ids],
        dataset["valid"][heldout_ids],
        dataset["remaining_time"][heldout_ids],
        dataset["success"][heldout_ids],
    )


def _summary(entries: list[dict]) -> dict:
    summary = {}
    for variant in ("zero_time", "real_time"):
        rmse = [entry[variant]["metrics"]["trajectory"]["episode_equal_rmse"] for entry in entries]
        ev = [entry[variant]["metrics"]["trajectory"]["pooled"]["explained_variance"] for entry in entries]
        summary[variant] = {
            "episode_equal_rmse_mean": statistics.fmean(rmse),
            "episode_equal_rmse_std": statistics.pstdev(rmse),
            "pooled_explained_variance_mean": statistics.fmean(ev),
            "pooled_explained_variance_std": statistics.pstdev(ev),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--critic_seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--split_seed", type=int, default=13579)
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--learning_rate", type=float, default=1.0e-3)
    args = parser.parse_args()
    if len(set(args.critic_seeds)) != len(args.critic_seeds):
        raise ValueError("critic_seeds must be unique")

    dataset_path = Path(args.dataset).resolve()
    dataset = load_critic_dataset(dataset_path)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_ids, heldout_ids = _episode_split(
        dataset["policy_observation"].shape[0], args.validation_fraction, args.split_seed
    )
    valid = dataset["valid"]
    targets = undiscounted_return_to_go(dataset["cost"], valid)
    zero_feature = torch.cat(
        (dataset["policy_observation"], torch.zeros_like(dataset["remaining_time"]).unsqueeze(-1)), dim=-1
    )
    real_feature = torch.cat(
        (dataset["policy_observation"], dataset["remaining_time"].unsqueeze(-1)), dim=-1
    )
    train_target = _masked_flatten(targets, train_ids, valid)

    entries = []
    for seed in args.critic_seeds:
        seed_entry = {"seed": seed}
        paired_initial_hash = None
        for variant, feature in (("zero_time", zero_feature), ("real_time", real_feature)):
            train_feature = _masked_flatten(feature, train_ids, valid)
            model, training = train_offline_cost_critic(
                train_feature,
                train_target,
                seed=seed,
                device=args.device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
            )
            if paired_initial_hash is None:
                paired_initial_hash = training["initial_state_sha256"]
            elif training["initial_state_sha256"] != paired_initial_hash:
                raise RuntimeError("paired critics did not start from identical parameters")
            flat_prediction = predict_in_batches(
                model,
                feature.reshape(-1, feature.shape[-1]),
                device=args.device,
                batch_size=args.batch_size,
            )
            prediction = flat_prediction.reshape(feature.shape[:2])
            model_path = output_dir / f"{variant}_seed_{seed}.pt"
            atomic_torch_save(
                model_path,
                {
                    "schema_version": 1,
                    "variant": variant,
                    "seed": seed,
                    "dataset_sha256": file_sha256(dataset_path),
                    "source_actor_checkpoint_sha256": dataset["checkpoint_sha256"],
                    "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                    "training": training,
                },
            )
            seed_entry[variant] = {
                "training": training,
                "model_path": str(model_path),
                "model_sha256": file_sha256(model_path),
                "metrics": _heldout_metrics(prediction, dataset, heldout_ids),
            }
        entries.append(seed_entry)

    train_time = _masked_flatten(dataset["remaining_time"], train_ids, valid)
    coefficients = fit_time_only_linear_baseline(train_time, train_target)
    time_prediction = predict_time_only(dataset["remaining_time"], coefficients)
    result = {
        "schema_version": 1,
        "dataset_path": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "train_episode_ids": train_ids.tolist(),
        "heldout_episode_ids": heldout_ids.tolist(),
        "split_seed": args.split_seed,
        "critic_seeds": args.critic_seeds,
        "controls": entries,
        "time_only": {
            "model": "linear_slope_and_intercept",
            "coefficients": coefficients.tolist(),
            "metrics": _heldout_metrics(time_prediction, dataset, heldout_ids),
        },
        "paired_summary": _summary(entries),
    }
    result_path = output_dir / "comparison.json"
    atomic_write_json(result_path, result)
    print(result_path)


if __name__ == "__main__":
    main()
