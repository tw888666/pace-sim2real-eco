"""Backfill episode-level energy feasibility metrics from an archived dataset.

This utility is CPU-only. It never modifies the source dataset and publishes
an optional JSON summary atomically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pace_sim2real.dual import (
    atomic_write_json,
    energy_feasibility_metrics,
    file_sha256,
    load_critic_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    dataset = load_critic_dataset(dataset_path)
    valid = dataset["valid"]
    episode_augmented_cost = torch.where(
        valid,
        dataset["cost"],
        torch.zeros_like(dataset["cost"]),
    ).sum(dim=1)
    physical_energy = dataset["physical_energy_j"]
    success = dataset["success"]
    budget_j = float(dataset["budget_j"])

    payload = {
        "schema_version": 1,
        "source": "archived_complete_episode_dataset",
        "dataset_path": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "checkpoint_path": dataset["checkpoint_path"],
        "checkpoint_sha256": dataset["checkpoint_sha256"],
        "evaluation_seed": int(dataset["evaluation_seed"]),
        "budget_j": budget_j,
        "mean_physical_energy_j": float(physical_energy.float().mean().item()),
        "mean_augmented_cost": float(episode_augmented_cost.float().mean().item()),
        "success_rate": float(success.float().mean().item()),
        "energy_feasibility_metrics": energy_feasibility_metrics(
            physical_energy,
            success,
            budget_j,
        ).to_dict(),
    }
    if args.output is not None:
        output_path = Path(args.output).resolve()
        atomic_write_json(output_path, payload)
        print(output_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
