"""Check the two-consecutive-evaluation cost-critic admission gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pace_sim2real.dual import DualEvaluationResult


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs=2, help="Two consecutive dual result JSON files")
    parser.add_argument("--max_bias", type=float, default=0.05)
    parser.add_argument("--min_explained_variance", type=float, default=0.5)
    parser.add_argument("--max_initial_rmse", type=float, default=None)
    parser.add_argument("--max_episode_equal_rmse", type=float, default=None)
    parser.add_argument("--minimum_episodes", type=int, default=256)
    args = parser.parse_args()

    results = []
    for path in args.results:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        result = DualEvaluationResult.from_dict(payload)
        result.validate()
        results.append(result)
    if results[1].cycle_id != results[0].cycle_id + 1:
        raise SystemExit("FAIL: results are not consecutive cycles")
    if any(result.schema_version < 2 for result in results):
        raise SystemExit("FAIL: the revised admission gate requires schema-v2 full-trajectory evaluations")
    if results[0].checkpoint_sha256 != results[1].checkpoint_sha256:
        raise SystemExit("FAIL: the two diagnostic evaluations do not use the same frozen checkpoint")
    if results[0].evaluation_seed == results[1].evaluation_seed:
        raise SystemExit("FAIL: the two diagnostic evaluations must use different evaluation seeds")

    failures = []
    for result in results:
        if result.num_episodes < args.minimum_episodes:
            failures.append(f"cycle {result.cycle_id}: only {result.num_episodes} episodes")
        if result.initial_absolute_mean_bias > args.max_bias:
            failures.append(
                f"cycle {result.cycle_id}: initial absolute mean bias="
                f"{result.initial_absolute_mean_bias:.6f} > {args.max_bias}"
            )
        if result.trajectory_explained_variance < args.min_explained_variance:
            failures.append(
                f"cycle {result.cycle_id}: pooled trajectory explained variance="
                f"{result.trajectory_explained_variance:.6f} "
                f"< {args.min_explained_variance}"
            )
        metrics = result.cost_value_metrics
        assert metrics is not None
        if args.max_initial_rmse is not None and metrics["initial"]["rmse"] > args.max_initial_rmse:
            failures.append(
                f"cycle {result.cycle_id}: initial RMSE={metrics['initial']['rmse']:.6f} "
                f"> {args.max_initial_rmse}"
            )
        episode_rmse = metrics["trajectory"]["episode_equal_rmse"]
        if args.max_episode_equal_rmse is not None and episode_rmse > args.max_episode_equal_rmse:
            failures.append(
                f"cycle {result.cycle_id}: episode-equal RMSE={episode_rmse:.6f} "
                f"> {args.max_episode_equal_rmse}"
            )
    if failures:
        raise SystemExit("FAIL:\n" + "\n".join(failures))
    print("PASS: one frozen cost critic met all configured thresholds under two independent evaluation seeds")


if __name__ == "__main__":
    main()
