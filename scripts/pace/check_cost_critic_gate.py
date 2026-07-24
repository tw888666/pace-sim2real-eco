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

    failures = []
    for result in results:
        if result.num_episodes < args.minimum_episodes:
            failures.append(f"cycle {result.cycle_id}: only {result.num_episodes} episodes")
        if result.cost_value_initial_bias > args.max_bias:
            failures.append(
                f"cycle {result.cycle_id}: |V_C(s0)-return|={result.cost_value_initial_bias:.6f} > {args.max_bias}"
            )
        if result.cost_explained_variance < args.min_explained_variance:
            failures.append(
                f"cycle {result.cycle_id}: explained variance={result.cost_explained_variance:.6f} "
                f"< {args.min_explained_variance}"
            )
    if failures:
        raise SystemExit("FAIL:\n" + "\n".join(failures) + "\nIncrease rollout length to 48, then 64 if needed.")
    print("PASS: cost critic met both calibration thresholds in two consecutive cycles")


if __name__ == "__main__":
    main()
