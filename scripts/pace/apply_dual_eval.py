"""Atomically and idempotently apply one dual-evaluation JSON result."""

from __future__ import annotations

import argparse

from pace_sim2real.dual import DualState, apply_evaluation_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--budget_j", required=True, type=float)
    parser.add_argument("--lambda_lr", required=True, type=float, choices=(0.01, 0.05, 0.1))
    parser.add_argument("--lambda_max", type=float, default=100.0)
    parser.add_argument("--minimum_episodes", type=int, default=256)
    args = parser.parse_args()
    state, changed = apply_evaluation_result(
        args.state,
        args.result,
        initial_state=DualState(
            budget_j=args.budget_j,
            learning_rate=args.lambda_lr,
            multiplier_max=args.lambda_max,
        ),
        minimum_episodes=args.minimum_episodes,
    )
    print(f"changed={changed} cycle={state.last_cycle_id} lambda={state.multiplier:.8f}")


if __name__ == "__main__":
    main()
