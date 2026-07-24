"""Freeze a segment checkpoint for an independent dual evaluation process."""

from __future__ import annotations

import argparse

from pace_sim2real.dual import freeze_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cycle_id", required=True, type=int)
    parser.add_argument("--budget_j", required=True, type=float)
    args = parser.parse_args()
    request = freeze_checkpoint(
        args.checkpoint,
        args.output_dir,
        cycle_id=args.cycle_id,
        budget_j=args.budget_j,
    )
    print(request)


if __name__ == "__main__":
    main()
