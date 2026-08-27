#!/usr/bin/env python3
"""在查看模型评估结果前生成 v2.3 Mixed calibration/holdout manifests。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pace_eco_lab.direction_conditioned_v2_3_mixed_protocol import (
    DESIRED_DIRECTION_W,
    EVAL_BATCHES,
    EVAL_NUM_ENVS,
    EVAL_TERRAIN_COLS,
    EVAL_TERRAIN_ROWS,
    MANIFEST_VERSION,
    PROTOCOL_VERSION,
    TARGET_SPEED_M_S_V2,
    evaluation_batch_seed,
    subtask_name,
    terrain_seed,
)
from pace_eco_lab.evaluation_states import (
    MULTI_TERRAIN_CALIBRATION_STATE_SET,
    MULTI_TERRAIN_HOLDOUT_STATE_SET,
    evaluation_state_count,
    evaluation_state_definition_sha256,
)
from pace_eco_lab.multi_terrain_protocol import difficulty_label, terrain_column_metadata
from pace_eco_lab.terrain_geometry import curriculum_difficulty_table


parser = argparse.ArgumentParser(description="冻结 v2.3 Mixed 两份评估 manifest。")
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()


def _payload(split: str) -> dict[str, object]:
    state_set = (
        MULTI_TERRAIN_CALIBRATION_STATE_SET
        if split == "calibration"
        else MULTI_TERRAIN_HOLDOUT_STATE_SET
    )
    state_count = evaluation_state_count(state_set)
    labels = terrain_column_metadata("mixed", EVAL_TERRAIN_COLS)
    episodes: list[dict[str, object]] = []
    for batch in range(EVAL_BATCHES):
        batch_seed = evaluation_batch_seed(split, batch)
        difficulties = curriculum_difficulty_table(
            batch_seed, EVAL_TERRAIN_ROWS, EVAL_TERRAIN_COLS, (0.10, 0.90)
        )
        for env_number in range(EVAL_NUM_ENVS):
            global_id = batch * EVAL_NUM_ENVS + env_number
            level = env_number % EVAL_TERRAIN_ROWS
            terrain_type = env_number // EVAL_TERRAIN_ROWS
            category, direction = labels[terrain_type].split(":", maxsplit=1)
            episodes.append(
                {
                    "episode_id": f"{split}-{global_id:04d}",
                    "batch_id": f"{split}-batch-{batch}",
                    "batch_index": batch,
                    "batch_env_number": env_number,
                    "terrain_category": category,
                    "direction": direction,
                    "subtask": subtask_name(category, direction),
                    "difficulty_level": level,
                    "difficulty_label": difficulty_label(level, EVAL_TERRAIN_ROWS),
                    "difficulty": float(difficulties[level, terrain_type]),
                    "terrain_seed": terrain_seed(split),
                    "batch_seed": batch_seed,
                    "initial_state_number": global_id % state_count,
                    "direction_command_w": list(DESIRED_DIRECTION_W),
                    "target_speed_m_s": TARGET_SPEED_M_S_V2,
                    "protocol_version": PROTOCOL_VERSION,
                    "initial_state_set": state_set,
                    "initial_state_set_sha256": evaluation_state_definition_sha256(state_set),
                }
            )
    return {
        "冻结状态": "已冻结",
        "manifest版本": MANIFEST_VERSION,
        "协议版本": PROTOCOL_VERSION,
        "数据拆分": split,
        "回合数": len(episodes),
        "初始状态集": state_set,
        "初始状态集SHA256": evaluation_state_definition_sha256(state_set),
        "逐回合": episodes,
    }


def main() -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        split: output_dir / f"{split}_manifest.json"
        for split in ("calibration", "holdout")
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"拒绝覆盖已冻结 manifest：{existing}")
    for split, path in paths.items():
        path.write_text(
            json.dumps(_payload(split), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(path)


if __name__ == "__main__":
    main()
