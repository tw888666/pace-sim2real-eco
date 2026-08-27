#!/usr/bin/env python3
"""适配既有正式回放器，录制方向条件 v2.1 最终 model_2999。"""

from __future__ import annotations

from pathlib import Path


source_path = Path(
    "/home/xy.chen/tw/PACE-ECO-multi-terrain/scripts/pace_eco/"
    "gpt_录制多地形策略回放.py"
)
source = source_path.read_text(encoding="utf-8")


def replace_once(old: str, new: str) -> None:
    global source
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"录像适配锚点数量错误：{count}，锚点={old[:80]!r}")
    source = source.replace(old, new)


replace_once(
    "from pace_eco_lab.evaluation_states import MULTI_TERRAIN_HOLDOUT_STATE_SET\n"
    "from pace_eco_lab.multi_terrain_protocol import (\n"
    "    EVAL_NUM_ENVS,\n"
    "    EVAL_TERRAIN_ROWS,\n"
    "    TASK_IDS,\n"
    "    evaluation_batch_seed,\n"
    "    terrain_seed,\n"
    ")",
    "from pace_eco_lab.evaluation_states import MULTI_TERRAIN_CALIBRATION_STATE_SET\n"
    "from pace_eco_lab.direction_conditioned_protocol import (\n"
    "    EVAL_NUM_ENVS,\n"
    "    EVAL_TERRAIN_ROWS,\n"
    "    TASK_IDS,\n"
    "    evaluation_batch_seed,\n"
    "    terrain_seed,\n"
    ")",
)
replace_once(
    "def _task_parts() -> tuple[str, str]:\n"
    "    inverse = {task_id: pair for pair, task_id in TASK_IDS.items()}\n"
    "    try:\n"
    "        return inverse[args_cli.task]\n"
    "    except KeyError as error:\n"
    "        raise ValueError(\"只允许 Terrain20sWide 多地形任务。\") from error",
    "def _task_parts() -> tuple[str, str, str]:\n"
    "    inverse = {task_id: triple for triple, task_id in TASK_IDS.items()}\n"
    "    try:\n"
    "        return inverse[args_cli.task]\n"
    "    except KeyError as error:\n"
    "        raise ValueError(\"只允许方向条件 v2.1 任务。\") from error",
)
replace_once("method, terrain = _task_parts()", "variant, method, terrain = _task_parts()")
replace_once(
    'base_terrain_seed = terrain_seed("stage1_holdout", terrain)',
    'base_terrain_seed = terrain_seed("stage1_calibration", terrain)',
)
replace_once(
    "MULTI_TERRAIN_HOLDOUT_STATE_SET,",
    "MULTI_TERRAIN_CALIBRATION_STATE_SET,",
)
replace_once(
    'final_video = output_dir / "gpt_多地形策略回放.mp4"',
    'final_video = output_dir / f"gpt_{terrain}方向条件model2999行走回放.mp4"',
)
replace_once(
    '"正式评估环境编号": formal_env_index,',
    '"方向条件标定环境编号": formal_env_index,\n'
    '        "实验变体": variant,\n'
    '        "评估性质": "stage1 calibration 最终 model_2999 代表回放",',
)
replace_once(
    'report_path = output_dir / "gpt_多地形策略回放说明.json"',
    'report_path = output_dir / f"gpt_{terrain}方向条件model2999行走回放说明.json"',
)

exec(compile(source, str(source_path), "exec"), {"__name__": "__main__", "__file__": str(source_path)})
