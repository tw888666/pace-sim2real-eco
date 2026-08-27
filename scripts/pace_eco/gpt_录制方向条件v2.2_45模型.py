#!/usr/bin/env python3
"""适配既有只读回放器，录制v2.2授权45模型的固定holdout实例。"""

from __future__ import annotations

from pathlib import Path


source_path = Path(__file__).with_name("gpt_录制多地形策略回放.py")
source = source_path.read_text(encoding="utf-8")


def replace_once(old: str, new: str) -> None:
    global source
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"录像适配锚点数量错误：{count}，锚点={old[:100]!r}")
    source = source.replace(old, new)


replace_once(
    'parser.add_argument("--energy_reference_json", default=None)',
    'parser.add_argument("--energy_reference_json", default=None)\n'
    'parser.add_argument("--holdout_authorization", required=True)',
)
replace_once(
    "from pace_eco_lab.multi_terrain_protocol import (\n"
    "    EVAL_NUM_ENVS,\n"
    "    EVAL_TERRAIN_ROWS,\n"
    "    TASK_IDS,\n"
    "    evaluation_batch_seed,\n"
    "    terrain_seed,\n"
    ")",
    "from pace_eco_lab.direction_conditioned_protocol import (\n"
    "    EVAL_NUM_ENVS,\n"
    "    EVAL_TERRAIN_ROWS,\n"
    "    evaluation_batch_seed,\n"
    "    terrain_seed,\n"
    ")\n"
    "from pace_eco_lab.direction_conditioned_v2_2_protocol import (\n"
    "    EVALUATION_PROTOCOL_VERSION,\n"
    "    FIXED_ENERGY_REWARD_WEIGHT,\n"
    "    TASK_IDS,\n"
    ")",
)
replace_once(
    "def _task_parts() -> tuple[str, str]:\n"
    "    inverse = {task_id: pair for pair, task_id in TASK_IDS.items()}\n"
    "    try:\n"
    "        return inverse[args_cli.task]\n"
    "    except KeyError as error:\n"
    '        raise ValueError("只允许 Terrain20sWide 多地形任务。") from error',
    "def _task_parts() -> tuple[str, str, str]:\n"
    "    inverse = {task_id: triple for triple, task_id in TASK_IDS.items()}\n"
    "    try:\n"
    "        return inverse[args_cli.task]\n"
    "    except KeyError as error:\n"
    '        raise ValueError("只允许方向条件v2.2三方法任务。") from error',
)
replace_once("method, terrain = _task_parts()", "variant, method, terrain = _task_parts()")
replace_once(
    "    if method != \"eco\":\n"
    "        if args_cli.energy_reference_json is not None:\n"
    '            raise ValueError("非 PACE-ECO 回放不应提供能耗参考文件。")\n'
    "        return None, None",
    "    if method != \"eco\":\n"
    "        if args_cli.energy_reference_json is not None:\n"
    '            raise ValueError("非ECO回放不应提供能耗参考文件。")\n'
    "        return None, None",
)
replace_once(
    '        raise FileNotFoundError("回放只接受正式终点 model_2999.pt。")',
    '        raise FileNotFoundError("回放只接受正式终点 model_2999.pt。")\n'
    '    authorization_path = Path(args_cli.holdout_authorization).expanduser().resolve()\n'
    '    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))\n'
    '    authorized = [item for item in authorization.get("模型", []) if Path(item.get("检查点", "")).resolve() == checkpoint and item.get("任务") == args_cli.task and int(item.get("PPO_seed", -1)) == args_cli.ppo_seed]\n'
    '    if authorization.get("协议版本") != EVALUATION_PROTOCOL_VERSION or len(authorized) != 1:\n'
    '        raise ValueError("当前模型不在冻结的45模型holdout授权中。")\n'
    '    if authorized[0].get("检查点SHA256") != _file_sha256(checkpoint):\n'
    '        raise ValueError("当前checkpoint哈希与45模型授权不一致。")\n'
    '    authorization_sha256 = _file_sha256(authorization_path)',
)
replace_once(
    '    if method == "eco":\n        agent_cfg.algorithm.energy_budget_j = energy_budget',
    '    if method == "eco":\n'
    '        agent_cfg.algorithm.energy_budget_j = energy_budget\n'
    '    if method == "fixed_weight" and abs(float(env_cfg.rewards.energy.weight) - FIXED_ENERGY_REWARD_WEIGHT) > 1.0e-12:\n'
    '        raise ValueError("Fixed-weight回放配置不是冻结的-0.00016。")',
)
replace_once(
    'final_video = output_dir / "gpt_多地形策略回放.mp4"',
    'final_video = output_dir / f"gpt_{terrain}_{method}_seed{args_cli.ppo_seed}_holdout回放.mp4"',
)
replace_once(
    '        "任务": args_cli.task,',
    '        "协议版本": EVALUATION_PROTOCOL_VERSION,\n'
    '        "任务": args_cli.task,\n'
    '        "实验变体": variant,',
)
replace_once(
    '        "检查点SHA256": _file_sha256(checkpoint),',
    '        "检查点SHA256": _file_sha256(checkpoint),\n'
    '        "holdout授权SHA256": authorization_sha256,\n'
    '        "固定录像选择规则": "batch0/env2；与历史最终模型回放一致的中等难度实例，不按表现挑选",',
)
replace_once(
    'report_path = output_dir / "gpt_多地形策略回放说明.json"',
    'report_path = output_dir / f"gpt_{terrain}_{method}_seed{args_cli.ppo_seed}_holdout回放说明.json"',
)

exec(compile(source, str(source_path), "exec"), {"__name__": "__main__", "__file__": str(source_path)})
