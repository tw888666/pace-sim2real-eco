#!/usr/bin/env python3
"""适配只读回放器，录制 v2.3 Mixed 授权模型的固定地形实例。"""

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
    'parser.add_argument("--energy_reference_json", required=True)\n'
    'parser.add_argument("--holdout_authorization", required=True)\n'
    'parser.add_argument("--scene", choices=("flat", "boxes", "stairs_up"), default="flat")',
)
replace_once(
    "from pace_eco_lab.multi_terrain_protocol import (\n"
    "    EVAL_NUM_ENVS,\n"
    "    EVAL_TERRAIN_ROWS,\n"
    "    TASK_IDS,\n"
    "    evaluation_batch_seed,\n"
    "    terrain_seed,\n"
    ")",
    "from pace_eco_lab.multi_terrain_protocol import EVAL_NUM_ENVS, EVAL_TERRAIN_ROWS\n"
    "from pace_eco_lab.direction_conditioned_v2_3_mixed_protocol import (\n"
    "    FIXED_ENERGY_REWARD_WEIGHT,\n"
    "    PROTOCOL_VERSION,\n"
    "    TASK_IDS,\n"
    "    evaluation_batch_seed,\n"
    ")",
)
replace_once(
    "def _task_parts() -> tuple[str, str]:\n"
    "    inverse = {task_id: pair for pair, task_id in TASK_IDS.items()}\n"
    "    try:\n"
    "        return inverse[args_cli.task]\n"
    "    except KeyError as error:\n"
    '        raise ValueError("只允许 Terrain20sWide 多地形任务。") from error',
    "def _task_parts() -> tuple[str, str]:\n"
    "    inverse = {task_id: method for method, task_id in TASK_IDS.items()}\n"
    "    try:\n"
    '        return inverse[args_cli.task], "mixed"\n'
    "    except KeyError as error:\n"
    '        raise ValueError("只允许方向条件 v2.3 Mixed 三方法任务。") from error',
)
replace_once(
    "def _load_energy_budget(method: str, terrain: str) -> tuple[float | None, str | None]:\n"
    "    if method != \"eco\":\n"
    "        if args_cli.energy_reference_json is not None:\n"
    '            raise ValueError("非 PACE-ECO 回放不应提供能耗参考文件。")\n'
    "        return None, None\n"
    "    if args_cli.energy_reference_json is None:\n"
    '        raise ValueError("PACE-ECO 回放必须提供训练时使用的 B_ref 冻结文件。")\n'
    "    path = Path(args_cli.energy_reference_json).expanduser().resolve()\n"
    '    data = json.loads(path.read_text(encoding="utf-8"))\n'
    '    reference = float(data.get("B_ref_J", {}).get(terrain, 0.0))\n'
    "    if reference <= 0.0:\n"
    '        raise ValueError(f"B_ref 文件缺少 {terrain} 的正值。")\n'
    "    return 0.8 * reference, _file_sha256(path)",
    "def _load_energy_budget(method: str, terrain: str) -> tuple[float | None, str | None]:\n"
    "    path = Path(args_cli.energy_reference_json).expanduser().resolve()\n"
    '    data = json.loads(path.read_text(encoding="utf-8"))\n'
    '    if data.get("冻结状态") != "已冻结" or data.get("协议版本") != PROTOCOL_VERSION:\n'
    '        raise ValueError("v2.3 Mixed预算冻结文件错误。")\n'
    '    budget = float(data.get("B80_mixed_J", 0.0))\n'
    "    if budget <= 0.0:\n"
    '        raise ValueError("v2.3 Mixed B80必须为正。")\n'
    '    return (budget if method == "eco" else None), _file_sha256(path)',
)
replace_once(
    '        raise FileNotFoundError("回放只接受正式终点 model_2999.pt。")',
    '        raise FileNotFoundError("回放只接受正式终点 model_2999.pt。")\n'
    '    authorization_path = Path(args_cli.holdout_authorization).expanduser().resolve()\n'
    '    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))\n'
    '    authorized = [item for item in authorization.get("模型", []) if Path(item.get("检查点", "")).resolve() == checkpoint and item.get("任务") == args_cli.task and int(item.get("PPO_seed", -1)) == args_cli.ppo_seed]\n'
    '    if authorization.get("协议版本") != PROTOCOL_VERSION or authorization.get("冻结状态") != "holdout已授权" or len(authorized) != 1:\n'
    '        raise ValueError("当前模型不在冻结的v2.3九模型holdout授权中。")\n'
    '    if authorized[0].get("检查点SHA256") != _file_sha256(checkpoint):\n'
    '        raise ValueError("当前checkpoint哈希与v2.3授权不一致。")\n'
    '    authorization_sha256 = _file_sha256(authorization_path)',
)
replace_once(
    '    if method == "eco":\n        agent_cfg.algorithm.energy_budget_j = energy_budget',
    '    if method == "eco":\n'
    '        agent_cfg.algorithm.energy_budget_j = energy_budget\n'
    '    if method == "fixed_weight" and abs(float(env_cfg.rewards.energy.weight) - FIXED_ENERGY_REWARD_WEIGHT) > 1.0e-12:\n'
    '        raise ValueError("Fixed-weight回放配置不是冻结的W100=-0.00016。")',
)
replace_once(
    '    base_terrain_seed = terrain_seed("stage1_holdout", terrain)\n'
    "    batch_terrain_seed = evaluation_batch_seed(base_terrain_seed, args_cli.batch_index)",
    '    batch_terrain_seed = evaluation_batch_seed("holdout", args_cli.batch_index)',
)
replace_once(
    'final_video = output_dir / "gpt_多地形策略回放.mp4"',
    'final_video = output_dir / f"gpt_v2.3_Mixed_{args_cli.scene}_{method}_seed{args_cli.ppo_seed}_holdout回放.mp4"',
)
replace_once(
    "    category, direction, difficulty = metadata_labels(\n"
    "        category_code,\n"
    "        direction_code,\n"
    "        level,\n"
    "        EVAL_TERRAIN_ROWS,\n"
    "    )",
    "    category, direction, difficulty = metadata_labels(\n"
    "        category_code,\n"
    "        direction_code,\n"
    "        level,\n"
    "        EVAL_TERRAIN_ROWS,\n"
    "    )\n"
    '    expected_scene = {"flat": ("flat", "level", 2), "boxes": ("boxes", "level", 32), "stairs_up": ("stairs", "up", 22)}[args_cli.scene]\n'
    '    if (category, direction, formal_env_index) != expected_scene:\n'
    '        raise RuntimeError(f"固定录像实例与{args_cli.scene}不匹配：{category}/{direction}/env{formal_env_index}")',
)
replace_once(
    '        "任务": args_cli.task,',
    '        "协议版本": PROTOCOL_VERSION,\n'
    '        "任务": args_cli.task,\n'
    '        "实验地形分布": "mixed",',
)
replace_once(
    '        "B_ref文件SHA256": reference_sha256,',
    '        "预算冻结文件SHA256": reference_sha256,\n'
    '        "holdout授权SHA256": authorization_sha256,\n'
    '        "固定录像选择规则": f"PPO seed1、holdout batch0/env{formal_env_index}；Mixed网格{category}/{direction}中等难度，不按表现挑选",',
)
replace_once(
    'report_path = output_dir / "gpt_多地形策略回放说明.json"',
    'report_path = output_dir / f"gpt_v2.3_Mixed_{args_cli.scene}_{method}_seed{args_cli.ppo_seed}_holdout回放说明.json"',
)

exec(compile(source, str(source_path), "exec"), {"__name__": "__main__", "__file__": str(source_path)})
