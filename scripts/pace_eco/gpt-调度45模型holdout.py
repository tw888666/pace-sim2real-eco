#!/usr/bin/env python3
"""在指定 GPU 上持久化串行执行45模型正式 holdout 队列。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts/pace_eco/gpt-运行方向条件v2.2评估.sh"
REFERENCE = ROOT / "results/supplementary/direction_conditioned_v2_1/stage1/gpt-阶段一v2_B_ref冻结文件.json"
DEFAULT_AUTHORIZATION = ROOT / "results/supplementary/direction_conditioned_v2_2/gpt-v2.2_45模型holdout授权.json"
PLAN_ROOT = ROOT / "results/supplementary/direction_conditioned_v2_2/gpt_45模型调度"
SESSION_PREFIX = "gpt_v2_2_holdout45_gpu"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="调度45模型方向条件正式holdout。")
    parser.add_argument("--authorization", default=str(DEFAULT_AUTHORIZATION))
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 3, 4, 5])
    parser.add_argument("--worker-gpu", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--plan", default=None, help=argparse.SUPPRESS)
    return parser


def _load_json(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON不是对象：{path}")
    return data


def _validate_authorization(path: Path) -> list[dict[str, object]]:
    data = _load_json(path)
    models = data.get("模型")
    if (
        data.get("冻结状态") != "holdout已授权"
        or data.get("协议版本") != "gpt-direction-conditioned-v2.2-45model-holdout-v1"
        or data.get("模型数量") != 45
        or not isinstance(models, list)
        or len(models) != 45
    ):
        raise ValueError("授权不是冻结的45模型正式holdout授权。")
    combinations = {(item["地形"], item["方法"], int(item["PPO_seed"])) for item in models}
    if len(combinations) != 45 or any(seed not in (1, 2, 3) for _, _, seed in combinations):
        raise ValueError("授权模型不满足45个唯一地形×方法×seed1-3组合。")
    return models


def _assign(models: list[dict[str, object]], gpus: list[int]) -> dict[int, list[dict[str, object]]]:
    if len(gpus) != 4 or len(set(gpus)) != 4 or any(gpu not in range(6) for gpu in gpus):
        raise ValueError("正式队列必须提供4张不同的GPU，编号范围0--5。")
    queues = {gpu: [] for gpu in gpus}
    for index, model in enumerate(models):
        queues[gpus[index % len(gpus)]].append(model)
    return queues


def _worker(gpu: int, plan_path: Path, authorization: Path) -> None:
    plan = _load_json(plan_path)
    queue = plan.get("GPU队列", {}).get(str(gpu))
    if not isinstance(queue, list) or not queue:
        raise ValueError(f"GPU{gpu}在计划中没有队列。")
    for queue_index, item in enumerate(queue, start=1):
        checkpoint = Path(str(item["检查点"]))
        command = [
            "bash", str(LAUNCHER), str(gpu), "stage1", "holdout", "directional",
            str(item["地形"]), str(item["方法"]), str(item["PPO_seed"]),
            str(checkpoint), str(REFERENCE), str(authorization),
        ]
        print(
            f"[45模型队列] GPU{gpu} 开始 {queue_index}/{len(queue)}："
            f"{item['地形']}/{item['方法']}/seed{item['PPO_seed']}",
            flush=True,
        )
        subprocess.run(command, cwd=ROOT, check=True)
        print(f"[45模型队列] GPU{gpu} 完成 {queue_index}/{len(queue)}。", flush=True)
    print(f"[45模型队列] GPU{gpu} 队列全部完成。", flush=True)


def _controller(gpus: list[int], authorization: Path, models: list[dict[str, object]]) -> None:
    queues = _assign(models, gpus)
    existing = subprocess.run(
        ["tmux", "list-sessions", "-F", "#{session_name}"],
        text=True, capture_output=True, check=False,
    ).stdout.splitlines()
    conflicts = [f"{SESSION_PREFIX}{gpu}" for gpu in gpus if f"{SESSION_PREFIX}{gpu}" in existing]
    if conflicts:
        raise RuntimeError(f"拒绝覆盖已有正式评估会话：{conflicts}")
    PLAN_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    plan_path = PLAN_ROOT / f"gpt-45模型GPU分配计划_{timestamp}.json"
    payload = {
        "状态": "已冻结并启动",
        "创建时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "授权": str(authorization),
        "GPU顺序": gpus,
        "分配规则": "按授权顺序轮转GPU；单卡串行，单模型四批完整结束后进入下一项",
        "GPU队列": {str(gpu): queue for gpu, queue in queues.items()},
    }
    plan_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for gpu in gpus:
        session = f"{SESSION_PREFIX}{gpu}"
        log_path = PLAN_ROOT / f"gpt-GPU{gpu}_45模型队列_{timestamp}.log"
        worker_command = (
            f"set -o pipefail; cd {ROOT} && PYTHONUNBUFFERED=1 {sys.executable} {Path(__file__).resolve()} "
            f"--authorization {authorization} --worker-gpu {gpu} --plan {plan_path} "
            f"2>&1 | tee {log_path}"
        )
        subprocess.run(["tmux", "new-session", "-d", "-s", session, worker_command], check=True)
    print(plan_path)
    for gpu in gpus:
        print(f"{SESSION_PREFIX}{gpu}")


def main() -> None:
    args = _parser().parse_args()
    authorization = Path(args.authorization).expanduser().resolve()
    models = _validate_authorization(authorization)
    if args.worker_gpu is not None:
        if args.plan is None:
            raise ValueError("worker模式必须提供计划文件。")
        _worker(args.worker_gpu, Path(args.plan).resolve(), authorization)
    else:
        _controller(args.gpus, authorization, models)


if __name__ == "__main__":
    main()
