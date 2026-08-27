#!/usr/bin/env python3
"""使用GPU0/1串行队列录制45个授权模型的固定holdout回放。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RECORDER = ROOT / "scripts/pace_eco/gpt_录制方向条件v2.2_45模型.py"
ISAACLAB = Path("/home/xy.chen/tw/isaaclab_ws/IsaacLab/isaaclab.sh")
CONDA = Path("/home/xy.chen/miniconda3/bin/conda")
CONDA_ENV = "env_isaaclab_v100"
PACE_DATA = ROOT / "pace_data"
REFERENCE = ROOT / "results/supplementary/direction_conditioned_v2_1/stage1/gpt-阶段一v2_B_ref冻结文件.json"
DEFAULT_AUTHORIZATION = ROOT / "results/supplementary/direction_conditioned_v2_2/gpt-v2.2_45模型holdout授权.json"
DEFAULT_OUTPUT_PARENT = ROOT / "results/supplementary/direction_conditioned_v2_2"
SESSION_PREFIX = "gpt_v2_2_video45_gpu"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON不是对象：{path}")
    return data


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="调度v2.2授权45模型录像。")
    parser.add_argument("--authorization", default=str(DEFAULT_AUTHORIZATION))
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--worker_gpu", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--plan", default=None, help=argparse.SUPPRESS)
    return parser


def _authorized_models(path: Path) -> list[dict[str, object]]:
    data = _load_json(path)
    models = data.get("模型", [])
    if data.get("模型数量") != 45 or not isinstance(models, list) or len(models) != 45:
        raise ValueError("录像只接受冻结的45模型授权。")
    if len({(item["地形"], item["方法"], int(item["PPO_seed"])) for item in models}) != 45:
        raise ValueError("授权模型组合不唯一。")
    return models


def _video_dir(output_root: Path, item: dict[str, object]) -> Path:
    return output_root / f"gpt_{item['地形']}_{item['方法']}_seed{item['PPO_seed']}"


def _is_complete(directory: Path, item: dict[str, object]) -> bool:
    reports = list(directory.glob("*回放说明.json")) if directory.is_dir() else []
    videos = list(directory.glob("*.mp4")) if directory.is_dir() else []
    if len(reports) != 1 or len(videos) != 1 or videos[0].stat().st_size <= 0:
        return False
    try:
        report = _load_json(reports[0])
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(videos[0])],
            text=True, capture_output=True, check=True,
        )
    except (OSError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError):
        return False
    return (
        report.get("检查点SHA256") == item.get("检查点SHA256")
        and int(report.get("录像帧数", -1)) == 200
        and report.get("固定录像选择规则") == "batch0/env2；与历史最终模型回放一致的中等难度实例，不按表现挑选"
        and float(probe.stdout.strip()) >= 19.9
    )


def _record_one(gpu: int, item: dict[str, object], output_root: Path, authorization: Path) -> None:
    destination = _video_dir(output_root, item)
    if _is_complete(destination, item):
        print(f"[45模型录像] GPU{gpu} 已存在合格视频，跳过：{destination.name}", flush=True)
        return
    if destination.exists():
        failed = destination.with_name(destination.name + "_失败保留_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        destination.rename(failed)
    command = [
        "timeout", "--signal=TERM", "--kill-after=60s", "20m",
        str(CONDA), "run", "-n", CONDA_ENV, "--no-capture-output",
        str(ISAACLAB), "-p", str(RECORDER),
        "--task", str(item["任务"]), "--checkpoint", str(item["检查点"]),
        "--ppo_seed", str(item["PPO_seed"]), "--batch_index", "0", "--env_index", "2",
        "--holdout_authorization", str(authorization), "--duration_s", "20",
        "--video_fps", "10", "--output_dir", str(destination), "--resolution", "960", "540",
        "--headless", "--device", "cuda:0",
    ]
    if item["方法"] == "eco":
        command[command.index("--holdout_authorization"):command.index("--holdout_authorization")] = ["--energy_reference_json", str(REFERENCE)]
    environment = {
        **dict(__import__("os").environ),
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PACE_ECO_DATA_ROOT": str(PACE_DATA),
        "PYTHONPATH": str(ROOT),
        "PYTHONUNBUFFERED": "1",
        "TERM": "xterm",
    }
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    if not _is_complete(destination, item):
        raise RuntimeError(f"录像进程成功退出但产物质检失败：{destination}")


def _worker(gpu: int, plan_path: Path, authorization: Path) -> None:
    plan = _load_json(plan_path)
    output_root = Path(str(plan["输出目录"]))
    queue = plan.get("GPU队列", {}).get(str(gpu), [])
    if not isinstance(queue, list):
        raise ValueError(f"GPU{gpu}队列格式错误。")
    for index, item in enumerate(queue, start=1):
        print(f"[45模型录像] GPU{gpu} 开始 {index}/{len(queue)}：{item['地形']}/{item['方法']}/seed{item['PPO_seed']}", flush=True)
        _record_one(gpu, item, output_root, authorization)
        print(f"[45模型录像] GPU{gpu} 完成 {index}/{len(queue)}。", flush=True)
    print(f"[45模型录像] GPU{gpu} 队列全部完成。", flush=True)


def _controller(gpus: list[int], models: list[dict[str, object]], authorization: Path, requested_output: str | None) -> None:
    if gpus != [0, 1]:
        raise ValueError("本次45模型录像冻结为GPU0和GPU1。")
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_root = Path(requested_output).expanduser().resolve() if requested_output else DEFAULT_OUTPUT_PARENT / f"gpt_45模型正式视频_{timestamp}"
    if output_root.exists():
        raise FileExistsError(f"拒绝覆盖录像输出目录：{output_root}")
    output_root.mkdir(parents=True)
    queues = {gpu: [] for gpu in gpus}
    for index, item in enumerate(models):
        queues[gpus[index % 2]].append(item)
    plan_path = output_root / "gpt-45模型录像GPU分配计划.json"
    plan_path.write_text(json.dumps({
        "状态": "已冻结并启动", "创建时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "授权": str(authorization), "授权SHA256": _sha256(authorization), "输出目录": str(output_root),
        "固定录像条件": {"评估批次": 0, "正式环境编号": 2, "难度": "中", "stairs_slope方向": "up", "时长_s": 20, "帧率": 10, "分辨率": [960, 540]},
        "分配规则": "授权顺序轮转GPU0/1；每卡同时只录制一个模型",
        "GPU队列": {str(gpu): queue for gpu, queue in queues.items()},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for gpu in gpus:
        session = f"{SESSION_PREFIX}{gpu}"
        subprocess.run(["tmux", "has-session", "-t", session], capture_output=True, check=False)
        log_path = output_root / f"gpt-GPU{gpu}_录像队列.log"
        worker = (
            f"set -o pipefail; cd {ROOT} && PYTHONUNBUFFERED=1 {sys.executable} {Path(__file__).resolve()} "
            f"--authorization {authorization} --worker_gpu {gpu} --plan {plan_path} 2>&1 | tee {log_path}"
        )
        result = subprocess.run(["tmux", "new-session", "-d", "-s", session, worker], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"启动tmux会话失败：{session}：{result.stderr}")
    print(output_root)


def main() -> None:
    args = _parser().parse_args()
    authorization = Path(args.authorization).expanduser().resolve()
    models = _authorized_models(authorization)
    if args.worker_gpu is not None:
        if args.plan is None:
            raise ValueError("worker模式必须提供计划文件。")
        _worker(args.worker_gpu, Path(args.plan).resolve(), authorization)
    else:
        _controller(args.gpus, models, authorization, args.output_root)


if __name__ == "__main__":
    main()
