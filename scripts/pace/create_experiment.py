#!/usr/bin/env python3
"""Create a reproducible PACE-ECO experiment directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence


@dataclass(frozen=True)
class GitSnapshot:
    repo_name: str
    repo_path: Path
    branch: str
    commit: str
    dirty: bool
    status: str


@dataclass(frozen=True)
class GpuSnapshot:
    accelerator: str
    physical_index: int | None
    name: str | None
    memory_total_mib: int | None
    memory_used_mib: int | None
    utilization_percent: int | None
    uuid: str | None
    driver_version: str | None


def _run(command: Sequence[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "无错误详情"
        raise RuntimeError(f"命令失败：{' '.join(command)}\n{detail}")
    return completed.stdout.strip()


def capture_git_snapshot(repo_path: Path, *, allow_dirty: bool) -> GitSnapshot:
    repo_path = repo_path.expanduser().resolve()
    if not repo_path.is_dir():
        raise ValueError(f"Git 仓库目录不存在：{repo_path}")

    top_level = Path(_run(["git", "rev-parse", "--show-toplevel"], cwd=repo_path)).resolve()
    if top_level != repo_path:
        raise ValueError(f"--repo 必须指向工作树根目录：{top_level}")

    commit = _run(["git", "rev-parse", "HEAD"], cwd=repo_path)
    branch_result = subprocess.run(
        ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
        cwd=repo_path,
        check=False,
        capture_output=True,
        text=True,
    )
    branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "DETACHED"
    status = _run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo_path,
    )
    dirty = bool(status)
    if dirty and not allow_dirty:
        raise ValueError(
            "Git 工作树存在未提交修改；正式实验默认拒绝创建。"
            "如确需开发实验，请显式添加 --allow-dirty。"
        )

    return GitSnapshot(
        repo_name=repo_path.name,
        repo_path=repo_path,
        branch=branch,
        commit=commit,
        dirty=dirty,
        status=status,
    )


def query_gpu(physical_index: int) -> GpuSnapshot:
    output = _run(
        [
            "nvidia-smi",
            "-i",
            str(physical_index),
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,uuid,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = list(csv.reader(output.splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 7:
        raise RuntimeError(f"无法解析 GPU {physical_index} 的 nvidia-smi 输出：{output}")

    fields = [field.strip() for field in rows[0]]
    returned_index = int(fields[0])
    if returned_index != physical_index:
        raise RuntimeError(
            f"nvidia-smi 返回物理 GPU {returned_index}，与请求的 {physical_index} 不一致"
        )
    return GpuSnapshot(
        accelerator="gpu",
        physical_index=returned_index,
        name=fields[1],
        memory_total_mib=int(fields[2]),
        memory_used_mib=int(fields[3]),
        utilization_percent=int(fields[4]),
        uuid=fields[5],
        driver_version=fields[6],
    )


def cpu_snapshot() -> GpuSnapshot:
    return GpuSnapshot(
        accelerator="cpu",
        physical_index=None,
        name=None,
        memory_total_mib=None,
        memory_used_mib=None,
        utilization_percent=None,
        uuid=None,
        driver_version=None,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _yaml_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def _validate_experiment_name(name: str) -> None:
    if not name or name in {".", ".."}:
        raise ValueError("实验名称不能为空或使用点目录")
    if Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("实验名称必须是单个目录名，不能包含路径分隔符")
    if name.startswith("."):
        raise ValueError("实验名称不能以点开头")


def render_metadata(
    args: argparse.Namespace,
    *,
    destination: Path,
    created_at: str,
    git_snapshot: GitSnapshot,
    gpu_snapshot: GpuSnapshot,
    config_source: Path,
    config_sha256: str,
) -> str:
    values = {
        "name": args.name,
        "created_at": created_at,
        "directory": str(destination),
        "repo_name": git_snapshot.repo_name,
        "repo_path": str(git_snapshot.repo_path),
        "branch": git_snapshot.branch,
        "commit": git_snapshot.commit,
        "dirty": git_snapshot.dirty,
        "config_source": str(config_source),
        "config_sha256": config_sha256,
        "algorithm": args.algorithm,
        "train_seed": args.train_seed,
        "terrain_seed": args.terrain_seed,
        "terrain": args.terrain,
        "accelerator": gpu_snapshot.accelerator,
        "gpu_index": gpu_snapshot.physical_index,
        "gpu_name": gpu_snapshot.name,
        "gpu_memory_total": gpu_snapshot.memory_total_mib,
        "gpu_memory_used": gpu_snapshot.memory_used_mib,
        "gpu_utilization": gpu_snapshot.utilization_percent,
        "gpu_uuid": gpu_snapshot.uuid,
        "driver_version": gpu_snapshot.driver_version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "iterations": args.iterations,
        "budget_j": args.budget_j,
        "training_command": args.training_command,
        "notes": args.notes,
    }
    q = _yaml_scalar
    return f"""schema_version: 1
experiment:
  name: {q(values['name'])}
  created_at: {q(values['created_at'])}
  directory: {q(values['directory'])}
code:
  repo: {q(values['repo_name'])}
  repo_path: {q(values['repo_path'])}
  branch: {q(values['branch'])}
  commit: {q(values['commit'])}
  dirty: {q(values['dirty'])}
  status_snapshot: "git-status.txt"
config:
  source: {q(values['config_source'])}
  snapshot: "config.yaml"
  sha256: {q(values['config_sha256'])}
algorithm:
  name: {q(values['algorithm'])}
seed:
  training: {q(values['train_seed'])}
  terrain: {q(values['terrain_seed'])}
environment:
  terrain: {q(values['terrain'])}
hardware:
  accelerator: {q(values['accelerator'])}
  hostname: {q(values['hostname'])}
  physical_gpu_index: {q(values['gpu_index'])}
  gpu_name: {q(values['gpu_name'])}
  gpu_memory_total_mib: {q(values['gpu_memory_total'])}
  gpu_memory_used_mib_at_creation: {q(values['gpu_memory_used'])}
  gpu_utilization_percent_at_creation: {q(values['gpu_utilization'])}
  gpu_uuid: {q(values['gpu_uuid'])}
  driver_version: {q(values['driver_version'])}
  cuda_visible_devices_at_creation: {q(values['cuda_visible_devices'])}
  expected_cuda_visible_devices: {q(values['gpu_index'])}
  process_local_cuda_index: {q(0 if values['gpu_index'] is not None else None)}
runtime:
  python_version: {q(values['python'])}
training:
  iterations: {q(values['iterations'])}
  budget_j: {q(values['budget_j'])}
  command: {q(values['training_command'])}
  notes: {q(values['notes'])}
outputs:
  checkpoints: "checkpoints"
  evaluation: "evaluation"
  logs: "logs"
  videos: "videos"
  figures: "figures"
"""


def render_readme(
    args: argparse.Namespace,
    *,
    created_at: str,
    git_snapshot: GitSnapshot,
    gpu_snapshot: GpuSnapshot,
    config_sha256: str,
) -> str:
    if gpu_snapshot.physical_index is None:
        hardware = "CPU"
        launch_prefix = "不适用"
    else:
        hardware = (
            f"物理 GPU {gpu_snapshot.physical_index}：{gpu_snapshot.name} "
            f"（{gpu_snapshot.memory_total_mib} MiB）"
        )
        launch_prefix = f"CUDA_VISIBLE_DEVICES={gpu_snapshot.physical_index}"
    terrain_seed = "未设置" if args.terrain_seed is None else str(args.terrain_seed)
    budget = "未设置" if args.budget_j is None else f"{args.budget_j} J"
    command = args.training_command or "尚未记录"
    notes = args.notes or "无"
    return f"""# {args.name}

该目录由 `scripts/pace/create_experiment.py` 自动创建。`metadata.yaml` 是机器可读的完整记录，`config.yaml` 是创建时配置快照。

## 实验信息

- 创建时间：{created_at}
- 算法：{args.algorithm}
- 地形：{args.terrain}
- 训练 seed：{args.train_seed}
- 地形 seed：{terrain_seed}
- 训练迭代数：{args.iterations}
- 约束预算：{budget}

## 代码

- 仓库：{git_snapshot.repo_name}
- 分支：`{git_snapshot.branch}`
- commit：`{git_snapshot.commit}`
- 创建时工作树：{"有未提交修改" if git_snapshot.dirty else "干净"}

## 配置

- 快照：`config.yaml`
- SHA-256：`{config_sha256}`

## 硬件

- {hardware}
- 训练命令应使用的显卡前缀：`{launch_prefix}`
- 创建时 GPU 已用显存和利用率记录在 `metadata.yaml`，它们只代表创建瞬间，不代替正式启动前的占用与错误检查。

## 运行

- 训练命令：`{command}`
- 备注：{notes}

## 输出目录

- `checkpoints/`：模型检查点；
- `evaluation/`：评估结果；
- `logs/`：训练与启动日志；
- `videos/`：视频；
- `figures/`：图表。
"""


def create_experiment(
    args: argparse.Namespace,
    *,
    gpu_query: Callable[[int], GpuSnapshot] = query_gpu,
) -> Path:
    _validate_experiment_name(args.name)
    output_root = Path(args.output_root).expanduser().resolve()
    if not output_root.is_dir():
        raise ValueError(f"实验资产根目录不存在：{output_root}")

    destination = output_root / args.name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"实验目录已存在，拒绝覆盖：{destination}")

    config_source = Path(args.config).expanduser().resolve()
    if not config_source.is_file():
        raise ValueError(f"配置文件不存在：{config_source}")
    if config_source.suffix.lower() not in {".yaml", ".yml", ".json"}:
        raise ValueError("配置快照只接受 .yaml、.yml 或 .json 文件")

    git_snapshot = capture_git_snapshot(Path(args.repo), allow_dirty=args.allow_dirty)
    if args.gpu == "cpu":
        gpu_snapshot = cpu_snapshot()
    else:
        try:
            physical_index = int(args.gpu)
        except ValueError as error:
            raise ValueError("--gpu 必须是非负物理 GPU 编号或 cpu") from error
        if physical_index < 0:
            raise ValueError("--gpu 必须是非负物理 GPU 编号或 cpu")
        gpu_snapshot = gpu_query(physical_index)

    config_sha256 = sha256_file(config_source)
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    staging = Path(tempfile.mkdtemp(prefix=f".{args.name}.creating-", dir=output_root))
    try:
        for directory_name in ("checkpoints", "evaluation", "logs", "videos", "figures"):
            (staging / directory_name).mkdir()
        shutil.copy2(config_source, staging / "config.yaml")
        (staging / "commit.txt").write_text(git_snapshot.commit + "\n", encoding="utf-8")
        status_text = git_snapshot.status + "\n" if git_snapshot.status else "clean\n"
        (staging / "git-status.txt").write_text(status_text, encoding="utf-8")
        (staging / "metadata.yaml").write_text(
            render_metadata(
                args,
                destination=destination,
                created_at=created_at,
                git_snapshot=git_snapshot,
                gpu_snapshot=gpu_snapshot,
                config_source=config_source,
                config_sha256=config_sha256,
            ),
            encoding="utf-8",
        )
        (staging / "README.md").write_text(
            render_readme(
                args,
                created_at=created_at,
                git_snapshot=git_snapshot,
                gpu_snapshot=gpu_snapshot,
                config_sha256=config_sha256,
            ),
            encoding="utf-8",
        )
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def build_parser() -> argparse.ArgumentParser:
    script_repo = Path(__file__).resolve().parents[2]
    default_output_root = script_repo.parent / "experiment-data" / "PACE-ECO"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="唯一实验目录名")
    parser.add_argument("--repo", default=str(script_repo), help="Git 工作树根目录")
    parser.add_argument(
        "--output-root",
        default=os.environ.get("PACE_ECO_EXPERIMENT_ROOT", str(default_output_root)),
        help="实验资产根目录",
    )
    parser.add_argument("--config", required=True, help="需要冻结的 YAML 或 JSON 配置")
    parser.add_argument("--algorithm", required=True, help="算法名称")
    parser.add_argument("--train-seed", required=True, type=int, help="训练 seed")
    parser.add_argument("--terrain-seed", type=int, help="地形 seed；平地可省略")
    parser.add_argument("--terrain", required=True, help="地形名称")
    parser.add_argument("--gpu", required=True, help="物理 GPU 编号，或 cpu")
    parser.add_argument("--iterations", required=True, type=int, help="训练迭代数")
    parser.add_argument("--budget-j", type=float, help="可选的能耗预算，单位 J")
    parser.add_argument("--training-command", help="可选的完整训练命令记录")
    parser.add_argument("--notes", help="可选备注")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="允许从有未提交修改的工作树创建开发实验",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.iterations <= 0:
        parser.error("--iterations 必须大于 0")
    try:
        destination = create_experiment(args)
    except (FileExistsError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(f"实验目录已创建：{destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
