"""PACE-ECO 运行的配置指纹和复现记录。"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from hashlib import sha256
from pathlib import Path
from typing import Any

from pace_eco_lab.constants import (
    PACE_FITTING_PATH,
    PACE_FITTING_SHA256,
    PACE_REPLAY_PATH,
    PACE_REPLAY_SHA256,
    PROJECT_ROOT,
)
from pace_eco_lab.mdp.parameters import file_sha256

_FINGERPRINT_FILE = "gpt_配置指纹.json"
_REPRODUCIBILITY_FILE = "gpt_复现信息.json"
_ENV_CONFIG_FILE = "gpt_环境配置.json"
_AGENT_CONFIG_FILE = "gpt_算法配置.json"
_GIT_STATUS_FILE = "gpt_Git状态.txt"


def _plain(value: Any) -> Any:
    """把配置对象转换为稳定、可 JSON 序列化的结构。"""

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        module = getattr(value, "__module__", type(value).__module__)
        name = getattr(value, "__qualname__", type(value).__qualname__)
        return f"{module}:{name}"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _fingerprint_agent_config(agent_cfg: Any) -> dict[str, Any]:
    config = _plain(agent_cfg)
    for transient_key in ("max_iterations", "run_name", "resume", "load_run", "load_checkpoint"):
        config.pop(transient_key, None)
    return config


def implementation_source_hashes() -> dict[str, str]:
    """返回会影响环境或训练语义的本地 Python 源码哈希。"""

    package_root = PROJECT_ROOT / "pace_eco_lab"
    files = sorted(path for path in package_root.rglob("*.py") if "__pycache__" not in path.parts)
    for relative in (
        "scripts/pace_eco/train.py",
        "scripts/pace_eco/multi_terrain_eval.py",
        "scripts/pace_eco/freeze_multi_terrain_budgets.py",
        "scripts/pace_eco/freeze_multi_terrain_holdout.py",
        "scripts/pace_eco/汇总多地形结果.py",
        "scripts/pace_eco/direction_conditioned_eval.py",
        "scripts/pace_eco/freeze_direction_conditioned_budgets.py",
        "scripts/pace_eco/freeze_direction_conditioned_holdout.py",
    ):
        path = PROJECT_ROOT / relative
        if path.is_file():
            files.append(path)
    return {str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in files}


def configuration_payload(task: str, seed: int, env_cfg: Any, agent_cfg: Any) -> dict[str, Any]:
    """生成恢复训练时必须保持不变的配置载荷。"""

    return {
        "task": task,
        "seed": int(seed),
        "environment": _plain(env_cfg),
        "agent": _fingerprint_agent_config(agent_cfg),
        "implementation_source_hashes": implementation_source_hashes(),
    }


def configuration_fingerprint(task: str, seed: int, env_cfg: Any, agent_cfg: Any) -> tuple[str, dict[str, Any]]:
    payload = configuration_payload(task, seed, env_cfg, agent_cfg)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest(), payload


def _run_git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else f"不可用：{completed.stderr.strip()}"


def dependency_versions() -> dict[str, str]:
    distributions = {
        "torch": "torch",
        "isaaclab": "isaaclab",
        "rsl-rl-lib": "rsl-rl-lib",
        "isaacsim": "isaacsim",
    }
    versions: dict[str, str] = {}
    for label, distribution in distributions.items():
        try:
            versions[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[label] = "未安装"
    return versions


def verify_dependency_versions() -> dict[str, str]:
    """拒绝在未经本计划验证的核心依赖版本上运行。"""

    actual = dependency_versions()
    expected_prefixes = {
        "torch": "2.7.0",
        "isaaclab": "0.54.4",
        "rsl-rl-lib": "5.0.1",
        "isaacsim": "5.1.",
    }
    mismatches = {
        name: (expected, actual[name])
        for name, expected in expected_prefixes.items()
        if not actual[name].startswith(expected)
    }
    if mismatches:
        raise RuntimeError(f"核心依赖版本不符合固定基线：{mismatches}")
    return actual


def verify_fixed_inputs() -> dict[str, str]:
    actual = {
        "fitting.npy": file_sha256(PACE_FITTING_PATH),
        "data.npy": file_sha256(PACE_REPLAY_PATH),
    }
    expected = {
        "fitting.npy": PACE_FITTING_SHA256,
        "data.npy": PACE_REPLAY_SHA256,
    }
    for name, digest in actual.items():
        if digest != expected[name]:
            raise ValueError(f"{name} 哈希不匹配：期望 {expected[name]}，实际 {digest}。")
    return actual


def write_run_records(
    run_dir: str | Path,
    *,
    task: str,
    seed: int,
    run_name: str,
    env_cfg: Any,
    agent_cfg: Any,
) -> str:
    """写出完整配置、复现信息和用于恢复校验的指纹。"""

    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint, payload = configuration_fingerprint(task, seed, env_cfg, agent_cfg)
    fixed_hashes = verify_fixed_inputs()
    git_status = _run_git("status", "--short")
    record = {
        "task": task,
        "seed": int(seed),
        "run_name": run_name,
        "configuration_fingerprint": fingerprint,
        "python": platform.python_version(),
        "dependencies": verify_dependency_versions(),
        "data_hashes": fixed_hashes,
        "implementation_source_hashes": implementation_source_hashes(),
        "git_commit": _run_git("rev-parse", "HEAD"),
        "git_worktree_clean": not bool(git_status),
    }
    (output / _ENV_CONFIG_FILE).write_text(
        json.dumps(_plain(env_cfg), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / _AGENT_CONFIG_FILE).write_text(
        json.dumps(_plain(agent_cfg), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / _FINGERPRINT_FILE).write_text(
        json.dumps({"sha256": fingerprint, "payload": payload}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / _REPRODUCIBILITY_FILE).write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / _GIT_STATUS_FILE).write_text((git_status or "工作树干净") + "\n", encoding="utf-8")
    return fingerprint


def validate_resume_records(
    run_dir: str | Path,
    *,
    task: str,
    seed: int,
    run_name: str,
    env_cfg: Any,
    agent_cfg: Any,
) -> str:
    """拒绝从不同环境、超参数或种子的运行中恢复。"""

    path = Path(run_dir) / _FINGERPRINT_FILE
    if not path.is_file():
        raise FileNotFoundError(f"恢复目录缺少 {path.name}，不能证明是同一次运行。")
    saved = json.loads(path.read_text(encoding="utf-8"))
    reproducibility_path = Path(run_dir) / _REPRODUCIBILITY_FILE
    if not reproducibility_path.is_file():
        raise FileNotFoundError(f"恢复目录缺少 {reproducibility_path.name}。")
    record = json.loads(reproducibility_path.read_text(encoding="utf-8"))
    if record.get("run_name") != run_name:
        raise ValueError(f"恢复运行名不一致：原运行 {record.get('run_name')}，当前 {run_name}。")
    actual, _ = configuration_fingerprint(task, seed, env_cfg, agent_cfg)
    if saved.get("sha256") != actual:
        raise ValueError(
            "恢复配置指纹不一致；环境、算法超参数、预算、环境数或随机种子发生了变化，"
            "必须从随机初始化开始。"
        )
    verify_fixed_inputs()
    return actual


def validate_evaluation_checkpoint(
    run_dir: str | Path,
    *,
    task: str,
    energy_budget_j: float | None,
    require_current_implementation: bool = False,
) -> dict[str, Any]:
    """校验评估检查点所属运行，不要求评估环境数等于训练环境数。"""

    output = Path(run_dir)
    reproducibility_path = output / _REPRODUCIBILITY_FILE
    agent_path = output / _AGENT_CONFIG_FILE
    fingerprint_path = output / _FINGERPRINT_FILE
    for path in (reproducibility_path, agent_path, fingerprint_path):
        if not path.is_file():
            raise FileNotFoundError(f"检查点目录缺少 {path.name}。")
    record = json.loads(reproducibility_path.read_text(encoding="utf-8"))
    agent = json.loads(agent_path.read_text(encoding="utf-8"))
    if record.get("task") != task:
        raise ValueError(f"检查点任务为 {record.get('task')}，评估请求为 {task}。")
    if "smoke" in str(record.get("run_name", "")).lower():
        raise ValueError("smoke 权重只验证训练链路，禁止用于策略评估或论文结果。")
    saved_budget = agent.get("algorithm", {}).get("energy_budget_j")
    if saved_budget is not None:
        if energy_budget_j is None or abs(float(saved_budget) - float(energy_budget_j)) > 1.0e-9:
            raise ValueError(f"训练预算为 {saved_budget} J，评估预算必须完全一致。")
    if require_current_implementation:
        recorded_hashes = record.get("implementation_source_hashes")
        current_hashes = implementation_source_hashes()
        if recorded_hashes != current_hashes:
            raise ValueError("评估实现源码与训练记录不一致；正式多地形评估禁止跨实现加载。")
    verify_fixed_inputs()
    return record


__all__ = [
    "configuration_fingerprint",
    "dependency_versions",
    "validate_evaluation_checkpoint",
    "validate_resume_records",
    "verify_dependency_versions",
    "verify_fixed_inputs",
    "write_run_records",
]
