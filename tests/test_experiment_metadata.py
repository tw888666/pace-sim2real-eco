from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "pace" / "create_experiment.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("create_experiment", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _clean_git_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "PACE Test")
    _git(repo, "config", "user.email", "pace-test@example.invalid")
    (repo / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "test snapshot")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_create_experiment_records_git_config_seed_and_gpu(tmp_path: Path) -> None:
    module = _load_module()
    repo, commit = _clean_git_repo(tmp_path)
    output_root = tmp_path / "experiment-data"
    output_root.mkdir()
    config = tmp_path / "frozen.json"
    config.write_text(json.dumps({"learning_rate": 0.001}), encoding="utf-8")

    args = module.build_parser().parse_args(
        [
            "--name",
            "flat-v2-seed3",
            "--repo",
            str(repo),
            "--output-root",
            str(output_root),
            "--config",
            str(config),
            "--algorithm",
            "PPO-Lagrangian",
            "--train-seed",
            "3",
            "--terrain-seed",
            "103",
            "--terrain",
            "flat",
            "--gpu",
            "0",
            "--iterations",
            "3000",
        ]
    )
    fake_gpu = module.GpuSnapshot(
        accelerator="gpu",
        physical_index=0,
        name="NVIDIA A100 80GB PCIe",
        memory_total_mib=81920,
        memory_used_mib=123,
        utilization_percent=4,
        uuid="GPU-test",
        driver_version="test-driver",
    )

    destination = module.create_experiment(args, gpu_query=lambda index: fake_gpu)

    assert destination == output_root / "flat-v2-seed3"
    assert (destination / "config.yaml").read_bytes() == config.read_bytes()
    assert (destination / "commit.txt").read_text(encoding="utf-8") == commit + "\n"
    assert (destination / "git-status.txt").read_text(encoding="utf-8") == "clean\n"
    for directory_name in ("checkpoints", "evaluation", "logs", "videos", "figures"):
        assert (destination / directory_name).is_dir()

    metadata = (destination / "metadata.yaml").read_text(encoding="utf-8")
    assert f'  commit: "{commit}"' in metadata
    assert '  name: "PPO-Lagrangian"' in metadata
    assert "  training: 3" in metadata
    assert "  terrain: 103" in metadata
    assert "  physical_gpu_index: 0" in metadata
    assert '  gpu_name: "NVIDIA A100 80GB PCIe"' in metadata
    assert "  iterations: 3000" in metadata

    readme = (destination / "README.md").read_text(encoding="utf-8")
    assert "物理 GPU 0：NVIDIA A100 80GB PCIe" in readme
    assert "CUDA_VISIBLE_DEVICES=0" in readme


def test_existing_destination_and_dirty_repo_are_rejected(tmp_path: Path) -> None:
    module = _load_module()
    repo, _ = _clean_git_repo(tmp_path)
    output_root = tmp_path / "experiment-data"
    output_root.mkdir()
    config = tmp_path / "frozen.yaml"
    config.write_text("learning_rate: 0.001\n", encoding="utf-8")
    args = module.build_parser().parse_args(
        [
            "--name",
            "existing",
            "--repo",
            str(repo),
            "--output-root",
            str(output_root),
            "--config",
            str(config),
            "--algorithm",
            "PPO",
            "--train-seed",
            "1",
            "--terrain",
            "flat",
            "--gpu",
            "cpu",
            "--iterations",
            "10",
        ]
    )

    (repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    try:
        module.create_experiment(args)
    except ValueError as error:
        assert "未提交修改" in str(error)
    else:
        raise AssertionError("dirty repository should be rejected")

    (repo / "dirty.txt").unlink()
    (output_root / "existing").mkdir()
    try:
        module.create_experiment(args)
    except FileExistsError as error:
        assert "拒绝覆盖" in str(error)
    else:
        raise AssertionError("existing destination should be rejected")
