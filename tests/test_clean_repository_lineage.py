from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY_SNAPSHOT = "82be138"
FORBIDDEN_TOKENS = ("bru" + "ce", "big" + "ai", "eco" + "-humanoid")


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def test_tracked_tree_contains_no_legacy_robot_paths_or_references():
    tracked = [Path(path) for path in _git("ls-files").stdout.splitlines()]
    lowered_paths = [path.as_posix().lower() for path in tracked]
    assert not any(token in path for path in lowered_paths for token in FORBIDDEN_TOKENS)

    offenders: list[str] = []
    for relative in tracked:
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        try:
            content = path.read_text(encoding="utf-8").lower()
        except UnicodeDecodeError:
            continue
        if any(token in content for token in FORBIDDEN_TOKENS):
            offenders.append(relative.as_posix())
    assert offenders == []


def test_history_does_not_descend_from_legacy_snapshot():
    ancestors = set(_git("rev-list", "HEAD").stdout.splitlines())
    resolved = _git("rev-parse", "--verify", f"{LEGACY_SNAPSHOT}^{{commit}}", check=False)
    if resolved.returncode != 0:
        return
    legacy = resolved.stdout.strip()
    assert legacy not in ancestors
