"""Workflow directory management for experiments.

This module provides utilities for creating and managing experiment
directories with consistent naming and structure.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

__all__ = [
    "next_experiment_dir",
    "write_run_files",
    "try_get_git_commit",
]


def next_experiment_dir(workflow_dir: Path) -> Path:
    """Create the next experiment directory with zero-padded naming.

    Creates directories:
    - workflow_dir/exp_XXXX/
    - workflow_dir/exp_XXXX/artifacts/

    Naming convention: exp_0000, exp_0001, exp_0002, ...
    Policy: next index after the maximum existing index.

    Args:
        workflow_dir: Parent directory for all experiments.

    Returns:
        Path to the newly created experiment directory.

    Example:
        >>> exp_dir = next_experiment_dir(Path("workflow"))
        >>> exp_dir
        PosixPath('workflow/exp_0000')
    """
    # Ensure workflow directory exists
    workflow_dir.mkdir(parents=True, exist_ok=True)

    # Find existing experiment directories
    pattern = re.compile(r"^exp_(\d{4})$")
    max_index = -1

    for entry in workflow_dir.iterdir():
        if entry.is_dir():
            match = pattern.match(entry.name)
            if match:
                index = int(match.group(1))
                max_index = max(max_index, index)

    # Next index
    next_index = max_index + 1
    exp_name = f"exp_{next_index:04d}"
    exp_dir = workflow_dir / exp_name

    # Create directories
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "artifacts").mkdir(exist_ok=True)

    return exp_dir


def write_run_files(
    exp_dir: Path,
    *,
    meta: dict[str, Any],
    config: dict[str, Any],
    readme_text: str,
) -> None:
    """Write experiment metadata files.

    Writes:
    - exp_dir/meta.json
    - exp_dir/config.json
    - exp_dir/README.md

    Args:
        exp_dir: Experiment directory path.
        meta: Metadata dictionary (timestamp, git commit, argv, etc.).
        config: Resolved configuration dictionary.
        readme_text: Human-readable README content.
    """
    # Write meta.json
    meta_path = exp_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    # Write config.json
    config_path = exp_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    # Write README.md
    readme_path = exp_dir / "README.md"
    with readme_path.open("w", encoding="utf-8") as f:
        f.write(readme_text)


def try_get_git_commit() -> str | None:
    """Attempt to get the current git commit hash.

    Returns:
        The git commit hash (short form), or None if git is unavailable
        or not in a git repository.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None
