from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from benchmarks import runner


@dataclass
class DummyExperiment:
    workflow_dir: str

    def run(self) -> dict[str, Any]:
        exp_dir = Path(self.workflow_dir) / "exp_test"
        exp_dir.mkdir(parents=True, exist_ok=True)
        return {"workflow_dir": self.workflow_dir, "exp_dir": str(exp_dir)}


def test_runner_main_executes_config(tmp_path: Path) -> None:
    config = {
        "experiment": {
            "class": "tests.benchmarks.test_runner_config.DummyExperiment",
            "params": {"workflow_dir": str(tmp_path)},
        }
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json_dumps(config))

    status = runner.main(["--config", str(config_path)])
    assert status == 0
    assert (tmp_path / "exp_test" / "meta.json").exists()


def test_runner_missing_experiment_raises() -> None:
    with pytest.raises(ValueError):
        runner.run_from_config({})


class NoRunExperiment:
    def __init__(self) -> None:
        pass


def test_runner_requires_run_method() -> None:
    config = {"experiment": {"class": "tests.benchmarks.test_runner_config.NoRunExperiment", "params": {}}}
    with pytest.raises(TypeError):
        runner.run_from_config(config)


def test_runner_main_overrides(tmp_path: Path) -> None:
    config = {
        "experiment": {
            "class": "tests.benchmarks.test_runner_config.DummyExperiment",
            "params": {"workflow_dir": str(tmp_path)},
        }
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json_dumps(config))

    status = runner.main(
        [
            "--config",
            str(config_path),
            "--workflow-dir",
            str(tmp_path / "override"),
            "--set",
            "experiment.params.workflow_dir=\"override\"",
        ]
    )
    assert status == 0


def json_dumps(data: dict[str, Any]) -> str:
    import json

    return json.dumps(data)
