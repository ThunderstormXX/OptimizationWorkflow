from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pytest

from experiments.base_training import ExperimentBaseTraining


@dataclass
class DummyTask:
    value: int = 1


@dataclass
class DummyModel:
    scale: float = 1.0


@dataclass
class DummyOptimizer:
    lr: float = 0.1


@dataclass
class DummyTrainer:
    def train(
        self,
        *,
        model: Any,
        optimizer: Any,
        task: Any,
        seed: int,
        run_dir: Path,
    ) -> tuple[list[dict[str, float]], dict[str, Any]]:
        history = [
            {"loss": float(task.value), "accuracy": 0.5},
            {"loss": float(task.value) / 2.0, "accuracy": 0.75},
        ]
        summary = {"seed": seed, "lr": optimizer.lr, "value": task.value}
        steps_path = run_dir / "steps.jsonl"
        steps_path.write_text(
            "\n".join(
                [
                    json.dumps({"iter": 1.0, "batch_loss": 1.0, "loss": 0.9}),
                    json.dumps({"iter": 2.0, "batch_loss": 0.8}),
                ]
            )
        )
        return history, summary


class RaisingTrainer(DummyTrainer):
    def train(self, *args, **kwargs):  # type: ignore[override]
        raise RuntimeError("should not be called")


def test_experiment_base_training_runs(tmp_path: Path) -> None:
    tasks = [
        {
            "name": "dummy_task",
            "task": {"class": "tests.experiments.test_base_training.DummyTask", "params": {"value": 4}},
            "models": [
                {"name": "dummy_model", "class": "tests.experiments.test_base_training.DummyModel", "params": {"scale": 2.0}}
            ],
            "optimizers": [
                {"name": "dummy_opt", "class": "tests.experiments.test_base_training.DummyOptimizer", "params": {"lr": 0.2}}
            ],
        }
    ]

    exp = ExperimentBaseTraining(
        tasks=tasks,
        trainer={"class": "tests.experiments.test_base_training.DummyTrainer", "params": {}},
        workflow_dir=str(tmp_path),
        num_runs=1,
        seed=0,
    )

    summary = exp.run()
    assert summary["summaries"]
    exp_dir = Path(summary["exp_dir"])
    assert (exp_dir / "plots" / "dummy_task" / "dummy_model" / "loss.png").exists()
    assert (exp_dir / "plots" / "dummy_task" / "dummy_model" / "steps" / "batch_loss.png").exists()


def test_experiment_task_level_trainer(tmp_path: Path) -> None:
    tasks = [
        {
            "name": "dummy_task",
            "task": {"class": "tests.experiments.test_base_training.DummyTask", "params": {"value": 3}},
            "models": [
                {"name": "dummy_model", "class": "tests.experiments.test_base_training.DummyModel", "params": {"scale": 1.0}}
            ],
            "optimizers": [
                {"name": "dummy_opt", "class": "tests.experiments.test_base_training.DummyOptimizer", "params": {"lr": 0.1}}
            ],
            "trainer": {"class": "tests.experiments.test_base_training.DummyTrainer", "params": {}},
        }
    ]

    exp = ExperimentBaseTraining(tasks=tasks, workflow_dir=str(tmp_path), num_runs=1, seed=0)
    summary = exp.run()
    assert summary["summaries"][0]["task"] == "dummy_task"


def test_experiment_task_overrides() -> None:
    exp = ExperimentBaseTraining(task_overrides={"value": 7, "nested": {"b": 2}})
    task_spec = {
        "class": "tests.experiments.test_base_training.DummyTask",
        "params": {"value": 2, "nested": {"b": 1}},
    }
    updated = exp._apply_task_overrides(task_spec)
    assert updated["params"]["value"] == 7
    assert updated["params"]["nested"]["b"] == 2


def test_experiment_load_tasks_requires_input() -> None:
    exp = ExperimentBaseTraining()
    with pytest.raises(ValueError):
        exp._load_tasks()


def test_experiment_aggregate_empty() -> None:
    exp = ExperimentBaseTraining(tasks=[])
    assert exp._aggregate([]) == {"metrics": {}}


torch = pytest.importorskip("torch")


def test_build_optimizer_with_torch() -> None:
    exp = ExperimentBaseTraining(tasks=[])
    model = torch.nn.Linear(2, 1)
    spec = {"class": "torch.optim.SGD", "params": {"lr": 0.1}}
    optimizer = exp._build_optimizer(spec, model)
    assert optimizer.__class__.__name__ == "SGD"


def test_build_optimizer_stochastic_fw_skip_constraint() -> None:
    from optim.legacy_frankwolfe import StochasticFrankWolfe

    exp = ExperimentBaseTraining(tasks=[])
    model = torch.nn.Linear(2, 1)
    spec = {
        "class": "optim.legacy_frankwolfe.StochasticFrankWolfe",
        "params": {
            "constraint": {"class": "optim.legacy_frankwolfe.L2BallConstraint", "params": {"radius": "auto"}},
            "step_size": {"type": "constant", "gamma": 0.5},
        },
    }
    optimizer = exp._build_optimizer(spec, model)
    assert isinstance(optimizer, StochasticFrankWolfe)


def test_load_tasks_from_dir(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.json").write_text("{\"class\": \"tests.experiments.test_base_training.DummyTask\", \"params\": {\"value\": 1}}")
    (task_dir / "models.json").write_text("[]")
    (task_dir / "optimizers.json").write_text("[]")
    exp = ExperimentBaseTraining(task_dirs=[str(task_dir)])
    tasks = exp._load_tasks()
    assert tasks[0]["name"] == "task"


def test_experiment_apply_model_init(tmp_path: Path) -> None:
    tasks = [
        {
            "name": "torch_task",
            "task": {"class": "tests.experiments.test_base_training.DummyTask", "params": {"value": 1}},
            "models": [
                {
                    "name": "linear",
                    "class": "torch.nn.Linear",
                    "params": {"in_features": 2, "out_features": 1},
                    "init": {"type": "normal", "mean": 0.0, "std": 0.1},
                }
            ],
            "optimizers": [
                {"name": "dummy_opt", "class": "tests.experiments.test_base_training.DummyOptimizer", "params": {"lr": 0.1}}
            ],
        }
    ]
    exp = ExperimentBaseTraining(
        tasks=tasks,
        trainer={"class": "tests.experiments.test_base_training.DummyTrainer", "params": {}},
        workflow_dir=str(tmp_path),
        num_runs=1,
        seed=0,
    )
    summary = exp.run()
    assert summary["summaries"][0]["model"] == "linear"


def test_seed_everything_importerror(monkeypatch: pytest.MonkeyPatch) -> None:
    exp = ExperimentBaseTraining(tasks=[])

    import builtins

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch":
            raise ImportError("no torch")
        return builtins.__import__(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    exp._seed_everything(0)


def test_experiment_skip_existing(tmp_path: Path) -> None:
    tasks = [
        {
            "name": "skip_task",
            "task": {"class": "tests.experiments.test_base_training.DummyTask", "params": {"value": 1}},
            "models": [
                {"name": "skip_model", "class": "tests.experiments.test_base_training.DummyModel", "params": {}}
            ],
            "optimizers": [
                {"name": "skip_opt", "class": "tests.experiments.test_base_training.DummyOptimizer", "params": {"lr": 0.1}}
            ],
        }
    ]

    exp_dir = tmp_path / "exp_0001" / "runs"
    exp_dir.mkdir(parents=True, exist_ok=True)
    run_id = "skip_task__skip_model__skip_opt__run00"
    run_dir = exp_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "history.jsonl").write_text("{\"loss\": 1.0}\n")
    (run_dir / "summary.json").write_text("{\"ok\": true}")

    exp = ExperimentBaseTraining(
        tasks=tasks,
        trainer={"class": "tests.experiments.test_base_training.RaisingTrainer", "params": {}},
        workflow_dir=str(tmp_path),
        exp_id=1,
        num_runs=1,
        seed=0,
        skip_existing=True,
    )

    summary = exp.run()
    assert summary["summaries"][0]["task"] == "skip_task"
