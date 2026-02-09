"""Experiment for supervised training across tasks/models/optimizers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.workflow import next_experiment_dir
from experiments.config import import_class, load_task_dir, resolve_spec, resolve_values
from experiments.plotting import plot_metric_by_optimizer
from experiments.utils import apply_model_init, is_torch_optimizer_class
from trainers.supervised import SupervisedTrainer


def _safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")

def _log(msg: str) -> None:
    print(f"[Experiment] {msg}", flush=True)


@dataclass
class ExperimentBaseTraining:
    """Run supervised training for multiple optimizers and models."""

    task_dirs: list[str] | None = None
    tasks: list[dict[str, Any]] | None = None
    trainer: dict[str, Any] | None = None
    task_overrides: dict[str, Any] | None = None
    workflow_dir: str = "workflow"
    num_runs: int = 1
    seed: int = 0
    exp_id: int | None = None
    skip_existing: bool = True

    def _load_tasks(self) -> list[dict[str, Any]]:
        if self.tasks:
            return self.tasks
        if not self.task_dirs:
            raise ValueError("Either tasks or task_dirs must be provided")
        return [load_task_dir(Path(d)) for d in self.task_dirs]

    def run(self) -> dict[str, Any]:
        workflow_dir = Path(self.workflow_dir)
        if self.exp_id is None:
            exp_dir = next_experiment_dir(workflow_dir)
        else:
            exp_dir = workflow_dir / f"exp_{int(self.exp_id):04d}"
            exp_dir.mkdir(parents=True, exist_ok=True)
            (exp_dir / "artifacts").mkdir(exist_ok=True)
        exp_dir.mkdir(parents=True, exist_ok=True)
        runs_dir = exp_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        _log(f"exp_dir={exp_dir}")

        trainer_obj = resolve_spec(self.trainer) if self.trainer else SupervisedTrainer()

        summaries: list[dict[str, Any]] = []

        for task_bundle in self._load_tasks():
            task_name = task_bundle.get("name") or "task"
            _log(f"task={task_name} (initializing)")
            task_spec = self._apply_task_overrides(task_bundle["task"])
            models_spec = task_bundle["models"]
            optimizers_spec = task_bundle["optimizers"]
            task_trainer = trainer_obj
            if task_bundle.get("trainer") is not None:
                task_trainer = resolve_spec(task_bundle["trainer"])

            task_obj = resolve_spec(task_spec)
            task_split = getattr(task_obj, "split_report", None)
            if task_split:
                _log(f"task={task_name} split={task_split}")

            for model_spec in models_spec:
                model_name = model_spec.get("name") or model_spec.get("class", "model")
                model_params = model_spec.get("params", {})
                init_spec = model_spec.get("init")

                _log(f"model={model_name}")
                opt_aggregates: dict[str, dict[str, Any]] = {}
                opt_step_aggregates: dict[str, dict[str, Any]] = {}

                for opt_spec in optimizers_spec:
                    opt_name = opt_spec.get("name") or opt_spec.get("class", "optimizer")
                    _log(f"optimizer={opt_name}")
                    histories: list[list[dict[str, float]]] = []
                    run_summaries: list[dict[str, Any]] = []
                    step_histories: list[list[dict[str, float]]] = []

                    for run_idx in range(self.num_runs):
                        _log(f"run {run_idx + 1}/{self.num_runs} -> {task_name}/{model_name}/{opt_name}")
                        run_id = f"{_safe_name(task_name)}__{_safe_name(model_name)}__{_safe_name(opt_name)}__run{run_idx:02d}"
                        run_dir = runs_dir / run_id
                        run_dir.mkdir(parents=True, exist_ok=True)

                        if self.skip_existing:
                            history_path = run_dir / "history.jsonl"
                            summary_path = run_dir / "summary.json"
                            if history_path.exists() and summary_path.exists():
                                _log(f"skipping existing run: {run_id}")
                                histories.append(self._read_history(history_path))
                                run_summaries.append(self._read_summary(summary_path))
                                step_histories.append(self._read_steps(run_dir / "steps.jsonl"))
                                continue

                        self._seed_everything(self.seed + run_idx)
                        model_obj = resolve_spec(
                            {"class": model_spec["class"], "params": model_params}
                        )
                        if init_spec:
                            apply_model_init(model_obj, init_spec, seed=self.seed + run_idx)

                        optimizer_obj = self._build_optimizer(opt_spec, model_obj)
                        init_fn = getattr(optimizer_obj, "init", None)
                        if callable(init_fn):
                            init_fn()

                        history, summary = task_trainer.train(
                            model=model_obj,
                            optimizer=optimizer_obj,
                            task=task_obj,
                            seed=self.seed + run_idx,
                            run_dir=run_dir,
                        )
                        histories.append(history)
                        run_summaries.append(summary)
                        step_histories.append(self._read_steps(run_dir / "steps.jsonl"))

                    aggregate = self._aggregate(histories)
                    opt_aggregates[opt_name] = aggregate
                    if any(step_histories):
                        opt_step_aggregates[opt_name] = self._aggregate(step_histories)

                    summaries.append(
                        {
                            "task": task_name,
                            "model": model_name,
                            "optimizer": opt_name,
                            "task_split": task_split,
                            "aggregate": aggregate,
                            "runs": run_summaries,
                        }
                    )

                plots_dir = exp_dir / "plots" / _safe_name(task_name) / _safe_name(model_name)
                plots_dir.mkdir(parents=True, exist_ok=True)
                metrics = sorted(
                    {
                        metric
                        for aggregate in opt_aggregates.values()
                        for metric in aggregate.get("metrics", {}).keys()
                    }
                )
                if metrics:
                    _log(f"plotting metrics for {task_name}/{model_name}: {', '.join(metrics)}")
                for metric in metrics:
                    plot_metric_by_optimizer(
                        opt_aggregates,
                        metric,
                        plots_dir / f"{metric}.png",
                        title=f"{task_name} / {model_name} / {metric}",
                    )

                if opt_step_aggregates:
                    step_metrics = sorted(
                        {
                            metric
                            for aggregate in opt_step_aggregates.values()
                            for metric in aggregate.get("metrics", {}).keys()
                        }
                    )
                    if step_metrics:
                        _log(
                            f"plotting step metrics for {task_name}/{model_name}: {', '.join(step_metrics)}"
                        )
                    step_dir = plots_dir / "steps"
                    step_dir.mkdir(parents=True, exist_ok=True)
                    for metric in step_metrics:
                        is_full_loss = metric == "loss"
                        is_batch_loss = metric == "batch_loss"
                        plot_metric_by_optimizer(
                            opt_step_aggregates,
                            metric,
                            step_dir / f"{metric}.png",
                            title=f"{task_name} / {model_name} / step {metric}",
                            upper_only=is_full_loss,
                            loglog=is_full_loss or is_batch_loss,
                        )

        summary = {
            "exp_dir": str(exp_dir),
            "workflow_dir": str(self.workflow_dir),
            "num_runs": self.num_runs,
            "summaries": summaries,
        }
        return summary

    def _build_optimizer(self, spec: dict[str, Any], model: Any) -> Any:
        opt_class = import_class(spec["class"])
        skip_keys = set()
        if opt_class.__name__ in {
            "StochasticFrankWolfe",
            "StochasticFrankWolfeMomentumPre",
            "StochasticFrankWolfeMomentumPost",
            "OrthogonalSGDM",
        }:
            skip_keys.add("constraint")
        params = resolve_values(spec.get("params", {}), skip_keys=skip_keys)

        if is_torch_optimizer_class(opt_class):
            return opt_class(model.parameters(), **params)  # type: ignore[arg-type]
        return opt_class(**params)

    @staticmethod
    def _seed_everything(seed: int) -> None:
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
        except ImportError:
            pass

    def _apply_task_overrides(self, task_spec: dict[str, Any]) -> dict[str, Any]:
        if not self.task_overrides:
            return task_spec
        updated = dict(task_spec)
        params = dict(updated.get("params", {}))
        self._merge_dict(params, self.task_overrides)
        updated["params"] = params
        return updated

    @staticmethod
    def _merge_dict(target: dict[str, Any], updates: dict[str, Any]) -> None:
        for key, value in updates.items():
            if (
                key in target
                and isinstance(target[key], dict)
                and isinstance(value, dict)
            ):
                ExperimentBaseTraining._merge_dict(target[key], value)
            else:
                target[key] = value

    def _aggregate(self, histories: list[list[dict[str, float]]]) -> dict[str, Any]:
        if not histories:
            return {"metrics": {}}

        metric_keys = sorted(
            {
                k
                for h in histories
                for step in h
                for k in step.keys()
                if k not in {"epoch", "iter"}
            }
        )
        max_len = max(len(h) for h in histories)

        metrics: dict[str, dict[str, list[float]]] = {}
        for key in metric_keys:
            values = np.full((len(histories), max_len), np.nan, dtype=np.float64)
            for i, hist in enumerate(histories):
                for t, row in enumerate(hist):
                    if key in row:
                        values[i, t] = row[key]
            mask = np.isfinite(values)
            count = mask.sum(axis=0)
            total = np.nansum(values, axis=0)
            mean = np.where(count > 0, total / count, np.nan)
            diffs = np.where(mask, values - mean, 0.0)
            var = np.where(count > 0, np.nansum(diffs**2, axis=0) / count, np.nan)
            metrics[key] = {
                "mean": mean.tolist(),
                "std": np.sqrt(var).tolist(),
            }

        return {"metrics": metrics, "epochs": max_len}

    @staticmethod
    def _read_steps(path: Path) -> list[dict[str, float]]:
        if not path.exists():
            return []
        rows: list[dict[str, float]] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows

    @staticmethod
    def _read_history(path: Path) -> list[dict[str, float]]:
        if not path.exists():
            return []
        rows: list[dict[str, float]] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows

    @staticmethod
    def _read_summary(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}
