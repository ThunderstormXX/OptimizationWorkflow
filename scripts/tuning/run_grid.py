from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RunResult:
    name: str
    exp_dir: Path
    mean_val_acc: float
    std_val_acc: float
    mean_test_acc: float
    std_test_acc: float
    mean_val_loss: float
    mean_test_loss: float


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return safe.strip("_") or "run"


def _find_python() -> str:
    env_bin = os.environ.get("PYTHON_BIN")
    if env_bin:
        return env_bin
    venv_bin = Path(".venv/bin/python")
    if venv_bin.exists():
        return str(venv_bin)
    return "python"


def _next_exp_id(workflow_dir: Path) -> int:
    workflow_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r"^exp_(\d{4})$")
    max_idx = -1
    for entry in workflow_dir.iterdir():
        if entry.is_dir():
            match = pattern.match(entry.name)
            if match:
                max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1


def _copy_task_dir(base_task_dir: Path, out_dir: Path, optimizer_spec: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("task.json", "models.json"):
        shutil.copy2(base_task_dir / fname, out_dir / fname)
    optimizers = [optimizer_spec]
    (out_dir / "optimizers.json").write_text(json.dumps(optimizers, indent=2))


def _collect_metrics(exp_dir: Path) -> RunResult:
    run_summaries = list(exp_dir.glob("runs/*/summary.json"))
    vals = []
    tests = []
    val_losses = []
    test_losses = []
    for path in run_summaries:
        data = json.loads(path.read_text())
        final = data.get("final", {})
        if "val_accuracy" in final:
            vals.append(final.get("val_accuracy"))
        if "test_accuracy" in final:
            tests.append(final.get("test_accuracy"))
        if "val_loss" in final:
            val_losses.append(final.get("val_loss"))
        if "test_loss" in final:
            test_losses.append(final.get("test_loss"))
    if not vals:
        raise RuntimeError(f"No summary.json found in {exp_dir}")

    def mean_std(items: list[float]) -> tuple[float, float]:
        mean = sum(items) / len(items)
        var = sum((x - mean) ** 2 for x in items) / max(len(items), 1)
        return mean, var ** 0.5

    mean_val, std_val = mean_std(vals)
    mean_test, std_test = mean_std(tests)
    mean_val_loss = sum(val_losses) / len(val_losses) if val_losses else float("nan")
    mean_test_loss = sum(test_losses) / len(test_losses) if test_losses else float("nan")
    return RunResult(
        name=exp_dir.name,
        exp_dir=exp_dir,
        mean_val_acc=mean_val,
        std_val_acc=std_val,
        mean_test_acc=mean_test,
        std_test_acc=std_test,
        mean_val_loss=mean_val_loss,
        mean_test_loss=mean_test_loss,
    )


def _run_runner(
    *,
    config: Path,
    workflow_dir: Path,
    exp_id: int,
    task_dir: Path,
    epochs: int,
    num_runs: int,
    seed: int,
    device: str,
    batch_size: int,
    train_size: int,
    val_size: int,
    test_size: int,
    data_root: str,
    full_loss_every: int,
    batch_log_every: int,
    extra_sets: list[str],
) -> None:
    python_bin = _find_python()
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "src")
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        python_bin,
        "-m",
        "benchmarks.runner",
        "--config",
        str(config),
        "--workflow-dir",
        str(workflow_dir),
        "--exp-id",
        str(exp_id),
        "--set",
        f"experiment.params.task_dirs={json.dumps([str(task_dir)])}",
        "--set",
        f"experiment.params.seed={seed}",
        "--set",
        f"experiment.params.num_runs={num_runs}",
        "--set",
        f"experiment.params.trainer.params.epochs={epochs}",
        "--set",
        f"experiment.params.trainer.params.device=\"{device}\"",
        "--set",
        f"experiment.params.trainer.params.full_loss_every={full_loss_every}",
        "--set",
        f"experiment.params.trainer.params.batch_log_every={batch_log_every}",
        "--set",
        f"experiment.params.task_overrides.data_root=\"{data_root}\"",
        "--set",
        f"experiment.params.task_overrides.batch_size={batch_size}",
        "--set",
        f"experiment.params.task_overrides.train_size={train_size}",
        "--set",
        f"experiment.params.task_overrides.val_size={val_size}",
        "--set",
        f"experiment.params.task_overrides.test_size={test_size}",
        "--set",
        f"experiment.params.task_overrides.device=\"{device}\"",
        "--set",
        "experiment.params.task_overrides.verbose=true",
    ]
    for item in extra_sets:
        cmd.extend(["--set", item])
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--config", default="configs/experiments/mnist_cnn.json")
    parser.add_argument("--task-dir", default="configs/tasks/mnist_cnn")
    parser.add_argument("--workflow-dir", default="workflow/tuning")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--data-root", default=".data")
    parser.add_argument("--full-loss-every", type=int, default=10)
    parser.add_argument("--batch-log-every", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    method = _safe_name(args.method)
    grid_path = Path(args.grid)
    grid = json.loads(grid_path.read_text())

    base_task_dir = Path(args.task_dir)
    workflow_dir = Path(args.workflow_dir) / method
    workflow_dir.mkdir(parents=True, exist_ok=True)

    results_path = workflow_dir / "results.json"
    best_path = workflow_dir / "best.json"
    results: list[dict[str, Any]] = []

    start_id = _next_exp_id(workflow_dir)

    for idx, spec in enumerate(grid):
        name = spec.get("name") or spec.get("class", f"opt_{idx}")
        spec_name = _safe_name(name)
        task_dir = workflow_dir / f"task_{idx:02d}_{spec_name}"
        _copy_task_dir(base_task_dir, task_dir, spec)

        exp_id = start_id + idx
        exp_dir = workflow_dir / f"exp_{exp_id:04d}"
        if exp_dir.exists() and not args.force:
            # Try to reuse existing results
            try:
                result = _collect_metrics(exp_dir)
                results.append(
                    {
                        "name": name,
                        "spec": spec,
                        "exp_dir": str(exp_dir),
                        "mean_val_acc": result.mean_val_acc,
                        "std_val_acc": result.std_val_acc,
                        "mean_test_acc": result.mean_test_acc,
                        "std_test_acc": result.std_test_acc,
                        "mean_val_loss": result.mean_val_loss,
                        "mean_test_loss": result.mean_test_loss,
                    }
                )
                continue
            except Exception:
                pass

        _run_runner(
            config=Path(args.config),
            workflow_dir=workflow_dir,
            exp_id=exp_id,
            task_dir=task_dir,
            epochs=args.epochs,
            num_runs=args.num_runs,
            seed=args.seed,
            device=args.device,
            batch_size=args.batch_size,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            data_root=args.data_root,
            full_loss_every=args.full_loss_every,
            batch_log_every=args.batch_log_every,
            extra_sets=[],
        )

        result = _collect_metrics(exp_dir)
        results.append(
            {
                "name": name,
                "spec": spec,
                "exp_dir": str(exp_dir),
                "mean_val_acc": result.mean_val_acc,
                "std_val_acc": result.std_val_acc,
                "mean_test_acc": result.mean_test_acc,
                "std_test_acc": result.std_test_acc,
                "mean_val_loss": result.mean_val_loss,
                "mean_test_loss": result.mean_test_loss,
            }
        )

    results.sort(key=lambda r: r["mean_val_acc"], reverse=True)
    results_path.write_text(json.dumps(results, indent=2))

    if results:
        best = results[0]
        best_path.write_text(json.dumps(best, indent=2))
        print(f"Best {args.method}: {best['name']} val_acc={best['mean_val_acc']:.4f}")


if __name__ == "__main__":
    main()
