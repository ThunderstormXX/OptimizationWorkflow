from __future__ import annotations

import json
from pathlib import Path

from benchmarks.legacy_runner import generate_matrix_grid, parse_args, run_matrix_mode


def test_generate_matrix_grid_mnist_dist_big() -> None:
    args = parse_args(
        [
            "--mode",
            "matrix",
            "--matrix",
            "mnist_dist_big",
            "--env",
            "gossip",
            "--task",
            "mnist",
            "--n-nodes",
            "2",
            "--steps",
            "1",
            "--batch-size",
            "4",
        ]
    )
    grid = generate_matrix_grid(args, seeds=[0])
    assert grid
    assert all(cfg["task"] == "mnist" for cfg in grid)


def test_generate_matrix_grid_large_and_medium() -> None:
    args_large = parse_args(["--mode", "matrix", "--matrix", "large"])
    grid_large = generate_matrix_grid(args_large, seeds=[0])
    assert grid_large

    args_medium = parse_args(["--mode", "matrix", "--matrix", "medium"])
    grid_medium = generate_matrix_grid(args_medium, seeds=[0])
    assert grid_medium


def test_run_matrix_mode_with_ablation_and_animation(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "base": {
                    "env": "single",
                    "task": "logistic",
                    "steps": 2,
                    "dim": 3,
                    "optimizer": "gd",
                    "lr": 0.1,
                    "n_samples": 50,
                    "batch_size": 10,
                },
                "grid": {
                    "optimizer": ["gd"],
                    "topology": ["ring"],
                    "strategy": ["local_then_gossip"],
                    "step_schedule": ["harmonic"],
                    "heterogeneity": ["iid"],
                },
                "seeds": [0],
            }
        )
    )

    args = parse_args(
        [
            "--mode",
            "matrix",
            "--env",
            "single",
            "--task",
            "logistic",
            "--steps",
            "2",
            "--dim",
            "3",
            "--n-samples",
            "50",
            "--batch-size",
            "10",
            "--seeds",
            "0",
            "--ablation-spec",
            str(spec_path),
            "--animate",
            "--animate-format",
            "gif",
            "--animate-top-k",
            "1",
            "--animate-metric",
            "final_mean_accuracy",
        ]
    )

    exp_dir = tmp_path / "exp_0000"
    exp_dir.mkdir(parents=True, exist_ok=True)

    summary = run_matrix_mode(args, exp_dir, render=True, animate=True)
    assert summary["number_of_runs"] == 1
    results_csv = exp_dir / "artifacts" / "results.csv"
    assert results_csv.exists()
