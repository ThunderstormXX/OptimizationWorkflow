from __future__ import annotations

import json
from pathlib import Path

from benchmarks.checks import CheckResult, ChecksSummary
from benchmarks.report import (
    _compute_mean_std,
    render_checks_markdown,
    render_matrix_markdown,
    render_matrix_visual_report_md,
    render_single_run_report_md,
)


def _write_plot(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")
    return path


def test_render_checks_markdown_covers_failure() -> None:
    summary = ChecksSummary(
        passed=False,
        results=[
            CheckResult(
                name="single_decreases_suboptimality",
                passed=False,
                details={
                    "initial_suboptimality": 1.0,
                    "final_suboptimality": 2.0,
                    "ratio_achieved": 2.0,
                    "ratio_threshold": 0.8,
                    "error": "boom",
                },
            ),
            CheckResult(
                name="constraint_feasibility",
                passed=True,
                details={"final_norm": 0.5, "bound": 1.0},
            ),
            CheckResult(
                name="custom_check",
                passed=True,
                details={"foo": "bar"},
            ),
            CheckResult(
                name="skipped_check",
                passed=True,
                details={"skipped": True, "reason": "not applicable"},
            ),
        ],
    )

    config = {
        "env": "gossip",
        "optimizer": "fw",
        "steps": 5,
        "seed": 1,
        "dim": 4,
        "cond": 10.0,
        "constraint": "l2ball",
        "radius": 2.0,
        "step_schedule": "constant",
        "gamma": 0.2,
        "n_nodes": 3,
        "topology": "ring",
        "strategy": "local_then_gossip",
    }

    md = render_checks_markdown(summary, config)
    assert "FAILED" in md
    assert "single_decreases_suboptimality" in md
    assert "skipped_check" in md


def test_compute_mean_std_empty() -> None:
    mean, std = _compute_mean_std([])
    assert mean == 0.0
    assert std == 0.0


def test_render_matrix_markdown(tmp_path: Path) -> None:
    results_csv = tmp_path / "results.csv"
    results_csv.write_text(
        "run_id,optimizer,topology,strategy,schedule,seed,final_suboptimality,final_mean_loss,final_consensus_error\n"
        "run_0000,fw,ring,local_then_gossip,harmonic,0,0.1,1.0,0.05\n"
        "run_0001,gd,complete,gossip_then_local,na,1,0.2,0.9,0.02\n"
    )

    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "number_of_runs": 2,
                "runs_directory": "runs/",
                "best_by_final_suboptimality": {
                    "run_id": "run_0000",
                    "optimizer": "fw",
                    "topology": "ring",
                    "strategy": "local_then_gossip",
                    "step_schedule": "harmonic",
                    "seed": 0,
                    "final_suboptimality": 0.1,
                },
                "best_by_final_mean_loss": {
                    "run_id": "run_0001",
                    "optimizer": "gd",
                    "topology": "complete",
                    "strategy": "gossip_then_local",
                    "step_schedule": "na",
                    "seed": 1,
                    "final_mean_loss": 0.9,
                },
                "averages": {
                    "mean_final_suboptimality": 0.15,
                    "mean_final_mean_loss": 0.95,
                    "mean_final_consensus_error": 0.03,
                },
            }
        )
    )

    md = render_matrix_markdown(results_csv, summary_path)
    assert "Matrix Experiment Report" in md
    assert "Best Runs" in md


def test_render_single_run_report_md(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp_0000"
    artifacts_dir = exp_dir / "artifacts"
    plots_dir = artifacts_dir / "plots"

    plots = {
        "loss": _write_plot(plots_dir / "loss.png"),
        "suboptimality": _write_plot(plots_dir / "suboptimality.png"),
        "accuracy": _write_plot(plots_dir / "accuracy.png"),
        "consensus": _write_plot(plots_dir / "consensus.png"),
        "budgets": _write_plot(plots_dir / "budgets.png"),
    }

    summary = {
        "env": "gossip",
        "task": "logistic",
        "optimizer": "gd",
        "steps": 5,
        "seed": 0,
        "dim": 4,
        "n_nodes": 3,
        "topology": "ring",
        "strategy": "local_then_gossip",
        "n_samples": 100,
        "batch_size": 10,
        "heterogeneity": "iid",
        "final_mean_loss": 0.5,
        "final_suboptimality": 0.1,
        "final_dist_to_opt": 0.3,
        "final_accuracy": 0.9,
        "final_mean_accuracy": 0.88,
        "final_consensus_error": 0.02,
    }

    md = render_single_run_report_md(exp_dir, summary=summary, plots=plots)
    assert "Experiment Report" in md
    assert "Plots" in md


def test_render_matrix_visual_report_md_with_plots(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp_0000"
    artifacts_dir = exp_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    results_csv_path = artifacts_dir / "results.csv"
    results_csv_path.write_text(
        "run_id,model,optimizer,constraint,radius,topology,strategy,seed,final_mean_loss,final_mean_accuracy,final_suboptimality,final_consensus_error,heterogeneity\n"
        "run_0000,mlp3,adam,l2ball,10,ring,local_then_gossip,0,0.5,0.95,0.1,0.05,iid\n"
        "run_0001,cnn,fw,l1ball,5,complete,gossip_then_local,1,0.4,0.97,0.08,0.02,label_skew\n"
    )

    global_summary = {
        "task": "mnist",
        "number_of_runs": 2,
        "best_by_final_suboptimality": {"run_id": "run_0001", "final_suboptimality": 0.08},
        "best_by_final_mean_loss": {"run_id": "run_0001", "final_mean_loss": 0.4},
    }

    plots_dir = artifacts_dir / "plots"
    plot_keys = [
        "bar_final_loss_by_optimizer",
        "bar_final_loss_by_model",
        "bar_final_loss_by_model_optimizer",
        "bar_final_accuracy_by_optimizer",
        "bar_final_accuracy_by_model",
        "bar_final_accuracy_by_model_optimizer",
        "bar_final_suboptimality_by_optimizer",
        "bar_final_consensus_by_strategy",
        "bar_final_loss_by_topology",
        "bar_final_loss_by_optimizer_strategy",
        "bar_final_accuracy_by_topology_strategy",
        "bar_final_accuracy_by_model_constraint",
        "bar_final_accuracy_by_constraint",
        "bar_final_accuracy_by_heterogeneity",
    ]

    plots = {key: _write_plot(plots_dir / f"{key}.png") for key in plot_keys}

    md = render_matrix_visual_report_md(
        exp_dir,
        global_summary=global_summary,
        results_csv_path=results_csv_path,
        plots=plots,
    )

    assert "Matrix Experiment Report" in md
    assert "Sanity Targets" in md


def test_render_matrix_visual_report_md_no_accuracy(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp_0001"
    artifacts_dir = exp_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    results_csv_path = artifacts_dir / "results.csv"
    results_csv_path.write_text(
        "run_id,model,optimizer,topology,strategy,seed,final_mean_loss,final_consensus_error\\n"
        "run_0000,mlp3,gd,ring,local_then_gossip,0,,0.05\\n"
        "run_0001,cnn,gd,ring,local_then_gossip,1,,0.02\\n"
    )

    global_summary = {"task": "quadratic", "number_of_runs": 2}

    md = render_matrix_visual_report_md(
        exp_dir,
        global_summary=global_summary,
        results_csv_path=results_csv_path,
        plots={},
    )

    assert "Top 10 Runs" in md


def test_render_matrix_visual_report_md_mnist_low_accuracy(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp_0002"
    artifacts_dir = exp_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    results_csv_path = artifacts_dir / "results.csv"
    results_csv_path.write_text(
        "run_id,model,optimizer,topology,strategy,seed,final_mean_loss,final_mean_accuracy\n"
        "run_0000,mlp3,adam,ring,local_then_gossip,0,0.9,0.5\n"
    )

    global_summary = {"task": "mnist", "number_of_runs": 1}

    import csv

    with results_csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["final_mean_accuracy"] == "0.5"

    md = render_matrix_visual_report_md(
        exp_dir,
        global_summary=global_summary,
        results_csv_path=results_csv_path,
        plots={},
    )

    assert "MNIST Baseline Accuracy Target" in md
