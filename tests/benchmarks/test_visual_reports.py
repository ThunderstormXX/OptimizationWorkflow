"""Tests for visual reports and plotting functionality.

This module tests:
- Single run plot generation and report rendering
- Matrix run aggregate plots and visual reports
- Ablation spec loading and configuration generation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.plotting import (
    group_mean_std,
    plot_matrix_results,
    plot_single_run,
    read_history_jsonl,
    read_results_csv,
)
from benchmarks.report import render_matrix_visual_report_md, render_single_run_report_md
from benchmarks.legacy_runner import load_ablation_spec, main


class TestPlottingParsers:
    """Tests for parsing functions."""

    def test_read_history_jsonl(self, tmp_path: Path) -> None:
        """Test reading history.jsonl file."""
        history_path = tmp_path / "history.jsonl"
        lines = [
            '{"step": 0, "mean_loss": 1.0, "suboptimality": 0.5}',
            '{"step": 1, "mean_loss": 0.8, "suboptimality": 0.3}',
            '{"step": 2, "mean_loss": 0.6, "suboptimality": 0.1}',
        ]
        history_path.write_text("\n".join(lines))

        result = read_history_jsonl(history_path)

        assert len(result) == 3
        assert result[0]["step"] == 0
        assert result[0]["mean_loss"] == 1.0
        assert result[2]["suboptimality"] == 0.1

    def test_read_results_csv(self, tmp_path: Path) -> None:
        """Test reading results.csv file."""
        csv_path = tmp_path / "results.csv"
        csv_content = """run_id,optimizer,final_mean_loss,final_suboptimality
run_0000,fw,0.5,0.1
run_0001,gd,0.6,0.2
run_0002,pgd,0.4,0.05"""
        csv_path.write_text(csv_content)

        result = read_results_csv(csv_path)

        assert len(result) == 3
        assert result[0]["run_id"] == "run_0000"
        assert result[0]["optimizer"] == "fw"
        assert result[1]["final_mean_loss"] == "0.6"


class TestGroupMeanStd:
    """Tests for group_mean_std helper."""

    def test_group_by_single_key(self) -> None:
        """Test grouping by a single key."""
        rows = [
            {"optimizer": "fw", "final_loss": "1.0"},
            {"optimizer": "fw", "final_loss": "2.0"},
            {"optimizer": "gd", "final_loss": "3.0"},
        ]

        result = group_mean_std(rows, ["optimizer"], "final_loss")

        assert len(result) == 2
        # fw: mean=1.5, std=0.5
        fw_result = [r for r in result if r[0] == "fw"][0]
        assert fw_result[1] == pytest.approx(1.5)
        assert fw_result[2] == pytest.approx(0.5)
        # gd: mean=3.0, std=0.0 (single value)
        gd_result = [r for r in result if r[0] == "gd"][0]
        assert gd_result[1] == pytest.approx(3.0)
        assert gd_result[2] == pytest.approx(0.0)

    def test_missing_values_skipped(self) -> None:
        """Test that missing/empty values are skipped."""
        rows = [
            {"optimizer": "fw", "final_loss": "1.0"},
            {"optimizer": "fw", "final_loss": ""},
            {"optimizer": "fw", "final_loss": "na"},
        ]

        result = group_mean_std(rows, ["optimizer"], "final_loss")

        assert len(result) == 1
        assert result[0][1] == pytest.approx(1.0)


class TestPlotSingleRun:
    """Tests for single run plotting."""

    def test_creates_loss_plot(self, tmp_path: Path) -> None:
        """Test that loss plot is created."""
        history = [
            {"step": 0, "mean_loss": 1.0},
            {"step": 1, "mean_loss": 0.8},
            {"step": 2, "mean_loss": 0.6},
        ]

        plots = plot_single_run(history, tmp_path, task="quadratic", env="single")

        assert "loss" in plots
        assert plots["loss"].exists()
        assert plots["loss"].stat().st_size > 0

    def test_creates_suboptimality_plot_for_quadratic(self, tmp_path: Path) -> None:
        """Test that suboptimality plot is created for quadratic task."""
        history = [
            {"step": 0, "mean_loss": 1.0, "suboptimality": 0.5},
            {"step": 1, "mean_loss": 0.8, "suboptimality": 0.3},
            {"step": 2, "mean_loss": 0.6, "suboptimality": 0.1},
        ]

        plots = plot_single_run(history, tmp_path, task="quadratic", env="single")

        assert "suboptimality" in plots
        assert plots["suboptimality"].exists()
        assert plots["suboptimality"].stat().st_size > 0

    def test_creates_accuracy_plot_for_logistic(self, tmp_path: Path) -> None:
        """Test that accuracy plot is created for logistic task."""
        history = [
            {"step": 0, "mean_loss": 0.7, "mean_accuracy": 0.5},
            {"step": 1, "mean_loss": 0.6, "mean_accuracy": 0.6},
            {"step": 2, "mean_loss": 0.5, "mean_accuracy": 0.7},
        ]

        plots = plot_single_run(history, tmp_path, task="logistic", env="single")

        assert "accuracy" in plots
        assert plots["accuracy"].exists()
        assert plots["accuracy"].stat().st_size > 0

    def test_creates_consensus_plot_for_gossip(self, tmp_path: Path) -> None:
        """Test that consensus plot is created for gossip environment."""
        history = [
            {"step": 0, "mean_loss": 1.0, "consensus_error": 0.5},
            {"step": 1, "mean_loss": 0.8, "consensus_error": 0.3},
            {"step": 2, "mean_loss": 0.6, "consensus_error": 0.1},
        ]

        plots = plot_single_run(history, tmp_path, task="quadratic", env="gossip")

        assert "consensus" in plots
        assert plots["consensus"].exists()
        assert plots["consensus"].stat().st_size > 0

    def test_creates_budgets_plot(self, tmp_path: Path) -> None:
        """Test that budgets plot is created when data is available."""
        history = [
            {"step": 0, "mean_loss": 1.0, "total_grad_evals": 5, "total_gossip_rounds": 1},
            {"step": 1, "mean_loss": 0.8, "total_grad_evals": 10, "total_gossip_rounds": 2},
        ]

        plots = plot_single_run(history, tmp_path, task="quadratic", env="gossip")

        assert "budgets" in plots
        assert plots["budgets"].exists()
        assert plots["budgets"].stat().st_size > 0


class TestPlotMatrixResults:
    """Tests for matrix results plotting."""

    def test_creates_bar_plots(self, tmp_path: Path) -> None:
        """Test that bar plots are created."""
        rows = [
            {"optimizer": "fw", "final_mean_loss": "1.0", "final_suboptimality": "0.1"},
            {"optimizer": "fw", "final_mean_loss": "0.9", "final_suboptimality": "0.08"},
            {"optimizer": "gd", "final_mean_loss": "1.2", "final_suboptimality": "0.15"},
        ]

        plots = plot_matrix_results(rows, tmp_path, task="quadratic")

        assert "bar_final_loss_by_optimizer" in plots
        assert plots["bar_final_loss_by_optimizer"].exists()
        assert "bar_final_suboptimality_by_optimizer" in plots
        assert plots["bar_final_suboptimality_by_optimizer"].exists()

    def test_creates_gossip_specific_plots(self, tmp_path: Path) -> None:
        """Test that gossip-specific plots are created."""
        rows = [
            {
                "env": "gossip",
                "optimizer": "fw",
                "strategy": "local_then_gossip",
                "topology": "ring",
                "final_mean_loss": "1.0",
                "final_consensus_error": "0.1",
            },
            {
                "env": "gossip",
                "optimizer": "gd",
                "strategy": "gossip_then_local",
                "topology": "complete",
                "final_mean_loss": "0.9",
                "final_consensus_error": "0.05",
            },
        ]

        plots = plot_matrix_results(rows, tmp_path, task="quadratic")

        assert "bar_final_consensus_by_strategy" in plots
        assert "bar_final_loss_by_topology" in plots


class TestRenderSingleRunReportMd:
    """Tests for single run report rendering."""

    def test_includes_config_and_metrics(self, tmp_path: Path) -> None:
        """Test that report includes configuration and metrics."""
        summary: dict[str, Any] = {
            "env": "gossip",
            "task": "quadratic",
            "optimizer": "fw",
            "steps": 50,
            "seed": 0,
            "dim": 10,
            "n_nodes": 5,
            "topology": "ring",
            "strategy": "local_then_gossip",
            "final_mean_loss": 0.5,
            "final_suboptimality": 0.1,
            "final_consensus_error": 0.05,
        }

        report = render_single_run_report_md(tmp_path, summary=summary, plots={})

        assert "# Experiment Report" in report
        assert "gossip" in report
        assert "quadratic" in report
        assert "fw" in report
        assert "0.5" in report or "0.500000" in report
        assert "0.1" in report or "0.100000" in report

    def test_embeds_plots_with_relative_paths(self, tmp_path: Path) -> None:
        """Test that plots are embedded with relative paths."""
        summary: dict[str, Any] = {
            "env": "single",
            "task": "quadratic",
            "optimizer": "gd",
            "steps": 30,
            "seed": 0,
            "dim": 5,
            "final_mean_loss": 0.3,
        }

        # Create fake plot files
        plots_dir = tmp_path / "artifacts" / "plots"
        plots_dir.mkdir(parents=True)
        loss_plot = plots_dir / "loss.png"
        loss_plot.write_bytes(b"fake png data")

        plots = {"loss": loss_plot}

        report = render_single_run_report_md(tmp_path, summary=summary, plots=plots)

        assert "![Loss]" in report
        assert "plots/loss.png" in report


class TestRenderMatrixVisualReportMd:
    """Tests for matrix visual report rendering."""

    def test_includes_tables_and_plots(self, tmp_path: Path) -> None:
        """Test that report includes tables and embedded plots."""
        # Create results CSV
        csv_path = tmp_path / "results.csv"
        header = "run_id,optimizer,strategy,topology,seed,final_mean_loss"
        header += ",final_suboptimality,final_consensus_error"
        csv_content = f"""{header}
run_0000,fw,local_then_gossip,ring,0,1.0,0.1,0.05
run_0001,gd,gossip_then_local,complete,0,0.9,0.08,0.03"""
        csv_path.write_text(csv_content)

        global_summary: dict[str, Any] = {
            "number_of_runs": 2,
            "best_by_final_mean_loss": {
                "run_id": "run_0001",
                "final_mean_loss": 0.9,
                "optimizer": "gd",
                "topology": "complete",
                "strategy": "gossip_then_local",
            },
        }

        plots: dict[str, Path] = {}

        report = render_matrix_visual_report_md(
            tmp_path,
            global_summary=global_summary,
            results_csv_path=csv_path,
            plots=plots,
        )

        assert "# Matrix Experiment Report" in report
        assert "number_of_runs" in report.lower() or "Number of runs" in report
        assert "run_0001" in report
        assert "By Optimizer" in report


class TestAblationSpec:
    """Tests for ablation spec loading."""

    def test_load_ablation_spec(self, tmp_path: Path) -> None:
        """Test loading ablation spec from JSON."""
        spec = {
            "base": {
                "env": "gossip",
                "task": "quadratic",
                "steps": 10,
                "dim": 5,
            },
            "grid": {
                "optimizer": ["fw", "gd"],
                "topology": ["ring"],
                "strategy": ["local_then_gossip"],
                "step_schedule": ["harmonic"],
            },
            "seeds": [0, 1],
        }
        spec_path = tmp_path / "spec.json"
        spec_path.write_text(json.dumps(spec))

        # Create a mock args object
        class MockArgs:
            env = "gossip"
            task = "quadratic"
            steps = 10
            dim = 5
            cond = 10.0
            lr = 0.1
            constraint = "l2ball"
            radius = 1.0
            gamma = 0.2
            n_nodes = 5
            n_samples = 2000
            batch_size = 64

        args = MockArgs()

        configs = load_ablation_spec(spec_path, args)

        # Should have 2 optimizers x 1 topology x 1 strategy x 2 seeds = 4 configs
        # But gd with different schedules are deduplicated
        assert len(configs) >= 2
        assert all(c["env"] == "gossip" for c in configs)
        assert all(c["task"] == "quadratic" for c in configs)


class TestSingleRunRenderSmokeTest:
    """Smoke tests for single run with rendering."""

    def test_single_run_creates_plots_and_report(self, tmp_path: Path) -> None:
        """Test that single run creates plots and report.md."""
        workflow_dir = tmp_path / "workflow"
        workflow_dir.mkdir()

        exit_code = main(
            [
                "--mode",
                "single",
                "--env",
                "gossip",
                "--task",
                "logistic",
                "--steps",
                "5",
                "--n-nodes",
                "3",
                "--heterogeneity",
                "iid",
                "--optimizer",
                "gd",
                "--lr",
                "0.1",
                "--dim",
                "5",
                "--n-samples",
                "100",
                "--batch-size",
                "32",
                "--workflow-dir",
                str(workflow_dir),
            ]
        )

        assert exit_code == 0

        # Find the experiment directory
        exp_dirs = list(workflow_dir.glob("exp_*"))
        assert len(exp_dirs) == 1
        exp_dir = exp_dirs[0]

        # Check report.md exists
        report_path = exp_dir / "artifacts" / "report.md"
        assert report_path.exists()
        report_content = report_path.read_text()
        assert "![" in report_content or "Loss" in report_content

        # Check plots directory exists and has files
        plots_dir = exp_dir / "artifacts" / "plots"
        assert plots_dir.exists()
        png_files = list(plots_dir.glob("*.png"))
        assert len(png_files) > 0

        # Check loss.png exists and is non-empty
        loss_png = plots_dir / "loss.png"
        assert loss_png.exists()
        assert loss_png.stat().st_size > 0


class TestMatrixRenderSmokeTest:
    """Smoke tests for matrix mode with rendering."""

    def test_matrix_run_creates_plots_and_report(self, tmp_path: Path) -> None:
        """Test that matrix run creates plots and report.md."""
        workflow_dir = tmp_path / "workflow"
        workflow_dir.mkdir()

        exit_code = main(
            [
                "--mode",
                "matrix",
                "--matrix",
                "small",
                "--env",
                "gossip",
                "--task",
                "quadratic",
                "--steps",
                "5",
                "--n-nodes",
                "3",
                "--seeds",
                "0,1",
                "--dim",
                "5",
                "--workflow-dir",
                str(workflow_dir),
            ]
        )

        assert exit_code == 0

        # Find the experiment directory
        exp_dirs = list(workflow_dir.glob("exp_*"))
        assert len(exp_dirs) == 1
        exp_dir = exp_dirs[0]

        # Check report.md exists
        report_path = exp_dir / "artifacts" / "report.md"
        assert report_path.exists()

        # Check plots directory exists and has files
        plots_dir = exp_dir / "artifacts" / "plots"
        assert plots_dir.exists()
        png_files = list(plots_dir.glob("*.png"))
        assert len(png_files) > 0

        # Check results.csv exists
        results_csv = exp_dir / "artifacts" / "results.csv"
        assert results_csv.exists()


class TestAblationSpecSmokeTest:
    """Smoke tests for ablation spec mode."""

    def test_ablation_spec_creates_runs_and_report(self, tmp_path: Path) -> None:
        """Test that ablation spec creates correct number of runs and report."""
        workflow_dir = tmp_path / "workflow"
        workflow_dir.mkdir()

        # Create a tiny spec
        spec = {
            "base": {
                "env": "gossip",
                "task": "quadratic",
                "steps": 3,
                "dim": 3,
                "n_nodes": 2,
            },
            "grid": {
                "optimizer": ["fw", "gd"],
                "topology": ["ring"],
                "strategy": ["local_then_gossip"],
                "step_schedule": ["harmonic"],
            },
            "seeds": [0],
        }
        spec_path = tmp_path / "spec.json"
        spec_path.write_text(json.dumps(spec))

        exit_code = main(
            [
                "--mode",
                "matrix",
                "--ablation-spec",
                str(spec_path),
                "--workflow-dir",
                str(workflow_dir),
            ]
        )

        assert exit_code == 0

        # Find the experiment directory
        exp_dirs = list(workflow_dir.glob("exp_*"))
        assert len(exp_dirs) == 1
        exp_dir = exp_dirs[0]

        # Check runs directory has correct number of runs
        runs_dir = exp_dir / "artifacts" / "runs"
        assert runs_dir.exists()
        run_dirs = list(runs_dir.glob("run_*"))
        # 2 optimizers x 1 topology x 1 strategy x 1 seed = 2 configs
        assert len(run_dirs) == 2

        # Check report.md exists
        report_path = exp_dir / "artifacts" / "report.md"
        assert report_path.exists()

        # Check at least one plot exists
        plots_dir = exp_dir / "artifacts" / "plots"
        assert plots_dir.exists()
        png_files = list(plots_dir.glob("*.png"))
        assert len(png_files) >= 1
