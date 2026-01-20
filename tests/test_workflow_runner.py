"""Tests for workflow management and experiment runner.

This module tests:
- Experiment directory creation and naming
- Runner smoke tests for single and gossip environments
- Artifact generation (history.jsonl, summary.json)
- Matrix mode with results.csv
- Strategy selection
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from benchmarks.runner import CSV_COLUMNS, main
from benchmarks.workflow import next_experiment_dir, try_get_git_commit, write_run_files

# =============================================================================
# Tests for next_experiment_dir
# =============================================================================


class TestNextExperimentDir:
    """Tests for experiment directory creation."""

    def test_creates_first_exp_dir(self, tmp_path: Path) -> None:
        """First experiment should be exp_0000."""
        workflow_dir = tmp_path / "workflow"

        exp_dir = next_experiment_dir(workflow_dir)

        assert exp_dir.name == "exp_0000"
        assert exp_dir.exists()
        assert (exp_dir / "artifacts").exists()

    def test_increments_after_existing(self, tmp_path: Path) -> None:
        """Should create next index after max existing."""
        workflow_dir = tmp_path / "workflow"
        workflow_dir.mkdir()

        # Create exp_0000 and exp_0002 (gap at 0001)
        (workflow_dir / "exp_0000").mkdir()
        (workflow_dir / "exp_0002").mkdir()

        exp_dir = next_experiment_dir(workflow_dir)

        # Should be exp_0003 (next after max=2)
        assert exp_dir.name == "exp_0003"
        assert exp_dir.exists()
        assert (exp_dir / "artifacts").exists()

    def test_ignores_non_exp_dirs(self, tmp_path: Path) -> None:
        """Should ignore directories not matching exp_XXXX pattern."""
        workflow_dir = tmp_path / "workflow"
        workflow_dir.mkdir()

        # Create some non-matching directories
        (workflow_dir / "other_dir").mkdir()
        (workflow_dir / "exp_invalid").mkdir()
        (workflow_dir / "exp_00001").mkdir()  # 5 digits, not 4

        exp_dir = next_experiment_dir(workflow_dir)

        assert exp_dir.name == "exp_0000"

    def test_creates_workflow_dir_if_missing(self, tmp_path: Path) -> None:
        """Should create workflow directory if it doesn't exist."""
        workflow_dir = tmp_path / "new_workflow"

        exp_dir = next_experiment_dir(workflow_dir)

        assert workflow_dir.exists()
        assert exp_dir.exists()


# =============================================================================
# Tests for write_run_files
# =============================================================================


class TestWriteRunFiles:
    """Tests for writing experiment metadata files."""

    def test_writes_all_files(self, tmp_path: Path) -> None:
        """Should write meta.json, config.json, and README.md."""
        exp_dir = tmp_path / "exp_0000"
        exp_dir.mkdir()

        meta = {"created_at": "2024-01-01T00:00:00Z", "argv": ["test"]}
        config = {"env": "single", "steps": 10}
        readme = "# Test Experiment\n\nDescription here."

        write_run_files(exp_dir, meta=meta, config=config, readme_text=readme)

        assert (exp_dir / "meta.json").exists()
        assert (exp_dir / "config.json").exists()
        assert (exp_dir / "README.md").exists()

        # Verify content
        with (exp_dir / "meta.json").open() as f:
            loaded_meta = json.load(f)
        assert loaded_meta["created_at"] == "2024-01-01T00:00:00Z"

        with (exp_dir / "config.json").open() as f:
            loaded_config = json.load(f)
        assert loaded_config["env"] == "single"

        with (exp_dir / "README.md").open() as f:
            loaded_readme = f.read()
        assert "Test Experiment" in loaded_readme


# =============================================================================
# Tests for try_get_git_commit
# =============================================================================


class TestTryGetGitCommit:
    """Tests for git commit retrieval."""

    def test_returns_string_or_none(self) -> None:
        """Should return a string (commit hash) or None."""
        result = try_get_git_commit()
        assert result is None or isinstance(result, str)

    def test_does_not_crash(self) -> None:
        """Should never raise an exception."""
        # This test just verifies it doesn't crash
        try:
            try_get_git_commit()
        except Exception as e:
            pytest.fail(f"try_get_git_commit raised: {e}")


# =============================================================================
# Tests for runner (single env)
# =============================================================================


class TestRunnerSingleEnv:
    """Smoke tests for single-process environment runner."""

    def test_runner_single_creates_artifacts(self, tmp_path: Path) -> None:
        """Runner should create experiment directory and artifacts."""
        workflow_dir = tmp_path / "workflow"

        exit_code = main(
            [
                "--workflow-dir",
                str(workflow_dir),
                "--env",
                "single",
                "--steps",
                "5",
                "--seed",
                "0",
                "--dim",
                "5",
            ]
        )

        assert exit_code == 0

        # Find created experiment directory
        exp_dirs = list(workflow_dir.glob("exp_*"))
        assert len(exp_dirs) == 1
        exp_dir = exp_dirs[0]

        # Check metadata files exist and are non-empty
        assert (exp_dir / "meta.json").exists()
        assert (exp_dir / "meta.json").stat().st_size > 0

        assert (exp_dir / "config.json").exists()
        assert (exp_dir / "config.json").stat().st_size > 0

        assert (exp_dir / "README.md").exists()
        assert (exp_dir / "README.md").stat().st_size > 0

        # Check artifacts
        artifacts_dir = exp_dir / "artifacts"
        assert artifacts_dir.exists()

        # history.jsonl should have exactly 5 lines
        history_path = artifacts_dir / "history.jsonl"
        assert history_path.exists()
        with history_path.open() as f:
            lines = f.readlines()
        assert len(lines) == 5

        # Each line should be valid JSON
        for line in lines:
            data = json.loads(line)
            assert "step" in data
            assert "mean_loss" in data

        # summary.json should exist with expected keys
        summary_path = artifacts_dir / "summary.json"
        assert summary_path.exists()
        with summary_path.open() as f:
            summary = json.load(f)
        assert "final_mean_loss" in summary
        assert "final_suboptimality" in summary
        assert "env" in summary
        assert summary["env"] == "single"


# =============================================================================
# Tests for runner (gossip env)
# =============================================================================


class TestRunnerGossipEnv:
    """Smoke tests for gossip environment runner."""

    def test_runner_gossip_creates_artifacts(self, tmp_path: Path) -> None:
        """Runner should create experiment directory and artifacts for gossip."""
        workflow_dir = tmp_path / "workflow"

        exit_code = main(
            [
                "--workflow-dir",
                str(workflow_dir),
                "--env",
                "gossip",
                "--n-nodes",
                "3",
                "--topology",
                "ring",
                "--steps",
                "5",
                "--seed",
                "0",
                "--dim",
                "5",
            ]
        )

        assert exit_code == 0

        # Find created experiment directory
        exp_dirs = list(workflow_dir.glob("exp_*"))
        assert len(exp_dirs) == 1
        exp_dir = exp_dirs[0]

        # Check artifacts
        artifacts_dir = exp_dir / "artifacts"

        # summary.json should include gossip-specific fields
        summary_path = artifacts_dir / "summary.json"
        assert summary_path.exists()
        with summary_path.open() as f:
            summary = json.load(f)

        assert "final_consensus_error" in summary
        assert "final_mean_loss" in summary
        assert "final_suboptimality" in summary
        assert summary["env"] == "gossip"
        assert summary["n_nodes"] == 3
        assert summary["topology"] == "ring"

        # history.jsonl lines should include consensus_error
        history_path = artifacts_dir / "history.jsonl"
        with history_path.open() as f:
            lines = f.readlines()

        assert len(lines) == 5
        for line in lines:
            data = json.loads(line)
            assert "consensus_error" in data

    def test_runner_gossip_complete_topology(self, tmp_path: Path) -> None:
        """Runner should work with complete topology."""
        workflow_dir = tmp_path / "workflow"

        exit_code = main(
            [
                "--workflow-dir",
                str(workflow_dir),
                "--env",
                "gossip",
                "--n-nodes",
                "4",
                "--topology",
                "complete",
                "--steps",
                "3",
                "--seed",
                "42",
                "--dim",
                "4",
            ]
        )

        assert exit_code == 0

        exp_dirs = list(workflow_dir.glob("exp_*"))
        assert len(exp_dirs) == 1


# =============================================================================
# Tests for runner options
# =============================================================================


class TestRunnerOptions:
    """Tests for various runner options."""

    def test_simplex_constraint(self, tmp_path: Path) -> None:
        """Runner should work with simplex constraint."""
        workflow_dir = tmp_path / "workflow"

        exit_code = main(
            [
                "--workflow-dir",
                str(workflow_dir),
                "--env",
                "single",
                "--constraint",
                "simplex",
                "--steps",
                "3",
                "--dim",
                "5",
            ]
        )

        assert exit_code == 0

    def test_constant_step_size(self, tmp_path: Path) -> None:
        """Runner should work with constant step size."""
        workflow_dir = tmp_path / "workflow"

        exit_code = main(
            [
                "--workflow-dir",
                str(workflow_dir),
                "--env",
                "single",
                "--step-schedule",
                "constant",
                "--gamma",
                "0.1",
                "--steps",
                "3",
                "--dim",
                "5",
            ]
        )

        assert exit_code == 0

    def test_exp_name_and_description(self, tmp_path: Path) -> None:
        """Runner should include exp-name and description in README."""
        workflow_dir = tmp_path / "workflow"

        exit_code = main(
            [
                "--workflow-dir",
                str(workflow_dir),
                "--env",
                "single",
                "--steps",
                "2",
                "--dim",
                "3",
                "--exp-name",
                "my_experiment",
                "--description",
                "Test description",
            ]
        )

        assert exit_code == 0

        exp_dirs = list(workflow_dir.glob("exp_*"))
        exp_dir = exp_dirs[0]

        with (exp_dir / "README.md").open() as f:
            readme = f.read()

        assert "my_experiment" in readme
        assert "Test description" in readme


# =============================================================================
# Tests for multiple runs
# =============================================================================


class TestMultipleRuns:
    """Tests for running multiple experiments."""

    def test_multiple_runs_increment_dir(self, tmp_path: Path) -> None:
        """Multiple runs should create incrementing directories."""
        workflow_dir = tmp_path / "workflow"

        # Run 3 experiments
        for _ in range(3):
            exit_code = main(
                [
                    "--workflow-dir",
                    str(workflow_dir),
                    "--env",
                    "single",
                    "--steps",
                    "2",
                    "--dim",
                    "3",
                ]
            )
            assert exit_code == 0

        # Should have exp_0000, exp_0001, exp_0002
        exp_dirs = sorted(workflow_dir.glob("exp_*"))
        assert len(exp_dirs) == 3
        assert exp_dirs[0].name == "exp_0000"
        assert exp_dirs[1].name == "exp_0001"
        assert exp_dirs[2].name == "exp_0002"


# =============================================================================
# Tests for optimizer selection
# =============================================================================


class TestOptimizerSelection:
    """Tests for optimizer selection via CLI."""

    def test_fw_optimizer(self, tmp_path: Path) -> None:
        """Runner should work with Frank-Wolfe optimizer."""
        workflow_dir = tmp_path / "workflow"

        exit_code = main(
            [
                "--workflow-dir",
                str(workflow_dir),
                "--env",
                "single",
                "--optimizer",
                "fw",
                "--steps",
                "3",
                "--dim",
                "5",
            ]
        )

        assert exit_code == 0

        exp_dirs = list(workflow_dir.glob("exp_*"))
        exp_dir = exp_dirs[0]

        with (exp_dir / "artifacts" / "summary.json").open() as f:
            summary = json.load(f)

        assert summary["optimizer"] == "fw"

    def test_gd_optimizer(self, tmp_path: Path) -> None:
        """Runner should work with Gradient Descent optimizer."""
        workflow_dir = tmp_path / "workflow"

        exit_code = main(
            [
                "--workflow-dir",
                str(workflow_dir),
                "--env",
                "single",
                "--optimizer",
                "gd",
                "--lr",
                "0.01",
                "--steps",
                "3",
                "--dim",
                "5",
            ]
        )

        assert exit_code == 0

        exp_dirs = list(workflow_dir.glob("exp_*"))
        exp_dir = exp_dirs[0]

        with (exp_dir / "artifacts" / "summary.json").open() as f:
            summary = json.load(f)

        assert summary["optimizer"] == "gd"
        assert summary["lr"] == 0.01

    def test_pgd_optimizer(self, tmp_path: Path) -> None:
        """Runner should work with Projected GD optimizer."""
        workflow_dir = tmp_path / "workflow"

        exit_code = main(
            [
                "--workflow-dir",
                str(workflow_dir),
                "--env",
                "single",
                "--optimizer",
                "pgd",
                "--lr",
                "0.05",
                "--radius",
                "2.0",
                "--steps",
                "3",
                "--dim",
                "5",
            ]
        )

        assert exit_code == 0

        exp_dirs = list(workflow_dir.glob("exp_*"))
        exp_dir = exp_dirs[0]

        with (exp_dir / "artifacts" / "summary.json").open() as f:
            summary = json.load(f)

        assert summary["optimizer"] == "pgd"
        assert summary["lr"] == 0.05

    def test_pgd_gossip_optimizer(self, tmp_path: Path) -> None:
        """Runner should work with PGD optimizer in gossip mode."""
        workflow_dir = tmp_path / "workflow"

        exit_code = main(
            [
                "--workflow-dir",
                str(workflow_dir),
                "--env",
                "gossip",
                "--optimizer",
                "pgd",
                "--lr",
                "0.05",
                "--n-nodes",
                "3",
                "--steps",
                "3",
                "--dim",
                "5",
            ]
        )

        assert exit_code == 0

        exp_dirs = list(workflow_dir.glob("exp_*"))
        exp_dir = exp_dirs[0]

        with (exp_dir / "artifacts" / "summary.json").open() as f:
            summary = json.load(f)

        assert summary["optimizer"] == "pgd"
        assert "final_consensus_error" in summary


# =============================================================================
# Tests for strategy selection
# =============================================================================


class TestStrategySelection:
    """Tests for gossip strategy selection."""

    def test_local_then_gossip_strategy(self, tmp_path: Path) -> None:
        """Runner should work with local_then_gossip strategy."""
        workflow_dir = tmp_path / "workflow"

        exit_code = main(
            [
                "--workflow-dir",
                str(workflow_dir),
                "--env",
                "gossip",
                "--n-nodes",
                "3",
                "--topology",
                "ring",
                "--strategy",
                "local_then_gossip",
                "--steps",
                "3",
                "--seed",
                "0",
                "--dim",
                "5",
            ]
        )

        assert exit_code == 0

        exp_dirs = list(workflow_dir.glob("exp_*"))
        exp_dir = exp_dirs[0]

        with (exp_dir / "artifacts" / "summary.json").open() as f:
            summary = json.load(f)

        assert summary["strategy"] == "local_then_gossip"

    def test_gossip_then_local_strategy(self, tmp_path: Path) -> None:
        """Runner should work with gossip_then_local strategy."""
        workflow_dir = tmp_path / "workflow"

        exit_code = main(
            [
                "--workflow-dir",
                str(workflow_dir),
                "--env",
                "gossip",
                "--n-nodes",
                "3",
                "--topology",
                "ring",
                "--strategy",
                "gossip_then_local",
                "--steps",
                "3",
                "--seed",
                "0",
                "--dim",
                "5",
            ]
        )

        assert exit_code == 0

        exp_dirs = list(workflow_dir.glob("exp_*"))
        exp_dir = exp_dirs[0]

        with (exp_dir / "artifacts" / "summary.json").open() as f:
            summary = json.load(f)

        assert summary["strategy"] == "gossip_then_local"

    def test_strategies_produce_different_results(self, tmp_path: Path) -> None:
        """Different strategies should produce different final parameters."""
        workflow_dir = tmp_path / "workflow"

        # Run with local_then_gossip
        main(
            [
                "--workflow-dir",
                str(workflow_dir),
                "--env",
                "gossip",
                "--n-nodes",
                "3",
                "--topology",
                "ring",
                "--strategy",
                "local_then_gossip",
                "--steps",
                "5",
                "--seed",
                "42",
                "--dim",
                "5",
            ]
        )

        exp_dir_1 = list(workflow_dir.glob("exp_*"))[0]
        with (exp_dir_1 / "artifacts" / "summary.json").open() as f:
            summary_1 = json.load(f)

        # Run with gossip_then_local
        main(
            [
                "--workflow-dir",
                str(workflow_dir),
                "--env",
                "gossip",
                "--n-nodes",
                "3",
                "--topology",
                "ring",
                "--strategy",
                "gossip_then_local",
                "--steps",
                "5",
                "--seed",
                "42",
                "--dim",
                "5",
            ]
        )

        exp_dirs = sorted(workflow_dir.glob("exp_*"))
        exp_dir_2 = exp_dirs[1]
        with (exp_dir_2 / "artifacts" / "summary.json").open() as f:
            summary_2 = json.load(f)

        # Different strategies should generally produce different results
        # (with same seed, problem is the same, but execution order differs)
        assert summary_1["final_mean_loss"] != summary_2["final_mean_loss"]


# =============================================================================
# Tests for matrix mode
# =============================================================================


class TestMatrixMode:
    """Tests for matrix mode experiment grid."""

    def test_matrix_mode_creates_artifacts(self, tmp_path: Path) -> None:
        """Matrix mode should create results.csv and run directories."""
        workflow_dir = tmp_path / "workflow"

        exit_code = main(
            [
                "--mode",
                "matrix",
                "--matrix",
                "small",
                "--env",
                "gossip",
                "--steps",
                "5",
                "--n-nodes",
                "3",
                "--dim",
                "5",
                "--cond",
                "5",
                "--radius",
                "2.0",
                "--seeds",
                "0",
                "--workflow-dir",
                str(workflow_dir),
            ]
        )

        assert exit_code == 0

        # Find created experiment directory
        exp_dirs = list(workflow_dir.glob("exp_*"))
        assert len(exp_dirs) == 1
        exp_dir = exp_dirs[0]

        # Check artifacts/results.csv exists
        results_csv = exp_dir / "artifacts" / "results.csv"
        assert results_csv.exists()

        # Check artifacts/runs/ exists
        runs_dir = exp_dir / "artifacts" / "runs"
        assert runs_dir.exists()

        # For small matrix with 1 seed:
        # fw: 2 topologies * 2 strategies * 2 schedules = 8 runs
        # pgd: 2 topologies * 2 strategies * 1 (no schedule) = 4 runs
        # Total: 12 runs
        run_dirs = sorted(runs_dir.glob("run_*"))
        assert len(run_dirs) == 12

        # Each run dir should have config.json and summary.json
        for run_dir in run_dirs:
            assert (run_dir / "config.json").exists()
            assert (run_dir / "summary.json").exists()

    def test_matrix_mode_csv_columns(self, tmp_path: Path) -> None:
        """CSV should have exact required columns."""
        workflow_dir = tmp_path / "workflow"

        main(
            [
                "--mode",
                "matrix",
                "--matrix",
                "small",
                "--env",
                "gossip",
                "--steps",
                "3",
                "--n-nodes",
                "3",
                "--dim",
                "3",
                "--seeds",
                "0",
                "--workflow-dir",
                str(workflow_dir),
            ]
        )

        exp_dir = list(workflow_dir.glob("exp_*"))[0]
        results_csv = exp_dir / "artifacts" / "results.csv"

        with results_csv.open() as f:
            reader = csv.reader(f)
            header = next(reader)

        # Check exact column order
        assert header == CSV_COLUMNS

    def test_matrix_mode_multiple_seeds(self, tmp_path: Path) -> None:
        """Matrix mode with multiple seeds should create more runs."""
        workflow_dir = tmp_path / "workflow"

        exit_code = main(
            [
                "--mode",
                "matrix",
                "--matrix",
                "small",
                "--env",
                "gossip",
                "--steps",
                "3",
                "--n-nodes",
                "3",
                "--dim",
                "3",
                "--seeds",
                "0,1",
                "--workflow-dir",
                str(workflow_dir),
            ]
        )

        assert exit_code == 0

        exp_dir = list(workflow_dir.glob("exp_*"))[0]
        runs_dir = exp_dir / "artifacts" / "runs"

        # fw: 2 topologies * 2 strategies * 2 schedules * 2 seeds = 16 runs
        # pgd: 2 topologies * 2 strategies * 1 * 2 seeds = 8 runs
        # Total: 24 runs
        run_dirs = list(runs_dir.glob("run_*"))
        assert len(run_dirs) == 24

    def test_matrix_mode_save_histories(self, tmp_path: Path) -> None:
        """With --save-histories, each run should have history.jsonl."""
        workflow_dir = tmp_path / "workflow"

        exit_code = main(
            [
                "--mode",
                "matrix",
                "--matrix",
                "small",
                "--env",
                "gossip",
                "--steps",
                "5",
                "--n-nodes",
                "3",
                "--dim",
                "3",
                "--seeds",
                "0",
                "--save-histories",
                "--workflow-dir",
                str(workflow_dir),
            ]
        )

        assert exit_code == 0

        exp_dir = list(workflow_dir.glob("exp_*"))[0]
        runs_dir = exp_dir / "artifacts" / "runs"

        for run_dir in runs_dir.glob("run_*"):
            history_path = run_dir / "history.jsonl"
            assert history_path.exists()

            with history_path.open() as f:
                lines = f.readlines()
            assert len(lines) == 5

    def test_matrix_mode_without_save_histories(self, tmp_path: Path) -> None:
        """Without --save-histories, runs should not have history.jsonl."""
        workflow_dir = tmp_path / "workflow"

        exit_code = main(
            [
                "--mode",
                "matrix",
                "--matrix",
                "small",
                "--env",
                "gossip",
                "--steps",
                "3",
                "--n-nodes",
                "3",
                "--dim",
                "3",
                "--seeds",
                "0",
                "--workflow-dir",
                str(workflow_dir),
            ]
        )

        assert exit_code == 0

        exp_dir = list(workflow_dir.glob("exp_*"))[0]
        runs_dir = exp_dir / "artifacts" / "runs"

        for run_dir in runs_dir.glob("run_*"):
            history_path = run_dir / "history.jsonl"
            assert not history_path.exists()

    def test_matrix_mode_global_summary(self, tmp_path: Path) -> None:
        """Matrix mode should create global summary.json with statistics."""
        workflow_dir = tmp_path / "workflow"

        main(
            [
                "--mode",
                "matrix",
                "--matrix",
                "small",
                "--env",
                "gossip",
                "--steps",
                "3",
                "--n-nodes",
                "3",
                "--dim",
                "3",
                "--seeds",
                "0",
                "--workflow-dir",
                str(workflow_dir),
            ]
        )

        exp_dir = list(workflow_dir.glob("exp_*"))[0]
        summary_path = exp_dir / "artifacts" / "summary.json"

        with summary_path.open() as f:
            summary = json.load(f)

        assert "number_of_runs" in summary
        # fw: 2*2*2 = 8, pgd: 2*2*1 = 4, total = 12
        assert summary["number_of_runs"] == 12
        assert "best_by_final_suboptimality" in summary
        assert "best_by_final_mean_loss" in summary
        assert "averages" in summary
        assert "mean_final_suboptimality" in summary["averages"]
        assert "mean_final_consensus_error" in summary["averages"]

    def test_matrix_mode_csv_row_count(self, tmp_path: Path) -> None:
        """CSV should have correct number of data rows."""
        workflow_dir = tmp_path / "workflow"

        main(
            [
                "--mode",
                "matrix",
                "--matrix",
                "small",
                "--env",
                "gossip",
                "--steps",
                "3",
                "--n-nodes",
                "3",
                "--dim",
                "3",
                "--seeds",
                "0,1,2",
                "--workflow-dir",
                str(workflow_dir),
            ]
        )

        exp_dir = list(workflow_dir.glob("exp_*"))[0]
        results_csv = exp_dir / "artifacts" / "results.csv"

        with results_csv.open() as f:
            reader = csv.reader(f)
            rows = list(reader)

        # fw: 2*2*2*3 = 24 runs, pgd: 2*2*1*3 = 12 runs, total = 36 data rows
        assert len(rows) == 37  # 1 header + 36 data rows
