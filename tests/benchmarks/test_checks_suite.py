"""Tests for the convergence check suite.

This module tests:
- Check suite passes for stable configurations
- Gossip consensus check passes for complete topology
- Failing scenarios are detected correctly
- Runner --mode checks creates expected artifacts
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.checks import (
    CheckResult,
    ChecksSummary,
    check_constraint_feasibility,
    check_gossip_consensus_decreases,
    check_single_decreases_suboptimality,
    run_checks,
)
from benchmarks.legacy_runner import main

# =============================================================================
# Tests for CheckResult and ChecksSummary
# =============================================================================


class TestCheckDataclasses:
    """Tests for check result dataclasses."""

    def test_check_result_to_dict(self) -> None:
        """CheckResult should convert to dict correctly."""
        result = CheckResult(
            name="test_check",
            passed=True,
            details={"value": 1.0, "threshold": 0.5},
        )
        d = result.to_dict()

        assert d["name"] == "test_check"
        assert d["passed"] is True
        assert d["details"]["value"] == 1.0

    def test_checks_summary_to_json(self) -> None:
        """ChecksSummary should convert to JSON dict correctly."""
        results = [
            CheckResult(name="check1", passed=True, details={}),
            CheckResult(name="check2", passed=False, details={"error": "failed"}),
        ]
        summary = ChecksSummary(passed=False, results=results)
        j = summary.to_json()

        assert j["passed"] is False
        assert j["num_checks"] == 2
        assert j["num_passed"] == 1
        assert j["num_failed"] == 1
        assert len(j["results"]) == 2


# =============================================================================
# Tests for individual checks
# =============================================================================


class TestSuboptimalityCheck:
    """Tests for the suboptimality decrease check."""

    def test_gd_decreases_suboptimality(self) -> None:
        """GD should decrease suboptimality on quadratic."""
        config = {
            "optimizer": "gd",
            "dim": 5,
            "cond": 5.0,
            "seed": 0,
            "steps": 30,
            "lr": 0.05,
            "constraint": "l2ball",
            "radius": 1.0,
            "step_schedule": "harmonic",
            "gamma": 0.2,
        }
        result = check_single_decreases_suboptimality(config)

        assert result.passed is True
        assert result.details["ratio_achieved"] < result.details["ratio_threshold"]

    def test_fw_decreases_suboptimality(self) -> None:
        """FW should decrease suboptimality on quadratic."""
        config = {
            "optimizer": "fw",
            "dim": 5,
            "cond": 5.0,
            "seed": 0,
            "steps": 30,
            "constraint": "l2ball",
            "radius": 2.0,
            "step_schedule": "harmonic",
            "gamma": 0.2,
            "lr": 0.1,
        }
        result = check_single_decreases_suboptimality(config)

        assert result.passed is True

    def test_pgd_decreases_suboptimality(self) -> None:
        """PGD should decrease suboptimality on quadratic."""
        config = {
            "optimizer": "pgd",
            "dim": 5,
            "cond": 5.0,
            "seed": 0,
            "steps": 30,
            "lr": 0.05,
            "constraint": "l2ball",
            "radius": 2.0,
            "step_schedule": "harmonic",
            "gamma": 0.2,
        }
        result = check_single_decreases_suboptimality(config)

        assert result.passed is True


class TestFeasibilityCheck:
    """Tests for the constraint feasibility check."""

    def test_fw_preserves_l2ball_feasibility(self) -> None:
        """FW should preserve L2 ball feasibility."""
        config = {
            "env": "single",
            "optimizer": "fw",
            "dim": 5,
            "cond": 5.0,
            "seed": 0,
            "steps": 30,
            "constraint": "l2ball",
            "radius": 1.0,
            "step_schedule": "harmonic",
            "gamma": 0.2,
            "lr": 0.1,
        }
        result = check_constraint_feasibility(config)

        assert result.passed is True
        assert result.details["final_norm"] <= result.details["bound"]

    def test_pgd_preserves_l2ball_feasibility(self) -> None:
        """PGD should preserve L2 ball feasibility."""
        config = {
            "env": "single",
            "optimizer": "pgd",
            "dim": 5,
            "cond": 5.0,
            "seed": 0,
            "steps": 30,
            "lr": 0.1,
            "constraint": "l2ball",
            "radius": 1.0,
            "step_schedule": "harmonic",
            "gamma": 0.2,
        }
        result = check_constraint_feasibility(config)

        assert result.passed is True

    def test_gd_skips_feasibility_check(self) -> None:
        """GD should skip feasibility check (no constraint)."""
        config = {
            "env": "single",
            "optimizer": "gd",
            "dim": 5,
            "cond": 5.0,
            "seed": 0,
            "steps": 30,
            "lr": 0.1,
            "constraint": "l2ball",
            "radius": 1.0,
            "step_schedule": "harmonic",
            "gamma": 0.2,
        }
        result = check_constraint_feasibility(config)

        assert result.passed is True
        assert result.details.get("skipped") is True


class TestConsensusCheck:
    """Tests for the gossip consensus check."""

    def test_complete_topology_achieves_consensus(self) -> None:
        """Complete topology should achieve consensus quickly."""
        config = {
            "env": "gossip",
            "optimizer": "fw",
            "dim": 5,
            "cond": 5.0,
            "seed": 0,
            "steps": 30,
            "n_nodes": 4,
            "topology": "complete",
            "strategy": "local_then_gossip",
            "constraint": "l2ball",
            "radius": 2.0,
            "step_schedule": "harmonic",
            "gamma": 0.2,
            "lr": 0.1,
        }
        result = check_gossip_consensus_decreases(config)

        assert result.passed is True
        assert result.details["final_consensus_error"] < result.details["initial_consensus_error"]

    def test_ring_topology_achieves_consensus(self) -> None:
        """Ring topology should achieve consensus (slower than complete)."""
        config = {
            "env": "gossip",
            "optimizer": "fw",
            "dim": 5,
            "cond": 5.0,
            "seed": 0,
            "steps": 30,
            "n_nodes": 4,
            "topology": "ring",
            "strategy": "local_then_gossip",
            "constraint": "l2ball",
            "radius": 2.0,
            "step_schedule": "harmonic",
            "gamma": 0.2,
            "lr": 0.1,
        }
        result = check_gossip_consensus_decreases(config)

        assert result.passed is True

    def test_single_env_skips_consensus_check(self) -> None:
        """Single env should skip consensus check."""
        config = {
            "env": "single",
            "optimizer": "fw",
            "dim": 5,
            "cond": 5.0,
            "seed": 0,
            "steps": 30,
            "constraint": "l2ball",
            "radius": 2.0,
            "step_schedule": "harmonic",
            "gamma": 0.2,
            "lr": 0.1,
        }
        result = check_gossip_consensus_decreases(config)

        assert result.passed is True
        assert result.details.get("skipped") is True


# =============================================================================
# Tests for run_checks
# =============================================================================


class TestRunChecks:
    """Tests for the run_checks function."""

    def test_stable_single_gd_passes(self) -> None:
        """Stable single GD config should pass all checks."""
        config = {
            "env": "single",
            "optimizer": "gd",
            "dim": 5,
            "cond": 5.0,
            "seed": 0,
            "steps": 30,
            "lr": 0.05,
            "constraint": "l2ball",
            "radius": 1.0,
            "step_schedule": "harmonic",
            "gamma": 0.2,
        }
        summary = run_checks(config)

        assert summary.passed is True
        # Should have suboptimality check (gd skips feasibility)
        assert len(summary.results) >= 1

    def test_gossip_complete_passes(self) -> None:
        """Gossip with complete topology should pass all checks."""
        config = {
            "env": "gossip",
            "optimizer": "fw",
            "dim": 5,
            "cond": 5.0,
            "seed": 0,
            "steps": 30,
            "n_nodes": 4,
            "topology": "complete",
            "strategy": "local_then_gossip",
            "constraint": "l2ball",
            "radius": 2.0,
            "step_schedule": "harmonic",
            "gamma": 0.2,
            "lr": 0.1,
        }
        summary = run_checks(config)

        assert summary.passed is True
        # Should have suboptimality, feasibility, and consensus checks
        assert len(summary.results) == 3

        # Find consensus check
        consensus_check = next(
            (r for r in summary.results if r.name == "gossip_consensus_decreases"), None
        )
        assert consensus_check is not None
        assert consensus_check.passed is True


class TestFailingScenarios:
    """Tests for detecting failing scenarios."""

    def test_large_lr_causes_failure(self) -> None:
        """Very large learning rate should cause check failure."""
        config = {
            "env": "single",
            "optimizer": "gd",
            "dim": 5,
            "cond": 5.0,
            "seed": 0,
            "steps": 10,
            "lr": 10.0,  # Too large - will diverge
            "constraint": "l2ball",
            "radius": 1.0,
            "step_schedule": "harmonic",
            "gamma": 0.2,
        }
        summary = run_checks(config)

        # Should fail due to divergence (NaN/inf or increased suboptimality)
        assert summary.passed is False

        # Find suboptimality check
        subopt_check = next(
            (r for r in summary.results if r.name == "single_decreases_suboptimality"), None
        )
        assert subopt_check is not None
        assert subopt_check.passed is False


# =============================================================================
# Tests for runner --mode checks
# =============================================================================


class TestRunnerChecksMode:
    """Tests for runner in checks mode."""

    def test_checks_mode_creates_artifacts(self, tmp_path: Path) -> None:
        """Runner --mode checks should create checks.json and checks.md."""
        workflow_dir = tmp_path / "workflow"

        exit_code = main(
            [
                "--mode",
                "checks",
                "--workflow-dir",
                str(workflow_dir),
                "--env",
                "single",
                "--optimizer",
                "gd",
                "--lr",
                "0.05",
                "--steps",
                "20",
                "--dim",
                "5",
                "--seed",
                "0",
            ]
        )

        assert exit_code == 0

        # Find experiment directory
        exp_dirs = list(workflow_dir.glob("exp_*"))
        assert len(exp_dirs) == 1
        exp_dir = exp_dirs[0]

        # Check artifacts exist
        checks_json = exp_dir / "artifacts" / "checks.json"
        checks_md = exp_dir / "artifacts" / "checks.md"

        assert checks_json.exists()
        assert checks_md.exists()

        # Validate checks.json content
        with checks_json.open() as f:
            data = json.load(f)

        assert "passed" in data
        assert "num_checks" in data
        assert "results" in data

    def test_checks_mode_gossip(self, tmp_path: Path) -> None:
        """Runner --mode checks should work with gossip env."""
        workflow_dir = tmp_path / "workflow"

        exit_code = main(
            [
                "--mode",
                "checks",
                "--workflow-dir",
                str(workflow_dir),
                "--env",
                "gossip",
                "--optimizer",
                "fw",
                "--n-nodes",
                "3",
                "--topology",
                "complete",
                "--strategy",
                "local_then_gossip",
                "--steps",
                "20",
                "--dim",
                "5",
                "--seed",
                "0",
            ]
        )

        assert exit_code == 0

        exp_dir = list(workflow_dir.glob("exp_*"))[0]
        checks_json = exp_dir / "artifacts" / "checks.json"

        with checks_json.open() as f:
            data = json.load(f)

        assert data["passed"] is True
        # Should have 3 checks for gossip with FW
        assert data["num_checks"] == 3

    def test_checks_mode_returns_exit_code_2_on_failure(self, tmp_path: Path) -> None:
        """Runner should return exit code 2 when checks fail."""
        workflow_dir = tmp_path / "workflow"

        exit_code = main(
            [
                "--mode",
                "checks",
                "--workflow-dir",
                str(workflow_dir),
                "--env",
                "single",
                "--optimizer",
                "gd",
                "--lr",
                "10.0",  # Too large - will fail
                "--steps",
                "10",
                "--dim",
                "5",
                "--seed",
                "0",
            ]
        )

        assert exit_code == 2

        exp_dir = list(workflow_dir.glob("exp_*"))[0]
        checks_json = exp_dir / "artifacts" / "checks.json"

        with checks_json.open() as f:
            data = json.load(f)

        assert data["passed"] is False


class TestMatrixReportGeneration:
    """Tests for matrix mode report generation."""

    def test_matrix_mode_generates_report_md(self, tmp_path: Path) -> None:
        """Matrix mode should generate report.md."""
        workflow_dir = tmp_path / "workflow"

        exit_code = main(
            [
                "--mode",
                "matrix",
                "--matrix",
                "small",
                "--workflow-dir",
                str(workflow_dir),
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
            ]
        )

        assert exit_code == 0

        exp_dir = list(workflow_dir.glob("exp_*"))[0]
        report_md = exp_dir / "artifacts" / "report.md"

        assert report_md.exists()

        # Check content
        content = report_md.read_text()
        assert "# Matrix Experiment Report" in content
        assert "## Best Runs" in content
        assert "## Top 10 Runs" in content
        assert "## Statistics by Group" in content
