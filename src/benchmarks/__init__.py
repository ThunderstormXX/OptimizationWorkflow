"""Benchmarks module for experiment management.

This package provides utilities for running, tracking, and analyzing
optimization experiments:

- workflow: Experiment directory management
- metrics: Metric computation helpers
- runner: CLI for running experiments
- checks: Convergence/regression check suite
- report: Markdown report generation
"""

from __future__ import annotations

from benchmarks.checks import CheckResult, ChecksSummary, run_checks
from benchmarks.metrics import consensus_error, mean_params, suboptimality
from benchmarks.report import render_checks_markdown, render_matrix_markdown
from benchmarks.runner import main as run_experiment
from benchmarks.workflow import next_experiment_dir, try_get_git_commit, write_run_files

__all__ = [
    # Workflow
    "next_experiment_dir",
    "write_run_files",
    "try_get_git_commit",
    # Metrics
    "suboptimality",
    "consensus_error",
    "mean_params",
    # Runner
    "run_experiment",
    # Checks
    "CheckResult",
    "ChecksSummary",
    "run_checks",
    # Report
    "render_checks_markdown",
    "render_matrix_markdown",
]
