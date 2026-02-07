"""Markdown report generation for checks and matrix experiments.

This module provides functions to render check results and matrix experiment
summaries as human-readable Markdown reports with embedded plots.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from benchmarks.checks import ChecksSummary

__all__ = [
    "render_checks_markdown",
    "render_matrix_markdown",
    "render_single_run_report_md",
    "render_matrix_visual_report_md",
]


def _compute_mean_std(values: list[float]) -> tuple[float, float]:
    """Compute mean and standard deviation of a list of values."""
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std = variance**0.5
    return float(mean), float(std)


def render_checks_markdown(summary: ChecksSummary, config: dict[str, Any]) -> str:
    """Render checks summary as Markdown.

    Args:
        summary: ChecksSummary from run_checks().
        config: Configuration dictionary.

    Returns:
        Markdown string.
    """
    lines: list[str] = []

    # Title
    status = "✅ PASSED" if summary.passed else "❌ FAILED"
    lines.append(f"# Convergence Check Report — {status}")
    lines.append("")

    # Config summary
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- **Environment**: {config.get('env', 'single')}")
    lines.append(f"- **Optimizer**: {config.get('optimizer', 'fw')}")
    lines.append(f"- **Steps**: {config.get('steps', 30)}")
    lines.append(f"- **Seed**: {config.get('seed', 0)}")
    lines.append(f"- **Dimension**: {config.get('dim', 10)}")
    lines.append(f"- **Condition number**: {config.get('cond', 10.0)}")

    if config.get("optimizer") in ("fw", "pgd"):
        lines.append(f"- **Constraint**: {config.get('constraint', 'l2ball')}")
        if config.get("constraint") == "l2ball":
            lines.append(f"- **Radius**: {config.get('radius', 1.0)}")

    if config.get("optimizer") in ("gd", "pgd"):
        lines.append(f"- **Learning rate**: {config.get('lr', 0.1)}")

    if config.get("optimizer") == "fw":
        lines.append(f"- **Step schedule**: {config.get('step_schedule', 'harmonic')}")
        if config.get("step_schedule") == "constant":
            lines.append(f"- **Gamma**: {config.get('gamma', 0.2)}")

    if config.get("env") == "gossip":
        lines.append(f"- **Nodes**: {config.get('n_nodes', 5)}")
        lines.append(f"- **Topology**: {config.get('topology', 'ring')}")
        lines.append(f"- **Strategy**: {config.get('strategy', 'local_then_gossip')}")

    lines.append("")

    # Results table
    lines.append("## Check Results")
    lines.append("")
    lines.append("| Check | Status | Key Metrics |")
    lines.append("|-------|--------|-------------|")

    for result in summary.results:
        status_icon = "✅" if result.passed else "❌"
        details = result.details

        # Skip skipped checks with minimal info
        if details.get("skipped"):
            lines.append(f"| {result.name} | ⏭️ Skipped | {details.get('reason', 'N/A')} |")
            continue

        # Format key metrics based on check type
        if result.name == "single_decreases_suboptimality":
            metrics = (
                f"initial={details.get('initial_suboptimality', 0):.4f}, "
                f"final={details.get('final_suboptimality', 0):.4f}, "
                f"ratio={details.get('ratio_achieved', 0):.3f} "
                f"(threshold={details.get('ratio_threshold', 0):.2f})"
            )
        elif result.name == "constraint_feasibility":
            if "max_norm" in details:
                metrics = (
                    f"max_norm={details.get('max_norm', 0):.6f}, "
                    f"bound={details.get('bound', 0):.6f}"
                )
            else:
                metrics = (
                    f"final_norm={details.get('final_norm', 0):.6f}, "
                    f"bound={details.get('bound', 0):.6f}"
                )
        elif result.name == "gossip_consensus_decreases":
            metrics = (
                f"initial={details.get('initial_consensus_error', 0):.6f}, "
                f"final={details.get('final_consensus_error', 0):.6f}, "
                f"ratio={details.get('ratio_achieved', 0):.3f} "
                f"(threshold={details.get('ratio_threshold', 0):.2f})"
            )
        else:
            # Generic fallback
            metrics = ", ".join(f"{k}={v}" for k, v in details.items() if k != "error")

        lines.append(f"| {result.name} | {status_icon} | {metrics} |")

    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    num_passed = sum(1 for r in summary.results if r.passed)
    num_failed = sum(1 for r in summary.results if not r.passed)
    lines.append(f"- **Total checks**: {len(summary.results)}")
    lines.append(f"- **Passed**: {num_passed}")
    lines.append(f"- **Failed**: {num_failed}")
    lines.append("")

    if summary.passed:
        lines.append("**Overall: ✅ ALL CHECKS PASSED**")
    else:
        lines.append("**Overall: ❌ SOME CHECKS FAILED**")
        lines.append("")
        lines.append("Failed checks:")
        for result in summary.results:
            if not result.passed:
                error = result.details.get("error", "See details above")
                lines.append(f"- `{result.name}`: {error}")

    lines.append("")

    return "\n".join(lines)


def render_matrix_markdown(results_csv_path: Path, global_summary_json_path: Path) -> str:
    """Render matrix experiment results as Markdown.

    Args:
        results_csv_path: Path to results.csv.
        global_summary_json_path: Path to summary.json.

    Returns:
        Markdown string.
    """
    lines: list[str] = []

    # Load data
    with results_csv_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    with global_summary_json_path.open() as f:
        global_summary = json.load(f)

    # Title
    lines.append("# Matrix Experiment Report")
    lines.append("")

    # Overview
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- **Number of runs**: {global_summary.get('number_of_runs', len(rows))}")
    lines.append(f"- **Artifacts directory**: `{global_summary.get('runs_directory', 'runs/')}`")
    lines.append("")

    # Best runs
    lines.append("## Best Runs")
    lines.append("")

    best_subopt = global_summary.get("best_by_final_suboptimality", {})
    if best_subopt:
        lines.append("### Best by Final Suboptimality")
        lines.append("")
        lines.append(f"- **Run ID**: `{best_subopt.get('run_id', 'N/A')}`")
        lines.append(f"- **Optimizer**: {best_subopt.get('optimizer', 'N/A')}")
        lines.append(f"- **Topology**: {best_subopt.get('topology', 'N/A')}")
        lines.append(f"- **Strategy**: {best_subopt.get('strategy', 'N/A')}")
        lines.append(f"- **Step schedule**: {best_subopt.get('step_schedule', 'N/A')}")
        lines.append(f"- **Seed**: {best_subopt.get('seed', 'N/A')}")
        lines.append(f"- **Final suboptimality**: {best_subopt.get('final_suboptimality', 0):.6f}")
        lines.append("")

    best_loss = global_summary.get("best_by_final_mean_loss", {})
    if best_loss:
        lines.append("### Best by Final Mean Loss")
        lines.append("")
        lines.append(f"- **Run ID**: `{best_loss.get('run_id', 'N/A')}`")
        lines.append(f"- **Optimizer**: {best_loss.get('optimizer', 'N/A')}")
        lines.append(f"- **Topology**: {best_loss.get('topology', 'N/A')}")
        lines.append(f"- **Strategy**: {best_loss.get('strategy', 'N/A')}")
        lines.append(f"- **Step schedule**: {best_loss.get('step_schedule', 'N/A')}")
        lines.append(f"- **Seed**: {best_loss.get('seed', 'N/A')}")
        lines.append(f"- **Final mean loss**: {best_loss.get('final_mean_loss', 0):.6f}")
        lines.append("")

    # Top-5 runs by suboptimality
    lines.append("## Top 5 Runs by Final Suboptimality")
    lines.append("")
    lines.append(
        "| Rank | Run ID | Optimizer | Topology | Strategy | Schedule | Seed | Suboptimality |"
    )
    lines.append(
        "|------|--------|-----------|----------|----------|----------|------|---------------|"
    )

    # Sort rows by final_suboptimality
    sorted_rows = sorted(
        rows,
        key=lambda r: float(r.get("final_suboptimality", float("inf"))),
    )

    for i, row in enumerate(sorted_rows[:5], 1):
        run_id = row.get("run_id", "N/A")
        optimizer = row.get("optimizer", "N/A")
        topology = row.get("topology", "N/A")
        strategy = row.get("strategy", "N/A")
        schedule = row.get("schedule", "N/A")
        seed = row.get("seed", "N/A")
        subopt = float(row.get("final_suboptimality", 0))
        line = (
            f"| {i} | `{run_id}` | {optimizer} | {topology} | {strategy} "
            f"| {schedule} | {seed} | {subopt:.6f} |"
        )
        lines.append(line)

    lines.append("")

    # Averages
    lines.append("## Averages")
    lines.append("")

    averages = global_summary.get("averages", {})
    if averages:
        lines.append(
            f"- **Mean final suboptimality**: {averages.get('mean_final_suboptimality', 0):.6f}"
        )
        lines.append(f"- **Mean final mean loss**: {averages.get('mean_final_mean_loss', 0):.6f}")
        if "mean_final_consensus_error" in averages:
            mean_cons = averages.get("mean_final_consensus_error", 0)
            lines.append(f"- **Mean final consensus error**: {mean_cons:.6f}")
        lines.append("")

    # Means by optimizer
    lines.append("## Means by Optimizer")
    lines.append("")

    # Group by optimizer
    optimizer_groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        opt = row.get("optimizer", "unknown")
        if opt not in optimizer_groups:
            optimizer_groups[opt] = []
        optimizer_groups[opt].append(row)

    lines.append("| Optimizer | Runs | Mean Suboptimality | Mean Consensus Error |")
    lines.append("|-----------|------|--------------------|-----------------------|")

    for opt, opt_rows in sorted(optimizer_groups.items()):
        n_runs = len(opt_rows)
        subopt_values = [float(r.get("final_suboptimality", 0)) for r in opt_rows]
        mean_subopt = sum(subopt_values) / len(subopt_values) if subopt_values else 0

        cons_values = [
            float(r.get("final_consensus_error", 0))
            for r in opt_rows
            if r.get("final_consensus_error", "")
        ]
        mean_cons = sum(cons_values) / len(cons_values) if cons_values else 0

        cons_str = f"{mean_cons:.6f}" if cons_values else "N/A"
        lines.append(f"| {opt} | {n_runs} | {mean_subopt:.6f} | {cons_str} |")

    lines.append("")

    # Means by topology (for gossip)
    if any(r.get("topology") for r in rows):
        lines.append("## Means by Topology")
        lines.append("")

        topology_groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            topo = row.get("topology", "")
            if topo:
                if topo not in topology_groups:
                    topology_groups[topo] = []
                topology_groups[topo].append(row)

        lines.append("| Topology | Runs | Mean Suboptimality | Mean Consensus Error |")
        lines.append("|----------|------|--------------------|-----------------------|")

        for topo, topo_rows in sorted(topology_groups.items()):
            n_runs = len(topo_rows)
            subopt_values = [float(r.get("final_suboptimality", 0)) for r in topo_rows]
            mean_subopt = sum(subopt_values) / len(subopt_values) if subopt_values else 0

            cons_values = [
                float(r.get("final_consensus_error", 0))
                for r in topo_rows
                if r.get("final_consensus_error", "")
            ]
            mean_cons = sum(cons_values) / len(cons_values) if cons_values else 0

            cons_str = f"{mean_cons:.6f}" if cons_values else "N/A"
            lines.append(f"| {topo} | {n_runs} | {mean_subopt:.6f} | {cons_str} |")

        lines.append("")

    # Means by strategy (for gossip)
    if any(r.get("strategy") for r in rows):
        lines.append("## Means by Strategy")
        lines.append("")

        strategy_groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            strat = row.get("strategy", "")
            if strat:
                if strat not in strategy_groups:
                    strategy_groups[strat] = []
                strategy_groups[strat].append(row)

        lines.append("| Strategy | Runs | Mean Suboptimality | Mean Consensus Error |")
        lines.append("|----------|------|--------------------|-----------------------|")

        for strat, strat_rows in sorted(strategy_groups.items()):
            n_runs = len(strat_rows)
            subopt_values = [float(r.get("final_suboptimality", 0)) for r in strat_rows]
            mean_subopt = sum(subopt_values) / len(subopt_values) if subopt_values else 0

            cons_values = [
                float(r.get("final_consensus_error", 0))
                for r in strat_rows
                if r.get("final_consensus_error", "")
            ]
            mean_cons = sum(cons_values) / len(cons_values) if cons_values else 0

            cons_str = f"{mean_cons:.6f}" if cons_values else "N/A"
            lines.append(f"| {strat} | {n_runs} | {mean_subopt:.6f} | {cons_str} |")

        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Visual report generation (with embedded plots)
# =============================================================================


def render_single_run_report_md(
    exp_dir: Path,
    *,
    summary: dict[str, Any],
    plots: dict[str, Path],
) -> str:
    """Render a single run report with embedded plots.

    Args:
        exp_dir: Experiment directory (for computing relative paths).
        summary: Summary dictionary from summary.json.
        plots: Dictionary mapping plot name to absolute path.

    Returns:
        Markdown string with embedded plot images.
    """
    lines: list[str] = []

    # Title
    lines.append("# Experiment Report")
    lines.append("")

    # Configuration
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- **Environment**: {summary.get('env', 'single')}")
    lines.append(f"- **Task**: {summary.get('task', 'quadratic')}")
    lines.append(f"- **Optimizer**: {summary.get('optimizer', 'N/A')}")
    lines.append(f"- **Steps**: {summary.get('steps', 'N/A')}")
    lines.append(f"- **Seed**: {summary.get('seed', 'N/A')}")
    lines.append(f"- **Dimension**: {summary.get('dim', 'N/A')}")

    if summary.get("env") == "gossip":
        lines.append(f"- **Nodes**: {summary.get('n_nodes', 'N/A')}")
        lines.append(f"- **Topology**: {summary.get('topology', 'N/A')}")
        lines.append(f"- **Strategy**: {summary.get('strategy', 'N/A')}")

    if summary.get("task") == "logistic":
        lines.append(f"- **Samples**: {summary.get('n_samples', 'N/A')}")
        lines.append(f"- **Batch size**: {summary.get('batch_size', 'N/A')}")
        if "heterogeneity" in summary:
            lines.append(f"- **Heterogeneity**: {summary.get('heterogeneity', 'N/A')}")

    lines.append("")

    # Final Metrics
    lines.append("## Final Metrics")
    lines.append("")
    lines.append(f"- **Final mean loss**: {summary.get('final_mean_loss', 'N/A'):.6f}")

    if "final_suboptimality" in summary:
        lines.append(f"- **Final suboptimality**: {summary.get('final_suboptimality', 0):.6f}")

    if "final_dist_to_opt" in summary:
        lines.append(f"- **Final distance to optimum**: {summary.get('final_dist_to_opt', 0):.6f}")

    if "final_accuracy" in summary:
        lines.append(f"- **Final accuracy**: {summary.get('final_accuracy', 0):.4f}")

    if "final_mean_accuracy" in summary:
        lines.append(f"- **Final mean accuracy**: {summary.get('final_mean_accuracy', 0):.4f}")

    if "final_consensus_error" in summary:
        lines.append(f"- **Final consensus error**: {summary.get('final_consensus_error', 0):.6f}")

    lines.append("")

    # Plots section
    if plots:
        lines.append("## Plots")
        lines.append("")

        # Compute relative paths from exp_dir/artifacts to plots
        artifacts_dir = exp_dir / "artifacts"

        if "loss" in plots:
            rel_path = plots["loss"].relative_to(artifacts_dir)
            lines.append("### Loss vs Step")
            lines.append("")
            lines.append(f"![Loss]({rel_path})")
            lines.append("")

        if "suboptimality" in plots:
            rel_path = plots["suboptimality"].relative_to(artifacts_dir)
            lines.append("### Suboptimality vs Step")
            lines.append("")
            lines.append(f"![Suboptimality]({rel_path})")
            lines.append("")

        if "accuracy" in plots:
            rel_path = plots["accuracy"].relative_to(artifacts_dir)
            lines.append("### Accuracy vs Step")
            lines.append("")
            lines.append(f"![Accuracy]({rel_path})")
            lines.append("")

        if "consensus" in plots:
            rel_path = plots["consensus"].relative_to(artifacts_dir)
            lines.append("### Consensus Error vs Step")
            lines.append("")
            lines.append(f"![Consensus]({rel_path})")
            lines.append("")

        if "budgets" in plots:
            rel_path = plots["budgets"].relative_to(artifacts_dir)
            lines.append("### Budget Usage")
            lines.append("")
            lines.append(f"![Budgets]({rel_path})")
            lines.append("")

    return "\n".join(lines)


def render_matrix_visual_report_md(
    exp_dir: Path,
    *,
    global_summary: dict[str, Any],
    results_csv_path: Path,
    plots: dict[str, Path],
) -> str:
    """Render a matrix experiment report with embedded plots.

    Args:
        exp_dir: Experiment directory (for computing relative paths).
        global_summary: Global summary dictionary.
        results_csv_path: Path to results.csv.
        plots: Dictionary mapping plot name to absolute path.

    Returns:
        Markdown string with embedded plot images and tables.
    """
    lines: list[str] = []

    # Load results
    with results_csv_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Title
    lines.append("# Matrix Experiment Report")
    lines.append("")

    # Overview
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- **Number of runs**: {global_summary.get('number_of_runs', len(rows))}")
    lines.append("")

    # Best runs
    lines.append("## Best Runs")
    lines.append("")

    best_subopt = global_summary.get("best_by_final_suboptimality", {})
    if best_subopt and best_subopt.get("final_suboptimality") is not None:
        lines.append("### Best by Final Suboptimality")
        lines.append("")
        lines.append(f"- **Run ID**: `{best_subopt.get('run_id', 'N/A')}`")
        lines.append(f"- **Optimizer**: {best_subopt.get('optimizer', 'N/A')}")
        lines.append(f"- **Topology**: {best_subopt.get('topology', 'N/A')}")
        lines.append(f"- **Strategy**: {best_subopt.get('strategy', 'N/A')}")
        lines.append(f"- **Final suboptimality**: {best_subopt.get('final_suboptimality', 0):.6f}")
        lines.append("")

    best_loss = global_summary.get("best_by_final_mean_loss", {})
    if best_loss:
        lines.append("### Best by Final Mean Loss")
        lines.append("")
        lines.append(f"- **Run ID**: `{best_loss.get('run_id', 'N/A')}`")
        lines.append(f"- **Optimizer**: {best_loss.get('optimizer', 'N/A')}")
        lines.append(f"- **Topology**: {best_loss.get('topology', 'N/A')}")
        lines.append(f"- **Strategy**: {best_loss.get('strategy', 'N/A')}")
        lines.append(f"- **Final mean loss**: {best_loss.get('final_mean_loss', 0):.6f}")
        lines.append("")

    # Check for multiple models
    models = {row.get("model", "NumpyVector") for row in rows}
    has_multiple_models = len(models) > 1

    # Check for accuracy metric
    has_accuracy = any(row.get("final_mean_accuracy", "") not in ("", "N/A") for row in rows)

    # Best run per model (if multiple models)
    if has_multiple_models:
        lines.append("## Best Run per Model")
        lines.append("")
        lines.append(
            "| Model | Run ID | Optimizer | Constraint | Radius | Topology | Strategy | "
            "Accuracy | Loss |"
        )
        lines.append(
            "|-------|--------|-----------|------------|--------|----------|----------|"
            "----------|------|"
        )

        for model in sorted(models):
            model_rows = [r for r in rows if r.get("model", "NumpyVector") == model]
            if not model_rows:
                continue

            # Sort by accuracy (desc) then loss (asc)
            if has_accuracy:
                model_rows_sorted = sorted(
                    model_rows,
                    key=lambda r: (
                        -float(r.get("final_mean_accuracy", 0))
                        if r.get("final_mean_accuracy", "") not in ("", "N/A")
                        else 0,
                        float(r.get("final_mean_loss", float("inf")))
                        if r.get("final_mean_loss", "") not in ("", "N/A")
                        else float("inf"),
                    ),
                )
            else:
                model_rows_sorted = sorted(
                    model_rows,
                    key=lambda r: float(r.get("final_mean_loss", float("inf")))
                    if r.get("final_mean_loss", "") not in ("", "N/A")
                    else float("inf"),
                )

            best = model_rows_sorted[0]
            run_id = best.get("run_id", "N/A")
            optimizer = best.get("optimizer", "N/A")
            constraint = best.get("constraint", "N/A")
            radius = best.get("radius", "N/A")
            topology = best.get("topology", "N/A")
            strategy = best.get("strategy", "N/A")

            acc_val = best.get("final_mean_accuracy", "")
            acc_str = f"{float(acc_val):.4f}" if acc_val not in ("", "N/A") else "N/A"

            loss_val = best.get("final_mean_loss", "")
            loss_str = f"{float(loss_val):.4f}" if loss_val not in ("", "N/A") else "N/A"

            lines.append(
                f"| {model} | `{run_id}` | {optimizer} | {constraint} | {radius} | "
                f"{topology} | {strategy} | {acc_str} | {loss_str} |"
            )

        lines.append("")

    # Best run per optimizer
    optimizers = {row.get("optimizer", "N/A") for row in rows}
    if len(optimizers) > 1:
        lines.append("## Best Run per Optimizer")
        lines.append("")
        lines.append(
            "| Optimizer | Run ID | Model | Constraint | Radius | Topology | Strategy | "
            "Accuracy | Loss |"
        )
        lines.append(
            "|-----------|--------|-------|------------|--------|----------|----------|"
            "----------|------|"
        )

        for opt in sorted(optimizers):
            opt_rows = [r for r in rows if r.get("optimizer", "N/A") == opt]
            if not opt_rows:
                continue

            # Sort by accuracy (desc) then loss (asc)
            if has_accuracy:
                opt_rows_sorted = sorted(
                    opt_rows,
                    key=lambda r: (
                        -float(r.get("final_mean_accuracy", 0))
                        if r.get("final_mean_accuracy", "") not in ("", "N/A")
                        else 0,
                        float(r.get("final_mean_loss", float("inf")))
                        if r.get("final_mean_loss", "") not in ("", "N/A")
                        else float("inf"),
                    ),
                )
            else:
                opt_rows_sorted = sorted(
                    opt_rows,
                    key=lambda r: float(r.get("final_mean_loss", float("inf")))
                    if r.get("final_mean_loss", "") not in ("", "N/A")
                    else float("inf"),
                )

            best = opt_rows_sorted[0]
            run_id = best.get("run_id", "N/A")
            model = best.get("model", "NumpyVector")
            constraint = best.get("constraint", "N/A")
            radius = best.get("radius", "N/A")
            topology = best.get("topology", "N/A")
            strategy = best.get("strategy", "N/A")

            acc_val = best.get("final_mean_accuracy", "")
            acc_str = f"{float(acc_val):.4f}" if acc_val not in ("", "N/A") else "N/A"

            loss_val = best.get("final_mean_loss", "")
            loss_str = f"{float(loss_val):.4f}" if loss_val not in ("", "N/A") else "N/A"

            lines.append(
                f"| {opt} | `{run_id}` | {model} | {constraint} | {radius} | "
                f"{topology} | {strategy} | {acc_str} | {loss_str} |"
            )

        lines.append("")

    # Top-10 runs table
    lines.append("## Top 10 Runs")
    lines.append("")

    # Determine sort key based on task
    has_subopt = any(row.get("final_suboptimality", "") not in ("", "N/A") for row in rows)
    if has_accuracy:
        # For tasks with accuracy, sort by accuracy (desc) then loss (asc)
        sorted_rows = sorted(
            rows,
            key=lambda r: (
                -float(r.get("final_mean_accuracy", 0))
                if r.get("final_mean_accuracy", "") not in ("", "N/A")
                else 0,
                float(r.get("final_mean_loss", float("inf")))
                if r.get("final_mean_loss", "") not in ("", "N/A")
                else float("inf"),
            ),
        )
        sort_key = "Accuracy"
    elif has_subopt:
        sorted_rows = sorted(
            rows,
            key=lambda r: float(r.get("final_suboptimality", float("inf")))
            if r.get("final_suboptimality", "") not in ("", "N/A")
            else float("inf"),
        )
        sort_key = "Suboptimality"
    else:
        sorted_rows = sorted(
            rows,
            key=lambda r: float(r.get("final_mean_loss", float("inf")))
            if r.get("final_mean_loss", "") not in ("", "N/A")
            else float("inf"),
        )
        sort_key = "Mean Loss"

    # Include model and optimizer columns
    lines.append(f"| Rank | Run ID | Model | Optimizer | Strategy | Topology | Seed | {sort_key} |")
    lines.append("|------|--------|-------|-----------|----------|----------|------|------------|")

    for i, row in enumerate(sorted_rows[:10], 1):
        run_id = row.get("run_id", "N/A")
        model = row.get("model", "NumpyVector")
        optimizer = row.get("optimizer", "N/A")
        strategy = row.get("strategy", "N/A")
        topology = row.get("topology", "N/A")
        seed = row.get("seed", "N/A")
        if has_accuracy:
            val = row.get("final_mean_accuracy", "N/A")
            val_str = f"{float(val):.4f}" if val not in ("", "N/A") else "N/A"
        elif has_subopt:
            val = row.get("final_suboptimality", "N/A")
            val_str = f"{float(val):.6f}" if val not in ("", "N/A") else "N/A"
        else:
            val = row.get("final_mean_loss", "N/A")
            val_str = f"{float(val):.6f}" if val not in ("", "N/A") else "N/A"
        row_line = (
            f"| {i} | `{run_id}` | {model} | {optimizer} | {strategy} | "
            f"{topology} | {seed} | {val_str} |"
        )
        lines.append(row_line)

    lines.append("")

    # Grouped statistics
    lines.append("## Statistics by Group")
    lines.append("")

    # By optimizer
    lines.append("### By Optimizer")
    lines.append("")
    lines.append("| Optimizer | Runs | Mean Loss ± Std | Mean Consensus ± Std |")
    lines.append("|-----------|------|-----------------|----------------------|")

    optimizer_groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        opt = row.get("optimizer", "unknown")
        if opt not in optimizer_groups:
            optimizer_groups[opt] = []
        optimizer_groups[opt].append(row)

    for opt, opt_rows in sorted(optimizer_groups.items()):
        n_runs = len(opt_rows)
        loss_vals = [
            float(r.get("final_mean_loss", 0))
            for r in opt_rows
            if r.get("final_mean_loss", "") not in ("", "N/A")
        ]
        if loss_vals:
            mean_loss, std_loss = _compute_mean_std(loss_vals)
            loss_str = f"{mean_loss:.4f} ± {std_loss:.4f}"
        else:
            loss_str = "N/A"

        cons_vals = [
            float(r.get("final_consensus_error", 0))
            for r in opt_rows
            if r.get("final_consensus_error", "") not in ("", "N/A")
        ]
        if cons_vals:
            mean_cons, std_cons = _compute_mean_std(cons_vals)
            cons_str = f"{mean_cons:.4f} ± {std_cons:.4f}"
        else:
            cons_str = "N/A"

        lines.append(f"| {opt} | {n_runs} | {loss_str} | {cons_str} |")

    lines.append("")

    # By strategy (if gossip)
    strategy_groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        strat = row.get("strategy", "")
        if strat and strat not in ("", "N/A"):
            if strat not in strategy_groups:
                strategy_groups[strat] = []
            strategy_groups[strat].append(row)

    if strategy_groups:
        lines.append("### By Strategy")
        lines.append("")
        lines.append("| Strategy | Runs | Mean Loss ± Std | Mean Consensus ± Std |")
        lines.append("|----------|------|-----------------|----------------------|")

        for strat, strat_rows in sorted(strategy_groups.items()):
            n_runs = len(strat_rows)
            loss_vals = [
                float(r.get("final_mean_loss", 0))
                for r in strat_rows
                if r.get("final_mean_loss", "") not in ("", "N/A")
            ]
            if loss_vals:
                mean_loss, std_loss = _compute_mean_std(loss_vals)
                loss_str = f"{mean_loss:.4f} ± {std_loss:.4f}"
            else:
                loss_str = "N/A"

            cons_vals = [
                float(r.get("final_consensus_error", 0))
                for r in strat_rows
                if r.get("final_consensus_error", "") not in ("", "N/A")
            ]
            if cons_vals:
                mean_cons, std_cons = _compute_mean_std(cons_vals)
                cons_str = f"{mean_cons:.4f} ± {std_cons:.4f}"
            else:
                cons_str = "N/A"

            lines.append(f"| {strat} | {n_runs} | {loss_str} | {cons_str} |")

        lines.append("")

    # By topology (if gossip)
    topology_groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        topo = row.get("topology", "")
        if topo and topo not in ("", "N/A"):
            if topo not in topology_groups:
                topology_groups[topo] = []
            topology_groups[topo].append(row)

    if topology_groups:
        lines.append("### By Topology")
        lines.append("")
        lines.append("| Topology | Runs | Mean Loss ± Std | Mean Consensus ± Std |")
        lines.append("|----------|------|-----------------|----------------------|")

        for topo, topo_rows in sorted(topology_groups.items()):
            n_runs = len(topo_rows)
            loss_vals = [
                float(r.get("final_mean_loss", 0))
                for r in topo_rows
                if r.get("final_mean_loss", "") not in ("", "N/A")
            ]
            if loss_vals:
                mean_loss, std_loss = _compute_mean_std(loss_vals)
                loss_str = f"{mean_loss:.4f} ± {std_loss:.4f}"
            else:
                loss_str = "N/A"

            cons_vals = [
                float(r.get("final_consensus_error", 0))
                for r in topo_rows
                if r.get("final_consensus_error", "") not in ("", "N/A")
            ]
            if cons_vals:
                mean_cons, std_cons = _compute_mean_std(cons_vals)
                cons_str = f"{mean_cons:.4f} ± {std_cons:.4f}"
            else:
                cons_str = "N/A"

            lines.append(f"| {topo} | {n_runs} | {loss_str} | {cons_str} |")

        lines.append("")

    # Plots section
    if plots:
        lines.append("## Plots")
        lines.append("")

        artifacts_dir = exp_dir / "artifacts"

        if "bar_final_loss_by_optimizer" in plots:
            rel_path = plots["bar_final_loss_by_optimizer"].relative_to(artifacts_dir)
            lines.append("### Final Loss by Optimizer")
            lines.append("")
            lines.append(f"![Loss by Optimizer]({rel_path})")
            lines.append("")

        if "bar_final_loss_by_model" in plots:
            rel_path = plots["bar_final_loss_by_model"].relative_to(artifacts_dir)
            lines.append("### Final Loss by Model")
            lines.append("")
            lines.append(f"![Loss by Model]({rel_path})")
            lines.append("")

        if "bar_final_loss_by_model_optimizer" in plots:
            rel_path = plots["bar_final_loss_by_model_optimizer"].relative_to(artifacts_dir)
            lines.append("### Final Loss by Model + Optimizer")
            lines.append("")
            lines.append(f"![Loss by Model+Optimizer]({rel_path})")
            lines.append("")

        if "bar_final_accuracy_by_optimizer" in plots:
            rel_path = plots["bar_final_accuracy_by_optimizer"].relative_to(artifacts_dir)
            lines.append("### Final Accuracy by Optimizer")
            lines.append("")
            lines.append(f"![Accuracy by Optimizer]({rel_path})")
            lines.append("")

        if "bar_final_accuracy_by_model" in plots:
            rel_path = plots["bar_final_accuracy_by_model"].relative_to(artifacts_dir)
            lines.append("### Final Accuracy by Model")
            lines.append("")
            lines.append(f"![Accuracy by Model]({rel_path})")
            lines.append("")

        if "bar_final_accuracy_by_model_optimizer" in plots:
            rel_path = plots["bar_final_accuracy_by_model_optimizer"].relative_to(artifacts_dir)
            lines.append("### Final Accuracy by Model + Optimizer")
            lines.append("")
            lines.append(f"![Accuracy by Model+Optimizer]({rel_path})")
            lines.append("")

        if "bar_final_suboptimality_by_optimizer" in plots:
            rel_path = plots["bar_final_suboptimality_by_optimizer"].relative_to(artifacts_dir)
            lines.append("### Final Suboptimality by Optimizer")
            lines.append("")
            lines.append(f"![Suboptimality by Optimizer]({rel_path})")
            lines.append("")

        if "bar_final_consensus_by_strategy" in plots:
            rel_path = plots["bar_final_consensus_by_strategy"].relative_to(artifacts_dir)
            lines.append("### Final Consensus Error by Strategy")
            lines.append("")
            lines.append(f"![Consensus by Strategy]({rel_path})")
            lines.append("")

        if "bar_final_loss_by_topology" in plots:
            rel_path = plots["bar_final_loss_by_topology"].relative_to(artifacts_dir)
            lines.append("### Final Loss by Topology")
            lines.append("")
            lines.append(f"![Loss by Topology]({rel_path})")
            lines.append("")

        if "bar_final_loss_by_optimizer_strategy" in plots:
            rel_path = plots["bar_final_loss_by_optimizer_strategy"].relative_to(artifacts_dir)
            lines.append("### Final Loss by Optimizer + Strategy")
            lines.append("")
            lines.append(f"![Loss by Optimizer+Strategy]({rel_path})")
            lines.append("")

        if "bar_final_accuracy_by_topology_strategy" in plots:
            rel_path = plots["bar_final_accuracy_by_topology_strategy"].relative_to(artifacts_dir)
            lines.append("### Final Accuracy by Topology + Strategy")
            lines.append("")
            lines.append(f"![Accuracy by Topology+Strategy]({rel_path})")
            lines.append("")

        if "bar_final_accuracy_by_model_constraint" in plots:
            rel_path = plots["bar_final_accuracy_by_model_constraint"].relative_to(artifacts_dir)
            lines.append("### Final Accuracy by Model + Constraint")
            lines.append("")
            lines.append(f"![Accuracy by Model+Constraint]({rel_path})")
            lines.append("")

        if "bar_final_accuracy_by_constraint" in plots:
            rel_path = plots["bar_final_accuracy_by_constraint"].relative_to(artifacts_dir)
            lines.append("### Final Accuracy by Constraint")
            lines.append("")
            lines.append(f"![Accuracy by Constraint]({rel_path})")
            lines.append("")

        if "bar_final_accuracy_by_heterogeneity" in plots:
            rel_path = plots["bar_final_accuracy_by_heterogeneity"].relative_to(artifacts_dir)
            lines.append("### Final Accuracy by Heterogeneity")
            lines.append("")
            lines.append(f"![Accuracy by Heterogeneity]({rel_path})")
            lines.append("")

    # Sanity targets section for MNIST
    task = global_summary.get("task", "quadratic")
    if task == "mnist" and has_accuracy:
        lines.append("## Sanity Targets")
        lines.append("")

        # Find best accuracy
        acc_vals = [
            float(r.get("final_mean_accuracy", 0))
            for r in rows
            if r.get("final_mean_accuracy", "") not in ("", "N/A")
        ]
        if acc_vals:
            best_acc = max(acc_vals)
            target_acc = 0.90
            status = "✅ PASSED" if best_acc >= target_acc else "❌ FAILED"

            lines.append(f"**MNIST Baseline Accuracy Target**: {status}")
            lines.append("")
            lines.append(f"- **Best validation accuracy**: {best_acc:.4f}")
            lines.append(f"- **Target**: ≥ {target_acc:.2f}")
            lines.append("")

            if best_acc >= target_acc:
                lines.append(
                    "The best run achieves the target accuracy of 90% on MNIST validation set."
                )
            else:
                lines.append(
                    f"⚠️ The best run ({best_acc:.4f}) does not reach the target accuracy "
                    f"({target_acc:.2f}). Consider increasing steps or adjusting hyperparameters."
                )
            lines.append("")

    return "\n".join(lines)
