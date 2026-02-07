"""Plotting utilities for benchmark visualization.

This module provides functions for:
- Parsing experiment history and results files
- Creating per-run plots (loss, accuracy, consensus, budgets)
- Creating aggregate plots for matrix/ablation experiments

Uses matplotlib only (no seaborn).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "read_history_jsonl",
    "read_results_csv",
    "plot_single_run",
    "plot_matrix_results",
    "group_mean_std",
]


# =============================================================================
# Parsers
# =============================================================================


def read_history_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a history.jsonl file and return list of step dictionaries.

    Each line in the file is a JSON object representing one step's metrics.

    Args:
        path: Path to the history.jsonl file.

    Returns:
        List of dictionaries, one per step.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If a line is not valid JSON.
    """
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_results_csv(path: Path) -> list[dict[str, str]]:
    """Read a results.csv file and return list of row dictionaries.

    Args:
        path: Path to the results.csv file.

    Returns:
        List of dictionaries with string values (as read from CSV).

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


# =============================================================================
# Helpers
# =============================================================================


def _safe_float(value: Any) -> float | None:
    """Convert value to float, returning None if not possible."""
    if value is None or value == "" or value == "na" or value == "N/A":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _ensure_plots_dir(out_dir: Path) -> Path:
    """Ensure the plots directory exists and return its path."""
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def group_mean_std(
    rows: list[dict[str, str]],
    group_keys: list[str],
    value_key: str,
) -> list[tuple[str, float, float]]:
    """Group rows by keys and compute mean and std of a value.

    Args:
        rows: List of row dictionaries.
        group_keys: List of column names to group by.
        value_key: Column name of the value to aggregate.

    Returns:
        List of (group_label, mean, std) tuples sorted by group_label.
        group_label is formed by joining group key values with "/".
    """
    groups: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        # Build group label
        label_parts = [str(row.get(k, "")) for k in group_keys]
        label = "/".join(label_parts)

        # Get value
        val = _safe_float(row.get(value_key))
        if val is not None:
            groups[label].append(val)

    # Compute mean and std
    results: list[tuple[str, float, float]] = []
    for label in sorted(groups.keys()):
        values = groups[label]
        if values:
            mean = float(np.mean(values))
            std = float(np.std(values)) if len(values) > 1 else 0.0
            results.append((label, mean, std))

    return results


# =============================================================================
# Per-run plotting
# =============================================================================


def plot_single_run(
    history: list[dict[str, Any]],
    out_dir: Path,
    *,
    task: str,
    env: str,
) -> dict[str, Path]:
    """Create plots for a single run and return mapping of plot names to paths.

    Creates the following plots (if data is available):
    - loss.png: mean_loss vs step
    - suboptimality.png: suboptimality vs step (quadratic only)
    - accuracy.png: mean_accuracy vs step (logistic only)
    - consensus.png: consensus_error vs step (gossip only)
    - budgets.png: cumulative grad_evals and gossip_rounds vs step

    Args:
        history: List of step dictionaries from history.jsonl.
        out_dir: Output directory (plots will be in out_dir/plots/).
        task: Task type ("quadratic" or "logistic").
        env: Environment type ("single" or "gossip").

    Returns:
        Dictionary mapping plot name to file path.
    """
    plots_dir = _ensure_plots_dir(out_dir)
    created_plots: dict[str, Path] = {}

    if not history:
        return created_plots

    # Extract step indices
    steps = [h.get("step", i) for i, h in enumerate(history)]

    # Plot loss
    losses = [_safe_float(h.get("mean_loss")) for h in history]
    if any(v is not None for v in losses):
        fig, ax = plt.subplots(figsize=(8, 5))
        valid_steps = [s for s, v in zip(steps, losses, strict=False) if v is not None]
        valid_losses = [v for v in losses if v is not None]
        ax.plot(valid_steps, valid_losses, marker="o", markersize=3)
        ax.set_xlabel("Step")
        ax.set_ylabel("Mean Loss")
        ax.set_title("Loss vs Step")
        ax.grid(True, alpha=0.3)
        path = plots_dir / "loss.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        created_plots["loss"] = path

    # Plot suboptimality (quadratic only)
    if task == "quadratic":
        subopt = [_safe_float(h.get("suboptimality")) for h in history]
        if any(v is not None for v in subopt):
            fig, ax = plt.subplots(figsize=(8, 5))
            valid_steps = [s for s, v in zip(steps, subopt, strict=False) if v is not None]
            valid_subopt = [v for v in subopt if v is not None]
            ax.plot(valid_steps, valid_subopt, marker="o", markersize=3)
            ax.set_xlabel("Step")
            ax.set_ylabel("Suboptimality")
            ax.set_title("Suboptimality vs Step")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            path = plots_dir / "suboptimality.png"
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            created_plots["suboptimality"] = path

    # Plot accuracy (logistic only)
    if task == "logistic":
        # Check for both accuracy and mean_accuracy
        acc = [_safe_float(h.get("mean_accuracy") or h.get("accuracy")) for h in history]
        if any(v is not None for v in acc):
            fig, ax = plt.subplots(figsize=(8, 5))
            valid_steps = [s for s, v in zip(steps, acc, strict=False) if v is not None]
            valid_acc = [v for v in acc if v is not None]
            ax.plot(valid_steps, valid_acc, marker="o", markersize=3)
            ax.set_xlabel("Step")
            ax.set_ylabel("Accuracy")
            ax.set_title("Accuracy vs Step")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            path = plots_dir / "accuracy.png"
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            created_plots["accuracy"] = path

    # Plot consensus error (gossip only)
    if env == "gossip":
        cons = [_safe_float(h.get("consensus_error")) for h in history]
        if any(v is not None for v in cons):
            fig, ax = plt.subplots(figsize=(8, 5))
            valid_steps = [s for s, v in zip(steps, cons, strict=False) if v is not None]
            valid_cons = [v for v in cons if v is not None]
            ax.plot(valid_steps, valid_cons, marker="o", markersize=3)
            ax.set_xlabel("Step")
            ax.set_ylabel("Consensus Error")
            ax.set_title("Consensus Error vs Step")
            if all(v > 0 for v in valid_cons):
                ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            path = plots_dir / "consensus.png"
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            created_plots["consensus"] = path

    # Plot budgets (grad_evals and gossip_rounds)
    grad_evals = [_safe_float(h.get("total_grad_evals")) for h in history]
    gossip_rounds = [_safe_float(h.get("total_gossip_rounds")) for h in history]

    has_grad = any(v is not None for v in grad_evals)
    has_gossip = any(v is not None for v in gossip_rounds)

    if has_grad or has_gossip:
        fig, ax = plt.subplots(figsize=(8, 5))
        if has_grad:
            valid_steps_g = [s for s, v in zip(steps, grad_evals, strict=False) if v is not None]
            valid_grad = [v for v in grad_evals if v is not None]
            ax.plot(valid_steps_g, valid_grad, marker="o", markersize=3, label="Grad Evals")
        if has_gossip:
            valid_steps_r = [s for s, v in zip(steps, gossip_rounds, strict=False) if v is not None]
            valid_rounds = [v for v in gossip_rounds if v is not None]
            ax.plot(valid_steps_r, valid_rounds, marker="s", markersize=3, label="Gossip Rounds")
        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative Count")
        ax.set_title("Budget Usage vs Step")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = plots_dir / "budgets.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        created_plots["budgets"] = path

    return created_plots


# =============================================================================
# Matrix/ablation aggregate plotting
# =============================================================================


def plot_matrix_results(
    rows: list[dict[str, str]],
    out_dir: Path,
    *,
    task: str = "quadratic",
) -> dict[str, Path]:
    """Create aggregate plots for matrix/ablation results.

    Creates bar charts showing mean±std grouped by various factors:
    - bar_final_loss_by_optimizer.png
    - bar_final_loss_by_model.png (if multiple models)
    - bar_final_loss_by_model_optimizer.png (if multiple models)
    - bar_final_accuracy_by_optimizer.png (logistic only)
    - bar_final_accuracy_by_model.png (logistic, if multiple models)
    - bar_final_suboptimality_by_optimizer.png (quadratic only)
    - bar_final_consensus_by_strategy.png (gossip only)
    - bar_final_loss_by_topology.png (gossip only)
    - bar_final_loss_by_optimizer_strategy.png (gossip only)

    Args:
        rows: List of row dictionaries from results.csv.
        out_dir: Output directory (plots will be in out_dir/plots/).
        task: Task type for conditional plots.

    Returns:
        Dictionary mapping plot name to file path.
    """
    plots_dir = _ensure_plots_dir(out_dir)
    created_plots: dict[str, Path] = {}

    if not rows:
        return created_plots

    # Check if gossip environment
    is_gossip = any(row.get("env") == "gossip" for row in rows)

    # Check if multiple models present
    models = {row.get("model", "NumpyVector") for row in rows}
    has_multiple_models = len(models) > 1

    # Check if logistic task (has accuracy)
    is_logistic = task == "logistic"
    has_accuracy = any(_safe_float(row.get("final_mean_accuracy")) is not None for row in rows)

    # Bar chart: final_mean_loss by optimizer
    stats = group_mean_std(rows, ["optimizer"], "final_mean_loss")
    if stats:
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = [s[0] for s in stats]
        means = [s[1] for s in stats]
        stds = [s[2] for s in stats]
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Optimizer")
        ax.set_ylabel("Final Mean Loss")
        ax.set_title("Final Loss by Optimizer (mean ± std)")
        ax.grid(True, alpha=0.3, axis="y")
        path = plots_dir / "bar_final_loss_by_optimizer.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        created_plots["bar_final_loss_by_optimizer"] = path

    # Bar chart: final_mean_loss by model (if multiple models)
    if has_multiple_models:
        stats = group_mean_std(rows, ["model"], "final_mean_loss")
        if stats:
            fig, ax = plt.subplots(figsize=(8, 5))
            labels = [s[0] for s in stats]
            means = [s[1] for s in stats]
            stds = [s[2] for s in stats]
            x = np.arange(len(labels))
            ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_xlabel("Model")
            ax.set_ylabel("Final Mean Loss")
            ax.set_title("Final Loss by Model (mean ± std)")
            ax.grid(True, alpha=0.3, axis="y")
            path = plots_dir / "bar_final_loss_by_model.png"
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            created_plots["bar_final_loss_by_model"] = path

        # Bar chart: final_mean_loss by (model, optimizer)
        stats = group_mean_std(rows, ["model", "optimizer"], "final_mean_loss")
        if stats and len(stats) <= 20:
            fig, ax = plt.subplots(figsize=(12, 6))
            labels = [s[0] for s in stats]
            means = [s[1] for s in stats]
            stds = [s[2] for s in stats]
            x = np.arange(len(labels))
            ax.bar(x, means, yerr=stds, capsize=3, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_xlabel("Model / Optimizer")
            ax.set_ylabel("Final Mean Loss")
            ax.set_title("Final Loss by Model+Optimizer (mean ± std)")
            ax.grid(True, alpha=0.3, axis="y")
            path = plots_dir / "bar_final_loss_by_model_optimizer.png"
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            created_plots["bar_final_loss_by_model_optimizer"] = path

    # Accuracy plots (logistic/mnist)
    if is_logistic or has_accuracy:
        # Bar chart: final_mean_accuracy by optimizer
        stats = group_mean_std(rows, ["optimizer"], "final_mean_accuracy")
        if stats:
            fig, ax = plt.subplots(figsize=(8, 5))
            labels = [s[0] for s in stats]
            means = [s[1] for s in stats]
            stds = [s[2] for s in stats]
            x = np.arange(len(labels))
            ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, color="green")
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_xlabel("Optimizer")
            ax.set_ylabel("Final Mean Accuracy")
            ax.set_title("Final Accuracy by Optimizer (mean ± std)")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3, axis="y")
            path = plots_dir / "bar_final_accuracy_by_optimizer.png"
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            created_plots["bar_final_accuracy_by_optimizer"] = path

        # Bar chart: final_mean_accuracy by model (if multiple models)
        if has_multiple_models:
            stats = group_mean_std(rows, ["model"], "final_mean_accuracy")
            if stats:
                fig, ax = plt.subplots(figsize=(8, 5))
                labels = [s[0] for s in stats]
                means = [s[1] for s in stats]
                stds = [s[2] for s in stats]
                x = np.arange(len(labels))
                ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, color="green")
                ax.set_xticks(x)
                ax.set_xticklabels(labels)
                ax.set_xlabel("Model")
                ax.set_ylabel("Final Mean Accuracy")
                ax.set_title("Final Accuracy by Model (mean ± std)")
                ax.set_ylim(0, 1.05)
                ax.grid(True, alpha=0.3, axis="y")
                path = plots_dir / "bar_final_accuracy_by_model.png"
                fig.savefig(path, dpi=100, bbox_inches="tight")
                plt.close(fig)
                created_plots["bar_final_accuracy_by_model"] = path

            # Bar chart: final_mean_accuracy by (model, optimizer)
            stats = group_mean_std(rows, ["model", "optimizer"], "final_mean_accuracy")
            if stats and len(stats) <= 20:
                fig, ax = plt.subplots(figsize=(12, 6))
                labels = [s[0] for s in stats]
                means = [s[1] for s in stats]
                stds = [s[2] for s in stats]
                x = np.arange(len(labels))
                ax.bar(x, means, yerr=stds, capsize=3, alpha=0.8, color="green")
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
                ax.set_xlabel("Model / Optimizer")
                ax.set_ylabel("Final Mean Accuracy")
                ax.set_title("Final Accuracy by Model+Optimizer (mean ± std)")
                ax.set_ylim(0, 1.05)
                ax.grid(True, alpha=0.3, axis="y")
                path = plots_dir / "bar_final_accuracy_by_model_optimizer.png"
                fig.savefig(path, dpi=100, bbox_inches="tight")
                plt.close(fig)
                created_plots["bar_final_accuracy_by_model_optimizer"] = path

    # Bar chart: final_suboptimality by optimizer (quadratic only)
    if task == "quadratic":
        stats = group_mean_std(rows, ["optimizer"], "final_suboptimality")
        if stats:
            fig, ax = plt.subplots(figsize=(8, 5))
            labels = [s[0] for s in stats]
            means = [s[1] for s in stats]
            stds = [s[2] for s in stats]
            x = np.arange(len(labels))
            ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_xlabel("Optimizer")
            ax.set_ylabel("Final Suboptimality")
            ax.set_title("Final Suboptimality by Optimizer (mean ± std)")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3, axis="y")
            path = plots_dir / "bar_final_suboptimality_by_optimizer.png"
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            created_plots["bar_final_suboptimality_by_optimizer"] = path

    # Gossip-specific plots
    if is_gossip:
        # Bar chart: final_consensus_error by strategy
        stats = group_mean_std(rows, ["strategy"], "final_consensus_error")
        if stats:
            fig, ax = plt.subplots(figsize=(8, 5))
            labels = [s[0] for s in stats]
            means = [s[1] for s in stats]
            stds = [s[2] for s in stats]
            x = np.arange(len(labels))
            ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15, ha="right")
            ax.set_xlabel("Strategy")
            ax.set_ylabel("Final Consensus Error")
            ax.set_title("Final Consensus Error by Strategy (mean ± std)")
            ax.grid(True, alpha=0.3, axis="y")
            path = plots_dir / "bar_final_consensus_by_strategy.png"
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            created_plots["bar_final_consensus_by_strategy"] = path

        # Bar chart: final_mean_loss by topology
        stats = group_mean_std(rows, ["topology"], "final_mean_loss")
        if stats:
            fig, ax = plt.subplots(figsize=(8, 5))
            labels = [s[0] for s in stats]
            means = [s[1] for s in stats]
            stds = [s[2] for s in stats]
            x = np.arange(len(labels))
            ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_xlabel("Topology")
            ax.set_ylabel("Final Mean Loss")
            ax.set_title("Final Loss by Topology (mean ± std)")
            ax.grid(True, alpha=0.3, axis="y")
            path = plots_dir / "bar_final_loss_by_topology.png"
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            created_plots["bar_final_loss_by_topology"] = path

        # Bar chart: final_mean_loss by (optimizer, strategy)
        stats = group_mean_std(rows, ["optimizer", "strategy"], "final_mean_loss")
        if stats and len(stats) <= 20:  # Keep readable
            fig, ax = plt.subplots(figsize=(12, 6))
            labels = [s[0] for s in stats]
            means = [s[1] for s in stats]
            stds = [s[2] for s in stats]
            x = np.arange(len(labels))
            ax.bar(x, means, yerr=stds, capsize=3, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_xlabel("Optimizer / Strategy")
            ax.set_ylabel("Final Mean Loss")
            ax.set_title("Final Loss by Optimizer+Strategy (mean ± std)")
            ax.grid(True, alpha=0.3, axis="y")
            path = plots_dir / "bar_final_loss_by_optimizer_strategy.png"
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            created_plots["bar_final_loss_by_optimizer_strategy"] = path

        # MNIST-specific: accuracy by topology+strategy
        if has_accuracy:
            stats = group_mean_std(rows, ["topology", "strategy"], "final_mean_accuracy")
            if stats and len(stats) <= 20:
                fig, ax = plt.subplots(figsize=(12, 6))
                labels = [s[0] for s in stats]
                means = [s[1] for s in stats]
                stds = [s[2] for s in stats]
                x = np.arange(len(labels))
                ax.bar(x, means, yerr=stds, capsize=3, alpha=0.8, color="green")
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
                ax.set_xlabel("Topology / Strategy")
                ax.set_ylabel("Final Mean Accuracy")
                ax.set_title("Final Accuracy by Topology+Strategy (mean ± std)")
                ax.set_ylim(0, 1.05)
                ax.grid(True, alpha=0.3, axis="y")
                path = plots_dir / "bar_final_accuracy_by_topology_strategy.png"
                fig.savefig(path, dpi=100, bbox_inches="tight")
                plt.close(fig)
                created_plots["bar_final_accuracy_by_topology_strategy"] = path

    # MNIST-specific plots: accuracy by constraint
    if task == "mnist" and has_accuracy:
        # Bar chart: accuracy by (model, constraint)
        stats = group_mean_std(rows, ["model", "constraint"], "final_mean_accuracy")
        if stats and len(stats) <= 20:
            fig, ax = plt.subplots(figsize=(12, 6))
            labels = [s[0] for s in stats]
            means = [s[1] for s in stats]
            stds = [s[2] for s in stats]
            x = np.arange(len(labels))
            ax.bar(x, means, yerr=stds, capsize=3, alpha=0.8, color="green")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_xlabel("Model / Constraint")
            ax.set_ylabel("Final Mean Accuracy")
            ax.set_title("Final Accuracy by Model+Constraint (mean ± std)")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3, axis="y")
            path = plots_dir / "bar_final_accuracy_by_model_constraint.png"
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            created_plots["bar_final_accuracy_by_model_constraint"] = path

        # Bar chart: accuracy by constraint alone
        stats = group_mean_std(rows, ["constraint"], "final_mean_accuracy")
        if stats:
            fig, ax = plt.subplots(figsize=(8, 5))
            labels = [s[0] for s in stats]
            means = [s[1] for s in stats]
            stds = [s[2] for s in stats]
            x = np.arange(len(labels))
            ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, color="green")
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_xlabel("Constraint")
            ax.set_ylabel("Final Mean Accuracy")
            ax.set_title("Final Accuracy by Constraint (mean ± std)")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3, axis="y")
            path = plots_dir / "bar_final_accuracy_by_constraint.png"
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            created_plots["bar_final_accuracy_by_constraint"] = path

        # Bar chart: accuracy by heterogeneity
        stats = group_mean_std(rows, ["heterogeneity"], "final_mean_accuracy")
        if stats:
            fig, ax = plt.subplots(figsize=(8, 5))
            labels = [s[0] for s in stats]
            means = [s[1] for s in stats]
            stds = [s[2] for s in stats]
            x = np.arange(len(labels))
            ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, color="green")
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_xlabel("Heterogeneity")
            ax.set_ylabel("Final Mean Accuracy")
            ax.set_title("Final Accuracy by Heterogeneity (mean ± std)")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3, axis="y")
            path = plots_dir / "bar_final_accuracy_by_heterogeneity.png"
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            created_plots["bar_final_accuracy_by_heterogeneity"] = path

    return created_plots
