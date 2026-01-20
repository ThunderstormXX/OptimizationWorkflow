"""Experiment runner CLI for benchmark experiments.

This module provides a command-line interface for running optimization
experiments with configurable environments, optimizers, and tasks.

Supports three modes:
- single: Run one experiment with specified parameters
- matrix: Run a grid of experiments varying optimizer, topology, strategy, and schedule
- checks: Run convergence/regression checks and produce a report

Optimizers:
- fw: Frank-Wolfe (uses constraint + schedule)
- gd: Gradient Descent (uses lr, ignores constraint)
- pgd: Projected Gradient Descent (uses lr + constraint)

Usage:
    python -m benchmarks.runner --env single --steps 50 --seed 0
    python -m benchmarks.runner --env gossip --n-nodes 5 --optimizer fw
    python -m benchmarks.runner --mode matrix --env gossip --steps 20 --seeds "0,1"
    python -m benchmarks.runner --mode checks --env gossip --optimizer fw --steps 30
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.checks import run_checks
from benchmarks.metrics import consensus_error, mean_params, suboptimality
from benchmarks.plotting import (
    plot_matrix_results,
    plot_single_run,
    read_history_jsonl,
    read_results_csv,
)
from benchmarks.registry import get_optimizer, get_strategy_with_config
from benchmarks.report import (
    render_checks_markdown,
    render_matrix_visual_report_md,
    render_single_run_report_md,
)
from benchmarks.workflow import next_experiment_dir, try_get_git_commit, write_run_files
from core.protocols import Topology
from core.types import NodeId, StepResult
from distributed.communicator import SynchronousGossipCommunicator
from distributed.strategies import GossipNode
from distributed.topology import CompleteTopology, RingTopology
from environments.gossip import GossipEnvironment
from environments.single_process import SingleProcessEnvironment
from models.numpy_vector import NumpyVectorModel
from optim.frank_wolfe import FWState
from optim.gradient_descent import GDState
from tasks.logistic_regression import (
    LogisticGradComputer,
    LogisticRegressionTask,
    make_logistic_data,
    split_across_nodes,
)
from tasks.synthetic_quadratic import (
    QuadraticGradComputer,
    SyntheticQuadraticTask,
    make_spd_quadratic,
)

__all__ = ["main", "run_once", "run_checks_mode", "CSV_COLUMNS"]

# CSV column order (stable)
CSV_COLUMNS = [
    "run_id",
    "seed",
    "env",
    "n_nodes",
    "topology",
    "strategy",
    "optimizer",
    "lr",
    "constraint",
    "schedule",
    "gamma",
    "radius",
    "dim",
    "cond",
    "steps",
    "final_mean_loss",
    "final_suboptimality",
    "final_consensus_error",
    "final_dist_to_opt",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run optimization experiment on synthetic quadratic task",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "matrix", "checks"],
        default="single",
        help="Run mode: single experiment, matrix of experiments, or checks suite",
    )

    # Workflow
    parser.add_argument(
        "--workflow-dir",
        type=str,
        default="workflow",
        help="Directory for experiment outputs",
    )

    # Environment
    parser.add_argument(
        "--env",
        type=str,
        choices=["single", "gossip"],
        default="single",
        help="Environment type",
    )

    # General
    parser.add_argument("--steps", type=int, default=50, help="Number of steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (single mode)")
    parser.add_argument(
        "--seeds", type=str, default="0", help="Comma-separated seeds (matrix mode)"
    )
    parser.add_argument("--dim", type=int, default=10, help="Problem dimension")
    parser.add_argument("--cond", type=float, default=10.0, help="Condition number for SPD matrix")

    # Optimizer
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["fw", "gd", "pgd"],
        default="fw",
        help="Optimizer: fw=Frank-Wolfe, gd=Gradient Descent, pgd=Projected GD",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for gd/pgd optimizers")

    # Constraint (for fw and pgd)
    parser.add_argument(
        "--constraint",
        type=str,
        choices=["l2ball", "simplex"],
        default="l2ball",
        help="Constraint type (for fw and pgd)",
    )
    parser.add_argument("--radius", type=float, default=1.0, help="Radius for L2 ball constraint")

    # Step schedule (for fw)
    parser.add_argument(
        "--step-schedule",
        type=str,
        choices=["harmonic", "constant"],
        default="harmonic",
        help="Step size schedule (for fw only)",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.2, help="Gamma for constant step size (fw)"
    )

    # Task selection
    parser.add_argument(
        "--task",
        type=str,
        choices=["quadratic", "logistic"],
        default="quadratic",
        help="Task type: quadratic or logistic regression",
    )

    # Logistic regression options
    parser.add_argument(
        "--n-samples", type=int, default=2000, help="Number of samples (logistic only)"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (logistic only)")
    parser.add_argument(
        "--heterogeneity",
        type=str,
        choices=["iid", "label_skew"],
        default="iid",
        help="Data heterogeneity type (logistic + gossip only)",
    )

    # Gossip-specific
    parser.add_argument("--n-nodes", type=int, default=5, help="Number of nodes (gossip only)")
    parser.add_argument(
        "--topology",
        type=str,
        choices=["ring", "complete"],
        default="ring",
        help="Topology (gossip only)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["local_then_gossip", "gossip_then_local", "gradient_tracking"],
        default="local_then_gossip",
        help="Gossip strategy (gossip only)",
    )

    # Matrix mode
    parser.add_argument(
        "--matrix",
        type=str,
        choices=["small", "medium", "large"],
        default="small",
        help="Matrix size (matrix mode only)",
    )
    parser.add_argument(
        "--save-histories",
        action="store_true",
        help="Save per-run history.jsonl files (matrix mode)",
    )
    parser.add_argument(
        "--ablation-spec",
        type=str,
        default=None,
        help="Path to JSON ablation spec file (matrix mode, overrides --matrix)",
    )

    # Rendering
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable automatic plot and report generation",
    )

    # Descriptive
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name")
    parser.add_argument("--description", type=str, default=None, help="Experiment description")

    return parser.parse_args(argv)


def build_config(args: argparse.Namespace, seed: int | None = None) -> dict[str, Any]:
    """Build configuration dictionary from parsed arguments."""
    config: dict[str, Any] = {
        "env": args.env,
        "task": args.task,
        "steps": args.steps,
        "seed": seed if seed is not None else args.seed,
        "dim": args.dim,
        "cond": args.cond,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "constraint": args.constraint,
        "step_schedule": args.step_schedule,
        "radius": args.radius,
        "gamma": args.gamma,
    }

    # Logistic regression options
    if args.task == "logistic":
        config["n_samples"] = args.n_samples
        config["batch_size"] = args.batch_size
        config["heterogeneity"] = args.heterogeneity

    if args.env == "gossip":
        config["n_nodes"] = args.n_nodes
        config["topology"] = args.topology
        config["strategy"] = args.strategy

    if args.exp_name:
        config["exp_name"] = args.exp_name
    if args.description:
        config["description"] = args.description

    return config


def _get_init_state(optimizer_name: str) -> FWState | GDState:
    """Get initial optimizer state based on optimizer type."""
    if optimizer_name == "fw":
        return FWState(t=0)
    else:
        return GDState(t=0)


def run_once(
    config: dict[str, Any],
    out_dir: Path,
    *,
    save_history: bool = True,
) -> dict[str, Any]:
    """Run a single experiment with the given configuration.

    Args:
        config: Configuration dictionary with all parameters.
        out_dir: Directory to write artifacts (summary.json, optionally history.jsonl).
        save_history: Whether to save history.jsonl.

    Returns:
        Summary dictionary with final metrics.
    """
    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    task_type = config.get("task", "quadratic")

    if task_type == "quadratic":
        # Build quadratic problem deterministically from seed
        seed = config["seed"]
        rng = np.random.default_rng(seed)
        problem = make_spd_quadratic(dim=config["dim"], rng=rng, cond=config["cond"])
        task: Any = SyntheticQuadraticTask(problem)

        if config["env"] == "single":
            summary = _run_single_env_quadratic(config, task, out_dir, save_history=save_history)
        else:
            summary = _run_gossip_env_quadratic(config, task, out_dir, save_history=save_history)
    else:
        # Logistic regression task
        if config["env"] == "single":
            summary = _run_single_env_logistic(config, out_dir, save_history=save_history)
        else:
            summary = _run_gossip_env_logistic(config, out_dir, save_history=save_history)

    # Write summary.json
    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    return summary


def _run_single_env_quadratic(
    config: dict[str, Any],
    task: SyntheticQuadraticTask,
    out_dir: Path,
    *,
    save_history: bool = True,
) -> dict[str, Any]:
    """Run single-process environment experiment with quadratic task."""
    seed = config["seed"]
    dim = config["dim"]
    steps = config["steps"]
    optimizer_name = config["optimizer"]

    # Build optimizer using registry
    optimizer = get_optimizer(optimizer_name, config)

    # Build model with deterministic initial params
    rng = np.random.default_rng(seed + 1000)
    x0 = rng.standard_normal(dim)
    model = NumpyVectorModel(x0)

    grad_computer = QuadraticGradComputer()

    # Build environment
    env: SingleProcessEnvironment[Any, Any, Any] = SingleProcessEnvironment(
        task=task,  # type: ignore[arg-type]
        model=model,
        optimizer=optimizer,
        grad_computer=grad_computer,  # type: ignore[arg-type]
    )

    # Run
    env.reset(seed=seed)

    # Collect per-step metrics
    history_lines: list[dict[str, Any]] = []
    for t in range(steps):
        result = env.step()
        step_result: StepResult = result
        line: dict[str, Any] = {
            "step": t,
            "mean_loss": step_result.loss,
        }
        if "grad_norm" in step_result.metrics:
            line["mean_grad_norm"] = step_result.metrics["grad_norm"]
        if "dist_to_opt" in step_result.metrics:
            line["mean_dist_to_opt"] = step_result.metrics["dist_to_opt"]
        history_lines.append(line)

    # Write history.jsonl if requested
    if save_history:
        history_path = out_dir / "history.jsonl"
        with history_path.open("w", encoding="utf-8") as f:
            for line in history_lines:
                f.write(json.dumps(line, sort_keys=True) + "\n")

    # Compute final metrics
    final_params = model.parameters_vector()
    final_loss = task.loss(model, None)
    final_subopt = suboptimality(task, final_params)
    final_metrics = task.metrics(model, None)

    summary: dict[str, Any] = {
        "env": "single",
        "steps": steps,
        "seed": seed,
        "dim": dim,
        "cond": config["cond"],
        "optimizer": optimizer_name,
        "lr": config["lr"],
        "constraint": config["constraint"],
        "step_schedule": config["step_schedule"] if optimizer_name == "fw" else "na",
        "gamma": config["gamma"],
        "radius": config["radius"],
        "final_mean_loss": final_loss,
        "final_suboptimality": final_subopt,
        "final_dist_to_opt": final_metrics["dist_to_opt"],
    }

    return summary


def _run_gossip_env_quadratic(
    config: dict[str, Any],
    task: SyntheticQuadraticTask,
    out_dir: Path,
    *,
    save_history: bool = True,
) -> dict[str, Any]:
    """Run gossip environment experiment with quadratic task."""
    seed = config["seed"]
    dim = config["dim"]
    steps = config["steps"]
    n_nodes = config["n_nodes"]
    topology_name = config["topology"]
    strategy_name = config["strategy"]
    optimizer_name = config["optimizer"]

    # Build topology and communicator
    topology: Topology
    if topology_name == "ring":
        topology = RingTopology(n=n_nodes)
    else:
        topology = CompleteTopology(n=n_nodes)
    communicator = SynchronousGossipCommunicator(topology=topology)

    # Build strategy using registry (with config for gradient_tracking)
    strategy = get_strategy_with_config(strategy_name, config)

    # Build nodes with different initial params
    master_rng = np.random.default_rng(seed + 2000)
    nodes: list[GossipNode[Any, Any, Any]] = []

    for i in range(n_nodes):
        x0 = master_rng.standard_normal(dim)
        model = NumpyVectorModel(x0)

        node_task = SyntheticQuadraticTask(task.problem)
        node_grad_computer = QuadraticGradComputer()

        # Build optimizer using registry
        optimizer = get_optimizer(optimizer_name, config)
        init_state = _get_init_state(optimizer_name)

        node: GossipNode[Any, Any, Any] = GossipNode(
            node_id=i,
            task=node_task,  # type: ignore[arg-type]
            model=model,
            optimizer=optimizer,
            grad_computer=node_grad_computer,  # type: ignore[arg-type]
            opt_state=init_state,
            rng=np.random.default_rng(0),
        )
        nodes.append(node)

    # Build environment
    env: GossipEnvironment[Any, Any, Any] = GossipEnvironment(
        nodes=nodes,
        communicator=communicator,
        strategy=strategy,
    )

    # Run (quadratic gossip)
    env.reset(seed=seed)

    # Collect per-step metrics
    history_lines: list[dict[str, Any]] = []
    for t in range(steps):
        step_record = env.step()
        result: Mapping[NodeId, StepResult] = step_record  # type: ignore[assignment]

        losses = [result[i].loss for i in range(n_nodes)]
        mean_loss = float(np.mean(losses))

        line: dict[str, Any] = {
            "step": t,
            "mean_loss": mean_loss,
        }

        if "grad_norm" in result[0].metrics:
            grad_norms = [result[i].metrics["grad_norm"] for i in range(n_nodes)]
            line["mean_grad_norm"] = float(np.mean(grad_norms))

        if "dist_to_opt" in result[0].metrics:
            dists = [result[i].metrics["dist_to_opt"] for i in range(n_nodes)]
            line["mean_dist_to_opt"] = float(np.mean(dists))

        params_by_node = env.get_params_by_node()
        line["consensus_error"] = consensus_error(params_by_node)

        history_lines.append(line)

    # Write history.jsonl if requested
    if save_history:
        history_path = out_dir / "history.jsonl"
        with history_path.open("w", encoding="utf-8") as f:
            for line in history_lines:
                f.write(json.dumps(line, sort_keys=True) + "\n")

    # Compute final metrics
    params_by_node = env.get_params_by_node()
    x_bar = mean_params(params_by_node)
    final_subopt = suboptimality(task, x_bar)
    final_cons_error = consensus_error(params_by_node)

    final_losses = [nodes[i].task.loss(nodes[i].model, None) for i in range(n_nodes)]
    final_mean_loss = float(np.mean(final_losses))

    final_dists = [float(np.linalg.norm(params_by_node[i] - task.x_star)) for i in range(n_nodes)]
    final_mean_dist = float(np.mean(final_dists))

    summary: dict[str, Any] = {
        "env": "gossip",
        "steps": steps,
        "seed": seed,
        "dim": dim,
        "cond": config["cond"],
        "optimizer": optimizer_name,
        "lr": config["lr"],
        "constraint": config["constraint"],
        "step_schedule": config["step_schedule"] if optimizer_name == "fw" else "na",
        "gamma": config["gamma"],
        "radius": config["radius"],
        "n_nodes": n_nodes,
        "topology": topology_name,
        "strategy": strategy_name,
        "final_mean_loss": final_mean_loss,
        "final_suboptimality": final_subopt,
        "final_consensus_error": final_cons_error,
        "final_dist_to_opt": final_mean_dist,
    }

    return summary


def _run_single_env_logistic(
    config: dict[str, Any],
    out_dir: Path,
    *,
    save_history: bool = True,
) -> dict[str, Any]:
    """Run single-process environment experiment with logistic task."""
    seed = config["seed"]
    dim = config["dim"]
    steps = config["steps"]
    optimizer_name = config["optimizer"]
    n_samples = config.get("n_samples", 2000)
    batch_size = config.get("batch_size", 64)

    # Generate logistic data
    rng = np.random.default_rng(seed)
    X, y = make_logistic_data(n=n_samples, dim=dim, rng=rng, separable=False)

    # Create single-node dataset
    from tasks.logistic_regression import NodeDataset

    dataset = NodeDataset(X=X, y=y, node_id=0)
    task = LogisticRegressionTask(dataset=dataset, batch_size=batch_size)
    grad_computer = LogisticGradComputer()

    # Build optimizer using registry
    optimizer = get_optimizer(optimizer_name, config)

    # Build model with deterministic initial params
    model_rng = np.random.default_rng(seed + 1000)
    x0 = model_rng.standard_normal(dim) * 0.1  # Small initial weights
    model = NumpyVectorModel(x0)

    # Build environment
    env: SingleProcessEnvironment[Any, Any, Any] = SingleProcessEnvironment(
        task=task,
        model=model,
        optimizer=optimizer,
        grad_computer=grad_computer,
    )

    # Run
    env.reset(seed=seed)

    # Collect per-step metrics
    history_lines: list[dict[str, Any]] = []
    for t in range(steps):
        result = env.step()
        step_result: StepResult = result
        line: dict[str, Any] = {
            "step": t,
            "mean_loss": step_result.loss,
        }
        if "accuracy" in step_result.metrics:
            line["accuracy"] = step_result.metrics["accuracy"]
        history_lines.append(line)

    # Write history.jsonl if requested
    if save_history:
        history_path = out_dir / "history.jsonl"
        with history_path.open("w", encoding="utf-8") as f:
            for line in history_lines:
                f.write(json.dumps(line, sort_keys=True) + "\n")

    # Compute final metrics
    final_loss = task.loss(model, None)
    final_metrics = task.metrics(model, None)

    summary: dict[str, Any] = {
        "env": "single",
        "task": "logistic",
        "steps": steps,
        "seed": seed,
        "dim": dim,
        "n_samples": n_samples,
        "batch_size": batch_size,
        "optimizer": optimizer_name,
        "lr": config["lr"],
        "constraint": config.get("constraint", "none"),
        "step_schedule": config.get("step_schedule", "na") if optimizer_name == "fw" else "na",
        "gamma": config.get("gamma", 0.0),
        "radius": config.get("radius", 0.0),
        "final_mean_loss": final_loss,
        "final_accuracy": final_metrics["accuracy"],
    }

    return summary


def _run_gossip_env_logistic(
    config: dict[str, Any],
    out_dir: Path,
    *,
    save_history: bool = True,
) -> dict[str, Any]:
    """Run gossip environment experiment with logistic task."""
    seed = config["seed"]
    dim = config["dim"]
    steps = config["steps"]
    n_nodes = config["n_nodes"]
    topology_name = config["topology"]
    strategy_name = config["strategy"]
    optimizer_name = config["optimizer"]
    n_samples = config.get("n_samples", 2000)
    batch_size = config.get("batch_size", 64)
    heterogeneity = config.get("heterogeneity", "iid")

    # Build topology and communicator
    topology: Topology
    if topology_name == "ring":
        topology = RingTopology(n=n_nodes)
    else:
        topology = CompleteTopology(n=n_nodes)
    communicator = SynchronousGossipCommunicator(topology=topology)

    # Build strategy using registry (with config for gradient_tracking)
    strategy = get_strategy_with_config(strategy_name, config)

    # Generate and split logistic data
    data_rng = np.random.default_rng(seed)
    X, y = make_logistic_data(n=n_samples, dim=dim, rng=data_rng, separable=False)
    node_datasets = split_across_nodes(X, y, n_nodes, heterogeneity, data_rng)

    # Build nodes
    master_rng = np.random.default_rng(seed + 2000)
    nodes: list[GossipNode[Any, Any, Any]] = []

    for i in range(n_nodes):
        x0 = master_rng.standard_normal(dim) * 0.1  # Small initial weights
        model = NumpyVectorModel(x0)

        node_task = LogisticRegressionTask(dataset=node_datasets[i], batch_size=batch_size)
        node_grad_computer = LogisticGradComputer()

        # Build optimizer using registry
        optimizer = get_optimizer(optimizer_name, config)
        init_state = _get_init_state(optimizer_name)

        node: GossipNode[Any, Any, Any] = GossipNode(
            node_id=i,
            task=node_task,
            model=model,
            optimizer=optimizer,
            grad_computer=node_grad_computer,
            opt_state=init_state,
            rng=np.random.default_rng(0),
        )
        nodes.append(node)

    # Build environment
    env: GossipEnvironment[Any, Any, Any] = GossipEnvironment(
        nodes=nodes,
        communicator=communicator,
        strategy=strategy,
    )

    # Run
    env.reset(seed=seed)

    # Collect per-step metrics
    history_lines: list[dict[str, Any]] = []
    for t in range(steps):
        step_record = env.step()
        result: Mapping[NodeId, StepResult] = step_record  # type: ignore[assignment]

        losses = [result[i].loss for i in range(n_nodes)]
        mean_loss = float(np.mean(losses))

        line: dict[str, Any] = {
            "step": t,
            "mean_loss": mean_loss,
        }

        if "accuracy" in result[0].metrics:
            accs = [result[i].metrics["accuracy"] for i in range(n_nodes)]
            line["mean_accuracy"] = float(np.mean(accs))

        params_by_node = env.get_params_by_node()
        line["consensus_error"] = consensus_error(params_by_node)

        history_lines.append(line)

    # Write history.jsonl if requested
    if save_history:
        history_path = out_dir / "history.jsonl"
        with history_path.open("w", encoding="utf-8") as f:
            for line in history_lines:
                f.write(json.dumps(line, sort_keys=True) + "\n")

    # Compute final metrics
    params_by_node = env.get_params_by_node()
    final_cons_error = consensus_error(params_by_node)

    final_losses = [nodes[i].task.loss(nodes[i].model, None) for i in range(n_nodes)]
    final_mean_loss = float(np.mean(final_losses))

    final_accs = [nodes[i].task.metrics(nodes[i].model, None)["accuracy"] for i in range(n_nodes)]
    final_mean_acc = float(np.mean(final_accs))

    summary: dict[str, Any] = {
        "env": "gossip",
        "task": "logistic",
        "steps": steps,
        "seed": seed,
        "dim": dim,
        "n_samples": n_samples,
        "batch_size": batch_size,
        "heterogeneity": heterogeneity,
        "optimizer": optimizer_name,
        "lr": config["lr"],
        "constraint": config.get("constraint", "none"),
        "step_schedule": config.get("step_schedule", "na") if optimizer_name == "fw" else "na",
        "gamma": config.get("gamma", 0.0),
        "radius": config.get("radius", 0.0),
        "n_nodes": n_nodes,
        "topology": topology_name,
        "strategy": strategy_name,
        "final_mean_loss": final_mean_loss,
        "final_mean_accuracy": final_mean_acc,
        "final_consensus_error": final_cons_error,
    }

    return summary


def summary_to_csv_row(run_id: str, summary: dict[str, Any]) -> dict[str, Any]:
    """Convert summary dict to CSV row dict with stable column order."""
    row: dict[str, Any] = {
        "run_id": run_id,
        "seed": summary.get("seed", ""),
        "env": summary.get("env", ""),
        "n_nodes": summary.get("n_nodes", ""),
        "topology": summary.get("topology", ""),
        "strategy": summary.get("strategy", ""),
        "optimizer": summary.get("optimizer", ""),
        "lr": summary.get("lr", ""),
        "constraint": summary.get("constraint", ""),
        "schedule": summary.get("step_schedule", ""),
        "gamma": summary.get("gamma", ""),
        "radius": summary.get("radius", ""),
        "dim": summary.get("dim", ""),
        "cond": summary.get("cond", ""),
        "steps": summary.get("steps", ""),
        "final_mean_loss": summary.get("final_mean_loss", ""),
        "final_suboptimality": summary.get("final_suboptimality", ""),
        "final_consensus_error": summary.get("final_consensus_error", ""),
        "final_dist_to_opt": summary.get("final_dist_to_opt", ""),
    }
    return row


def generate_matrix_grid(
    args: argparse.Namespace,
    seeds: list[int],
) -> list[dict[str, Any]]:
    """Generate the grid of configurations for matrix mode.

    For matrix=small:
    - optimizer: [fw, pgd]
    - topology: [ring, complete]
    - strategy: [local_then_gossip, gossip_then_local]
    - schedule: [harmonic, constant] (only for fw; pgd uses "na")
    - seed: from --seeds

    For matrix=large:
    - optimizer: [fw, pgd, gd]
    - topology: [ring, complete]
    - strategy: [local_then_gossip, gossip_then_local, gradient_tracking]
    - schedule: [harmonic, constant] (only for fw)
    """
    if args.matrix == "small":
        optimizers = ["fw", "pgd"]
        topologies = ["ring", "complete"]
        strategies = ["local_then_gossip", "gossip_then_local"]
        schedules = ["harmonic", "constant"]
    elif args.matrix == "large":
        optimizers = ["fw", "pgd", "gd"]
        topologies = ["ring", "complete"]
        strategies = ["local_then_gossip", "gossip_then_local", "gradient_tracking"]
        schedules = ["harmonic", "constant"]
    else:
        # medium: same as small for now (can be extended)
        optimizers = ["fw", "pgd"]
        topologies = ["ring", "complete"]
        strategies = ["local_then_gossip", "gossip_then_local"]
        schedules = ["harmonic", "constant"]

    configs: list[dict[str, Any]] = []

    # Determine task
    task = getattr(args, "task", "quadratic")

    # Heterogeneity options (only for logistic)
    heterogeneities = ["iid"]
    if task == "logistic":
        heterogeneities = ["iid", "label_skew"] if args.matrix == "large" else ["iid"]

    # Deterministic ordering: optimizer -> topology -> strategy -> schedule -> hetero -> seed
    for opt, topo, strat, sched, hetero, seed in product(
        optimizers, topologies, strategies, schedules, heterogeneities, seeds
    ):
        # For non-FW optimizers, schedule is not meaningful
        effective_schedule = sched if opt == "fw" else "na"

        # For non-logistic tasks, heterogeneity is not meaningful
        effective_hetero = hetero if task == "logistic" else "na"

        config: dict[str, Any] = {
            "env": "gossip",
            "task": task,
            "steps": args.steps,
            "seed": seed,
            "dim": args.dim,
            "cond": args.cond,
            "optimizer": opt,
            "lr": args.lr,
            "constraint": args.constraint,
            "radius": args.radius,
            "gamma": args.gamma,
            "n_nodes": args.n_nodes,
            "topology": topo,
            "strategy": strat,
            "step_schedule": effective_schedule,
        }

        # Add logistic-specific options
        if task == "logistic":
            config["n_samples"] = getattr(args, "n_samples", 2000)
            config["batch_size"] = getattr(args, "batch_size", 64)
            config["heterogeneity"] = effective_hetero

        configs.append(config)

    # Remove duplicate configs (pgd with different schedules are identical)
    seen: set[str] = set()
    unique_configs: list[dict[str, Any]] = []
    for config in configs:
        # Create a hashable key from the config (excluding schedule for non-fw)
        key_parts = [
            config["optimizer"],
            config["topology"],
            config["strategy"],
            str(config["seed"]),
        ]
        if config["optimizer"] == "fw":
            key_parts.append(config["step_schedule"])
        if config.get("task") == "logistic":
            key_parts.append(config.get("heterogeneity", "na"))

        key = "|".join(key_parts)
        if key not in seen:
            seen.add(key)
            unique_configs.append(config)

    return unique_configs


def load_ablation_spec(spec_path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Load ablation spec from JSON and generate configurations.

    Spec format:
    {
      "base": { ... CLI-like config fields ... },
      "grid": {
        "optimizer": ["fw","pgd","gd"],
        "strategy": ["local_then_gossip","gossip_then_local","gradient_tracking"],
        "topology": ["ring","complete"],
        "step_schedule": ["harmonic","constant"],
        "heterogeneity": ["iid","label_skew"]
      },
      "seeds": [0,1,2]
    }

    Compatibility filters applied:
    - if optimizer != "fw" then step_schedule="na"
    - if task != "logistic" then heterogeneity="na"
    - if env != "gossip" then strategy/topology/n_nodes="na"

    Args:
        spec_path: Path to JSON spec file.
        args: Parsed CLI arguments for defaults.

    Returns:
        List of configuration dictionaries.
    """
    with spec_path.open("r", encoding="utf-8") as f:
        spec = json.load(f)

    # Get base config
    base = spec.get("base", {})

    # Get grid dimensions
    grid = spec.get("grid", {})
    seeds = spec.get("seeds", [0])

    # Build defaults from args
    defaults: dict[str, Any] = {
        "env": getattr(args, "env", "gossip"),
        "task": getattr(args, "task", "quadratic"),
        "steps": getattr(args, "steps", 50),
        "dim": getattr(args, "dim", 10),
        "cond": getattr(args, "cond", 10.0),
        "optimizer": "fw",
        "lr": getattr(args, "lr", 0.1),
        "constraint": getattr(args, "constraint", "l2ball"),
        "radius": getattr(args, "radius", 1.0),
        "gamma": getattr(args, "gamma", 0.2),
        "n_nodes": getattr(args, "n_nodes", 5),
        "topology": "ring",
        "strategy": "local_then_gossip",
        "step_schedule": "harmonic",
    }

    # Merge base into defaults
    defaults.update(base)

    # Extract grid dimensions
    optimizers = grid.get("optimizer", [defaults["optimizer"]])
    topologies = grid.get("topology", [defaults["topology"]])
    strategies = grid.get("strategy", [defaults["strategy"]])
    schedules = grid.get("step_schedule", [defaults["step_schedule"]])
    heterogeneities = grid.get("heterogeneity", ["iid"])

    configs: list[dict[str, Any]] = []

    # Generate Cartesian product
    for opt, topo, strat, sched, hetero, seed in product(
        optimizers, topologies, strategies, schedules, heterogeneities, seeds
    ):
        config = dict(defaults)
        config["seed"] = seed
        config["optimizer"] = opt
        config["topology"] = topo
        config["strategy"] = strat

        # Apply compatibility filters
        # Filter: if optimizer != "fw" then step_schedule="na"
        config["step_schedule"] = sched if opt == "fw" else "na"

        # Filter: if task != "logistic" then heterogeneity="na"
        task = config.get("task", "quadratic")
        if task == "logistic":
            config["heterogeneity"] = hetero
            config["n_samples"] = config.get("n_samples", getattr(args, "n_samples", 2000))
            config["batch_size"] = config.get("batch_size", getattr(args, "batch_size", 64))
        else:
            config["heterogeneity"] = "na"

        # Filter: if env != "gossip" then strategy/topology="na"
        if config["env"] != "gossip":
            config["topology"] = "na"
            config["strategy"] = "na"
            config["n_nodes"] = 1

        configs.append(config)

    # Remove duplicates
    seen: set[str] = set()
    unique_configs: list[dict[str, Any]] = []
    for config in configs:
        key_parts = [
            config["optimizer"],
            config["topology"],
            config["strategy"],
            str(config["seed"]),
        ]
        if config["optimizer"] == "fw":
            key_parts.append(config["step_schedule"])
        if config.get("task") == "logistic":
            key_parts.append(config.get("heterogeneity", "na"))

        key = "|".join(key_parts)
        if key not in seen:
            seen.add(key)
            unique_configs.append(config)

    return unique_configs


def run_matrix_mode(
    args: argparse.Namespace, exp_dir: Path, *, render: bool = True
) -> dict[str, Any]:
    """Run matrix mode: execute grid of experiments and write results.csv.

    Args:
        args: Parsed command-line arguments.
        exp_dir: Experiment directory.
        render: Whether to generate plots and visual report.

    Returns:
        Global summary dictionary.
    """
    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # Generate grid - use ablation spec if provided
    if args.ablation_spec:
        spec_path = Path(args.ablation_spec)
        grid = load_ablation_spec(spec_path, args)
    else:
        grid = generate_matrix_grid(args, seeds)

    # Create runs directory
    runs_dir = exp_dir / "artifacts" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Prepare CSV
    csv_path = exp_dir / "artifacts" / "results.csv"
    csv_rows: list[dict[str, Any]] = []

    all_summaries: list[dict[str, Any]] = []

    # Determine task for later use
    task = getattr(args, "task", "quadratic")
    if grid:
        task = grid[0].get("task", task)

    for idx, config in enumerate(grid):
        run_id = f"run_{idx:04d}"
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Write per-run config.json
        config_path = run_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)

        # Run experiment
        summary = run_once(config, run_dir, save_history=args.save_histories)
        summary["run_id"] = run_id

        all_summaries.append(summary)

        # Add to CSV rows
        csv_rows.append(summary_to_csv_row(run_id, summary))

    # Write results.csv
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(csv_rows)

    # Compute global summary
    global_summary = _compute_global_summary(all_summaries, runs_dir, task=task)

    # Write global summary.json
    global_summary_path = exp_dir / "artifacts" / "summary.json"
    with global_summary_path.open("w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2, sort_keys=True)

    # Generate plots and visual report if rendering is enabled
    if render:
        artifacts_dir = exp_dir / "artifacts"

        # Read results CSV for plotting
        rows = read_results_csv(csv_path)

        # Generate aggregate plots
        plots = plot_matrix_results(rows, artifacts_dir, task=task)

        # Generate visual report
        report_md = render_matrix_visual_report_md(
            exp_dir,
            global_summary=global_summary,
            results_csv_path=csv_path,
            plots=plots,
        )
        report_path = artifacts_dir / "report.md"
        with report_path.open("w", encoding="utf-8") as f:
            f.write(report_md)

    return global_summary


def _compute_global_summary(
    summaries: list[dict[str, Any]],
    runs_dir: Path,
    *,
    task: str = "quadratic",
) -> dict[str, Any]:
    """Compute global summary statistics from all runs.

    Args:
        summaries: List of summary dictionaries from all runs.
        runs_dir: Directory containing run artifacts.
        task: Task type ("quadratic" or "logistic").

    Returns:
        Global summary dictionary.
    """
    n_runs = len(summaries)

    # Find best by mean loss
    best_loss_idx = min(range(n_runs), key=lambda i: summaries[i]["final_mean_loss"])
    best_loss = summaries[best_loss_idx]

    # Compute averages
    avg_loss = float(np.mean([s["final_mean_loss"] for s in summaries]))

    # Average consensus error (only for gossip runs)
    cons_errors = [s["final_consensus_error"] for s in summaries if "final_consensus_error" in s]
    avg_cons_error = float(np.mean(cons_errors)) if cons_errors else None

    global_summary: dict[str, Any] = {
        "number_of_runs": n_runs,
        "best_by_final_mean_loss": {
            "run_id": best_loss["run_id"],
            "final_mean_loss": best_loss["final_mean_loss"],
            "optimizer": best_loss.get("optimizer", ""),
            "topology": best_loss.get("topology", ""),
            "strategy": best_loss.get("strategy", ""),
            "step_schedule": best_loss.get("step_schedule", ""),
            "seed": best_loss.get("seed", ""),
        },
        "averages": {
            "mean_final_mean_loss": avg_loss,
        },
    }

    # Add suboptimality-related stats only for quadratic task
    if task == "quadratic":
        subopt_values = [
            s["final_suboptimality"]
            for s in summaries
            if "final_suboptimality" in s and s["final_suboptimality"] is not None
        ]
        if subopt_values:
            best_subopt_idx = min(
                range(n_runs),
                key=lambda i: summaries[i].get("final_suboptimality", float("inf")),
            )
            best_subopt = summaries[best_subopt_idx]
            avg_subopt = float(np.mean(subopt_values))

            global_summary["best_by_final_suboptimality"] = {
                "run_id": best_subopt["run_id"],
                "final_suboptimality": best_subopt["final_suboptimality"],
                "optimizer": best_subopt.get("optimizer", ""),
                "topology": best_subopt.get("topology", ""),
                "strategy": best_subopt.get("strategy", ""),
                "step_schedule": best_subopt.get("step_schedule", ""),
                "seed": best_subopt.get("seed", ""),
            }
            global_summary["averages"]["mean_final_suboptimality"] = avg_subopt

    # Add accuracy stats for logistic task
    if task == "logistic":
        acc_values: list[float] = [
            float(s.get("final_mean_accuracy") or s.get("final_accuracy") or 0)
            for s in summaries
            if s.get("final_mean_accuracy") or s.get("final_accuracy")
        ]
        if acc_values:
            avg_acc = float(np.mean(acc_values))
            global_summary["averages"]["mean_final_accuracy"] = avg_acc

    # Add runs directory
    global_summary["runs_directory"] = str(runs_dir)

    if avg_cons_error is not None:
        global_summary["averages"]["mean_final_consensus_error"] = avg_cons_error

    return global_summary


def generate_readme(
    exp_dir: Path,
    args: argparse.Namespace,
    config: dict[str, Any],
) -> str:
    """Generate README.md content."""
    exp_name = args.exp_name or exp_dir.name
    lines = [f"# {exp_name}", ""]

    if args.description:
        lines.extend([args.description, ""])

    # Add note about visual report
    lines.extend(
        [
            "> See `artifacts/report.md` for plots and aggregated tables.",
            "",
        ]
    )

    lines.extend(
        [
            "## Configuration",
            "",
            f"- Mode: {args.mode}",
            f"- Environment: {args.env}",
            f"- Steps: {args.steps}",
            f"- Dimension: {args.dim}",
            f"- Condition number: {args.cond}",
            f"- Optimizer: {args.optimizer}",
            f"- Learning rate: {args.lr}",
            f"- Constraint: {args.constraint}",
            f"- Radius: {args.radius}",
        ]
    )

    if args.mode in ("single", "checks"):
        lines.append(f"- Seed: {args.seed}")
        if args.optimizer == "fw":
            lines.append(f"- Step schedule: {args.step_schedule}")
            if args.step_schedule == "constant":
                lines.append(f"- Gamma: {args.gamma}")

    if args.env == "gossip":
        lines.append(f"- Nodes: {args.n_nodes}")
        if args.mode in ("single", "checks"):
            lines.extend(
                [
                    f"- Topology: {args.topology}",
                    f"- Strategy: {args.strategy}",
                ]
            )

    if args.mode == "matrix":
        lines.extend(
            [
                f"- Matrix: {args.matrix}",
                f"- Seeds: {args.seeds}",
                f"- Save histories: {args.save_histories}",
            ]
        )

    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "python -m benchmarks.runner \\",
        ]
    )

    # Build reproduce command
    cmd_parts = [
        f"    --mode {args.mode}",
        f"    --workflow-dir {args.workflow_dir}",
        f"    --env {args.env}",
        f"    --steps {args.steps}",
        f"    --dim {args.dim}",
        f"    --cond {args.cond}",
        f"    --optimizer {args.optimizer}",
        f"    --lr {args.lr}",
        f"    --constraint {args.constraint}",
        f"    --radius {args.radius}",
    ]

    if args.mode in ("single", "checks"):
        cmd_parts.append(f"    --seed {args.seed}")
        if args.optimizer == "fw":
            cmd_parts.append(f"    --step-schedule {args.step_schedule}")
            if args.step_schedule == "constant":
                cmd_parts.append(f"    --gamma {args.gamma}")
    elif args.mode == "matrix":
        cmd_parts.append(f'    --seeds "{args.seeds}"')
        cmd_parts.append(f"    --matrix {args.matrix}")
        if args.save_histories:
            cmd_parts.append("    --save-histories")

    if args.env == "gossip":
        cmd_parts.append(f"    --n-nodes {args.n_nodes}")
        if args.mode in ("single", "checks"):
            cmd_parts.append(f"    --topology {args.topology}")
            cmd_parts.append(f"    --strategy {args.strategy}")

    if args.exp_name:
        cmd_parts.append(f'    --exp-name "{args.exp_name}"')
    if args.description:
        cmd_parts.append(f'    --description "{args.description}"')

    # Join with backslash continuation
    for part in cmd_parts[:-1]:
        lines.append(part + " \\")
    lines.append(cmd_parts[-1])

    lines.extend(["```", ""])

    return "\n".join(lines)


def run_single_mode(
    args: argparse.Namespace, exp_dir: Path, *, render: bool = True
) -> dict[str, Any]:
    """Run single mode: execute one experiment.

    Args:
        args: Parsed command-line arguments.
        exp_dir: Experiment directory.
        render: Whether to generate plots and visual report.

    Returns:
        Summary dictionary with final metrics.
    """
    config = build_config(args)

    # Create artifacts directory
    artifacts_dir = exp_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Run the experiment
    summary = run_once(config, artifacts_dir, save_history=True)

    # Generate plots and report if rendering is enabled
    if render:
        history_path = artifacts_dir / "history.jsonl"
        if history_path.exists():
            history = read_history_jsonl(history_path)
            plots = plot_single_run(
                history,
                artifacts_dir,
                task=config.get("task", "quadratic"),
                env=config.get("env", "single"),
            )

            # Generate visual report
            report_md = render_single_run_report_md(
                exp_dir,
                summary=summary,
                plots=plots,
            )
            report_path = artifacts_dir / "report.md"
            with report_path.open("w", encoding="utf-8") as f:
                f.write(report_md)

    return summary


def run_checks_mode(args: argparse.Namespace, exp_dir: Path) -> tuple[dict[str, Any], bool]:
    """Run checks mode: execute convergence/regression checks.

    Args:
        args: Parsed command-line arguments.
        exp_dir: Experiment directory.

    Returns:
        Tuple of (checks_summary_dict, all_passed).
    """
    # Build config with defaults for checks mode
    # Use smaller steps if not explicitly set
    if args.steps == 50:  # Default value - use smaller for checks
        args.steps = 30

    config = build_config(args)

    # Create artifacts directory
    artifacts_dir = exp_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Run checks
    summary = run_checks(config)

    # Write checks.json
    checks_json_path = artifacts_dir / "checks.json"
    with checks_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary.to_json(), f, indent=2, sort_keys=True)

    # Write checks.md
    checks_md_path = artifacts_dir / "checks.md"
    md_content = render_checks_markdown(summary, config)
    with checks_md_path.open("w", encoding="utf-8") as f:
        f.write(md_content)

    return summary.to_json(), summary.passed


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point for the experiment runner.

    Args:
        argv: Command-line arguments. If None, uses sys.argv[1:].

    Returns:
        Exit code (0 for success, 2 for checks failure).
    """
    args = parse_args(argv)
    config = build_config(args)

    # Create experiment directory
    workflow_dir = Path(args.workflow_dir)
    exp_dir = next_experiment_dir(workflow_dir)

    print(f"Created experiment directory: {exp_dir}")

    # Determine if rendering is enabled (default: True)
    render = not getattr(args, "no_render", False)

    # Run based on mode
    checks_passed = True
    if args.mode == "single":
        summary = run_single_mode(args, exp_dir, render=render)
    elif args.mode == "matrix":
        summary = run_matrix_mode(args, exp_dir, render=render)
    else:  # checks mode
        summary, checks_passed = run_checks_mode(args, exp_dir)

    # Write metadata
    meta: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "argv": sys.argv if argv is None else ["runner"] + list(argv),
        "mode": args.mode,
    }
    git_commit = try_get_git_commit()
    if git_commit:
        meta["git_commit"] = git_commit

    readme_text = generate_readme(exp_dir, args, config)
    write_run_files(exp_dir, meta=meta, config=config, readme_text=readme_text)

    # Print summary
    if args.mode == "single":
        print(f"Experiment completed: {exp_dir.name}")
        print(f"  final_mean_loss: {summary['final_mean_loss']:.6f}")
        if "final_suboptimality" in summary:
            print(f"  final_suboptimality: {summary['final_suboptimality']:.6f}")
        if "final_accuracy" in summary:
            print(f"  final_accuracy: {summary['final_accuracy']:.6f}")
        if "final_mean_accuracy" in summary:
            print(f"  final_mean_accuracy: {summary['final_mean_accuracy']:.6f}")
        if "final_consensus_error" in summary:
            print(f"  final_consensus_error: {summary['final_consensus_error']:.6f}")
    elif args.mode == "matrix":
        print(f"Matrix experiment completed: {exp_dir.name}")
        print(f"  number_of_runs: {summary['number_of_runs']}")
        # Use best_by_final_mean_loss for logistic task (no suboptimality)
        if "best_by_final_suboptimality" in summary:
            best = summary["best_by_final_suboptimality"]
            print(f"  best_run_id: {best['run_id']}")
            print(f"  best_optimizer: {best['optimizer']}")
            print(f"  best_strategy: {best['strategy']}")
            print(f"  best_topology: {best['topology']}")
            print(f"  best_final_suboptimality: {best['final_suboptimality']:.6f}")
        elif "best_by_final_mean_loss" in summary:
            best = summary["best_by_final_mean_loss"]
            print(f"  best_run_id: {best['run_id']}")
            print(f"  best_optimizer: {best.get('optimizer', 'N/A')}")
            print(f"  best_strategy: {best.get('strategy', 'N/A')}")
            print(f"  best_topology: {best.get('topology', 'N/A')}")
            print(f"  best_final_mean_loss: {best['final_mean_loss']:.6f}")
        avgs = summary.get("averages", {})
        if "mean_final_consensus_error" in avgs:
            print(f"  mean_final_consensus_error: {avgs['mean_final_consensus_error']:.6f}")
        print(f"  report: {exp_dir / 'artifacts' / 'report.md'}")
    else:  # checks mode
        status = "✅ PASSED" if checks_passed else "❌ FAILED"
        print(f"Checks completed: {exp_dir.name}")
        print(f"  status: {status}")
        print(f"  num_checks: {summary['num_checks']}")
        print(f"  num_passed: {summary['num_passed']}")
        print(f"  num_failed: {summary['num_failed']}")
        print(f"  checks.json: {exp_dir / 'artifacts' / 'checks.json'}")
        print(f"  checks.md: {exp_dir / 'artifacts' / 'checks.md'}")

        # Return exit code 2 if checks failed (for CI)
        if not checks_passed:
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
