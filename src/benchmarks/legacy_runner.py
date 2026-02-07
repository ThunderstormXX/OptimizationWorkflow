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
import math

from benchmarks.animate import (
    build_metadata_strings,
    read_trace_jsonl,
    render_animation,
    write_trace_line,
)
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
from optim.adam import AdamOptimizer, AdamPGDOptimizer
from optim.constraints import L1BallConstraint, L2BallConstraint
from optim.frank_wolfe import FrankWolfeOptimizer, FWState, constant_step_size, harmonic_step_size
from optim.gradient_descent import (
    GDState,
    GradientDescentOptimizer,
    ProjectedGradientDescentOptimizer,
)
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

try:
    from tqdm import tqdm  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

__all__ = ["main", "run_once", "run_checks_mode", "CSV_COLUMNS"]

# CSV column order (stable)
CSV_COLUMNS = [
    "run_id",
    "task",
    "model",
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
    "heterogeneity",
    "dim",
    "cond",
    "seed",
    "steps",
    "final_mean_loss",
    "final_mean_accuracy",
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
        choices=["fw", "gd", "pgd", "adam"],
        default="fw",
        help="Optimizer: fw=Frank-Wolfe, gd=Gradient Descent, pgd=Projected GD, adam=Adam",
    )
    parser.add_argument(
        "--lr", type=float, default=0.1, help="Learning rate for gd/pgd/adam optimizers"
    )

    # Constraint (for fw, pgd, and adam with projection)
    parser.add_argument(
        "--constraint",
        type=str,
        choices=["l2ball", "l1ball", "simplex", "none"],
        default="l2ball",
        help="Constraint type (for fw, pgd, adam)",
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
        choices=["quadratic", "logistic", "mnist"],
        default="quadratic",
        help="Task type: quadratic, logistic regression, or mnist",
    )

    # Logistic regression options
    parser.add_argument(
        "--n-samples", type=int, default=2000, help="Number of samples (logistic only)"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (logistic/mnist)")
    parser.add_argument(
        "--heterogeneity",
        type=str,
        choices=["iid", "label_skew"],
        default="iid",
        help="Data heterogeneity type (logistic/mnist + gossip only)",
    )

    # MNIST-specific options
    parser.add_argument(
        "--model",
        type=str,
        choices=["mlp3", "cnn"],
        default="mlp3",
        help="Model architecture for MNIST (mlp3 or cnn)",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=256,
        help="Hidden layer size for MLP3 (mnist only)",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=60000,
        help="Number of training samples (mnist only, max 60000)",
    )
    parser.add_argument(
        "--n-val",
        type=int,
        default=10000,
        help="Number of validation samples (mnist only, max 10000)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=".data",
        help="Directory for MNIST data (mnist only)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download MNIST data if not present",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for MNIST training (cpu or cuda)",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=20,
        help="Evaluate validation accuracy every N steps (mnist only)",
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=50,
        help="Number of batches for validation evaluation (mnist only)",
    )
    parser.add_argument(
        "--use-fake-data",
        action="store_true",
        help="[Dev only] Use FakeData instead of real MNIST",
    )
    parser.add_argument(
        "--mnist-fixture",
        type=str,
        default=None,
        help="Path to MNIST tiny fixture directory (for tests, avoids download)",
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
        choices=["small", "medium", "large", "mnist_dist_big"],
        default="small",
        help="Matrix size/preset (matrix mode only). mnist_dist_big is for MNIST ablation.",
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

    # Animation
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Generate animated visualization (MP4) of the optimization process",
    )
    parser.add_argument(
        "--animate-format",
        type=str,
        choices=["mp4", "gif"],
        default="mp4",
        help="Animation output format (default: mp4)",
    )
    parser.add_argument(
        "--animate-top-k",
        type=int,
        default=5,
        help="For matrix mode: animate only top-K runs by best metric",
    )
    parser.add_argument(
        "--animate-metric",
        type=str,
        choices=["mean_loss", "final_suboptimality", "final_mean_loss", "final_mean_accuracy"],
        default="final_mean_loss",
        help="Metric to use for selecting top-K runs to animate",
    )
    parser.add_argument(
        "--animate-fps",
        type=int,
        default=8,
        help="Frames per second for animation",
    )
    parser.add_argument(
        "--animate-max-steps",
        type=int,
        default=200,
        help="Maximum frames in animation (subsample if steps > this)",
    )

    # Descriptive
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name")
    parser.add_argument("--description", type=str, default=None, help="Experiment description")

    return parser.parse_args(argv)


def build_config(args: argparse.Namespace, seed: int | None = None) -> dict[str, Any]:
    """Build configuration dictionary from parsed arguments.

    Returns a fully resolved config dict with explicit keys.
    Non-applicable fields are set to "na" for consistency.
    """
    task = args.task

    # Determine model name based on task
    if task == "mnist":
        model = getattr(args, "model", "mlp3")
    else:
        model = "NumpyVector"

    config: dict[str, Any] = {
        # Core experiment settings
        "env": args.env,
        "task": task,
        "model": model,
        "steps": args.steps,
        "seed": seed if seed is not None else args.seed,
        "dim": args.dim,
        "cond": args.cond if task == "quadratic" else "na",
        # Optimizer settings
        "optimizer": args.optimizer,
        "lr": args.lr,
        "constraint": args.constraint,
        "step_schedule": args.step_schedule if args.optimizer == "fw" else "na",
        "radius": args.radius,
        "gamma": args.gamma if args.optimizer == "fw" else "na",
        # Data settings (logistic/mnist)
        "n_samples": (getattr(args, "n_samples", 2000) if task in ("logistic", "mnist") else "na"),
        "batch_size": (getattr(args, "batch_size", 64) if task in ("logistic", "mnist") else "na"),
        "heterogeneity": (
            getattr(args, "heterogeneity", "iid") if task in ("logistic", "mnist") else "na"
        ),
        # MNIST-specific settings
        "hidden": getattr(args, "hidden", 256) if task == "mnist" else "na",
        "n_train": getattr(args, "n_train", 60000) if task == "mnist" else "na",
        "n_val": getattr(args, "n_val", 10000) if task == "mnist" else "na",
        "data_root": getattr(args, "data_root", ".data") if task == "mnist" else "na",
        "device": getattr(args, "device", "cpu") if task == "mnist" else "na",
        "eval_every": getattr(args, "eval_every", 20) if task == "mnist" else "na",
        "eval_batches": getattr(args, "eval_batches", 50) if task == "mnist" else "na",
        "use_fake_data": getattr(args, "use_fake_data", False) if task == "mnist" else "na",
        "mnist_fixture": getattr(args, "mnist_fixture", None) if task == "mnist" else "na",
        "download": getattr(args, "download", False) if task == "mnist" else "na",
        # Gossip settings
        "n_nodes": args.n_nodes if args.env == "gossip" else "na",
        "topology": args.topology if args.env == "gossip" else "na",
        "strategy": args.strategy if args.env == "gossip" else "na",
    }

    # Descriptive fields
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
    save_trace: bool = False,
) -> dict[str, Any]:
    """Run a single experiment with the given configuration.

    Args:
        config: Configuration dictionary with all parameters.
        out_dir: Directory to write artifacts (summary.json, optionally history.jsonl).
        save_history: Whether to save history.jsonl.
        save_trace: Whether to save trace.jsonl for animation.

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
            summary = _run_single_env_quadratic(
                config, task, out_dir, save_history=save_history, save_trace=save_trace
            )
        else:
            summary = _run_gossip_env_quadratic(
                config, task, out_dir, save_history=save_history, save_trace=save_trace
            )
    elif task_type == "mnist":
        # MNIST classification task (requires torch)
        if config["env"] == "single":
            summary = _run_single_env_mnist(
                config, out_dir, save_history=save_history, save_trace=save_trace
            )
        else:
            summary = _run_gossip_env_mnist(
                config, out_dir, save_history=save_history, save_trace=save_trace
            )
    else:
        # Logistic regression task
        if config["env"] == "single":
            summary = _run_single_env_logistic(
                config, out_dir, save_history=save_history, save_trace=save_trace
            )
        else:
            summary = _run_gossip_env_logistic(
                config, out_dir, save_history=save_history, save_trace=save_trace
            )

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
    save_trace: bool = False,
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

    # Open trace file if needed
    trace_file = None
    if save_trace:
        trace_path = out_dir / "trace.jsonl"
        trace_file = trace_path.open("w", encoding="utf-8")

    try:
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

            # Write trace line if requested
            if trace_file:
                params = model.parameters_vector()
                node_metrics = {
                    "0": {
                        "loss": step_result.loss,
                        "dist_to_opt": step_result.metrics.get("dist_to_opt", 0.0),
                        "param_norm": float(np.linalg.norm(params)),
                    }
                }
                mean_metrics = {
                    "mean_loss": step_result.loss,
                    "suboptimality": suboptimality(task, params),
                }
                if "dist_to_opt" in step_result.metrics:
                    mean_metrics["mean_dist_to_opt"] = step_result.metrics["dist_to_opt"]
                write_trace_line(
                    trace_file,
                    t=t,
                    env="single",
                    node_metrics=node_metrics,
                    mean_metrics=mean_metrics,
                    params_by_node={"0": params.tolist()},
                )
    finally:
        if trace_file:
            trace_file.close()

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
        "task": "quadratic",
        "model": "NumpyVector",
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
    save_trace: bool = False,
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

    # Extract comm_graph for animation (if save_trace is enabled)
    comm_graph: dict[str, Any] | None = None
    if save_trace:
        edge_weights = communicator.edge_weights()
        comm_graph = {
            "edges": [[e[0], e[1]] for e in edge_weights],
            "weights": [e[2] for e in edge_weights],
        }

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

    # Open trace file if needed
    trace_file = None
    if save_trace:
        trace_path = out_dir / "trace.jsonl"
        trace_file = trace_path.open("w", encoding="utf-8")

    try:
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
            cons_err = consensus_error(params_by_node)
            line["consensus_error"] = cons_err

            history_lines.append(line)

            # Write trace line if requested
            if trace_file:
                x_bar = mean_params(params_by_node)
                node_metrics_dict: dict[str, dict[str, float]] = {}
                for i in range(n_nodes):
                    node_metrics_dict[str(i)] = {
                        "loss": result[i].loss,
                        "dist_to_opt": result[i].metrics.get("dist_to_opt", 0.0),
                        "param_norm": float(np.linalg.norm(params_by_node[i])),
                    }
                mean_metrics_dict: dict[str, float] = {
                    "mean_loss": mean_loss,
                    "suboptimality": suboptimality(task, x_bar),
                    "consensus_error": cons_err,
                }
                if "dist_to_opt" in result[0].metrics:
                    mean_metrics_dict["mean_dist_to_opt"] = line["mean_dist_to_opt"]
                write_trace_line(
                    trace_file,
                    t=t,
                    env="gossip",
                    node_metrics=node_metrics_dict,
                    mean_metrics=mean_metrics_dict,
                    params_by_node={str(i): params_by_node[i].tolist() for i in range(n_nodes)},
                    comm={"mode": "sync"},
                )
    finally:
        if trace_file:
            trace_file.close()

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
        "task": "quadratic",
        "model": "NumpyVector",
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

    # Add comm_graph if we saved trace
    if comm_graph is not None:
        summary["comm_graph"] = comm_graph

    return summary


def _run_single_env_logistic(
    config: dict[str, Any],
    out_dir: Path,
    *,
    save_history: bool = True,
    save_trace: bool = False,
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

    # Open trace file if needed
    trace_file = None
    if save_trace:
        trace_path = out_dir / "trace.jsonl"
        trace_file = trace_path.open("w", encoding="utf-8")

    try:
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

            # Write trace line if requested
            if trace_file:
                params = model.parameters_vector()
                node_metrics = {
                    "0": {
                        "loss": step_result.loss,
                        "accuracy": step_result.metrics.get("accuracy", 0.0),
                        "param_norm": float(np.linalg.norm(params)),
                    }
                }
                mean_metrics = {
                    "mean_loss": step_result.loss,
                    "mean_accuracy": step_result.metrics.get("accuracy", 0.0),
                }
                write_trace_line(
                    trace_file,
                    t=t,
                    env="single",
                    node_metrics=node_metrics,
                    mean_metrics=mean_metrics,
                    params_by_node={"0": params.tolist()},
                )
    finally:
        if trace_file:
            trace_file.close()

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
        "model": "NumpyVector",
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
        "final_mean_accuracy": final_metrics["accuracy"],
    }

    return summary


def _run_gossip_env_logistic(
    config: dict[str, Any],
    out_dir: Path,
    *,
    save_history: bool = True,
    save_trace: bool = False,
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

    # Extract comm_graph for animation (if save_trace is enabled)
    comm_graph: dict[str, Any] | None = None
    if save_trace:
        edge_weights = communicator.edge_weights()
        comm_graph = {
            "edges": [[e[0], e[1]] for e in edge_weights],
            "weights": [e[2] for e in edge_weights],
        }

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

    # Open trace file if needed
    trace_file = None
    if save_trace:
        trace_path = out_dir / "trace.jsonl"
        trace_file = trace_path.open("w", encoding="utf-8")

    try:
        for t in range(steps):
            step_record = env.step()
            result: Mapping[NodeId, StepResult] = step_record  # type: ignore[assignment]

            losses = [result[i].loss for i in range(n_nodes)]
            mean_loss = float(np.mean(losses))

            line: dict[str, Any] = {
                "step": t,
                "mean_loss": mean_loss,
            }

            mean_acc = 0.0
            if "accuracy" in result[0].metrics:
                accs = [result[i].metrics["accuracy"] for i in range(n_nodes)]
                mean_acc = float(np.mean(accs))
                line["mean_accuracy"] = mean_acc

            params_by_node = env.get_params_by_node()
            cons_err = consensus_error(params_by_node)
            line["consensus_error"] = cons_err

            history_lines.append(line)

            # Write trace line if requested
            if trace_file:
                node_metrics_dict: dict[str, dict[str, float]] = {}
                for i in range(n_nodes):
                    node_metrics_dict[str(i)] = {
                        "loss": result[i].loss,
                        "accuracy": result[i].metrics.get("accuracy", 0.0),
                        "param_norm": float(np.linalg.norm(params_by_node[i])),
                    }
                mean_metrics_dict: dict[str, float] = {
                    "mean_loss": mean_loss,
                    "mean_accuracy": mean_acc,
                    "consensus_error": cons_err,
                }
                write_trace_line(
                    trace_file,
                    t=t,
                    env="gossip",
                    node_metrics=node_metrics_dict,
                    mean_metrics=mean_metrics_dict,
                    params_by_node={str(i): params_by_node[i].tolist() for i in range(n_nodes)},
                    comm={"mode": "sync"},
                )
    finally:
        if trace_file:
            trace_file.close()

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
        "model": "NumpyVector",
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

    # Add comm_graph if we saved trace
    if comm_graph is not None:
        summary["comm_graph"] = comm_graph

    return summary


# =============================================================================
# MNIST task functions
# =============================================================================


def _run_single_env_mnist(
    config: dict[str, Any],
    out_dir: Path,
    *,
    save_history: bool = True,
    save_trace: bool = False,
) -> dict[str, Any]:
    """Run single-process environment experiment with MNIST task."""
    # Import torch modules (check availability)
    try:
        from models.torch_adapter import TorchModelAdapter
        from models.torch_nets import build_mnist_torch_model
        from tasks.mnist import (
            MNISTClassificationTask,
            TorchGradComputer,
            load_mnist_or_fake,
        )
    except ImportError as e:
        raise RuntimeError(
            "PyTorch is required for MNIST task. Install with: pip install torch torchvision"
        ) from e

    seed = config["seed"]
    steps = config["steps"]
    optimizer_name = config["optimizer"]
    model_name = config.get("model", "mlp3")
    hidden = config.get("hidden", 256)
    batch_size = config.get("batch_size", 128)
    n_train = config.get("n_train", 60000)
    n_val = config.get("n_val", 10000)
    data_root = Path(config.get("data_root", ".data"))
    device = config.get("device", "cpu")
    eval_every = config.get("eval_every", 20)
    eval_batches = config.get("eval_batches", 50)
    use_fake_data = config.get("use_fake_data", False)
    mnist_fixture = config.get("mnist_fixture")
    download = config.get("download", False)

    # Set random seeds
    rng = np.random.default_rng(seed)
    import torch

    torch.manual_seed(seed)

    # Load datasets
    if mnist_fixture:
        # Use tiny fixture for tests
        from tasks.mnist import load_mnist_tiny

        train_ds, val_ds = load_mnist_tiny(Path(mnist_fixture))
    else:
        train_ds = load_mnist_or_fake(
            use_fake_data=use_fake_data,
            n_samples=n_train,
            seed=seed,
            root=data_root,
            train=True,
            download=download,
        )
        val_ds = load_mnist_or_fake(
            use_fake_data=use_fake_data,
            n_samples=n_val,
            seed=seed + 1000,
            root=data_root,
            train=False,
            download=download,
        )

    # Build model and task
    torch_model = build_mnist_torch_model(model_name, hidden=hidden)
    model = TorchModelAdapter(torch_model, device=device)
    task = MNISTClassificationTask(
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=batch_size,
        device=device,
        eval_every=eval_every,
        eval_batches=eval_batches,
    )
    grad_computer = TorchGradComputer()

    # Build optimizer
    constraint_type = config.get("constraint", "none")
    radius = config.get("radius", 1.0)

    # Auto radius: use init_norm * 1.5
    if radius == "auto" or (constraint_type != "none" and radius <= 0):
        init_norm = float(np.linalg.norm(model.parameters_vector()))
        radius = init_norm * 1.5

    optimizer, state = _build_mnist_optimizer(
        optimizer_name, config, model, constraint_type, radius
    )

    # History tracking
    history: list[dict[str, Any]] = []
    trace_file = None
    if save_trace:
        trace_path = out_dir / "trace.jsonl"
        trace_file = trace_path.open("w", encoding="utf-8")

    # Training loop with progress (epochs + samples) when tqdm is available
    steps_per_epoch = max((n_train + batch_size - 1) // batch_size, 1)
    num_epochs = math.ceil(steps / steps_per_epoch)

    use_tqdm = tqdm is not None
    epoch_bar = None
    if use_tqdm:
        epoch_bar = tqdm(total=num_epochs, desc="Epochs", unit="epoch", position=0)

    t = 0
    for _epoch in range(num_epochs):
        if t >= steps:
            break
        steps_this_epoch = min(steps_per_epoch, steps - t)

        batch_bar = None
        if use_tqdm:
            total_samples = steps_this_epoch * batch_size
            batch_bar = tqdm(
                total=total_samples,
                desc="Samples",
                unit="sample",
                position=1,
                leave=False,
            )

        for _ in range(steps_this_epoch):
            batch = task.sample_batch(rng=rng)
            state, result = optimizer.step(
                task=task,
                model=model,
                batch=batch,
                grad_computer=grad_computer,
                state=state,
                rng=rng,
            )

            # Compute metrics with step number for eval scheduling
            metrics = task.metrics(model, batch, t=t)

            line: dict[str, Any] = {
                "step": t,
                "mean_loss": result.loss,
                "train_accuracy": metrics.get("train_accuracy", 0.0),
            }
            if "val_accuracy" in metrics:
                line["mean_accuracy"] = metrics["val_accuracy"]
                line["val_loss"] = metrics.get("val_loss", 0.0)

            history.append(line)

            # Write trace line
            if save_trace and trace_file:
                node_metrics = {
                    "0": {
                        "loss": result.loss,
                        "train_accuracy": metrics.get("train_accuracy", 0.0),
                        "val_accuracy": metrics.get("val_accuracy", 0.0),
                        "param_norm": float(np.linalg.norm(model.parameters_vector())),
                    }
                }
                mean_metrics = {
                    "mean_loss": result.loss,
                    "mean_train_accuracy": metrics.get("train_accuracy", 0.0),
                }
                if "val_accuracy" in metrics:
                    mean_metrics["mean_accuracy"] = metrics["val_accuracy"]

                write_trace_line(
                    trace_file,
                    t=t,
                    env="single",
                    node_metrics=node_metrics,
                    mean_metrics=mean_metrics,
                    params_by_node=None,  # Don't dump full params for MNIST
                )

            if batch_bar is not None:
                batch_bar.update(batch_size)

            t += 1

        if batch_bar is not None:
            batch_bar.close()
        if epoch_bar is not None:
            epoch_bar.update(1)

    if epoch_bar is not None:
        epoch_bar.close()

    if trace_file:
        trace_file.close()

    # Write history
    if save_history:
        history_path = out_dir / "history.jsonl"
        with history_path.open("w", encoding="utf-8") as f:
            for line in history:
                f.write(json.dumps(line) + "\n")

    # Final metrics
    final_metrics = task.metrics(model, None, t=steps)

    summary: dict[str, Any] = {
        "env": "single",
        "task": "mnist",
        "model": model_name,
        "hidden": hidden,
        "steps": steps,
        "seed": seed,
        "n_train": n_train,
        "n_val": n_val,
        "batch_size": batch_size,
        "device": device,
        "optimizer": optimizer_name,
        "lr": config["lr"],
        "constraint": constraint_type,
        "radius": radius,
        "final_mean_loss": final_metrics.get("train_loss", 0.0),
        "final_train_accuracy": final_metrics.get("train_accuracy", 0.0),
        "final_mean_accuracy": final_metrics.get("val_accuracy", 0.0),
    }

    return summary


def _run_gossip_env_mnist(
    config: dict[str, Any],
    out_dir: Path,
    *,
    save_history: bool = True,
    save_trace: bool = False,
) -> dict[str, Any]:
    """Run gossip environment experiment with MNIST task."""
    # Import torch modules
    try:
        from models.torch_adapter import TorchModelAdapter
        from models.torch_nets import build_mnist_torch_model
        from tasks.mnist import (
            MNISTClassificationTask,
            TorchGradComputer,
            load_mnist_or_fake,
            split_mnist_across_nodes,
        )
    except ImportError as e:
        raise RuntimeError(
            "PyTorch is required for MNIST task. Install with: pip install torch torchvision"
        ) from e

    seed = config["seed"]
    steps = config["steps"]
    n_nodes = config["n_nodes"]
    topology_name = config["topology"]
    strategy_name = config["strategy"]
    optimizer_name = config["optimizer"]
    model_name = config.get("model", "mlp3")
    hidden = config.get("hidden", 256)
    batch_size = config.get("batch_size", 128)
    n_train = config.get("n_train", 60000)
    n_val = config.get("n_val", 10000)
    data_root = Path(config.get("data_root", ".data"))
    device = config.get("device", "cpu")
    eval_every = config.get("eval_every", 20)
    eval_batches = config.get("eval_batches", 50)
    use_fake_data = config.get("use_fake_data", False)
    mnist_fixture = config.get("mnist_fixture")
    download = config.get("download", False)
    heterogeneity = config.get("heterogeneity", "iid")

    # Set random seeds
    import torch

    torch.manual_seed(seed)

    # Load and split datasets
    if mnist_fixture:
        # Use tiny fixture for tests
        from tasks.mnist import load_mnist_tiny

        train_ds, val_ds = load_mnist_tiny(Path(mnist_fixture))
    else:
        train_ds = load_mnist_or_fake(
            use_fake_data=use_fake_data,
            n_samples=n_train,
            seed=seed,
            root=data_root,
            train=True,
            download=download,
        )
        val_ds = load_mnist_or_fake(
            use_fake_data=use_fake_data,
            n_samples=n_val,
            seed=seed + 1000,
            root=data_root,
            train=False,
            download=download,
        )

    # Split training data across nodes
    node_train_datasets = split_mnist_across_nodes(train_ds, n_nodes, heterogeneity, seed)

    # Build topology
    if topology_name == "ring":
        topology: Topology = RingTopology(n=n_nodes)
    else:
        topology = CompleteTopology(n=n_nodes)

    # Build communicator
    communicator = SynchronousGossipCommunicator(topology=topology)

    # Extract comm_graph for animation
    comm_graph: dict[str, Any] | None = None
    if save_trace:
        edge_weights = communicator.edge_weights()
        comm_graph = {
            "edges": [[e[0], e[1]] for e in edge_weights],
            "weights": [e[2] for e in edge_weights],
        }

    # Build nodes
    constraint_type = config.get("constraint", "none")
    radius = config.get("radius", 1.0)

    # Build initial model to get param count for auto radius
    torch_model_init = build_mnist_torch_model(model_name, hidden=hidden)
    model_init = TorchModelAdapter(torch_model_init, device=device)
    init_norm = float(np.linalg.norm(model_init.parameters_vector()))

    if radius == "auto" or (constraint_type != "none" and radius <= 0):
        radius = init_norm * 1.5

    # Build strategy using registry (with config for gradient_tracking)
    strategy = get_strategy_with_config(strategy_name, config)

    # Build nodes with their own models and tasks
    nodes: list[GossipNode[Any, Any, Any]] = []
    for i in range(n_nodes):
        torch_model = build_mnist_torch_model(model_name, hidden=hidden)
        # Initialize with same weights as first node for fair comparison
        torch_model.load_state_dict(torch_model_init.state_dict())
        node_model = TorchModelAdapter(torch_model, device=device)

        node_task = MNISTClassificationTask(
            train_dataset=node_train_datasets[i],
            val_dataset=val_ds,  # Shared validation set
            batch_size=batch_size,
            device=device,
            eval_every=eval_every,
            eval_batches=eval_batches,
        )

        node_optimizer, node_state = _build_mnist_optimizer(
            optimizer_name, config, node_model, constraint_type, radius
        )

        nodes.append(
            GossipNode(
                node_id=i,
                task=node_task,
                model=node_model,
                optimizer=node_optimizer,
                grad_computer=TorchGradComputer(),
                opt_state=node_state,
                rng=np.random.default_rng(seed + i),
            )
        )

    # Build environment
    env: GossipEnvironment[Any, Any, Any] = GossipEnvironment(
        nodes=nodes,
        communicator=communicator,
        strategy=strategy,
    )

    # Reset environment
    env.reset(seed=seed)

    # History and trace
    history: list[dict[str, Any]] = []
    trace_file = None
    if save_trace:
        trace_path = out_dir / "trace.jsonl"
        trace_file = trace_path.open("w", encoding="utf-8")

    # Training loop
    for t in range(steps):
        # Run environment step (handles batching internally)
        step_record = env.step()
        result: Mapping[NodeId, StepResult] = step_record  # type: ignore[assignment]

        # Compute metrics
        params_by_node = {i: nodes[i].model.parameters_vector() for i in range(n_nodes)}
        cons_err = consensus_error(params_by_node)

        # Aggregate metrics across nodes
        train_losses = [result[i].loss for i in range(n_nodes)]
        mean_train_loss = float(np.mean(train_losses))

        # Get train/val accuracy from metrics
        train_accs = []
        val_accs = []
        for i in range(n_nodes):
            train_acc = result[i].metrics.get("train_accuracy", 0.0)
            train_accs.append(train_acc)
            if "val_accuracy" in result[i].metrics:
                val_accs.append(result[i].metrics["val_accuracy"])

        mean_train_acc = float(np.mean(train_accs))
        mean_val_acc = float(np.mean(val_accs)) if val_accs else None

        line: dict[str, Any] = {
            "step": t,
            "mean_loss": mean_train_loss,
            "mean_train_accuracy": mean_train_acc,
            "consensus_error": cons_err,
        }
        if mean_val_acc is not None:
            line["mean_accuracy"] = mean_val_acc

        history.append(line)

        # Write trace
        if save_trace and trace_file:
            node_metrics_dict = {
                str(i): {
                    "loss": result[i].loss,
                    "train_accuracy": train_accs[i],
                    "param_norm": float(np.linalg.norm(params_by_node[i])),
                }
                for i in range(n_nodes)
            }
            mean_metrics_dict: dict[str, float] = {
                "mean_loss": mean_train_loss,
                "mean_train_accuracy": mean_train_acc,
                "consensus_error": cons_err,
            }
            if mean_val_acc is not None:
                mean_metrics_dict["mean_accuracy"] = mean_val_acc

            write_trace_line(
                trace_file,
                t=t,
                env="gossip",
                node_metrics=node_metrics_dict,
                mean_metrics=mean_metrics_dict,
                params_by_node=None,  # Don't dump full params
                comm={"mode": "sync"},
            )

    if trace_file:
        trace_file.close()

    # Write history
    if save_history:
        history_path = out_dir / "history.jsonl"
        with history_path.open("w", encoding="utf-8") as f:
            for line in history:
                f.write(json.dumps(line) + "\n")

    # Final metrics
    final_params_by_node = {i: nodes[i].model.parameters_vector() for i in range(n_nodes)}
    final_cons_error = consensus_error(final_params_by_node)

    final_train_losses = [nodes[i].task.loss(nodes[i].model, None) for i in range(n_nodes)]
    final_mean_loss = float(np.mean(final_train_losses))

    # Get final validation accuracy (force eval)
    final_val_accs = []
    for node in nodes:
        # Compute val metrics directly using private method
        if hasattr(node.task, "_compute_val_metrics"):
            val_loss, val_acc = node.task._compute_val_metrics(node.model)  # noqa: SLF001
            final_val_accs.append(val_acc)

    final_mean_val_acc = float(np.mean(final_val_accs)) if final_val_accs else 0.0

    summary: dict[str, Any] = {
        "env": "gossip",
        "task": "mnist",
        "model": model_name,
        "hidden": hidden,
        "steps": steps,
        "seed": seed,
        "n_train": n_train,
        "n_val": n_val,
        "batch_size": batch_size,
        "device": device,
        "heterogeneity": heterogeneity,
        "optimizer": optimizer_name,
        "lr": config["lr"],
        "constraint": constraint_type,
        "radius": radius,
        "n_nodes": n_nodes,
        "topology": topology_name,
        "strategy": strategy_name,
        "final_mean_loss": final_mean_loss,
        "final_mean_accuracy": final_mean_val_acc,
        "final_consensus_error": final_cons_error,
    }

    if comm_graph is not None:
        summary["comm_graph"] = comm_graph

    return summary


def _build_mnist_optimizer(
    optimizer_name: str,
    config: dict[str, Any],
    model: Any,
    constraint_type: str,
    radius: float,
) -> tuple[Any, Any]:
    """Build optimizer and initial state for MNIST.

    Args:
        optimizer_name: Name of optimizer (adam, gd, pgd, fw).
        config: Configuration dictionary.
        model: TorchModelAdapter.
        constraint_type: Constraint type (none, l1ball, l2ball).
        radius: Constraint radius.

    Returns:
        Tuple of (optimizer, initial_state).
    """
    lr = config.get("lr", 0.001)

    # Build constraint if needed
    constraint: L1BallConstraint | L2BallConstraint | None = None
    if constraint_type == "l2ball":
        constraint = L2BallConstraint(radius=radius)
    elif constraint_type == "l1ball":
        constraint = L1BallConstraint(radius=radius)

    optimizer: Any
    state: Any

    if optimizer_name == "adam":
        if constraint is not None:
            optimizer = AdamPGDOptimizer(constraint=constraint, lr=lr)
        else:
            optimizer = AdamOptimizer(lr=lr)
        state = optimizer.init_state(model)
    elif optimizer_name == "gd":
        optimizer = GradientDescentOptimizer(lr=lr)
        state = optimizer.init_state(model)
    elif optimizer_name == "pgd":
        if constraint is None:
            raise ValueError("PGD requires a constraint (l1ball or l2ball)")
        optimizer = ProjectedGradientDescentOptimizer(lr=lr, constraint=constraint)
        state = optimizer.init_state(model)
    elif optimizer_name == "fw":
        if constraint is None:
            raise ValueError("Frank-Wolfe requires a constraint (l1ball or l2ball)")
        step_schedule = config.get("step_schedule", "harmonic")
        gamma = config.get("gamma", 0.2)
        if step_schedule == "harmonic":
            step_size_fn = harmonic_step_size()
        else:
            step_size_fn = constant_step_size(gamma)
        optimizer = FrankWolfeOptimizer(constraint=constraint, step_size=step_size_fn)
        state = optimizer.init_state(model)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer, state


def summary_to_csv_row(run_id: str, summary: dict[str, Any]) -> dict[str, Any]:
    """Convert summary dict to CSV row dict with stable column order."""
    # Determine model name based on task
    task = summary.get("task", "quadratic")
    if task == "mnist":
        model = summary.get("model", "na")
    else:
        model = "NumpyVector"

    row: dict[str, Any] = {
        "run_id": run_id,
        "task": task,
        "model": model,
        "env": summary.get("env", ""),
        "n_nodes": summary.get("n_nodes", "na"),
        "topology": summary.get("topology", "na"),
        "strategy": summary.get("strategy", "na"),
        "optimizer": summary.get("optimizer", ""),
        "lr": summary.get("lr", ""),
        "constraint": summary.get("constraint", ""),
        "schedule": summary.get("step_schedule", "na"),
        "gamma": summary.get("gamma", ""),
        "radius": summary.get("radius", ""),
        "heterogeneity": summary.get("heterogeneity", "na"),
        "dim": summary.get("dim", ""),
        "cond": summary.get("cond", "na"),
        "seed": summary.get("seed", ""),
        "steps": summary.get("steps", ""),
        "final_mean_loss": summary.get("final_mean_loss", ""),
        "final_mean_accuracy": summary.get(
            "final_mean_accuracy", summary.get("final_accuracy", "")
        ),
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

    For matrix=mnist_dist_big:
    - model: [mlp3, cnn]
    - topology: [ring, complete]
    - strategy: [local_then_gossip, gossip_then_local]
    - heterogeneity: [iid, label_skew]
    - optimizer: adam
    - constraint: [none, l2ball, l1ball]
    - radius: [auto, 10.0, 20.0] (for constrained only)
    - lr: [0.001, 0.0005]
    """
    # Handle mnist_dist_big preset specially
    if args.matrix == "mnist_dist_big":
        return _generate_mnist_dist_big_grid(args, seeds)

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


def _generate_mnist_dist_big_grid(
    args: argparse.Namespace,
    seeds: list[int],
) -> list[dict[str, Any]]:
    """Generate grid for large distributed MNIST experiment.

    Grid:
    - model: [mlp3, cnn]
    - topology: [ring, complete]
    - strategy: [local_then_gossip, gossip_then_local]
    - heterogeneity: [iid, label_skew]
    - constraint: [none, l2ball, l1ball]
    - radius: [auto] for constrained, [na] for none
    - lr: [0.001, 0.0005]
    - seed: from --seeds
    """
    models = ["mlp3", "cnn"]
    topologies = ["ring", "complete"]
    strategies = ["local_then_gossip", "gossip_then_local"]
    heterogeneities = ["iid", "label_skew"]
    constraints = ["none", "l2ball", "l1ball"]
    lrs = [0.001, 0.0005]

    configs: list[dict[str, Any]] = []

    # Grid iteration
    for model, topo, strat, hetero, constraint, lr, seed in product(
        models, topologies, strategies, heterogeneities, constraints, lrs, seeds
    ):
        # For constrained optimizers, use auto radius
        # For none constraint, radius is not applicable
        radius: Any = "auto" if constraint != "none" else "na"

        config: dict[str, Any] = {
            "env": "gossip",
            "task": "mnist",
            "model": model,
            "hidden": getattr(args, "hidden", 256),
            "steps": args.steps,
            "seed": seed,
            "optimizer": "adam",
            "lr": lr,
            "constraint": constraint,
            "radius": radius,
            "n_nodes": args.n_nodes,
            "topology": topo,
            "strategy": strat,
            "heterogeneity": hetero,
            "n_train": getattr(args, "n_train", 30000),
            "n_val": getattr(args, "n_val", 10000),
            "batch_size": getattr(args, "batch_size", 128),
            "data_root": getattr(args, "data_root", ".data"),
            "download": getattr(args, "download", False),
            "device": getattr(args, "device", "cpu"),
            "eval_every": getattr(args, "eval_every", 50),
            "eval_batches": getattr(args, "eval_batches", 50),
            "use_fake_data": getattr(args, "use_fake_data", False),
            "mnist_fixture": getattr(args, "mnist_fixture", None),
            # Not applicable for MNIST
            "dim": "na",
            "cond": "na",
            "step_schedule": "na",
            "gamma": "na",
            "n_samples": "na",
        }

        configs.append(config)

    return configs


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
    args: argparse.Namespace,
    exp_dir: Path,
    *,
    render: bool = True,
    animate: bool = False,
) -> dict[str, Any]:
    """Run matrix mode: execute grid of experiments and write results.csv.

    Args:
        args: Parsed command-line arguments.
        exp_dir: Experiment directory.
        render: Whether to generate plots and visual report.
        animate: Whether to generate animated visualizations for top-K runs.

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

    # If animating, we need to save traces for all runs
    save_trace = animate

    for idx, config in enumerate(grid):
        run_id = f"run_{idx:04d}"
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Write per-run config.json
        config_path = run_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)

        # Run experiment (save trace if animating)
        summary = run_once(config, run_dir, save_history=args.save_histories, save_trace=save_trace)
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
    plots: dict[str, Path] = {}
    if render:
        artifacts_dir = exp_dir / "artifacts"

        # Read results CSV for plotting
        rows = read_results_csv(csv_path)

        # Generate aggregate plots
        plots = plot_matrix_results(rows, artifacts_dir, task=task)

    # Generate animations for top-K runs if requested
    animated_runs: list[dict[str, Any]] = []
    if animate:
        artifacts_dir = exp_dir / "artifacts"
        top_k = getattr(args, "animate_top_k", 5)
        animate_metric = getattr(args, "animate_metric", "final_mean_loss")
        anim_format = getattr(args, "animate_format", "mp4")
        ext = f".{anim_format}"

        # Sort runs by metric
        # For accuracy metrics: higher is better (negate for sorting)
        # For loss/suboptimality: lower is better
        if "accuracy" in animate_metric:
            sorted_summaries = sorted(
                all_summaries,
                key=lambda s: (-s.get(animate_metric, 0), s.get("final_mean_loss", float("inf"))),
            )
        else:
            sorted_summaries = sorted(
                all_summaries, key=lambda s: s.get(animate_metric, float("inf"))
            )

        # Animate top-K runs
        for run_summary in sorted_summaries[:top_k]:
            run_id = run_summary["run_id"]
            run_dir = runs_dir / run_id
            trace_path = run_dir / "trace.jsonl"

            if trace_path.exists():
                trace = read_trace_jsonl(trace_path)
                animation_path = run_dir / f"animation{ext}"

                # Get config for this run
                config_path = run_dir / "config.json"
                with config_path.open("r", encoding="utf-8") as f:
                    run_config = json.load(f)

                # Get comm_graph from summary if available
                comm_graph = run_summary.get("comm_graph")

                render_animation(
                    trace=trace,
                    out_path=animation_path,
                    topology=run_config.get("topology"),
                    fps=getattr(args, "animate_fps", 8),
                    max_steps=getattr(args, "animate_max_steps", 200),
                    title=f"{run_id} ({run_config.get('optimizer', 'N/A')})",
                    config=run_config,
                    comm_graph=comm_graph,
                    output_format=anim_format,
                )

                # Build metadata filename
                _, _, filename_base = build_metadata_strings(run_config)

                animated_runs.append(
                    {
                        "run_id": run_id,
                        "config_summary": f"opt={run_config.get('optimizer')}, "
                        f"strat={run_config.get('strategy')}, "
                        f"topo={run_config.get('topology')}",
                        "metric_value": run_summary.get(animate_metric, "N/A"),
                        "animation_path": f"runs/{run_id}/animation{ext}",
                        "filename_base": filename_base,
                    }
                )

        # Write animations.md index
        if animated_runs:
            animations_md_path = artifacts_dir / "animations.md"
            with animations_md_path.open("w", encoding="utf-8") as f:
                f.write("# Animated Runs\n\n")
                f.write(f"Top {len(animated_runs)} runs by `{animate_metric}`:\n\n")
                f.write(f"Format: {anim_format.upper()}\n\n")
                f.write("| Run ID | Config | Metric Value | Animation |\n")
                f.write("|--------|--------|--------------|------------|\n")
                for run in animated_runs:
                    metric_str = (
                        f"{run['metric_value']:.6f}"
                        if isinstance(run["metric_value"], float)
                        else str(run["metric_value"])
                    )
                    f.write(
                        f"| {run['run_id']} | {run['config_summary']} | {metric_str} "
                        f"| [{anim_format.upper()}]({run['animation_path']}) |\n"
                    )
                f.write("\n\n## Metadata Filenames\n\n")
                for run in animated_runs:
                    f.write(f"- `{run['run_id']}`: `{run['filename_base']}`\n")

    # Generate visual report
    if render:
        report_md = render_matrix_visual_report_md(
            exp_dir,
            global_summary=global_summary,
            results_csv_path=csv_path,
            plots=plots,
        )
        # Add link to animations if generated
        if animated_runs:
            report_md += "\n## Animations\n\n"
            report_md += "See [animations.md](animations.md) for animated visualizations "
            report_md += f"of the top {len(animated_runs)} runs.\n"
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
    args: argparse.Namespace,
    exp_dir: Path,
    *,
    render: bool = True,
    animate: bool = False,
) -> dict[str, Any]:
    """Run single mode: execute one experiment.

    Args:
        args: Parsed command-line arguments.
        exp_dir: Experiment directory.
        render: Whether to generate plots and visual report.
        animate: Whether to generate animated visualization.

    Returns:
        Summary dictionary with final metrics.
    """
    config = build_config(args)

    # Create artifacts directory
    artifacts_dir = exp_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Run the experiment (with trace if animating)
    summary = run_once(config, artifacts_dir, save_history=True, save_trace=animate)

    # Generate plots and report if rendering is enabled
    plots: dict[str, Path] = {}
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

    # Generate animation if requested
    animation_path: Path | None = None
    if animate:
        trace_path = artifacts_dir / "trace.jsonl"
        if trace_path.exists():
            trace = read_trace_jsonl(trace_path)

            # Determine format from args
            anim_format = getattr(args, "animate_format", "mp4")
            ext = f".{anim_format}"

            # Build metadata-based filename
            _, _, filename_base = build_metadata_strings(config)

            # Standard animation path (e.g., animation.mp4)
            animation_path = artifacts_dir / f"animation{ext}"

            # Metadata-named version (e.g., animation__SyntheticLogit__...mp4)
            metadata_path = artifacts_dir / f"{filename_base}{ext}"

            # Get comm_graph from summary if available
            comm_graph = summary.get("comm_graph")

            render_animation(
                trace=trace,
                out_path=animation_path,
                topology=config.get("topology"),
                fps=getattr(args, "animate_fps", 8),
                max_steps=getattr(args, "animate_max_steps", 200),
                title=config.get("exp_name", exp_dir.name),
                config=config,
                comm_graph=comm_graph,
                output_format=anim_format,
            )

            # Create symlink or copy with metadata name
            if animation_path.exists() and not metadata_path.exists():
                try:
                    metadata_path.symlink_to(animation_path.name)
                except OSError:
                    # Fall back to copy if symlink fails
                    import shutil

                    shutil.copy(animation_path, metadata_path)

    # Generate visual report
    if render:
        report_md = render_single_run_report_md(
            exp_dir,
            summary=summary,
            plots=plots,
        )
        # Add animation link if generated
        if animation_path and animation_path.exists():
            anim_format = getattr(args, "animate_format", "mp4")
            report_md += "\n## Animation\n\n"
            report_md += f"[View Animation](animation.{anim_format})\n"
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
    animate = getattr(args, "animate", False)

    # Run based on mode
    checks_passed = True
    if args.mode == "single":
        summary = run_single_mode(args, exp_dir, render=render, animate=animate)
    elif args.mode == "matrix":
        summary = run_matrix_mode(args, exp_dir, render=render, animate=animate)
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
