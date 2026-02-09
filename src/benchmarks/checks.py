"""Convergence and regression check suite.

This module provides a fast, deterministic check suite to verify that:
- Optimizers decrease suboptimality on SyntheticQuadraticTask
- Constrained optimizers preserve feasibility
- Gossip runs achieve consensus (consensus_error decreases)

The checks are designed to be run locally or in CI without manual inspection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from benchmarks.metrics import consensus_error, suboptimality
from benchmarks.registry import get_optimizer, get_strategy
from core.protocols import Topology
from core.types import NodeId
from distributed.communicator import SynchronousGossipCommunicator
from distributed.strategies import GossipNode
from distributed.topology import CompleteTopology, RingTopology
from environments.gossip import GossipEnvironment
from environments.single_process import SingleProcessEnvironment
from models.numpy_vector import NumpyVectorModel
from optim.legacy_frankwolfe import FWState, GDState
from tasks.synthetic_quadratic import (
    QuadraticGradComputer,
    SyntheticQuadraticTask,
    make_spd_quadratic,
)

__all__ = [
    "CheckResult",
    "ChecksSummary",
    "check_single_decreases_suboptimality",
    "check_constraint_feasibility",
    "check_gossip_consensus_decreases",
    "run_checks",
]


@dataclass
class CheckResult:
    """Result of a single check.

    Attributes:
        name: Name of the check.
        passed: Whether the check passed.
        details: Additional details (numeric values, thresholds, config).
    """

    name: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "passed": self.passed,
            "details": self.details,
        }


@dataclass
class ChecksSummary:
    """Summary of all checks.

    Attributes:
        passed: Whether all checks passed.
        results: List of individual check results.
    """

    passed: bool
    results: list[CheckResult] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "passed": self.passed,
            "num_checks": len(self.results),
            "num_passed": sum(1 for r in self.results if r.passed),
            "num_failed": sum(1 for r in self.results if not r.passed),
            "results": [r.to_dict() for r in self.results],
        }


def _get_init_state(optimizer_name: str) -> FWState | GDState:
    """Get initial optimizer state based on optimizer type."""
    if optimizer_name == "fw":
        return FWState(t=0)
    else:
        return GDState(t=0)


def _is_valid_number(x: float) -> bool:
    """Check if a number is finite (not NaN or inf)."""
    return math.isfinite(x)


def check_single_decreases_suboptimality(config: dict[str, Any]) -> CheckResult:
    """Check that optimizer decreases suboptimality on single env.

    Args:
        config: Configuration with keys: optimizer, dim, cond, seed, steps,
                constraint, radius, gamma, step_schedule, lr.

    Returns:
        CheckResult with pass/fail and details.
    """
    optimizer_name = config.get("optimizer", "fw")
    dim = config.get("dim", 10)
    cond = config.get("cond", 10.0)
    seed = config.get("seed", 0)
    steps = config.get("steps", 30)

    # Ratio thresholds by optimizer
    ratio_thresholds = {
        "fw": 0.8,
        "gd": 0.6,
        "pgd": 0.8,
    }
    ratio_threshold = ratio_thresholds.get(optimizer_name, 0.8)

    # Build problem
    rng = np.random.default_rng(seed)
    problem = make_spd_quadratic(dim=dim, rng=rng, cond=cond)
    task = SyntheticQuadraticTask(problem)
    grad_computer = QuadraticGradComputer()

    # Build optimizer
    optimizer = get_optimizer(optimizer_name, config)

    # Build model
    model_rng = np.random.default_rng(seed + 1000)
    x0 = model_rng.standard_normal(dim)
    model = NumpyVectorModel(x0)

    # Compute initial suboptimality
    initial_subopt = suboptimality(task, model.parameters_vector())

    # Check for invalid initial value
    if not _is_valid_number(initial_subopt):
        return CheckResult(
            name="single_decreases_suboptimality",
            passed=False,
            details={
                "error": "Initial suboptimality is NaN or inf",
                "optimizer": optimizer_name,
            },
        )

    # Build and run environment
    env: SingleProcessEnvironment[Any, Any, Any] = SingleProcessEnvironment(
        task=task,  # type: ignore[arg-type]
        model=model,
        optimizer=optimizer,
        grad_computer=grad_computer,  # type: ignore[arg-type]
    )
    env.reset(seed=seed)
    env.run(steps=steps)

    # Compute final suboptimality
    final_subopt = suboptimality(task, model.parameters_vector())

    # Check for invalid final value
    if not _is_valid_number(final_subopt):
        return CheckResult(
            name="single_decreases_suboptimality",
            passed=False,
            details={
                "error": "Final suboptimality is NaN or inf",
                "optimizer": optimizer_name,
                "initial_suboptimality": initial_subopt,
            },
        )

    # Check condition
    threshold = initial_subopt * ratio_threshold
    passed = final_subopt <= threshold

    return CheckResult(
        name="single_decreases_suboptimality",
        passed=passed,
        details={
            "optimizer": optimizer_name,
            "initial_suboptimality": initial_subopt,
            "final_suboptimality": final_subopt,
            "ratio_threshold": ratio_threshold,
            "threshold": threshold,
            "ratio_achieved": final_subopt / initial_subopt if initial_subopt > 0 else 0,
            "steps": steps,
        },
    )


def check_constraint_feasibility(config: dict[str, Any]) -> CheckResult:
    """Check that constrained optimizers preserve feasibility.

    For fw with l2ball and pgd with l2ball:
    - Assert ||x|| <= R + eps at the end

    Args:
        config: Configuration dictionary.

    Returns:
        CheckResult with pass/fail and details.
    """
    optimizer_name = config.get("optimizer", "fw")
    constraint_type = config.get("constraint", "l2ball")
    env_type = config.get("env", "single")
    dim = config.get("dim", 10)
    cond = config.get("cond", 10.0)
    seed = config.get("seed", 0)
    steps = config.get("steps", 30)
    radius = config.get("radius", 1.0)
    eps = 1e-8

    # Only check for l2ball constraint with fw or pgd
    if constraint_type != "l2ball" or optimizer_name not in ("fw", "pgd"):
        return CheckResult(
            name="constraint_feasibility",
            passed=True,
            details={
                "skipped": True,
                "reason": f"Not applicable for {optimizer_name} with {constraint_type}",
            },
        )

    # Build problem
    rng = np.random.default_rng(seed)
    problem = make_spd_quadratic(dim=dim, rng=rng, cond=cond)
    task = SyntheticQuadraticTask(problem)
    grad_computer = QuadraticGradComputer()

    # Build optimizer
    optimizer = get_optimizer(optimizer_name, config)

    if env_type == "single":
        # Build model
        model_rng = np.random.default_rng(seed + 1000)
        x0 = model_rng.standard_normal(dim)
        model = NumpyVectorModel(x0)

        # Build and run environment
        env: SingleProcessEnvironment[Any, Any, Any] = SingleProcessEnvironment(
            task=task,  # type: ignore[arg-type]
            model=model,
            optimizer=optimizer,
            grad_computer=grad_computer,  # type: ignore[arg-type]
        )
        env.reset(seed=seed)
        env.run(steps=steps)

        # Check feasibility
        final_params = model.parameters_vector()
        final_norm = float(np.linalg.norm(final_params))
        passed = final_norm <= radius + eps

        return CheckResult(
            name="constraint_feasibility",
            passed=passed,
            details={
                "optimizer": optimizer_name,
                "constraint": constraint_type,
                "radius": radius,
                "final_norm": final_norm,
                "eps": eps,
                "bound": radius + eps,
            },
        )
    else:
        # Gossip environment - check all nodes
        n_nodes = config.get("n_nodes", 5)
        topology_name = config.get("topology", "ring")
        strategy_name = config.get("strategy", "local_then_gossip")

        # Build topology and communicator
        topology: Topology
        if topology_name == "ring":
            topology = RingTopology(n=n_nodes)
        else:
            topology = CompleteTopology(n=n_nodes)
        communicator = SynchronousGossipCommunicator(topology=topology)

        # Build strategy
        strategy = get_strategy(strategy_name)

        # Build nodes
        master_rng = np.random.default_rng(seed + 2000)
        nodes: list[GossipNode[Any, Any, Any]] = []

        for i in range(n_nodes):
            x0 = master_rng.standard_normal(dim)
            model = NumpyVectorModel(x0)
            node_task = SyntheticQuadraticTask(problem)
            node_grad_computer = QuadraticGradComputer()
            node_optimizer = get_optimizer(optimizer_name, config)
            init_state = _get_init_state(optimizer_name)

            node: GossipNode[Any, Any, Any] = GossipNode(
                node_id=i,
                task=node_task,  # type: ignore[arg-type]
                model=model,
                optimizer=node_optimizer,
                grad_computer=node_grad_computer,  # type: ignore[arg-type]
                opt_state=init_state,
                rng=np.random.default_rng(0),
            )
            nodes.append(node)

        # Build and run environment
        gossip_env: GossipEnvironment[Any, Any, Any] = GossipEnvironment(
            nodes=nodes,
            communicator=communicator,
            strategy=strategy,
        )
        gossip_env.reset(seed=seed)
        gossip_env.run(steps=steps)

        # Check feasibility for all nodes
        params_by_node = gossip_env.get_params_by_node()
        norms: dict[int, float] = {}
        all_feasible = True

        for node_id in range(n_nodes):
            norm = float(np.linalg.norm(params_by_node[node_id]))
            norms[node_id] = norm
            if norm > radius + eps:
                all_feasible = False

        return CheckResult(
            name="constraint_feasibility",
            passed=all_feasible,
            details={
                "optimizer": optimizer_name,
                "constraint": constraint_type,
                "radius": radius,
                "eps": eps,
                "bound": radius + eps,
                "node_norms": norms,
                "max_norm": max(norms.values()),
            },
        )


def check_gossip_consensus_decreases(config: dict[str, Any]) -> CheckResult:
    """Check that gossip achieves consensus (consensus_error decreases).

    Args:
        config: Configuration dictionary.

    Returns:
        CheckResult with pass/fail and details.
    """
    env_type = config.get("env", "single")

    # Only applicable for gossip
    if env_type != "gossip":
        return CheckResult(
            name="gossip_consensus_decreases",
            passed=True,
            details={
                "skipped": True,
                "reason": "Not applicable for single env",
            },
        )

    optimizer_name = config.get("optimizer", "fw")
    dim = config.get("dim", 10)
    cond = config.get("cond", 10.0)
    seed = config.get("seed", 0)
    steps = config.get("steps", 30)
    n_nodes = config.get("n_nodes", 5)
    topology_name = config.get("topology", "ring")
    strategy_name = config.get("strategy", "local_then_gossip")

    # Ratio thresholds by topology
    ratio_thresholds = {
        "ring": 0.5,
        "complete": 0.1,
    }
    ratio_threshold = ratio_thresholds.get(topology_name, 0.5)

    # Build problem
    rng = np.random.default_rng(seed)
    problem = make_spd_quadratic(dim=dim, rng=rng, cond=cond)

    # Build topology and communicator
    topology: Topology
    if topology_name == "ring":
        topology = RingTopology(n=n_nodes)
    else:
        topology = CompleteTopology(n=n_nodes)
    communicator = SynchronousGossipCommunicator(topology=topology)

    # Build strategy
    strategy = get_strategy(strategy_name)

    # Build nodes
    master_rng = np.random.default_rng(seed + 2000)
    nodes: list[GossipNode[Any, Any, Any]] = []

    for i in range(n_nodes):
        x0 = master_rng.standard_normal(dim)
        model = NumpyVectorModel(x0)
        node_task = SyntheticQuadraticTask(problem)
        node_grad_computer = QuadraticGradComputer()
        node_optimizer = get_optimizer(optimizer_name, config)
        init_state = _get_init_state(optimizer_name)

        node: GossipNode[Any, Any, Any] = GossipNode(
            node_id=i,
            task=node_task,  # type: ignore[arg-type]
            model=model,
            optimizer=node_optimizer,
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
    env.reset(seed=seed)

    # Compute initial consensus error
    initial_params: dict[NodeId, Any] = {}
    for node in nodes:
        initial_params[node.node_id] = node.model.parameters_vector()
    initial_cons_error = consensus_error(initial_params)

    # Run
    env.run(steps=steps)

    # Compute final consensus error
    final_params = env.get_params_by_node()
    final_cons_error = consensus_error(final_params)

    # Check condition
    # Handle edge case where initial error is very small
    if initial_cons_error < 1e-12:
        passed = final_cons_error < 1e-12
    else:
        threshold = initial_cons_error * ratio_threshold
        passed = final_cons_error <= threshold

    return CheckResult(
        name="gossip_consensus_decreases",
        passed=passed,
        details={
            "topology": topology_name,
            "strategy": strategy_name,
            "n_nodes": n_nodes,
            "initial_consensus_error": initial_cons_error,
            "final_consensus_error": final_cons_error,
            "ratio_threshold": ratio_threshold,
            "threshold": initial_cons_error * ratio_threshold,
            "ratio_achieved": (
                final_cons_error / initial_cons_error if initial_cons_error > 1e-12 else 0
            ),
            "steps": steps,
        },
    )


def run_checks(config: dict[str, Any]) -> ChecksSummary:
    """Run all applicable checks for the given configuration.

    Args:
        config: Configuration dictionary with env, optimizer, constraint, etc.

    Returns:
        ChecksSummary with all check results.
    """
    results: list[CheckResult] = []

    # Always run suboptimality check
    results.append(check_single_decreases_suboptimality(config))

    # Run feasibility check for constrained optimizers
    optimizer_name = config.get("optimizer", "fw")
    constraint_type = config.get("constraint", "l2ball")
    if optimizer_name in ("fw", "pgd") and constraint_type == "l2ball":
        results.append(check_constraint_feasibility(config))

    # Run consensus check for gossip env
    env_type = config.get("env", "single")
    if env_type == "gossip":
        results.append(check_gossip_consensus_decreases(config))

    # Determine overall pass/fail
    all_passed = all(r.passed for r in results)

    return ChecksSummary(passed=all_passed, results=results)
