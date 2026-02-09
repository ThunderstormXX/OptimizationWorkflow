"""Registry for optimizers and gossip strategies.

This module provides a clean factory layer for creating optimizers and strategies,
making it easy to add new methods to the benchmarking framework.

Adding a new optimizer:
1. Implement the Optimizer protocol (see core.protocols)
2. Register it here with register_optimizer(name, factory_fn)
3. It becomes available in CLI via --optimizer name

Adding a new strategy:
1. Implement the GossipStrategy protocol (see distributed.strategies)
2. Register it here with register_strategy(name, factory_fn)
3. It becomes available in CLI via --strategy name

Example:
    >>> from benchmarks.registry import get_optimizer, get_strategy
    >>> opt = get_optimizer("fw", {"constraint": L2BallConstraint(1.0), ...})
    >>> strat = get_strategy("local_then_gossip")
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from distributed.strategies import (
    GossipStrategy,
    GossipThenLocalStep,
    GradientTrackingStrategy,
    LocalStepThenGossipParams,
)
from optim.legacy_frankwolfe import L2BallConstraint, SimplexConstraint
from optim.legacy_frankwolfe import (
    FrankWolfeOptimizer,
    constant_step_size,
    harmonic_step_size,
)

__all__ = [
    "OPTIMIZERS",
    "STRATEGIES",
    "register_optimizer",
    "get_optimizer",
    "register_strategy",
    "get_strategy",
]

# Type aliases for factory functions
OptimizerFactory = Callable[[dict[str, Any]], Any]
StrategyFactory = Callable[[], GossipStrategy[Any, Any, Any]]

# Global registries
OPTIMIZERS: dict[str, OptimizerFactory] = {}
STRATEGIES: dict[str, StrategyFactory] = {}


def register_optimizer(name: str, factory: OptimizerFactory) -> None:
    """Register an optimizer factory.

    Args:
        name: Unique name for the optimizer (used in CLI).
        factory: Callable that takes config dict and returns an Optimizer.

    Raises:
        ValueError: If name is already registered.
    """
    if name in OPTIMIZERS:
        raise ValueError(f"Optimizer '{name}' is already registered")
    OPTIMIZERS[name] = factory


def get_optimizer(name: str, config: dict[str, Any]) -> Any:
    """Get an optimizer instance from the registry.

    Args:
        name: Registered optimizer name.
        config: Configuration dictionary passed to the factory.

    Returns:
        An Optimizer instance.

    Raises:
        KeyError: If name is not registered.
    """
    if name not in OPTIMIZERS:
        available = ", ".join(sorted(OPTIMIZERS.keys()))
        raise KeyError(f"Unknown optimizer '{name}'. Available: {available}")
    return OPTIMIZERS[name](config)


def register_strategy(name: str, factory: StrategyFactory) -> None:
    """Register a gossip strategy factory.

    Args:
        name: Unique name for the strategy (used in CLI).
        factory: Callable that returns a GossipStrategy.

    Raises:
        ValueError: If name is already registered.
    """
    if name in STRATEGIES:
        raise ValueError(f"Strategy '{name}' is already registered")
    STRATEGIES[name] = factory


def get_strategy(name: str) -> GossipStrategy[Any, Any, Any]:
    """Get a strategy instance from the registry.

    Args:
        name: Registered strategy name.

    Returns:
        A GossipStrategy instance.

    Raises:
        KeyError: If name is not registered.
    """
    if name not in STRATEGIES:
        available = ", ".join(sorted(STRATEGIES.keys()))
        raise KeyError(f"Unknown strategy '{name}'. Available: {available}")
    return STRATEGIES[name]()


# =============================================================================
# Optimizer factories
# =============================================================================


def _build_constraint(config: dict[str, Any]) -> L2BallConstraint | SimplexConstraint:
    """Build constraint from config."""
    constraint_type = config.get("constraint", "l2ball")
    if constraint_type == "l2ball":
        return L2BallConstraint(radius=config.get("radius", 1.0))
    else:
        return SimplexConstraint(dim=config.get("dim", 10))


def _fw_factory(config: dict[str, Any]) -> FrankWolfeOptimizer[Any, Any]:
    """Factory for Frank-Wolfe optimizer."""
    constraint = _build_constraint(config)

    schedule_name = config.get("step_schedule", "harmonic")
    if schedule_name == "harmonic":
        step_size = harmonic_step_size()
    else:
        gamma = config.get("gamma", 0.2)
        step_size = constant_step_size(gamma)

    return FrankWolfeOptimizer(constraint=constraint, step_size=step_size)


def _gd_factory(config: dict[str, Any]) -> Any:
    """Factory for Gradient Descent optimizer."""
    # Import here to avoid circular dependency at module load
    from optim.legacy_frankwolfe import GradientDescentOptimizer

    lr = config.get("lr", 0.1)
    return GradientDescentOptimizer(lr=lr)


def _pgd_factory(config: dict[str, Any]) -> Any:
    """Factory for Projected Gradient Descent optimizer."""
    # Import here to avoid circular dependency at module load
    from optim.legacy_frankwolfe import ProjectedGradientDescentOptimizer

    lr = config.get("lr", 0.1)
    constraint = _build_constraint(config)
    return ProjectedGradientDescentOptimizer(lr=lr, constraint=constraint)


# =============================================================================
# Strategy factories
# =============================================================================


def _local_then_gossip_factory() -> LocalStepThenGossipParams:
    """Factory for LocalStepThenGossipParams strategy."""
    return LocalStepThenGossipParams()


def _gossip_then_local_factory() -> GossipThenLocalStep:
    """Factory for GossipThenLocalStep strategy."""
    return GossipThenLocalStep()


# Strategy factories that need config are handled differently
# We use a closure to capture the config
_gradient_tracking_lr: float = 0.1  # Module-level default


def _gradient_tracking_factory() -> GradientTrackingStrategy:
    """Factory for GradientTrackingStrategy."""
    return GradientTrackingStrategy(lr=_gradient_tracking_lr)


def get_strategy_with_config(name: str, config: dict[str, Any]) -> GossipStrategy[Any, Any, Any]:
    """Get a strategy instance with config support.

    Some strategies (like GradientTrackingStrategy) need config parameters.
    This function handles both simple and configurable strategies.

    Args:
        name: Registered strategy name.
        config: Configuration dictionary (used for configurable strategies).

    Returns:
        A GossipStrategy instance.

    Raises:
        KeyError: If name is not registered.
    """
    global _gradient_tracking_lr

    # Handle gradient_tracking specially to pass lr
    if name == "gradient_tracking":
        lr = config.get("lr", 0.1)
        return GradientTrackingStrategy(lr=lr)

    # Fall back to standard factory
    return get_strategy(name)


# =============================================================================
# Initial registrations
# =============================================================================

# Register optimizers
register_optimizer("fw", _fw_factory)
register_optimizer("gd", _gd_factory)
register_optimizer("pgd", _pgd_factory)

# Register strategies
register_strategy("local_then_gossip", _local_then_gossip_factory)
register_strategy("gossip_then_local", _gossip_then_local_factory)
register_strategy("gradient_tracking", _gradient_tracking_factory)
