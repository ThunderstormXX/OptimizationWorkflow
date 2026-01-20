"""Metrics computation for benchmark experiments.

This module provides metric helpers for computing optimization quality
measures like suboptimality and consensus error.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from core.types import NodeId, ParamVector

# Import consensus_error from environments to avoid duplication
from environments.gossip import consensus_error as _consensus_error
from tasks.synthetic_quadratic import SyntheticQuadraticTask

__all__ = [
    "suboptimality",
    "consensus_error",
    "mean_params",
]


def suboptimality(task: SyntheticQuadraticTask, x: ParamVector) -> float:
    """Compute the suboptimality gap: f(x) - f(x*).

    Args:
        task: The quadratic task containing the problem definition.
        x: Current parameter vector.

    Returns:
        The suboptimality gap (non-negative for convex problems).
    """
    f_x = task.problem.loss(x)
    f_star = task.problem.loss(task.x_star)
    return float(f_x - f_star)


def consensus_error(params_by_node: Mapping[NodeId, ParamVector]) -> float:
    """Compute the consensus error across nodes.

    Reuses the implementation from environments.gossip.

    Args:
        params_by_node: Mapping from node ID to parameter vector.

    Returns:
        Mean L2 distance from each node to the global average.
    """
    return _consensus_error(params_by_node)


def mean_params(params_by_node: Mapping[NodeId, ParamVector]) -> ParamVector:
    """Compute the mean parameter vector across nodes.

    Args:
        params_by_node: Mapping from node ID to parameter vector.

    Returns:
        The element-wise average of all parameter vectors.

    Raises:
        ValueError: If params_by_node is empty.
    """
    if not params_by_node:
        raise ValueError("Cannot compute mean of empty params")

    vectors = [params_by_node[i] for i in sorted(params_by_node.keys())]
    result: ParamVector = np.mean(vectors, axis=0)
    return result
