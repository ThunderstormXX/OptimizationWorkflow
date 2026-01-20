"""Core type definitions for the benchmark framework.

This module contains:
- Type aliases for arrays, scalars, and optimization-related types
- Data containers for representing optimization state and results
- Configuration dataclasses
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "ParamVector",
    "NodeId",
    "StepResult",
    "StepRecord",
    "StepMeta",
    "History",
    "ExperimentConfig",
    # Payload type aliases
    "Payload",
    "PayloadMap",
    "MultiPayload",
    "MultiPayloadMap",
]

# Type alias for parameter vectors (model weights flattened)
ParamVector = np.ndarray

# Type alias for node identifiers in distributed settings
NodeId = int

# Payload type aliases for multi-channel messaging
Payload = ParamVector  # Single-channel payload (backward compatible)
PayloadMap = Mapping[NodeId, Payload]  # Map from node to single payload
MultiPayload = Mapping[str, ParamVector]  # Multi-channel: {"x": params, "y": tracker}
MultiPayloadMap = Mapping[NodeId, MultiPayload]  # Map from node to multi-channel payload


@dataclass(frozen=True, slots=True)
class StepResult:
    """Result of a single optimization step.

    Attributes:
        loss: The loss value computed during this step.
        metrics: Additional metrics computed during this step (e.g., accuracy, grad_norm).
    """

    loss: float
    metrics: Mapping[str, float] = field(default_factory=dict)


# A step record can be either:
# - A single StepResult (single-process run)
# - A mapping from NodeId to StepResult (multi-node distributed run)
StepRecord = StepResult | Mapping[NodeId, StepResult]


@dataclass(frozen=True, slots=True)
class StepMeta:
    """Metadata for a single optimization step (budget accounting).

    Attributes:
        num_grad_evals: Number of gradient evaluations in this step.
        num_gossip_rounds: Number of gossip communication rounds in this step.
    """

    num_grad_evals: int = 0
    num_gossip_rounds: int = 0


@dataclass
class History:
    """Container for storing optimization history across steps.

    Supports both single-process and multi-node distributed runs:
    - Single-process: each step stores a StepResult directly
    - Multi-node: each step stores a Mapping[NodeId, StepResult]

    Also tracks per-step metadata for budget accounting (gradient evals, gossip rounds).

    The aggregation methods (mean_loss, mean_metric) handle both cases:
    - For StepResult: use the value directly
    - For Mapping: average across nodes first, then average across steps

    Example:
        >>> history = History()
        >>> history.append(StepResult(loss=1.0, metrics={"acc": 0.9}))
        >>> history.append(StepResult(loss=0.5, metrics={"acc": 0.95}))
        >>> history.mean_loss()
        0.75
    """

    steps: list[tuple[StepRecord, StepMeta]] = field(default_factory=list)

    def __len__(self) -> int:
        """Return the number of recorded steps."""
        return len(self.steps)

    def append(self, record: StepRecord, meta: StepMeta | None = None) -> None:
        """Append a step record to the history.

        Args:
            record: Either a StepResult (single-process) or
                    Mapping[NodeId, StepResult] (multi-node).
            meta: Optional step metadata for budget accounting. Defaults to empty StepMeta.
        """
        if meta is None:
            meta = StepMeta()
        self.steps.append((record, meta))

    def last(self) -> StepRecord:
        """Return the most recent step record.

        Raises:
            IndexError: If history is empty.
        """
        return self.steps[-1][0]

    def last_meta(self) -> StepMeta:
        """Return the most recent step metadata.

        Raises:
            IndexError: If history is empty.
        """
        return self.steps[-1][1]

    def _extract_loss(self, record: StepRecord) -> float:
        """Extract loss from a record, averaging across nodes if needed."""
        if isinstance(record, StepResult):
            return record.loss
        # Multi-node: average across nodes
        losses = [r.loss for r in record.values()]
        return sum(losses) / len(losses)

    def _extract_metric(self, record: StepRecord, key: str) -> float:
        """Extract a metric from a record, averaging across nodes if needed.

        Raises:
            KeyError: If the metric key is not found.
        """
        if isinstance(record, StepResult):
            return record.metrics[key]
        # Multi-node: average across nodes
        values = [r.metrics[key] for r in record.values()]
        return sum(values) / len(values)

    def mean_loss(self) -> float:
        """Compute mean loss across all steps.

        For multi-node records, first averages across nodes for each step,
        then averages across all steps.

        Returns:
            The mean loss value.

        Raises:
            ValueError: If history is empty.
        """
        if not self.steps:
            raise ValueError("Cannot compute mean_loss on empty history")
        losses = [self._extract_loss(record) for record, _meta in self.steps]
        return sum(losses) / len(losses)

    def mean_metric(self, key: str) -> float:
        """Compute mean of a metric across all steps.

        For multi-node records, first averages across nodes for each step,
        then averages across all steps.

        Args:
            key: The metric key to aggregate.

        Returns:
            The mean metric value.

        Raises:
            ValueError: If history is empty.
            KeyError: If the metric key is not found in any record.
        """
        if not self.steps:
            raise ValueError("Cannot compute mean_metric on empty history")
        values = [self._extract_metric(record, key) for record, _meta in self.steps]
        return sum(values) / len(values)

    def total_grad_evals(self) -> int:
        """Return total number of gradient evaluations across all steps."""
        return sum(meta.num_grad_evals for _record, meta in self.steps)

    def total_gossip_rounds(self) -> int:
        """Return total number of gossip communication rounds across all steps."""
        return sum(meta.num_gossip_rounds for _record, meta in self.steps)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Configuration for an experiment run.

    This is a placeholder that will be extended with additional fields
    for specific experiment types (e.g., distributed settings, optimizer params).

    Attributes:
        seed: Random seed for reproducibility.
        steps: Number of optimization steps to run.
    """

    seed: int
    steps: int
