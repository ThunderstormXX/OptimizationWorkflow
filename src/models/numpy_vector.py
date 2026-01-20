"""Simple numpy vector model for optimization benchmarks.

This module provides a minimal model that stores parameters as a 1D numpy array,
suitable for tasks where the model is simply a parameter vector (e.g., quadratic
optimization, linear regression with explicit features).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.types import ParamVector

__all__ = ["NumpyVectorModel"]


class NumpyVectorModel:
    """A simple model that stores parameters as a 1D numpy vector.

    This model is designed for optimization tasks where the "model" is simply
    a parameter vector x ∈ R^d. It satisfies the Model protocol.

    Note:
        The `forward` method is not used for tasks like quadratic optimization
        where loss is computed directly from parameters. It returns None.

    Attributes:
        _x: Internal parameter vector (1D float64 array).

    Example:
        >>> x0 = np.zeros(10)
        >>> model = NumpyVectorModel(x0)
        >>> print(model.dim)
        10
        >>> model.set_parameters_vector(np.ones(10))
        >>> print(model.parameters_vector())
        [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
    """

    def __init__(self, x: ParamVector) -> None:
        """Initialize the model with a parameter vector.

        Args:
            x: Initial parameter vector (1D array).

        Raises:
            ValueError: If x is not 1-dimensional.
        """
        if x.ndim != 1:
            raise ValueError(f"x must be 1-dimensional, got ndim={x.ndim}")
        self._x: ParamVector = np.array(x, dtype=np.float64, copy=True)

    @property
    def dim(self) -> int:
        """Dimensionality of the parameter vector."""
        return int(self._x.shape[0])

    def forward(self, batch: Any) -> None:
        """Forward pass (unused for direct parameter optimization tasks).

        This method exists to satisfy the Model protocol but is not used
        for tasks like quadratic optimization where loss is computed
        directly from parameters.

        Args:
            batch: Input batch (ignored).

        Returns:
            None (forward pass not applicable for this model type).
        """
        return None

    def parameters_vector(self) -> ParamVector:
        """Return a copy of the parameter vector.

        Returns:
            A copy of the internal parameter vector (1D float64 array).
        """
        return self._x.copy()

    def set_parameters_vector(self, v: ParamVector) -> None:
        """Set parameters from a vector.

        Args:
            v: New parameter vector (must match current dimensionality).

        Raises:
            ValueError: If v has wrong shape.
        """
        if v.shape != self._x.shape:
            raise ValueError(f"Shape mismatch: expected {self._x.shape}, got {v.shape}")
        self._x = np.array(v, dtype=np.float64, copy=True)
