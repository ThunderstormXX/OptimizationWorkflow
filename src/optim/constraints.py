"""Constraint sets for Frank-Wolfe optimization.

This module provides constraint set implementations that satisfy the
ConstraintSet protocol, including:
- L1BallConstraint: L1 ball {x : ||x||_1 <= radius}
- L2BallConstraint: Euclidean ball {x : ||x||_2 <= radius}
- SimplexConstraint: Probability simplex {x : x >= 0, sum(x) = 1}
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.types import ParamVector

__all__ = ["L1BallConstraint", "L2BallConstraint", "SimplexConstraint"]


@dataclass(frozen=True)
class L1BallConstraint:
    """L1 ball constraint set: {x : ||x||_1 <= radius}.

    This constraint set represents the L1 ball centered at the origin
    with the given radius. It supports both the Linear Minimization Oracle
    (LMO) and projection operations.

    Attributes:
        radius: The radius of the ball. Must be positive.
    """

    radius: float

    def __post_init__(self) -> None:
        """Validate that radius is positive."""
        if self.radius <= 0:
            raise ValueError(f"radius must be positive, got {self.radius}")

    def lmo(self, grad: ParamVector) -> ParamVector:
        """Linear Minimization Oracle for the L1 ball.

        Solves: argmin_{s : ||s||_1 <= radius} <grad, s>

        The solution is s = -radius * sign(grad[i]) * e_i where i = argmax(|grad|).

        Args:
            grad: Gradient direction.

        Returns:
            The minimizer on the constraint set boundary.
        """
        if np.all(grad == 0):
            return np.zeros_like(grad, dtype=np.float64)

        # Find index of maximum absolute gradient
        i = int(np.argmax(np.abs(grad)))
        result = np.zeros_like(grad, dtype=np.float64)
        result[i] = -self.radius * np.sign(grad[i])
        return result

    def project(self, x: ParamVector) -> ParamVector:
        """Project a point onto the L1 ball.

        Uses the efficient O(n log n) algorithm.

        Args:
            x: Point to project.

        Returns:
            The projection of x onto the ball.
        """
        x_norm = float(np.linalg.norm(x, ord=1))
        if x_norm <= self.radius:
            return np.array(x, dtype=np.float64, copy=True)

        # Soft thresholding: find threshold mu such that ||S_mu(x)||_1 = radius
        # where S_mu(x)_i = sign(x_i) * max(|x_i| - mu, 0)
        abs_x = np.abs(x)
        sorted_abs = np.sort(abs_x)[::-1]  # Descending order

        # Find the threshold
        cumsum = np.cumsum(sorted_abs)
        n_nonzero = np.arange(1, len(sorted_abs) + 1)
        mu_candidates = (cumsum - self.radius) / n_nonzero

        # Find smallest index where sorted_abs[i] > mu_candidates[i]
        valid = sorted_abs > mu_candidates
        if np.any(valid):
            k = np.where(valid)[0][-1]
            mu = mu_candidates[k]
        else:
            mu = 0.0

        # Soft thresholding
        result = np.sign(x) * np.maximum(abs_x - mu, 0)
        return result.astype(np.float64)  # type: ignore[no-any-return]


@dataclass(frozen=True)
class L2BallConstraint:
    """L2 ball constraint set: {x : ||x||_2 <= radius}.

    This constraint set represents the Euclidean ball centered at the origin
    with the given radius. It supports both the Linear Minimization Oracle
    (LMO) and projection operations.

    Attributes:
        radius: The radius of the ball. Must be positive.

    Example:
        >>> constraint = L2BallConstraint(radius=1.0)
        >>> grad = np.array([3.0, 4.0])
        >>> s = constraint.lmo(grad)
        >>> np.linalg.norm(s)  # Should be 1.0
        1.0
    """

    radius: float

    def __post_init__(self) -> None:
        """Validate that radius is positive."""
        if self.radius <= 0:
            raise ValueError(f"radius must be positive, got {self.radius}")

    def lmo(self, grad: ParamVector) -> ParamVector:
        """Linear Minimization Oracle for the L2 ball.

        Solves: argmin_{s : ||s|| <= radius} <grad, s>

        The solution is s = -radius * grad / ||grad|| when grad != 0,
        which is the point on the ball boundary in the direction opposite
        to the gradient.

        Args:
            grad: Gradient direction.

        Returns:
            The minimizer on the constraint set boundary.
            Returns zeros if grad is zero.
        """
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm == 0:
            # Any feasible point works; return zeros (center of ball)
            return np.zeros_like(grad, dtype=np.float64)
        return np.asarray(-self.radius * grad / grad_norm, dtype=np.float64)

    def project(self, x: ParamVector) -> ParamVector:
        """Project a point onto the L2 ball.

        If ||x|| <= radius, returns x unchanged.
        Otherwise, returns radius * x / ||x|| (point on boundary).

        Args:
            x: Point to project.

        Returns:
            The projection of x onto the ball (as a copy).
        """
        x_norm = float(np.linalg.norm(x))
        if x_norm <= self.radius:
            return np.array(x, dtype=np.float64, copy=True)
        return np.asarray(self.radius * x / x_norm, dtype=np.float64)


@dataclass(frozen=True)
class SimplexConstraint:
    """Probability simplex constraint: {x : x >= 0, sum(x) = 1}.

    This constraint set represents the standard probability simplex
    in R^dim. It supports the Linear Minimization Oracle (LMO).

    Note:
        The project() method is not implemented for the simplex.
        Frank-Wolfe preserves feasibility when starting from a feasible
        point and using step sizes gamma in [0, 1], so projection is
        not needed during optimization.

    Attributes:
        dim: Dimensionality of the simplex. Must be >= 1.

    Example:
        >>> constraint = SimplexConstraint(dim=3)
        >>> grad = np.array([3.0, 1.0, 2.0])
        >>> s = constraint.lmo(grad)
        >>> s  # Should be [0, 1, 0] (one-hot at argmin)
        array([0., 1., 0.])
    """

    dim: int

    def __post_init__(self) -> None:
        """Validate that dim is at least 1."""
        if self.dim < 1:
            raise ValueError(f"dim must be >= 1, got {self.dim}")

    def lmo(self, grad: ParamVector) -> ParamVector:
        """Linear Minimization Oracle for the simplex.

        Solves: argmin_{s in simplex} <grad, s>

        The solution is the vertex e_i where i = argmin(grad),
        i.e., a one-hot vector at the index of the minimum gradient entry.

        Args:
            grad: Gradient direction of shape (dim,).

        Returns:
            One-hot vector e_i where i = argmin(grad).
        """
        i = int(np.argmin(grad))
        result = np.zeros(self.dim, dtype=np.float64)
        result[i] = 1.0
        return result

    def project(self, x: ParamVector) -> ParamVector:
        """Project a point onto the simplex.

        Note:
            This method raises NotImplementedError as simplex projection
            is not required for Frank-Wolfe when starting feasible.
            Frank-Wolfe maintains feasibility via convex combinations.

        Args:
            x: Point to project.

        Raises:
            NotImplementedError: Always, as projection is not implemented.
        """
        raise NotImplementedError(
            "Simplex projection not implemented. "
            "Frank-Wolfe preserves feasibility when starting from a feasible point."
        )
