"""Synthetic quadratic optimization task.

This module provides a convex quadratic optimization problem for benchmarking:
    f(x) = 0.5 * x^T A x + b^T x

where A is symmetric positive definite (SPD) and b is a vector.

This is the primary sanity-check task for optimization algorithms because:
- It has a unique global minimum at x* = -A^{-1} b
- Gradients are exact: ∇f(x) = Ax + b
- Convergence rates are well-understood
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from core.types import ParamVector
from models.numpy_vector import NumpyVectorModel

__all__ = [
    "QuadraticProblem",
    "SyntheticQuadraticTask",
    "QuadraticGradComputer",
    "make_spd_quadratic",
]


@dataclass(frozen=True)
class QuadraticProblem:
    """A convex quadratic optimization problem.

    Defines the function:
        f(x) = 0.5 * x^T A x + b^T x

    where A is symmetric positive definite and b is a vector.

    Attributes:
        A: Symmetric positive definite matrix of shape (d, d).
        b: Linear term vector of shape (d,).

    Example:
        >>> A = np.array([[2.0, 0.0], [0.0, 1.0]])
        >>> b = np.array([1.0, -1.0])
        >>> problem = QuadraticProblem(A, b)
        >>> x = np.array([0.0, 0.0])
        >>> print(problem.loss(x))
        0.0
        >>> print(problem.grad(x))
        [ 1. -1.]
    """

    A: np.ndarray
    b: np.ndarray

    def __post_init__(self) -> None:
        """Validate that A is SPD and dimensions are consistent."""
        # Check dimensions
        if self.A.ndim != 2:
            raise ValueError(f"A must be 2D, got ndim={self.A.ndim}")
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError(f"A must be square, got shape {self.A.shape}")
        if self.b.ndim != 1:
            raise ValueError(f"b must be 1D, got ndim={self.b.ndim}")
        if self.b.shape[0] != self.A.shape[0]:
            raise ValueError(
                f"Dimension mismatch: A is {self.A.shape[0]}x{self.A.shape[0]}, "
                f"b has length {self.b.shape[0]}"
            )

        # Check symmetry (with tolerance)
        if not np.allclose(self.A, self.A.T, rtol=1e-10, atol=1e-10):
            raise ValueError("A must be symmetric")

        # Check positive definiteness via eigenvalues
        # For small matrices, compute all eigenvalues
        # For larger matrices (d > 100), use a tolerance-based check
        d = self.A.shape[0]
        if d <= 100:
            eigvals = np.linalg.eigvalsh(self.A)
            min_eig = float(eigvals.min())
            if min_eig <= 0:
                raise ValueError(f"A must be positive definite, but has min eigenvalue {min_eig}")
        else:
            # For large matrices, just check that Cholesky decomposition succeeds
            try:
                np.linalg.cholesky(self.A)
            except np.linalg.LinAlgError as e:
                raise ValueError("A must be positive definite") from e

    @property
    def dim(self) -> int:
        """Dimensionality of the problem."""
        return int(self.A.shape[0])

    def loss(self, x: ParamVector) -> float:
        """Compute the quadratic loss at x.

        f(x) = 0.5 * x^T A x + b^T x

        Args:
            x: Parameter vector of shape (d,).

        Returns:
            Scalar loss value.
        """
        return float(0.5 * x @ self.A @ x + self.b @ x)

    def grad(self, x: ParamVector) -> ParamVector:
        """Compute the gradient at x.

        ∇f(x) = Ax + b

        Args:
            x: Parameter vector of shape (d,).

        Returns:
            Gradient vector of shape (d,).
        """
        return np.asarray(self.A @ x + self.b, dtype=np.float64)

    def x_star(self) -> ParamVector:
        """Compute the optimal solution x*.

        Solves Ax* + b = 0, i.e., x* = -A^{-1} b

        Returns:
            Optimal parameter vector of shape (d,).
        """
        return np.linalg.solve(self.A, -self.b)


def make_spd_quadratic(
    *,
    dim: int,
    rng: np.random.Generator,
    cond: float = 10.0,
) -> QuadraticProblem:
    """Generate a random SPD quadratic problem with controlled condition number.

    Creates a symmetric positive definite matrix A with eigenvalues
    uniformly spaced between 1 and `cond`, and a random vector b.

    Args:
        dim: Dimensionality of the problem.
        rng: Random number generator for reproducibility.
        cond: Condition number of A (ratio of max to min eigenvalue).
              Must be >= 1. Default is 10.0.

    Returns:
        A QuadraticProblem with the generated A and b.

    Raises:
        ValueError: If dim < 1 or cond < 1.

    Example:
        >>> rng = np.random.default_rng(42)
        >>> problem = make_spd_quadratic(dim=5, rng=rng, cond=10.0)
        >>> print(problem.dim)
        5
    """
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    if cond < 1.0:
        raise ValueError(f"cond must be >= 1, got {cond}")

    # Generate random orthogonal matrix Q via QR decomposition
    random_matrix = rng.standard_normal((dim, dim))
    Q, _ = np.linalg.qr(random_matrix)

    # Create eigenvalues uniformly spaced between 1 and cond
    if dim == 1:
        eigenvalues = np.array([1.0])
    else:
        eigenvalues = np.linspace(1.0, cond, dim)

    # Construct A = Q * diag(eigenvalues) * Q^T
    A = Q @ np.diag(eigenvalues) @ Q.T

    # Ensure exact symmetry (numerical errors can break this)
    A = (A + A.T) / 2.0

    # Generate random b
    b = rng.standard_normal(dim).astype(np.float64)

    return QuadraticProblem(A=A, b=b)


class SyntheticQuadraticTask:
    """A batchless task for quadratic optimization.

    This task wraps a QuadraticProblem and provides the Task protocol interface.
    Since the quadratic function doesn't use batches, sample_batch returns None.

    Metrics provided:
    - "loss": Current loss value f(x)
    - "grad_norm": L2 norm of the gradient ||∇f(x)||
    - "dist_to_opt": L2 distance to optimum ||x - x*||

    Example:
        >>> problem = make_spd_quadratic(dim=5, rng=np.random.default_rng(42))
        >>> task = SyntheticQuadraticTask(problem)
        >>> model = NumpyVectorModel(np.zeros(5))
        >>> metrics = task.metrics(model, None)
        >>> print(metrics.keys())
        dict_keys(['loss', 'grad_norm', 'dist_to_opt'])
    """

    def __init__(self, problem: QuadraticProblem) -> None:
        """Initialize the task with a quadratic problem.

        Args:
            problem: The QuadraticProblem defining the optimization objective.
        """
        self.problem = problem
        self._x_star: ParamVector | None = None  # Cached optimal solution

    @property
    def x_star(self) -> ParamVector:
        """Optimal solution (cached)."""
        if self._x_star is None:
            self._x_star = self.problem.x_star()
        return self._x_star

    def sample_batch(self, *, rng: np.random.Generator) -> None:
        """Sample a batch (returns None for batchless task).

        Args:
            rng: Random number generator (unused).

        Returns:
            None (this task doesn't use batches).
        """
        return None

    def loss(self, model: NumpyVectorModel, batch: None) -> float:
        """Compute loss at current model parameters.

        Args:
            model: Model containing parameter vector.
            batch: Unused (always None).

        Returns:
            Loss value f(x).
        """
        x = model.parameters_vector()
        return self.problem.loss(x)

    def metrics(self, model: NumpyVectorModel, batch: None) -> Mapping[str, float]:
        """Compute metrics at current model parameters.

        Args:
            model: Model containing parameter vector.
            batch: Unused (always None).

        Returns:
            Dictionary with keys:
            - "loss": Current loss f(x)
            - "grad_norm": L2 norm of gradient ||∇f(x)||
            - "dist_to_opt": L2 distance to optimum ||x - x*||
        """
        x = model.parameters_vector()
        loss = self.problem.loss(x)
        grad = self.problem.grad(x)
        grad_norm = float(np.linalg.norm(grad))
        dist_to_opt = float(np.linalg.norm(x - self.x_star))

        return {
            "loss": loss,
            "grad_norm": grad_norm,
            "dist_to_opt": dist_to_opt,
        }


class QuadraticGradComputer:
    """Gradient computer for quadratic tasks.

    Computes the exact gradient ∇f(x) = Ax + b for a quadratic problem.
    """

    def grad(
        self,
        task: SyntheticQuadraticTask,
        model: NumpyVectorModel,
        batch: None,
    ) -> ParamVector:
        """Compute gradient of the quadratic loss.

        Args:
            task: The quadratic task (contains the problem definition).
            model: Model containing current parameters.
            batch: Unused (always None).

        Returns:
            Gradient vector ∇f(x) = Ax + b.
        """
        x = model.parameters_vector()
        return task.problem.grad(x)
