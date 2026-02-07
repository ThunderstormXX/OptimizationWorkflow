"""Stochastic Frank-Wolfe optimizer for torch models."""

from __future__ import annotations

import importlib
from typing import Any, Callable

import numpy as np

from optim.frank_wolfe import constant_step_size, harmonic_step_size

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch optional at import time
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]


StepSize = Callable[[int], float]


def _check_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for StochasticFrankWolfe.")


def _import_class(path: str) -> type[Any]:
    if ":" in path:
        module_name, class_name = path.split(":", 1)
    else:
        module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _parse_step_size(step_size: Any, gamma: float | None) -> StepSize:
    if callable(step_size):
        return step_size
    if isinstance(step_size, (int, float)):
        return constant_step_size(float(step_size))
    if isinstance(step_size, str):
        name = step_size.lower()
        if name == "harmonic":
            return harmonic_step_size()
        if name in {"constant", "const"}:
            if gamma is None:
                raise ValueError("gamma must be provided for constant step size")
            return constant_step_size(float(gamma))
    if isinstance(step_size, dict):
        name = str(step_size.get("type", "")).lower()
        if name == "harmonic":
            return harmonic_step_size()
        if name in {"constant", "const"}:
            value = step_size.get("gamma", step_size.get("value", gamma))
            if value is None:
                raise ValueError("gamma must be provided for constant step size")
            return constant_step_size(float(value))
    raise ValueError(f"Unsupported step_size: {step_size}")


class StochasticFrankWolfe(torch.optim.Optimizer):  # type: ignore[misc]
    """Stochastic Frank-Wolfe optimizer using torch gradients.

    Args:
        params: Iterable of parameters.
        constraint: Constraint set or spec dict with "class" and "params".
        step_size: Step size schedule or spec.
        gamma: Constant step size if step_size is "constant".
    """

    def __init__(
        self,
        params: Any,
        *,
        constraint: Any,
        step_size: Any = "harmonic",
        gamma: float | None = None,
    ) -> None:
        _check_torch()
        defaults = {}
        super().__init__(params, defaults)

        self._step_size = _parse_step_size(step_size, gamma)
        self._t = 0
        self._constraint = None
        self._constraint_spec: dict[str, Any] | None = None

        if isinstance(constraint, dict) and "class" in constraint:
            self._constraint_spec = constraint
        else:
            self._constraint = constraint

        self._ensure_constraint()
        self._project_initial_params()

    def _ensure_constraint(self) -> None:
        if self._constraint is not None:
            return
        if self._constraint_spec is None:
            raise ValueError("constraint must be provided")
        spec = self._constraint_spec
        params = spec.get("params", {})
        radius = params.get("radius", None)
        if radius == "auto":
            radius = float(np.linalg.norm(self._params_to_vector()))
            params = dict(params)
            params["radius"] = radius
        cls = _import_class(spec["class"])
        self._constraint = cls(**params)

    def _project_initial_params(self) -> None:
        project_fn = getattr(self._constraint, "project", None)
        if not callable(project_fn):
            return
        x = self._params_to_vector()
        try:
            x_proj = project_fn(x)
        except NotImplementedError:
            return
        if np.array_equal(x, x_proj):
            return
        self._set_params_from_vector(x_proj)

    def _params_to_vector(self) -> np.ndarray:
        params = []
        for group in self.param_groups:
            for p in group["params"]:
                params.append(p.detach().cpu().numpy().reshape(-1))
        if not params:
            return np.array([], dtype=np.float64)
        return np.concatenate(params).astype(np.float64)

    def _grads_to_vector(self) -> np.ndarray:
        grads = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    grads.append(np.zeros_like(p.detach().cpu().numpy().reshape(-1)))
                else:
                    grads.append(p.grad.detach().cpu().numpy().reshape(-1))
        if not grads:
            return np.array([], dtype=np.float64)
        return np.concatenate(grads).astype(np.float64)

    def _set_params_from_vector(self, vec: np.ndarray) -> None:
        offset = 0
        for group in self.param_groups:
            for p in group["params"]:
                numel = p.numel()
                chunk = vec[offset : offset + numel]
                offset += numel
                new_values = torch.from_numpy(chunk).view_as(p).to(p.device)
                p.data.copy_(new_values)

    @torch.no_grad()
    def step(self, closure: Callable[[], Any] | None = None) -> Any:
        _check_torch()
        if closure is not None:
            closure()

        self._ensure_constraint()

        grad = self._grads_to_vector()
        if grad.size == 0:
            return None

        s = self._constraint.lmo(grad)
        gamma = float(self._step_size(self._t))
        if not np.isfinite(gamma) or not (0 < gamma <= 1):
            raise ValueError(f"Invalid step size: {gamma}")

        x = self._params_to_vector()
        x_new = (1.0 - gamma) * x + gamma * s
        self._set_params_from_vector(x_new)
        self._t += 1
        return None
