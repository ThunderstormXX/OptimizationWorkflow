"""Legacy Frank-Wolfe and constraint utilities (plus torch SFW)."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np

from core.protocols import ConstraintSet, GradComputer, Model, Task
from core.types import ParamVector, StepResult

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch optional at import time
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]

__all__ = [
    # Constraints
    "L1BallConstraint",
    "L2BallConstraint",
    "OrthogonalMatrixConstraint",
    "SimplexConstraint",
    # Step sizes
    "StepSize",
    "constant_step_size",
    "harmonic_step_size",
    # Legacy Frank-Wolfe
    "FWState",
    "FrankWolfeOptimizer",
    # Legacy GD/PGD
    "GDState",
    "GradientDescentOptimizer",
    "ProjectedGradientDescentOptimizer",
    # Torch stochastic Frank-Wolfe
    "StochasticFrankWolfe",
    "StochasticFrankWolfeMomentumPost",
    "StochasticFrankWolfeMomentumPre",
    "OrthogonalSGDM",
]


# =============================================================================
# Constraints
# =============================================================================


@dataclass(frozen=True)
class L1BallConstraint:
    """L1 ball constraint set: {x : ||x||_1 <= radius}."""

    radius: float

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError(f"radius must be positive, got {self.radius}")

    def lmo(self, grad: ParamVector) -> ParamVector:
        if np.all(grad == 0):
            return np.zeros_like(grad, dtype=np.float64)
        i = int(np.argmax(np.abs(grad)))
        result = np.zeros_like(grad, dtype=np.float64)
        result[i] = -self.radius * np.sign(grad[i])
        return result

    def project(self, x: ParamVector) -> ParamVector:
        x_norm = float(np.linalg.norm(x, ord=1))
        if x_norm <= self.radius:
            return np.array(x, dtype=np.float64, copy=True)

        abs_x = np.abs(x)
        sorted_abs = np.sort(abs_x)[::-1]
        cumsum = np.cumsum(sorted_abs)
        n_nonzero = np.arange(1, len(sorted_abs) + 1)
        mu_candidates = (cumsum - self.radius) / n_nonzero
        valid = sorted_abs > mu_candidates
        if np.any(valid):
            k = np.where(valid)[0][-1]
            mu = mu_candidates[k]
        else:
            mu = 0.0
        result = np.sign(x) * np.maximum(abs_x - mu, 0)
        return result.astype(np.float64)  # type: ignore[no-any-return]


@dataclass(frozen=True)
class L2BallConstraint:
    """L2 ball constraint set: {x : ||x||_2 <= radius}."""

    radius: float

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError(f"radius must be positive, got {self.radius}")

    def lmo(self, grad: ParamVector) -> ParamVector:
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm == 0:
            return np.zeros_like(grad, dtype=np.float64)
        return np.asarray(-self.radius * grad / grad_norm, dtype=np.float64)

    def project(self, x: ParamVector) -> ParamVector:
        x_norm = float(np.linalg.norm(x))
        if x_norm <= self.radius:
            return np.array(x, dtype=np.float64, copy=True)
        return np.asarray(self.radius * x / x_norm, dtype=np.float64)


@dataclass(frozen=True)
class L2BallTensorConstraint:
    """Per-tensor L2 ball constraint set: {x : ||x||_2 <= radius} for each tensor."""

    radius: float

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError(f"radius must be positive, got {self.radius}")

    def lmo_tensors(self, grads: list["torch.Tensor"]) -> list["torch.Tensor"]:
        _check_torch()
        outputs: list["torch.Tensor"] = []
        for g in grads:
            if g.numel() == 0:
                outputs.append(torch.zeros_like(g))
                continue
            norm = torch.norm(g)
            if norm <= 0:
                outputs.append(torch.zeros_like(g))
            else:
                outputs.append((-float(self.radius) * g / (norm + 1e-12)).to(dtype=g.dtype))
        return outputs

    def project_tensors(self, params: list["torch.Tensor"]) -> list["torch.Tensor"]:
        _check_torch()
        outputs: list["torch.Tensor"] = []
        for p in params:
            if p.numel() == 0:
                outputs.append(torch.zeros_like(p))
                continue
            norm = torch.norm(p)
            if norm <= self.radius:
                outputs.append(p.clone())
            else:
                outputs.append((float(self.radius) * p / (norm + 1e-12)).to(dtype=p.dtype))
        return outputs

    def info_tensors(self, params: list["torch.Tensor"]) -> dict[str, float]:
        _check_torch()
        if not params:
            return {}
        norms = []
        ratios = []
        for p in params:
            if p.numel() == 0:
                continue
            norm = float(torch.norm(p).item())
            norms.append(norm)
            ratios.append(norm / float(self.radius))
        if not norms:
            return {}
        mean_norm = float(np.mean(norms))
        mean_ratio = float(np.mean(ratios))
        on_boundary = 1.0 if abs(mean_ratio - 1.0) <= 1e-3 else 0.0
        return {
            "constraint_norm": mean_norm,
            "constraint_radius": float(self.radius),
            "constraint_ratio": mean_ratio,
            "constraint_on_boundary": on_boundary,
        }


class OrthogonalMatrixConstraint:
    """Constraint set of (scaled) orthogonal matrices for tensor-based Frank-Wolfe."""

    def __init__(self, radius: float = 1.0, ns_steps: int = 5, eps: float = 1e-7) -> None:
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")
        if ns_steps < 1:
            raise ValueError(f"ns_steps must be >= 1, got {ns_steps}")
        self.radius = float(radius)
        self.ns_steps = int(ns_steps)
        self.eps = float(eps)
        self.lmo_sign = 1.0

    def lmo_tensors(self, grads: list["torch.Tensor"]) -> list["torch.Tensor"]:
        _check_torch()
        return [_muon_lmo_tensor(g, radius=self.radius, steps=self.ns_steps) for g in grads]

    def project_tensors(self, params: list["torch.Tensor"]) -> list["torch.Tensor"]:
        _check_torch()
        return [_muon_project_tensor(p, radius=self.radius, steps=self.ns_steps) for p in params]

    def info_tensors(self, params: list["torch.Tensor"]) -> dict[str, float]:
        _check_torch()
        if not params:
            return {}
        errors = []
        norms = []
        for p in params:
            if p.ndim < 2:
                continue
            proj = _muon_project_tensor(p, radius=self.radius, steps=self.ns_steps)
            diff = (p - proj).float().reshape(-1)
            errors.append(float(torch.norm(diff).item()))
            norms.append(float(torch.norm(p.float().reshape(-1)).item()))
        if not errors:
            return {}
        mean_error = float(np.mean(errors))
        mean_norm = float(np.mean(norms)) if norms else float("nan")
        on_boundary = 1.0 if mean_error <= 1e-6 * max(1.0, self.radius) else 0.0
        return {
            "constraint_error": mean_error,
            "constraint_norm": mean_norm,
            "constraint_radius": float(self.radius),
            "constraint_on_boundary": on_boundary,
        }


@dataclass(frozen=True)
class SimplexConstraint:
    """Probability simplex constraint: {x : x >= 0, sum(x) = 1}."""

    dim: int

    def __post_init__(self) -> None:
        if self.dim < 1:
            raise ValueError(f"dim must be >= 1, got {self.dim}")

    def lmo(self, grad: ParamVector) -> ParamVector:
        i = int(np.argmin(grad))
        result = np.zeros(self.dim, dtype=np.float64)
        result[i] = 1.0
        return result

    def project(self, x: ParamVector) -> ParamVector:
        raise NotImplementedError


# =============================================================================
# Step sizes
# =============================================================================


StepSize = Callable[[int], float]


def constant_step_size(gamma: float) -> StepSize:
    if not (0 < gamma <= 1):
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")

    def schedule(t: int) -> float:
        return gamma

    return schedule


def harmonic_step_size() -> StepSize:
    def schedule(t: int) -> float:
        return 2.0 / (t + 2)

    return schedule


# =============================================================================
# Legacy Frank-Wolfe (protocol-based)
# =============================================================================


_Batch = TypeVar("_Batch")
_Pred = TypeVar("_Pred")


@dataclass
class FWState:
    t: int


class FrankWolfeOptimizer(Generic[_Batch, _Pred]):
    def __init__(
        self,
        *,
        constraint: ConstraintSet,
        step_size: StepSize,
    ) -> None:
        self.constraint = constraint
        self.step_size = step_size

    def _try_project(self, x: ParamVector) -> ParamVector:
        project_fn: Callable[[ParamVector], ParamVector] | None = getattr(
            self.constraint, "project", None
        )
        if project_fn is not None and callable(project_fn):
            try:
                return project_fn(x)
            except NotImplementedError:
                return x
        return x

    def init_state(self, model: Model[Any, Any]) -> FWState:
        x = model.parameters_vector()
        x_proj = self._try_project(x)
        if not np.array_equal(x, x_proj):
            model.set_parameters_vector(x_proj)
        return FWState(t=0)

    def step(
        self,
        *,
        task: Task[_Batch, _Pred],
        model: Model[_Batch, _Pred],
        batch: _Batch,
        grad_computer: GradComputer[_Batch, _Pred],
        state: FWState,
        rng: np.random.Generator,
    ) -> tuple[FWState, StepResult]:
        x = model.parameters_vector()
        grad = grad_computer.grad(task, model, batch)
        s = self.constraint.lmo(grad)
        gamma = self.step_size(state.t)
        if not (0 < gamma <= 1):
            raise ValueError(f"Invalid step size gamma={gamma}")
        x_new = (1 - gamma) * x + gamma * s
        model.set_parameters_vector(x_new)
        loss = task.loss(model, batch)
        metrics = task.metrics(model, batch)
        return FWState(t=state.t + 1), StepResult(loss=loss, metrics=dict(metrics))


# =============================================================================
# Legacy GD / PGD
# =============================================================================


@dataclass
class GDState:
    t: int = 0


class GradientDescentOptimizer:
    def __init__(self, lr: float) -> None:
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        self.lr = lr

    def init_state(self, model: Model[Any, Any]) -> GDState:
        return GDState(t=0)

    def step(
        self,
        *,
        task: Task[Any, Any],
        model: Model[Any, Any],
        batch: Any,
        grad_computer: GradComputer[Any, Any],
        state: GDState,
        rng: np.random.Generator,
    ) -> tuple[GDState, StepResult]:
        x = model.parameters_vector()
        grad = grad_computer.grad(task, model, batch)
        x_new = x - self.lr * grad
        model.set_parameters_vector(x_new)
        loss = task.loss(model, batch)
        metrics = task.metrics(model, batch)
        return GDState(t=state.t + 1), StepResult(loss=loss, metrics=dict(metrics))


class ProjectedGradientDescentOptimizer:
    def __init__(self, lr: float, constraint: ConstraintSet) -> None:
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        project = getattr(constraint, "project", None)
        if not callable(project):
            raise ValueError(
                f"Constraint {type(constraint).__name__} does not have a project() method."
            )
        self.lr = lr
        self.constraint = constraint
        self._project = project

    def init_state(self, model: Model[Any, Any]) -> GDState:
        x = model.parameters_vector()
        x_proj = self._project(x)
        model.set_parameters_vector(x_proj)
        return GDState(t=0)

    def step(
        self,
        *,
        task: Task[Any, Any],
        model: Model[Any, Any],
        batch: Any,
        grad_computer: GradComputer[Any, Any],
        state: GDState,
        rng: np.random.Generator,
    ) -> tuple[GDState, StepResult]:
        x = model.parameters_vector()
        grad = grad_computer.grad(task, model, batch)
        x_temp = x - self.lr * grad
        x_new = self._project(x_temp)
        model.set_parameters_vector(x_new)
        loss = task.loss(model, batch)
        metrics = task.metrics(model, batch)
        return GDState(t=state.t + 1), StepResult(loss=loss, metrics=dict(metrics))


# =============================================================================
# Torch stochastic Frank-Wolfe (for supervised training)
# =============================================================================


def _check_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for StochasticFrankWolfe.")


def _muon_orthogonalize_matrix(mat: "torch.Tensor", steps: int) -> "torch.Tensor":
    from optim.muon import zeropower_via_newtonschulz5

    if mat.ndim != 2:
        raise ValueError("orthogonalize expects 2D tensor")
    return zeropower_via_newtonschulz5(mat, steps=steps)


def _muon_lmo_tensor(g: "torch.Tensor", *, radius: float, steps: int) -> "torch.Tensor":
    if g.numel() == 0:
        return torch.zeros_like(g)
    if g.ndim < 2:
        norm = torch.norm(g)
        if norm <= 0:
            return torch.zeros_like(g)
        return (g / (norm + 1e-12)).to(dtype=g.dtype) * float(radius)
    original_shape = g.shape
    if g.ndim == 4:
        mat = g.view(len(g), -1)
    else:
        mat = g.reshape(g.size(0), -1)
    ortho = _muon_orthogonalize_matrix(mat, steps)
    scale = max(1.0, ortho.size(-2) / ortho.size(-1)) ** 0.5
    ortho = ortho * float(scale) * float(radius)
    return ortho.view(original_shape).to(dtype=g.dtype)


def _muon_project_tensor(p: "torch.Tensor", *, radius: float, steps: int) -> "torch.Tensor":
    if p.numel() == 0:
        return torch.zeros_like(p)
    if p.ndim < 2:
        return p.clone()
    original_shape = p.shape
    if p.ndim == 4:
        mat = p.view(len(p), -1)
    else:
        mat = p.reshape(p.size(0), -1)
    ortho = _muon_orthogonalize_matrix(mat, steps)
    scale = max(1.0, ortho.size(-2) / ortho.size(-1)) ** 0.5
    ortho = ortho * float(scale) * float(radius)
    return ortho.view(original_shape).to(dtype=p.dtype)


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
    """Stochastic Frank-Wolfe optimizer using torch gradients."""

    def __init__(
        self,
        params: Any,
        *,
        constraint: Any,
        step_size: Any = "harmonic",
        gamma: float | None = None,
        weight_decay: float = 0.0,
    ) -> None:
        _check_torch()
        params_list = list(params)
        if not params_list:
            raise ValueError("params must not be empty")
        defaults = {}
        super().__init__(params_list, defaults)

        self._step_size = _parse_step_size(step_size, gamma)
        self._t = 0
        self._constraint = None
        self._constraint_spec: dict[str, Any] | None = None
        self._tensor_constraint = False
        self._last_constraint_info: dict[str, float] = {}
        self._weight_decay = float(weight_decay)

        if isinstance(constraint, dict) and "class" in constraint:
            self._constraint_spec = constraint
        else:
            self._constraint = constraint

        self._ensure_constraint()
        self._tensor_constraint = hasattr(self._constraint, "lmo_tensors")
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
            params = dict(params)
            radius_mult = float(params.pop("radius_mult", 1.0))
            radius = float(np.linalg.norm(self._params_to_vector())) * radius_mult
            params["radius"] = radius
        cls = _import_class(spec["class"])
        self._constraint = cls(**params)

    def _project_initial_params(self) -> None:
        if self._tensor_constraint:
            project_fn = getattr(self._constraint, "project_tensors", None)
            if callable(project_fn):
                params = self._params_list()
                projected = project_fn(params)
                for p, proj in zip(params, projected):
                    p.data.copy_(proj.to(dtype=p.dtype, device=p.device))
            return

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

    def _params_list(self) -> list["torch.Tensor"]:
        params: list["torch.Tensor"] = []
        for group in self.param_groups:
            params.extend(group["params"])
        return params

    def _grads_list(self) -> list["torch.Tensor"]:
        grads: list["torch.Tensor"] = []
        for p in self._params_list():
            if p.grad is None:
                grads.append(torch.zeros_like(p))
            else:
                grads.append(p.grad.detach())
        return grads

    def _params_to_vector(self) -> np.ndarray:
        params = []
        for group in self.param_groups:
            for p in group["params"]:
                params.append(p.detach().cpu().numpy().reshape(-1))
        if not params:
            return np.array([], dtype=np.float32)
        return np.concatenate(params).astype(np.float32)

    def _grads_to_vector(self) -> np.ndarray:
        grads = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    grads.append(np.zeros_like(p.detach().cpu().numpy().reshape(-1)))
                else:
                    grads.append(p.grad.detach().cpu().numpy().reshape(-1))
        if not grads:
            return np.array([], dtype=np.float32)
        return np.concatenate(grads).astype(np.float32)

    def _set_params_from_vector(self, vec: np.ndarray) -> None:
        offset = 0
        for group in self.param_groups:
            for p in group["params"]:
                numel = p.numel()
                chunk = vec[offset : offset + numel]
                offset += numel
                new_values = torch.tensor(chunk, dtype=p.dtype, device=p.device).view_as(p)
                p.data.copy_(new_values)

    def _set_params_from_tensors(self, updates: list["torch.Tensor"], gamma: float) -> None:
        for p, s in zip(self._params_list(), updates):
            p.data.mul_(1.0 - gamma)
            p.data.add_(s.to(dtype=p.dtype, device=p.device), alpha=gamma)

    def _apply_weight_decay(self, scale: float) -> None:
        if self._weight_decay == 0.0:
            return
        decay = 1.0 - scale * self._weight_decay
        for p in self._params_list():
            p.data.mul_(decay)

    def _apply_weight_decay_vector(self, x: np.ndarray, scale: float) -> np.ndarray:
        if self._weight_decay == 0.0:
            return x
        decay = 1.0 - scale * self._weight_decay
        return x * decay

    def _lmo_tensors_for_fw(self, grads: list["torch.Tensor"]) -> list["torch.Tensor"]:
        s_list = self._constraint.lmo_tensors(grads)
        lmo_sign = float(getattr(self._constraint, "lmo_sign", -1.0))
        if lmo_sign > 0:
            return [s.neg() for s in s_list]
        return s_list

    def _compute_constraint_info(self) -> dict[str, float]:
        info: dict[str, float] = {}
        if self._tensor_constraint:
            info_fn = getattr(self._constraint, "info_tensors", None)
            if callable(info_fn):
                info = info_fn(self._params_list()) or {}
            self._last_constraint_info = {
                k: float(v) for k, v in info.items() if isinstance(v, (int, float))
            }
            return self._last_constraint_info

        x = self._params_to_vector()
        radius = getattr(self._constraint, "radius", None)
        if radius is None:
            return {}
        radius = float(radius)
        if isinstance(self._constraint, L1BallConstraint):
            norm = float(np.linalg.norm(x, ord=1))
        else:
            norm = float(np.linalg.norm(x))
        distance = radius - norm
        ratio = norm / radius if radius > 0 else float("nan")
        on_boundary = 1.0 if abs(distance) <= 1e-6 * max(1.0, radius) else 0.0
        info = {
            "constraint_norm": float(norm),
            "constraint_radius": float(radius),
            "constraint_ratio": float(ratio),
            "constraint_distance": float(distance),
            "constraint_on_boundary": float(on_boundary),
        }
        self._last_constraint_info = info
        return info

    def constraint_info(self) -> dict[str, float]:
        return dict(self._last_constraint_info)

    @torch.no_grad()
    def step(self, closure: Callable[[], Any] | None = None) -> Any:
        _check_torch()
        if closure is not None:
            closure()

        self._ensure_constraint()

        gamma = float(self._step_size(self._t))
        if not np.isfinite(gamma) or not (0 < gamma <= 1):
            raise ValueError(f"Invalid step size: {gamma}")

        if self._tensor_constraint:
            grads = self._grads_list()
            if not grads:
                return None
            s_list = self._lmo_tensors_for_fw(grads)
            self._apply_weight_decay(gamma)
            self._set_params_from_tensors(s_list, gamma)
        else:
            grad = self._grads_to_vector()
            if grad.size == 0:
                return None
            s = self._constraint.lmo(grad)
            x = self._params_to_vector()
            x = self._apply_weight_decay_vector(x, gamma)
            x_new = (1.0 - gamma) * x + gamma * s
            self._set_params_from_vector(x_new)

        self._compute_constraint_info()
        self._t += 1
        return None


class StochasticFrankWolfeMomentumPre(StochasticFrankWolfe):
    """Stochastic Frank-Wolfe with momentum on gradients before LMO (Muon-like)."""

    def __init__(
        self,
        params: Any,
        *,
        constraint: Any,
        step_size: Any = "harmonic",
        gamma: float | None = None,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ) -> None:
        if not (0.0 <= float(momentum) <= 1.0):
            raise ValueError(f"momentum must be in [0, 1], got {momentum}")
        self.momentum = float(momentum)
        self._momentum_vec: np.ndarray | None = None
        super().__init__(
            params,
            constraint=constraint,
            step_size=step_size,
            gamma=gamma,
            weight_decay=weight_decay,
        )

    @torch.no_grad()
    def step(self, closure: Callable[[], Any] | None = None) -> Any:
        _check_torch()
        if closure is not None:
            closure()

        self._ensure_constraint()
        gamma = float(self._step_size(self._t))
        if not np.isfinite(gamma) or not (0 < gamma <= 1):
            raise ValueError(f"Invalid step size: {gamma}")

        if self._tensor_constraint:
            grads = self._grads_list()
            if not grads:
                return None
            mom_grads: list[torch.Tensor] = []
            for p, g in zip(self._params_list(), grads):
                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = torch.zeros_like(g)
                buf.mul_(self.momentum).add_(g, alpha=1.0 - self.momentum)
                state["momentum_buffer"] = buf
                mom_grads.append(buf)
            s_list = self._lmo_tensors_for_fw(mom_grads)
            self._apply_weight_decay(gamma)
            self._set_params_from_tensors(s_list, gamma)
        else:
            grad = self._grads_to_vector()
            if grad.size == 0:
                return None
            if self._momentum_vec is None or self._momentum_vec.shape != grad.shape:
                self._momentum_vec = np.zeros_like(grad, dtype=np.float32)
            self._momentum_vec = self.momentum * self._momentum_vec + (1.0 - self.momentum) * grad
            s = self._constraint.lmo(self._momentum_vec)
            x = self._params_to_vector()
            x = self._apply_weight_decay_vector(x, gamma)
            x_new = (1.0 - gamma) * x + gamma * s
            self._set_params_from_vector(x_new)

        self._compute_constraint_info()
        self._t += 1
        return None


class StochasticFrankWolfeMomentumPost(StochasticFrankWolfe):
    """Stochastic Frank-Wolfe with momentum on LMO direction (post)."""

    def __init__(
        self,
        params: Any,
        *,
        constraint: Any,
        step_size: Any = "harmonic",
        gamma: float | None = None,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ) -> None:
        if not (0.0 <= float(momentum) <= 1.0):
            raise ValueError(f"momentum must be in [0, 1], got {momentum}")
        self.momentum = float(momentum)
        self._direction_vec: np.ndarray | None = None
        super().__init__(
            params,
            constraint=constraint,
            step_size=step_size,
            gamma=gamma,
            weight_decay=weight_decay,
        )

    @torch.no_grad()
    def step(self, closure: Callable[[], Any] | None = None) -> Any:
        _check_torch()
        if closure is not None:
            closure()

        self._ensure_constraint()
        gamma = float(self._step_size(self._t))
        if not np.isfinite(gamma) or not (0 < gamma <= 1):
            raise ValueError(f"Invalid step size: {gamma}")

        if self._tensor_constraint:
            grads = self._grads_list()
            if not grads:
                return None
            s_list = self._lmo_tensors_for_fw(grads)
            direction: list[torch.Tensor] = []
            for p, s in zip(self._params_list(), s_list):
                state = self.state[p]
                buf = state.get("direction_buffer")
                if buf is None:
                    buf = torch.zeros_like(s)
                buf.mul_(self.momentum).add_(s, alpha=1.0 - self.momentum)
                state["direction_buffer"] = buf
                direction.append(buf)
            self._apply_weight_decay(gamma)
            self._set_params_from_tensors(direction, gamma)
        else:
            grad = self._grads_to_vector()
            if grad.size == 0:
                return None
            s = self._constraint.lmo(grad)
            if self._direction_vec is None or self._direction_vec.shape != s.shape:
                self._direction_vec = np.zeros_like(s, dtype=np.float32)
            self._direction_vec = self.momentum * self._direction_vec + (1.0 - self.momentum) * s
            x = self._params_to_vector()
            x = self._apply_weight_decay_vector(x, gamma)
            x_new = (1.0 - gamma) * x + gamma * self._direction_vec
            self._set_params_from_vector(x_new)

        self._compute_constraint_info()
        self._t += 1
        return None


class OrthogonalSGDM(StochasticFrankWolfe):
    """O-SGDM: LMO on stochastic gradients, then momentum on direction (SGD-style update)."""

    def __init__(
        self,
        params: Any,
        *,
        constraint: Any | None = None,
        step_size: Any = "harmonic",
        gamma: float | None = None,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ) -> None:
        if constraint is None:
            constraint = OrthogonalMatrixConstraint(radius=1.0)
        if not (0.0 <= float(momentum) <= 1.0):
            raise ValueError(f"momentum must be in [0, 1], got {momentum}")
        self.momentum = float(momentum)
        self._direction_vec: np.ndarray | None = None
        self._weight_decay = float(weight_decay)
        super().__init__(params, constraint=constraint, step_size=step_size, gamma=gamma)

    @torch.no_grad()
    def step(self, closure: Callable[[], Any] | None = None) -> Any:
        _check_torch()
        if closure is not None:
            closure()

        self._ensure_constraint()
        lr = float(self._step_size(self._t))
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError(f"Invalid step size (lr): {lr}")

        if self._tensor_constraint:
            grads = self._grads_list()
            if not grads:
                return None
            s_list = self._constraint.lmo_tensors(grads)
            for p, s in zip(self._params_list(), s_list):
                state = self.state[p]
                buf = state.get("direction_buffer")
                if buf is None:
                    buf = torch.zeros_like(s)
                buf.mul_(self.momentum).add_(s, alpha=1.0 - self.momentum)
                state["direction_buffer"] = buf
                if self._weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * self._weight_decay)
                p.data.add_(buf.to(dtype=p.dtype, device=p.device), alpha=-lr)
        else:
            grad = self._grads_to_vector()
            if grad.size == 0:
                return None
            s = self._constraint.lmo(grad)
            lmo_sign = float(getattr(self._constraint, "lmo_sign", -1.0))
            if lmo_sign < 0:
                s = -s
            if self._direction_vec is None or self._direction_vec.shape != s.shape:
                self._direction_vec = np.zeros_like(s, dtype=np.float32)
            self._direction_vec = self.momentum * self._direction_vec + (1.0 - self.momentum) * s
            x = self._params_to_vector()
            if self._weight_decay != 0.0:
                x = x * (1.0 - lr * self._weight_decay)
            x_new = x - lr * self._direction_vec
            self._set_params_from_vector(x_new)

        self._compute_constraint_info()
        self._t += 1
        return None
