"""Adam optimizer (unconstrained, legacy protocol-based)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.protocols import GradComputer, Model, Task
from core.types import ParamVector, StepResult

__all__ = ["AdamState", "AdamOptimizer"]


@dataclass
class AdamState:
    t: int = 0
    m: ParamVector = field(default_factory=lambda: np.array([]))
    v: ParamVector = field(default_factory=lambda: np.array([]))


class AdamOptimizer:
    """Adam optimizer without projection."""

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        if not (0 <= beta1 < 1):
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
        if not (0 <= beta2 < 1):
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}")
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def init_state(self, model: Model[Any, Any]) -> AdamState:
        x = model.parameters_vector()
        return AdamState(
            t=0,
            m=np.zeros_like(x),
            v=np.zeros_like(x),
        )

    def step(
        self,
        *,
        task: Task[Any, Any],
        model: Model[Any, Any],
        batch: Any,
        grad_computer: GradComputer[Any, Any],
        state: AdamState,
        rng: np.random.Generator,
    ) -> tuple[AdamState, StepResult]:
        x = model.parameters_vector()
        grad = grad_computer.grad(task, model, batch)

        m = self.beta1 * state.m + (1 - self.beta1) * grad
        v = self.beta2 * state.v + (1 - self.beta2) * (grad**2)
        t = state.t + 1
        m_hat = m / (1 - self.beta1**t)
        v_hat = v / (1 - self.beta2**t)
        x_new = x - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        model.set_parameters_vector(x_new)
        loss = task.loss(model, batch)
        metrics = task.metrics(model, batch)

        new_state = AdamState(t=t, m=m, v=v)
        return new_state, StepResult(loss=loss, metrics=dict(metrics))
