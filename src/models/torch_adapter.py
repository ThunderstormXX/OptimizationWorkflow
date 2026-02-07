"""Adapter to wrap PyTorch models for use with the optimization framework.

This module provides TorchModelAdapter which wraps a torch.nn.Module and
provides the Model protocol interface (parameters_vector, set_parameters_vector).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.types import ParamVector

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

__all__ = ["TorchModelAdapter", "TORCH_AVAILABLE"]


def _check_torch() -> None:
    """Raise ImportError if torch is not available."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for TorchModelAdapter. Install with: pip install torch torchvision"
        )


class TorchModelAdapter:
    """Adapter wrapping a PyTorch model for the optimization framework.

    Provides:
    - parameters_vector() -> np.ndarray: Flatten all params to 1D array
    - set_parameters_vector(v): Load params from 1D array
    - forward(images) -> logits: Run forward pass
    - zero_grad(): Zero all gradients
    - train() / eval(): Set training mode

    The parameter order is stable: iterate module.parameters() once and cache
    shapes/slices for consistent flattening/unflattening.

    Attributes:
        module: The wrapped PyTorch module.
        device: Device where the model lives.
    """

    def __init__(self, module: nn.Module, device: str = "cpu") -> None:
        """Initialize the adapter.

        Args:
            module: PyTorch model to wrap.
            device: Device to use ("cpu" or "cuda").
        """
        _check_torch()
        self.module = module.to(device)
        self.device = device

        # Cache parameter metadata for stable ordering
        self._param_shapes: list[tuple[int, ...]] = []
        self._param_slices: list[tuple[int, int]] = []
        self._total_params = 0

        offset = 0
        for param in self.module.parameters():
            shape = tuple(param.shape)
            numel = param.numel()
            self._param_shapes.append(shape)
            self._param_slices.append((offset, offset + numel))
            offset += numel

        self._total_params = offset

    @property
    def dim(self) -> int:
        """Total number of parameters."""
        return self._total_params

    def parameters_vector(self) -> ParamVector:
        """Return all parameters as a flat numpy array (float64).

        Returns:
            1D numpy array of shape (dim,) containing all model parameters.
        """
        params_list: list[np.ndarray] = []
        for param in self.module.parameters():
            params_list.append(param.detach().cpu().numpy().flatten())
        return np.concatenate(params_list).astype(np.float64)

    def set_parameters_vector(self, v: ParamVector) -> None:
        """Load parameters from a flat numpy array.

        Args:
            v: 1D numpy array of shape (dim,) containing parameter values.

        Raises:
            ValueError: If v has wrong shape.
        """
        if v.shape != (self._total_params,):
            raise ValueError(f"Expected shape ({self._total_params},), got {v.shape}")

        with torch.no_grad():
            for i, param in enumerate(self.module.parameters()):
                start, end = self._param_slices[i]
                shape = self._param_shapes[i]
                param_data = v[start:end].reshape(shape)
                param.copy_(torch.from_numpy(param_data).to(self.device))

    def forward(self, images: Any) -> Any:
        """Run forward pass on images.

        Args:
            images: Input tensor of shape (batch, 1, 28, 28) or numpy array.

        Returns:
            Logits tensor of shape (batch, 10).
        """
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float().to(self.device)
        elif not isinstance(images, torch.Tensor):
            images = torch.tensor(images, dtype=torch.float32, device=self.device)
        else:
            images = images.to(self.device)

        return self.module(images)

    def zero_grad(self) -> None:
        """Zero all gradients."""
        self.module.zero_grad()

    def train(self) -> None:
        """Set model to training mode."""
        self.module.train()

    def eval(self) -> None:
        """Set model to evaluation mode."""
        self.module.eval()

    def get_grad_vector(self) -> ParamVector:
        """Return gradients as a flat numpy array.

        Call this after loss.backward() to get the gradient vector.

        Returns:
            1D numpy array of shape (dim,) containing all gradients.

        Raises:
            RuntimeError: If gradients are not computed (call backward first).
        """
        grads_list: list[np.ndarray] = []
        for param in self.module.parameters():
            if param.grad is None:
                raise RuntimeError("Gradient not computed. Call loss.backward() first.")
            grads_list.append(param.grad.detach().cpu().numpy().flatten())
        return np.concatenate(grads_list).astype(np.float64)
