"""Models module for the benchmark framework.

This package contains model implementations that satisfy the Model protocol.

Available models:
- NumpyVectorModel: Simple parameter vector (for quadratic/logistic tasks)
- TorchModelAdapter: Wraps PyTorch models (for MNIST task)
- MLP3, CNN: PyTorch models for MNIST (requires torch)
"""

from __future__ import annotations

from models.numpy_vector import NumpyVectorModel
from models.torch_adapter import TORCH_AVAILABLE, TorchModelAdapter
from models.torch_nets import CNN, MLP3, build_mnist_torch_model

__all__ = [
    "NumpyVectorModel",
    "TorchModelAdapter",
    "TORCH_AVAILABLE",
    "MLP3",
    "CNN",
    "build_mnist_torch_model",
]
