"""PyTorch neural network models for MNIST classification.

This module provides:
- MLP3: 3-layer fully connected network
- CNN: Convolutional neural network
- Factory function to build models by name
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

__all__ = ["MLP3", "CNN", "build_mnist_torch_model", "TORCH_AVAILABLE"]


def _check_torch() -> None:
    """Raise ImportError if torch is not available."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for MNIST models. Install with: pip install torch torchvision"
        )


class MLP3(nn.Module):
    """3-layer MLP for MNIST classification.

    Architecture:
    - Flatten: 28x28 -> 784
    - Linear(784 -> hidden) + ReLU
    - Linear(hidden -> hidden) + ReLU
    - Linear(hidden -> 10)

    Attributes:
        hidden: Hidden layer size.
    """

    def __init__(self, hidden: int = 256) -> None:
        """Initialize the MLP.

        Args:
            hidden: Hidden layer size. Default 256.
        """
        _check_torch()
        super().__init__()
        self.hidden = hidden
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 10)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 1, 28, 28) or (batch, 784).

        Returns:
            Logits of shape (batch, 10).
        """
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    """Convolutional neural network for MNIST classification.

    Architecture:
    - Conv2d(1, 32, 3, padding=1) + ReLU + MaxPool2d(2)
    - Conv2d(32, 64, 3, padding=1) + ReLU + MaxPool2d(2)
    - Flatten
    - Linear(64*7*7 -> 128) + ReLU
    - Linear(128 -> 10)
    """

    def __init__(self) -> None:
        """Initialize the CNN."""
        _check_torch()
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 1, 28, 28).

        Returns:
            Logits of shape (batch, 10).
        """
        # Conv block 1: 28x28 -> 14x14
        x = self.pool(self.relu(self.conv1(x)))
        # Conv block 2: 14x14 -> 7x7
        x = self.pool(self.relu(self.conv2(x)))
        # FC layers
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def build_mnist_torch_model(name: str, *, hidden: int = 256) -> nn.Module:
    """Factory function to build MNIST models by name.

    Args:
        name: Model name. One of "mlp3", "cnn".
        hidden: Hidden layer size for MLP3. Default 256.

    Returns:
        PyTorch model.

    Raises:
        ValueError: If name is unknown.
        ImportError: If torch is not available.
    """
    _check_torch()

    if name == "mlp3":
        return MLP3(hidden=hidden)
    elif name == "cnn":
        return CNN()
    else:
        raise ValueError(f"Unknown model name: {name}. Choose from: mlp3, cnn")
