"""MNIST classification task for PyTorch models.

This module provides:
- Dataset loading with normalization
- Tiny fixture dataset for tests (no download needed)
- FakeData support for development (deprecated for normal usage)
- Data splitting for gossip environments
- MNISTClassificationTask implementing the Task protocol
- TorchGradComputer for gradient computation
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.types import ParamVector

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
    from torchvision import datasets, transforms  # type: ignore[import-untyped]

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch optional at import time
    TORCH_AVAILABLE = False
    # Provide stub types for type checking
    Dataset = object  # type: ignore[misc, assignment]
    DataLoader = object  # type: ignore[misc, assignment]

__all__ = [
    "load_mnist",
    "load_mnist_or_fake",
    "load_mnist_tiny",
    "MNISTTinyDataset",
    "split_mnist_across_nodes",
    "MNISTClassificationTask",
    "MNISTSupervisedTask",
    "TorchGradComputer",
    "TORCH_AVAILABLE",
]


def _check_torch() -> None:
    """Raise ImportError if torch is not available."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for MNIST task. Install with: pip install torch torchvision"
        )


# MNIST normalization constants
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


# =============================================================================
# Tiny Fixture Dataset (for tests, no download needed)
# =============================================================================


class MNISTTinyDataset(Dataset):  # type: ignore[type-arg]
    """Tiny MNIST dataset loaded from numpy files.

    This dataset loads pre-saved MNIST samples from .npy files,
    used for deterministic testing without network access.

    The fixture should contain:
    - train_images.npy (uint8, shape [N, 28, 28])
    - train_labels.npy (int64, shape [N])
    - test_images.npy (uint8, shape [N, 28, 28])
    - test_labels.npy (int64, shape [N])
    """

    def __init__(self, root: Path, train: bool = True) -> None:
        """Initialize the dataset.

        Args:
            root: Directory containing the .npy files.
            train: If True, load training set. If False, load test set.

        Raises:
            FileNotFoundError: If the required files don't exist.
        """
        _check_torch()

        prefix = "train" if train else "test"
        images_path = root / f"{prefix}_images.npy"
        labels_path = root / f"{prefix}_labels.npy"

        if not images_path.exists():
            raise FileNotFoundError(
                f"MNIST tiny fixture not found at {images_path}. "
                f"Run: python scripts/make_mnist_tiny.py"
            )
        if not labels_path.exists():
            raise FileNotFoundError(
                f"MNIST tiny fixture not found at {labels_path}. "
                f"Run: python scripts/make_mnist_tiny.py"
            )

        # Load data
        self.images = np.load(images_path)  # uint8, [N, 28, 28]
        self.labels = np.load(labels_path)  # int64, [N]
        self.targets = torch.from_numpy(self.labels)  # For compatibility

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        """Get a sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image_tensor, label).
        """
        # Convert to float tensor and normalize
        image = self.images[idx].astype(np.float32) / 255.0
        image = (image - MNIST_MEAN) / MNIST_STD
        image_tensor = torch.from_numpy(image).unsqueeze(0)  # [1, 28, 28]
        label = int(self.labels[idx])
        return image_tensor, label


def load_mnist_tiny(root: Path) -> tuple[Dataset, Dataset]:  # type: ignore[type-arg]
    """Load tiny MNIST fixture for testing.

    Args:
        root: Directory containing the fixture files.

    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    _check_torch()
    train_ds = MNISTTinyDataset(root, train=True)
    test_ds = MNISTTinyDataset(root, train=False)
    return train_ds, test_ds


# =============================================================================
# Real MNIST Dataset Loading
# =============================================================================


def load_mnist(
    root: Path,
    train: bool,
    download: bool = False,
) -> Dataset:  # type: ignore[type-arg]
    """Load MNIST dataset with standard normalization.

    Args:
        root: Directory to store/load data.
        train: If True, load training set. If False, load test set.
        download: If True, download data if not present.

    Returns:
        MNIST dataset with transforms applied.

    Raises:
        RuntimeError: If MNIST data not found and download=False.
    """
    _check_torch()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )

    # Check if data exists when not downloading
    mnist_dir = Path(root) / "MNIST" / "raw"
    if not download and not mnist_dir.exists():
        raise RuntimeError(
            f"MNIST data not found in {root}. "
            f"Pass --download to download it, or set --data-root to an existing cache."
        )

    return datasets.MNIST(  # type: ignore[no-any-return]
        root=str(root),
        train=train,
        download=download,
        transform=transform,
    )


def load_mnist_or_fake(
    *,
    use_fake_data: bool,
    n_samples: int,
    seed: int,
    root: Path | None = None,
    train: bool = True,
    download: bool = False,
) -> Dataset:  # type: ignore[type-arg]
    """Load MNIST or FakeData for testing.

    Args:
        use_fake_data: If True, use FakeData instead of real MNIST.
        n_samples: Number of samples (for FakeData or subset of MNIST).
        seed: Random seed for reproducibility.
        root: Directory for real MNIST data.
        train: If True, load training set.
        download: If True, download MNIST if not present.

    Returns:
        Dataset (MNIST or FakeData).
    """
    _check_torch()

    if use_fake_data:
        # Use FakeData for testing
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        return datasets.FakeData(  # type: ignore[no-any-return]
            size=n_samples,
            image_size=(1, 28, 28),
            num_classes=10,
            transform=transform,
            random_offset=seed,
        )
    else:
        if root is None:
            raise ValueError("root must be provided for real MNIST data")
        ds = load_mnist(root, train=train, download=download)
        # Optionally subset
        if n_samples < len(ds):  # type: ignore[arg-type]
            generator = torch.Generator().manual_seed(seed)
            indices = torch.randperm(len(ds), generator=generator)[:n_samples].tolist()  # type: ignore[arg-type]
            return Subset(ds, indices)
        return ds


def split_mnist_across_nodes(
    dataset: Dataset,  # type: ignore[type-arg]
    n_nodes: int,
    heterogeneity: str,
    seed: int,
) -> list[Subset]:  # type: ignore[type-arg]
    """Split MNIST dataset across nodes.

    Args:
        dataset: Full MNIST dataset.
        n_nodes: Number of nodes.
        heterogeneity: Split type:
            - "iid": Random uniform split
            - "label_skew": Each node gets biased digit distribution
        seed: Random seed.

    Returns:
        List of Subset, one per node.
    """
    _check_torch()

    n = len(dataset)  # type: ignore[arg-type]
    generator = torch.Generator().manual_seed(seed)

    if heterogeneity == "iid":
        # Random split
        lengths = [n // n_nodes] * n_nodes
        # Distribute remainder
        for i in range(n % n_nodes):
            lengths[i] += 1
        return list(random_split(dataset, lengths, generator=generator))

    elif heterogeneity == "label_skew":
        # Group indices by label
        label_indices: dict[int, list[int]] = {i: [] for i in range(10)}

        # Get labels (handle both Dataset and Subset)
        for idx in range(n):
            if hasattr(dataset, "targets"):
                # Direct MNIST dataset
                label = int(dataset.targets[idx])
            else:
                # Subset or other
                _, label = dataset[idx]
                label = int(label)
            label_indices[label].append(idx)

        # Shuffle within each label
        rng = np.random.default_rng(seed)
        for label in label_indices:
            rng.shuffle(label_indices[label])

        # Assign labels to nodes with bias
        # Node i primarily gets digits (2*i) % 10 and (2*i + 1) % 10
        node_indices: list[list[int]] = [[] for _ in range(n_nodes)]
        samples_per_node = n // n_nodes

        for node_id in range(n_nodes):
            # Primary digits for this node
            primary_digits = [(2 * node_id + d) % 10 for d in range(2)]
            other_digits = [d for d in range(10) if d not in primary_digits]

            # 70% from primary digits, 30% from others
            n_primary = int(0.7 * samples_per_node)
            n_other = samples_per_node - n_primary

            # Collect from primary digits
            for i, digit in enumerate(primary_digits):
                n_from_digit = n_primary // len(primary_digits)
                if i < n_primary % len(primary_digits):
                    n_from_digit += 1
                indices = label_indices[digit][:n_from_digit]
                label_indices[digit] = label_indices[digit][n_from_digit:]
                node_indices[node_id].extend(indices)

            # Collect from other digits
            for i, digit in enumerate(other_digits):
                n_from_digit = n_other // len(other_digits)
                if i < n_other % len(other_digits):
                    n_from_digit += 1
                indices = label_indices[digit][:n_from_digit]
                label_indices[digit] = label_indices[digit][n_from_digit:]
                node_indices[node_id].extend(indices)

            # Shuffle node's data
            rng.shuffle(node_indices[node_id])

        return [Subset(dataset, indices) for indices in node_indices]

    else:
        raise ValueError(f"Unknown heterogeneity type: {heterogeneity}")


# =============================================================================
# MNIST Task
# =============================================================================


@dataclass
class MNISTClassificationTask:
    """MNIST classification task.

    Implements the Task protocol for MNIST digit classification.

    Attributes:
        train_dataset: Training dataset.
        val_dataset: Validation dataset (shared across nodes for fair comparison).
        batch_size: Batch size for training.
        device: Device for computation.
        eval_every: Evaluate on validation set every N steps.
        eval_batches: Number of batches for validation (to keep eval cheap).
    """

    train_dataset: Dataset  # type: ignore[type-arg]
    val_dataset: Dataset  # type: ignore[type-arg]
    batch_size: int = 128
    device: str = "cpu"
    eval_every: int = 20
    eval_batches: int = 50
    _train_loader: Any = field(init=False, repr=False)
    _train_iter: Iterator[tuple[Any, Any]] = field(init=False, repr=False)
    _val_loader: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize data loaders."""
        _check_torch()
        self._train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        self._train_iter = iter(self._train_loader)
        self._val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    @property
    def n_train(self) -> int:
        """Number of training samples."""
        return len(self.train_dataset)  # type: ignore[arg-type]

    @property
    def n_val(self) -> int:
        """Number of validation samples."""
        return len(self.val_dataset)  # type: ignore[arg-type]

    def sample_batch(self, *, rng: np.random.Generator) -> tuple[Any, Any]:
        """Sample a mini-batch from training data.

        Args:
            rng: Random number generator (unused, DataLoader handles shuffling).

        Returns:
            Tuple of (images, labels) tensors on device.
        """
        try:
            images, labels = next(self._train_iter)
        except StopIteration:
            # Reset iterator
            self._train_iter = iter(self._train_loader)
            images, labels = next(self._train_iter)

        return images.to(self.device), labels.to(self.device)

    def loss(self, model: Any, batch: tuple[Any, Any] | None) -> float:
        """Compute cross-entropy loss.

        Args:
            model: TorchModelAdapter wrapping the neural network.
            batch: Tuple of (images, labels) or None.

        Returns:
            Loss value.
        """
        if batch is None:
            # Use a fresh batch
            images, labels = self.sample_batch(rng=np.random.default_rng())
        else:
            images, labels = batch

        model.eval()
        with torch.no_grad():
            logits = model.forward(images)
            loss = F.cross_entropy(logits, labels)
        return float(loss.item())

    def metrics(
        self,
        model: Any,
        batch: tuple[Any, Any] | None,
        t: int = 0,
    ) -> Mapping[str, float]:
        """Compute metrics including accuracy.

        Args:
            model: TorchModelAdapter wrapping the neural network.
            batch: Tuple of (images, labels) or None for training batch.
            t: Current step (for deciding whether to compute val metrics).

        Returns:
            Dict with metrics:
            - train_loss: Loss on training batch
            - train_accuracy: Accuracy on training batch
            - val_loss: Loss on validation (if t % eval_every == 0)
            - val_accuracy: Accuracy on validation (if t % eval_every == 0)
        """
        if batch is None:
            images, labels = self.sample_batch(rng=np.random.default_rng())
        else:
            images, labels = batch

        model.eval()
        with torch.no_grad():
            # Training batch metrics
            logits = model.forward(images)
            train_loss = F.cross_entropy(logits, labels).item()
            preds = logits.argmax(dim=1)
            train_acc = (preds == labels).float().mean().item()

        result: dict[str, float] = {
            "train_loss": train_loss,
            "train_accuracy": train_acc,
        }

        # Validation metrics (periodically)
        if t % self.eval_every == 0:
            val_loss, val_acc = self._compute_val_metrics(model)
            result["val_loss"] = val_loss
            result["val_accuracy"] = val_acc

        return result

    def _compute_val_metrics(self, model: Any) -> tuple[float, float]:
        """Compute validation loss and accuracy.

        Uses at most eval_batches batches for efficiency.

        Args:
            model: TorchModelAdapter.

        Returns:
            Tuple of (val_loss, val_accuracy).
        """
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for i, (images, labels) in enumerate(self._val_loader):
                if i >= self.eval_batches:
                    break

                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = model.forward(images)
                loss = F.cross_entropy(logits, labels, reduction="sum")
                preds = logits.argmax(dim=1)

                total_loss += loss.item()
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        if total_samples == 0:
            return 0.0, 0.0

        return total_loss / total_samples, total_correct / total_samples


# =============================================================================
# MNIST Supervised Task (dataloaders for Trainer)
# =============================================================================


@dataclass
class MNISTSupervisedTask:
    """MNIST task with train/val/test dataloaders for supervised training."""

    data_root: str = ".data"
    download: bool = False
    train_size: int | None = None
    val_size: int | None = None
    test_size: int | None = None
    batch_size: int = 128
    device: str = "cpu"
    seed: int = 0
    val_from: str = "train"
    test_from: str = "test"
    num_workers: int = 0
    pin_memory: bool | None = None
    drop_last: bool = False
    use_fake_data: bool = False
    fake_train_size: int | None = None
    fake_test_size: int | None = None
    mnist_fixture: str | None = None
    verbose: bool = False

    train_loader: Any = field(init=False, repr=False)
    val_loader: Any = field(init=False, repr=False)
    test_loader: Any = field(init=False, repr=False)
    split_report: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        _check_torch()
        self.device = self._resolve_device(self.device)
        if self.pin_memory is None:
            self.pin_memory = self.device.startswith("cuda")
        if self.val_from not in {"train", "test"}:
            raise ValueError(f"val_from must be 'train' or 'test', got {self.val_from}")
        if self.test_from not in {"train", "test"}:
            raise ValueError(f"test_from must be 'train' or 'test', got {self.test_from}")

        if self.verbose:
            print(
                "[MNIST] loading datasets "
                f"(use_fake_data={self.use_fake_data}, fixture={self.mnist_fixture}, "
                f"data_root={self.data_root}, download={self.download})",
                flush=True,
            )
        train_full, test_full = self._load_datasets()
        train_set, val_set, test_set = self._split_datasets(train_full, test_full)

        if self.verbose:
            print(
                "[MNIST] split sizes "
                f"train={self.split_report['train']} "
                f"val={self.split_report['val']} "
                f"test={self.split_report['test']} "
                f"(val_from={self.val_from}, test_from={self.test_from})",
                flush=True,
            )

        self.train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            pin_memory=bool(self.pin_memory),
        )
        self.val_loader = DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=bool(self.pin_memory),
        )
        self.test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=bool(self.pin_memory),
        )

    def loss_fn(self, logits: Any, labels: Any) -> Any:
        return F.cross_entropy(logits, labels)

    def metrics_fn(self, logits: Any, labels: Any) -> dict[str, float]:
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        return {"accuracy": float(acc)}

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_datasets(self) -> tuple[Dataset, Dataset]:  # type: ignore[type-arg]
        if self.mnist_fixture:
            return load_mnist_tiny(Path(self.mnist_fixture))

        if self.use_fake_data:
            train_size = self.fake_train_size
            test_size = self.fake_test_size
            if train_size is None:
                train_size = max(self.train_size or 0, (self.train_size or 0) + (self.val_size or 0))
                if train_size == 0:
                    train_size = 60000
            if test_size is None:
                test_size = max(self.test_size or 0, (self.val_size or 0) + (self.test_size or 0))
                if test_size == 0:
                    test_size = 10000

            train_ds = load_mnist_or_fake(
                use_fake_data=True,
                n_samples=int(train_size),
                seed=self.seed,
                root=None,
                train=True,
                download=False,
            )
            test_ds = load_mnist_or_fake(
                use_fake_data=True,
                n_samples=int(test_size),
                seed=self.seed + 1,
                root=None,
                train=False,
                download=False,
            )
            return train_ds, test_ds

        train_ds = load_mnist(Path(self.data_root), train=True, download=self.download)
        test_ds = load_mnist(Path(self.data_root), train=False, download=self.download)
        return train_ds, test_ds

    def _split_datasets(
        self,
        train_full: Dataset,  # type: ignore[type-arg]
        test_full: Dataset,  # type: ignore[type-arg]
    ) -> tuple[Dataset, Dataset, Dataset]:  # type: ignore[type-arg]
        train_size = self.train_size if self.train_size is not None else len(train_full)  # type: ignore[arg-type]
        val_size = self.val_size or 0
        test_size = self.test_size or 0

        train_size = min(int(train_size), len(train_full))  # type: ignore[arg-type]

        train_val_from_train = self.val_from == "train"
        test_from_train = self.test_from == "train"

        val_from_test = self.val_from == "test"
        test_from_test = self.test_from == "test"

        train_val_size = val_size if train_val_from_train else 0
        train_test_size = test_size if test_from_train else 0

        train_remaining = len(train_full) - train_size  # type: ignore[arg-type]
        train_val_size = min(train_val_size, max(train_remaining, 0))
        train_remaining -= train_val_size
        train_test_size = min(train_test_size, max(train_remaining, 0))

        test_val_size = val_size if val_from_test else 0
        test_test_size = test_size if test_from_test else 0

        test_remaining = len(test_full)  # type: ignore[arg-type]
        test_test_size = min(test_test_size, test_remaining)
        test_remaining -= test_test_size
        test_val_size = min(test_val_size, max(test_remaining, 0))

        self.split_report = {
            "train": train_size,
            "val": train_val_size + test_val_size,
            "test": train_test_size + test_test_size,
        }

        train_indices = self._split_indices(
            len(train_full),  # type: ignore[arg-type]
            [train_size, train_val_size, train_test_size],
            self.seed,
        )
        train_set = Subset(train_full, train_indices[0])

        val_parts: list[Dataset] = []
        test_parts: list[Dataset] = []

        if train_val_size > 0:
            val_parts.append(Subset(train_full, train_indices[1]))
        if train_test_size > 0:
            test_parts.append(Subset(train_full, train_indices[2]))

        if test_val_size > 0 or test_test_size > 0:
            test_indices = self._split_indices(
                len(test_full),  # type: ignore[arg-type]
                [test_test_size, test_val_size],
                self.seed + 1,
            )
            if test_test_size > 0:
                test_parts.append(Subset(test_full, test_indices[0]))
            if test_val_size > 0:
                val_parts.append(Subset(test_full, test_indices[1]))

        if not val_parts:
            val_parts.append(Subset(train_full, []))
        if not test_parts:
            test_parts.append(Subset(test_full, []))

        val_set = self._concat_subsets(val_parts)
        test_set = self._concat_subsets(test_parts)
        return train_set, val_set, test_set

    @staticmethod
    def _split_indices(n: int, sizes: list[int], seed: int) -> list[list[int]]:
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=generator).tolist()
        indices: list[list[int]] = []
        start = 0
        for size in sizes:
            end = start + size
            indices.append(perm[start:end])
            start = end
        return indices

    @staticmethod
    def _concat_subsets(parts: list[Dataset]) -> Dataset:  # type: ignore[type-arg]
        if len(parts) == 1:
            return parts[0]
        return ConcatDataset(parts)  # pragma: no cover


# =============================================================================
# Gradient Computer
# =============================================================================


class TorchGradComputer:
    """Gradient computer for PyTorch models.

    Computes gradients via backpropagation.
    """

    def grad(
        self,
        task: Any,
        model: Any,
        batch: tuple[Any, Any] | None,
    ) -> ParamVector:
        """Compute gradient of loss w.r.t. model parameters.

        Args:
            task: MNISTClassificationTask.
            model: TorchModelAdapter.
            batch: Tuple of (images, labels) or None.

        Returns:
            Gradient vector (numpy float64).
        """
        _check_torch()

        if batch is None:
            images, labels = task.sample_batch(rng=np.random.default_rng())
        else:
            images, labels = batch

        # Forward pass
        model.train()
        model.zero_grad()
        logits = model.forward(images)
        loss = F.cross_entropy(logits, labels)

        # Backward pass
        loss.backward()  # type: ignore[no-untyped-call]

        # Get gradient vector
        return model.get_grad_vector()  # type: ignore[no-any-return]
