"""Fast tests for MNIST functionality using tiny fixture.

These tests verify:
- Model architecture correctness (MLP3, CNN)
- TorchModelAdapter functionality
- MNIST task with tiny fixture (no network, deterministic)
- Training loop basics
- Runner integration

Note: Tests use tests/assets/mnist_tiny fixture. Run scripts/make_mnist_tiny.py
to regenerate if needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Skip all tests if torch is not available
torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")

# Path to the tiny MNIST fixture
MNIST_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "assets" / "mnist_tiny"


class TestTorchModels:
    """Tests for PyTorch model architectures."""

    def test_mlp3_forward(self) -> None:
        """MLP3 forward pass produces correct output shape."""
        from models.torch_nets import MLP3

        model = MLP3(hidden=64)
        x = torch.randn(4, 1, 28, 28)
        logits = model(x)

        assert logits.shape == (4, 10)
        assert torch.isfinite(logits).all()

    def test_cnn_forward(self) -> None:
        """CNN forward pass produces correct output shape."""
        from models.torch_nets import CNN

        model = CNN()
        x = torch.randn(4, 1, 28, 28)
        logits = model(x)

        assert logits.shape == (4, 10)
        assert torch.isfinite(logits).all()

    def test_build_mnist_torch_model_mlp(self) -> None:
        """Factory builds MLP3 correctly."""
        from models.torch_nets import MLP3, build_mnist_torch_model

        model = build_mnist_torch_model("mlp3", hidden=128)
        assert isinstance(model, MLP3)
        assert model.hidden == 128

    def test_build_mnist_torch_model_cnn(self) -> None:
        """Factory builds CNN correctly."""
        from models.torch_nets import CNN, build_mnist_torch_model

        model = build_mnist_torch_model("cnn")
        assert isinstance(model, CNN)

    def test_build_mnist_torch_model_invalid(self) -> None:
        """Factory raises on invalid name."""
        from models.torch_nets import build_mnist_torch_model

        with pytest.raises(ValueError, match="Unknown model"):
            build_mnist_torch_model("invalid_model")


class TestTorchAdapter:
    """Tests for TorchModelAdapter."""

    def test_parameters_vector_roundtrip(self) -> None:
        """Parameters can be extracted and set back."""
        from models.torch_adapter import TorchModelAdapter
        from models.torch_nets import MLP3

        torch_model = MLP3(hidden=32)
        adapter = TorchModelAdapter(torch_model, device="cpu")

        params = adapter.parameters_vector()
        assert params.dtype == np.float64
        assert params.ndim == 1
        assert len(params) == adapter.dim

        new_params = params + 0.1
        adapter.set_parameters_vector(new_params)

        params_after = adapter.parameters_vector()
        np.testing.assert_allclose(params_after, new_params, rtol=1e-5)

    def test_forward_pass(self) -> None:
        """Adapter forward pass works."""
        from models.torch_adapter import TorchModelAdapter
        from models.torch_nets import CNN

        torch_model = CNN()
        adapter = TorchModelAdapter(torch_model, device="cpu")

        images = torch.randn(2, 1, 28, 28)
        logits = adapter.forward(images)

        assert logits.shape == (2, 10)

    def test_grad_vector(self) -> None:
        """Gradient vector can be extracted after backward."""
        from models.torch_adapter import TorchModelAdapter
        from models.torch_nets import MLP3

        torch_model = MLP3(hidden=16)
        adapter = TorchModelAdapter(torch_model, device="cpu")

        images = torch.randn(2, 1, 28, 28)
        labels = torch.tensor([0, 5])

        adapter.train()
        adapter.zero_grad()
        logits = adapter.forward(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()

        grad = adapter.get_grad_vector()
        assert grad.dtype == np.float64
        assert grad.shape == (adapter.dim,)
        assert np.any(grad != 0)


class TestMNISTTinyFixture:
    """Tests for MNIST tiny fixture dataset."""

    @pytest.fixture(autouse=True)
    def check_fixture_exists(self) -> None:
        """Skip if fixture doesn't exist."""
        if not MNIST_FIXTURE_PATH.exists():
            pytest.skip(
                f"MNIST fixture not found at {MNIST_FIXTURE_PATH}. "
                "Run: python scripts/make_mnist_tiny.py"
            )

    def test_load_mnist_tiny(self) -> None:
        """Tiny fixture loads correctly."""
        from tasks.mnist import load_mnist_tiny

        train_ds, test_ds = load_mnist_tiny(MNIST_FIXTURE_PATH)

        assert len(train_ds) == 64
        assert len(test_ds) == 64

        img, label = train_ds[0]
        assert img.shape == (1, 28, 28)
        assert isinstance(label, int)
        assert 0 <= label <= 9

    def test_mnist_tiny_dataset_class(self) -> None:
        """MNISTTinyDataset works correctly."""
        from tasks.mnist import MNISTTinyDataset

        ds = MNISTTinyDataset(MNIST_FIXTURE_PATH, train=True)
        assert len(ds) == 64

        img, label = ds[0]
        assert img.shape == (1, 28, 28)
        assert torch.isfinite(img).all()

    def test_mnist_task_with_fixture(self) -> None:
        """MNIST task works with tiny fixture."""
        from tasks.mnist import MNISTClassificationTask, load_mnist_tiny

        train_ds, val_ds = load_mnist_tiny(MNIST_FIXTURE_PATH)

        task = MNISTClassificationTask(
            train_dataset=train_ds,
            val_dataset=val_ds,
            batch_size=8,
            device="cpu",
            eval_every=1,
        )

        rng = np.random.default_rng(0)
        images, labels = task.sample_batch(rng=rng)

        assert images.shape == (8, 1, 28, 28)
        assert labels.shape == (8,)

    def test_mnist_task_loss_with_fixture(self) -> None:
        """MNIST task computes loss with fixture."""
        from models.torch_adapter import TorchModelAdapter
        from models.torch_nets import MLP3
        from tasks.mnist import MNISTClassificationTask, load_mnist_tiny

        train_ds, val_ds = load_mnist_tiny(MNIST_FIXTURE_PATH)

        task = MNISTClassificationTask(
            train_dataset=train_ds,
            val_dataset=val_ds,
            batch_size=8,
            device="cpu",
        )

        torch_model = MLP3(hidden=32)
        model = TorchModelAdapter(torch_model, device="cpu")

        rng = np.random.default_rng(0)
        batch = task.sample_batch(rng=rng)
        loss = task.loss(model, batch)

        assert isinstance(loss, float)
        assert np.isfinite(loss)
        assert loss > 0


class TestMNISTTrainingLoop:
    """Tests for MNIST training integration."""

    @pytest.fixture(autouse=True)
    def check_fixture_exists(self) -> None:
        """Skip if fixture doesn't exist."""
        if not MNIST_FIXTURE_PATH.exists():
            pytest.skip(
                f"MNIST fixture not found at {MNIST_FIXTURE_PATH}. "
                "Run: python scripts/make_mnist_tiny.py"
            )

    def test_tiny_training_loop(self) -> None:
        """Run a tiny training loop (5 steps) with fixture."""
        from models.torch_adapter import TorchModelAdapter
        from models.torch_nets import MLP3
        from optim.adam import AdamOptimizer
        from tasks.mnist import MNISTClassificationTask, TorchGradComputer, load_mnist_tiny

        train_ds, val_ds = load_mnist_tiny(MNIST_FIXTURE_PATH)

        task = MNISTClassificationTask(
            train_dataset=train_ds,
            val_dataset=val_ds,
            batch_size=8,
            device="cpu",
            eval_every=1,
        )

        torch_model = MLP3(hidden=32)
        model = TorchModelAdapter(torch_model, device="cpu")
        grad_computer = TorchGradComputer()
        optimizer = AdamOptimizer(lr=0.01)
        state = optimizer.init_state(model)

        rng = np.random.default_rng(42)

        losses = []
        for _ in range(5):
            batch = task.sample_batch(rng=rng)
            state, result = optimizer.step(
                task=task,
                model=model,
                batch=batch,
                grad_computer=grad_computer,
                state=state,
                rng=rng,
            )
            losses.append(result.loss)

        assert all(np.isfinite(loss) for loss in losses)
        assert state.t == 5


class TestMNISTRunner:
    """Tests for MNIST via runner CLI."""

    @pytest.fixture(autouse=True)
    def check_fixture_exists(self) -> None:
        """Skip if fixture doesn't exist."""
        if not MNIST_FIXTURE_PATH.exists():
            pytest.skip(
                f"MNIST fixture not found at {MNIST_FIXTURE_PATH}. "
                "Run: python scripts/make_mnist_tiny.py"
            )

    def test_runner_mnist_single_with_fixture(self, tmp_path: Path) -> None:
        """Runner can run MNIST in single mode with tiny fixture."""
        from benchmarks.legacy_runner import main

        workflow_dir = tmp_path / "workflow"

        exit_code = main(
            [
                "--mode",
                "single",
                "--env",
                "single",
                "--task",
                "mnist",
                "--model",
                "mlp3",
                "--hidden",
                "32",
                "--optimizer",
                "adam",
                "--constraint",
                "none",
                "--lr",
                "0.01",
                "--steps",
                "5",
                "--batch-size",
                "8",
                "--eval-every",
                "1",
                "--mnist-fixture",
                str(MNIST_FIXTURE_PATH),
                "--workflow-dir",
                str(workflow_dir),
                "--exp-name",
                "mnist_fixture_smoke",
            ]
        )

        assert exit_code == 0

        exp_dir = list(workflow_dir.glob("exp_*"))[0]
        assert (exp_dir / "artifacts" / "summary.json").exists()
        assert (exp_dir / "artifacts" / "history.jsonl").exists()

    def test_runner_mnist_gossip_with_fixture(self, tmp_path: Path) -> None:
        """Runner can run MNIST in gossip mode with tiny fixture."""
        from benchmarks.legacy_runner import main

        workflow_dir = tmp_path / "workflow"

        exit_code = main(
            [
                "--mode",
                "single",
                "--env",
                "gossip",
                "--task",
                "mnist",
                "--model",
                "cnn",
                "--optimizer",
                "adam",
                "--constraint",
                "none",
                "--lr",
                "0.01",
                "--steps",
                "3",
                "--n-nodes",
                "2",
                "--topology",
                "ring",
                "--strategy",
                "local_then_gossip",
                "--batch-size",
                "8",
                "--heterogeneity",
                "iid",
                "--eval-every",
                "1",
                "--mnist-fixture",
                str(MNIST_FIXTURE_PATH),
                "--workflow-dir",
                str(workflow_dir),
                "--exp-name",
                "mnist_gossip_fixture_smoke",
            ]
        )

        assert exit_code == 0

        exp_dir = list(workflow_dir.glob("exp_*"))[0]
        summary_path = exp_dir / "artifacts" / "summary.json"
        assert summary_path.exists()

        import json

        with summary_path.open() as f:
            summary = json.load(f)

        assert summary["task"] == "mnist"
        assert summary["model"] == "cnn"
        assert "final_mean_accuracy" in summary
        assert "final_consensus_error" in summary

    def test_runner_mnist_with_trace(self, tmp_path: Path) -> None:
        """Runner creates trace.jsonl for MNIST."""
        from benchmarks.legacy_runner import main

        workflow_dir = tmp_path / "workflow"

        exit_code = main(
            [
                "--mode",
                "single",
                "--env",
                "single",
                "--task",
                "mnist",
                "--model",
                "mlp3",
                "--hidden",
                "16",
                "--optimizer",
                "adam",
                "--constraint",
                "none",
                "--steps",
                "3",
                "--batch-size",
                "8",
                "--eval-every",
                "1",
                "--mnist-fixture",
                str(MNIST_FIXTURE_PATH),
                "--workflow-dir",
                str(workflow_dir),
                "--animate",
                "--animate-format",
                "gif",
            ]
        )

        assert exit_code == 0

        exp_dir = list(workflow_dir.glob("exp_*"))[0]
        trace_path = exp_dir / "artifacts" / "trace.jsonl"
        assert trace_path.exists()

        anim_path = exp_dir / "artifacts" / "animation.gif"
        assert anim_path.exists()
