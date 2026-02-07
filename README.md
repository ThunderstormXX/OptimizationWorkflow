# gossip-fw-bench

Benchmark framework for Frank-Wolfe, Gradient Descent, and distributed optimization.

## Quickstart

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest -q
```

## Architecture

- **Task**: Owns datasets, splits, loss, and metrics. Exposes dataloaders for train/val/test. Example: `tasks.mnist.MNISTSupervisedTask`.
- **Model**: Agent to be trained (e.g., `torch.nn.Module`).
- **Optimizer**: Update rule for model parameters. Referenced by class path in JSON (no registry).
- **Trainer**: Executes batch/epoch loops on a task using a model and optimizer (e.g., `trainers.supervised.SupervisedTrainer`).
- **Environment**: Generic wrapper for agent–task interaction (used by legacy step-based benchmarks).
- **Experiment**: Orchestrates tasks/models/optimizers and multiple runs, aggregates metrics, plots mean ± std.
- **Runner**: Thin CLI that reads JSON config and executes an experiment.

## Config-Based Experiments

Task directories contain JSON specs:

- `task.json` – task class + params
- `models.json` – list of model specs
- `optimizers.json` – list of optimizer specs

Experiment configs point at task directories and a trainer.

Example config: `configs/experiments/mnist_cnn.json`

```bash
python -m benchmarks.runner \
  --config configs/experiments/mnist_cnn.json
```

A convenience script is available:

```bash
bash scripts/bash/frankwolfe/run_mnist_cnn_fw.sh
```

This writes runs to `workflow/frankwolfe/exp_XXXX/`.

### Notes on MNIST splits

The default config requests 60k train, 10k val, 1k test. MNIST has 60k train + 10k test, so the task clamps validation size to keep test size intact. The effective split is recorded in each run summary.

## Legacy CLI

The original CLI (matrix mode, checks, gossip, animations) is preserved as:

```bash
python -m benchmarks.legacy_runner ...
```

## Project Structure

```
src/
├── core/                # Protocols and core types
├── tasks/               # Tasks and datasets (MNIST, logistic, quadratic)
├── models/              # Torch models and adapters
├── optim/               # Optimizers and constraints
├── trainers/            # Training loops
├── experiments/         # Experiments + config utilities
├── environments/        # Environments (single-process, gossip)
├── distributed/         # Topologies, strategies, communicators
└── benchmarks/          # Runner, legacy runner, workflow utilities
```

## Development

```bash
uv run ruff check .
uv run ruff format .
uv run mypy src
uv run pytest -q
```
