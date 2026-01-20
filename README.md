# gossip-fw-bench

Benchmark framework for Frank-Wolfe, Gradient Descent, and gossip-based distributed optimization.

Supports:
- **Single-process** and **distributed (gossip)** environments
- **Frank-Wolfe**, **Gradient Descent**, and **Projected GD** optimizers
- **Gradient Tracking** strategy for decentralized optimization
- **Quadratic** and **Logistic Regression** tasks
- **Heterogeneous data** distribution across nodes (IID, label skew)
- **Matrix mode** for running grid experiments
- **Checks mode** for convergence regression testing

## Installation / Quickstart

This project uses [uv](https://github.com/astral-sh/uv) as the task runner.

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest -q

# Type check
uv run mypy src
```

## Run a Single Experiment (Quadratic)

```bash
# Single-process with Frank-Wolfe
uv run python -m benchmarks.runner \
    --env single \
    --task quadratic \
    --optimizer fw \
    --steps 50 \
    --dim 10 \
    --cond 10 \
    --seed 0

# Gossip with Gradient Descent
uv run python -m benchmarks.runner \
    --env gossip \
    --task quadratic \
    --optimizer gd \
    --lr 0.1 \
    --steps 50 \
    --n-nodes 5 \
    --topology ring \
    --strategy local_then_gossip \
    --seed 0
```

## Run Matrix Mode (Grid of Experiments)

```bash
# Matrix mode varies optimizer, topology, strategy, schedule across seeds
uv run python -m benchmarks.runner \
    --mode matrix \
    --env gossip \
    --task quadratic \
    --steps 30 \
    --n-nodes 5 \
    --seeds "0,1,2" \
    --matrix small
```

## Run Checks Mode (Regression Gates)

```bash
# Run convergence checks and generate report
uv run python -m benchmarks.runner \
    --mode checks \
    --env gossip \
    --optimizer fw \
    --steps 30 \
    --n-nodes 5 \
    --topology complete
```

## Run Logistic Regression (IID + Label Skew)

```bash
# Single-process logistic regression
uv run python -m benchmarks.runner \
    --env single \
    --task logistic \
    --optimizer gd \
    --lr 0.5 \
    --steps 50 \
    --dim 10 \
    --n-samples 2000 \
    --batch-size 64 \
    --seed 0

# Gossip with heterogeneous data (label skew)
uv run python -m benchmarks.runner \
    --env gossip \
    --task logistic \
    --optimizer gd \
    --lr 0.1 \
    --steps 30 \
    --n-nodes 5 \
    --topology ring \
    --strategy gradient_tracking \
    --heterogeneity label_skew \
    --n-samples 2000 \
    --batch-size 64 \
    --seed 0
```

## Run Gradient Tracking

```bash
# Gradient tracking on quadratic task
uv run python -m benchmarks.runner \
    --env gossip \
    --task quadratic \
    --optimizer gd \
    --lr 0.05 \
    --steps 30 \
    --n-nodes 5 \
    --topology ring \
    --strategy gradient_tracking \
    --seed 0
```

## How to Add New Components

### Adding a New Optimizer

1. **Implement the Optimizer protocol** in `src/optim/your_optimizer.py`:

```python
from dataclasses import dataclass
from core.protocols import Optimizer, Task, Model, GradComputer
from core.types import StepResult

@dataclass
class MyOptimizerState:
    t: int = 0

class MyOptimizer:
    def __init__(self, *, lr: float) -> None:
        self.lr = lr

    def init_state(self, model: Model) -> MyOptimizerState:
        return MyOptimizerState(t=0)

    def step(self, *, task, model, batch, grad_computer, state, rng):
        # Compute gradient and update model
        grad = grad_computer.grad(task, model, batch)
        x = model.parameters_vector()
        model.set_parameters_vector(x - self.lr * grad)
        
        loss = task.loss(model, batch)
        metrics = task.metrics(model, batch)
        return MyOptimizerState(t=state.t + 1), StepResult(loss=loss, metrics=metrics)
```

2. **Register in `src/benchmarks/registry.py`**:

```python
def _my_optimizer_factory(config: dict[str, Any]) -> MyOptimizer:
    return MyOptimizer(lr=config.get("lr", 0.1))

register_optimizer("my_opt", _my_optimizer_factory)
```

3. **Use via CLI**: `--optimizer my_opt`

### Adding a New Gossip Strategy

1. **Implement the GossipStrategy protocol** in `src/distributed/strategies.py`:

```python
class MyStrategy:
    def step(self, *, nodes, communicator) -> dict[NodeId, StepResult]:
        # Your strategy logic here
        results = {}
        for node in nodes:
            # ... perform local step and/or gossip
            results[node.node_id] = StepResult(loss=..., metrics={})
        return results
```

2. **Register in `src/benchmarks/registry.py`**:

```python
def _my_strategy_factory() -> MyStrategy:
    return MyStrategy()

register_strategy("my_strategy", _my_strategy_factory)
```

3. **Use via CLI**: `--strategy my_strategy`

### Adding a New Task

1. **Implement the Task and GradComputer protocols** in `src/tasks/your_task.py`:

```python
class MyTask:
    def sample_batch(self, *, rng):
        # Return a batch of data
        ...

    def loss(self, model, batch) -> float:
        # Compute and return loss
        ...

    def metrics(self, model, batch) -> dict[str, float]:
        # Compute and return metrics
        ...

class MyGradComputer:
    def grad(self, task, model, batch) -> np.ndarray:
        # Compute and return gradient
        ...
```

2. **Update `src/benchmarks/runner.py`** to support the new task.

## Project Structure

```
src/
├── core/
│   ├── types.py          # Core types: ParamVector, StepResult, History, etc.
│   └── protocols.py      # Protocol definitions: Model, Task, Optimizer, etc.
├── models/
│   └── numpy_vector.py   # Simple numpy-based model
├── tasks/
│   ├── synthetic_quadratic.py  # Quadratic optimization task
│   └── logistic_regression.py  # Logistic regression task
├── optim/
│   ├── constraints.py    # L2Ball, Simplex constraints
│   ├── frank_wolfe.py    # Frank-Wolfe optimizer
│   └── gradient_descent.py  # GD and Projected GD
├── distributed/
│   ├── topology.py       # Ring, Complete topologies
│   ├── weights.py        # Metropolis-Hastings weights
│   ├── communicator.py   # Gossip communicator (single + multi-channel)
│   └── strategies.py     # LocalStepThenGossip, GossipThenLocal, GradientTracking
├── environments/
│   ├── base.py           # BaseEnvironment ABC
│   ├── single_process.py # Single-process environment
│   └── gossip.py         # Gossip-based distributed environment
└── benchmarks/
    ├── registry.py       # Optimizer/strategy registry
    ├── runner.py         # CLI experiment runner
    ├── workflow.py       # Experiment directory management
    ├── metrics.py        # Suboptimality, consensus error, etc.
    ├── checks.py         # Convergence checks
    └── report.py         # Report generation
```

## Development Commands

```bash
# Lint check
uv run ruff check .

# Format code
uv run ruff format .

# Type check
uv run mypy src

# Run tests
uv run pytest -q

# Run all checks
uv run ruff check . && uv run ruff format . && uv run mypy src && uv run pytest -q
```
