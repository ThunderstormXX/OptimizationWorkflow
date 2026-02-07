from __future__ import annotations

import numpy as np
import pytest

from distributed.strategies import GradientTrackingStrategy, GossipNode
from distributed.topology import RingTopology
from distributed.communicator import SynchronousGossipCommunicator
from models.numpy_vector import NumpyVectorModel
from tasks.synthetic_quadratic import QuadraticGradComputer, QuadraticProblem, SyntheticQuadraticTask
from optim.gradient_descent import GradientDescentOptimizer, GDState


def _make_node(node_id: int) -> GossipNode:
    problem = QuadraticProblem(np.eye(2), np.zeros(2))
    task = SyntheticQuadraticTask(problem)
    model = NumpyVectorModel(np.array([1.0, -1.0]))
    optimizer = GradientDescentOptimizer(lr=0.1)
    return GossipNode(
        node_id=node_id,
        task=task,
        model=model,
        optimizer=optimizer,
        grad_computer=QuadraticGradComputer(),
        opt_state=GDState(t=0),
        rng=np.random.default_rng(0),
    )


def test_gradient_tracking_requires_reset() -> None:
    strategy = GradientTrackingStrategy(lr=0.1)
    communicator = SynchronousGossipCommunicator(topology=RingTopology(n=2))
    with pytest.raises(RuntimeError):
        strategy.step(nodes=[], communicator=communicator)


def test_gradient_tracking_requires_sync_communicator() -> None:
    strategy = GradientTrackingStrategy(lr=0.1)
    nodes = [_make_node(0), _make_node(1)]
    communicator = SynchronousGossipCommunicator(topology=RingTopology(n=2))
    strategy.reset(nodes=nodes, communicator=communicator)

    class DummyCommunicator:
        pass

    with pytest.raises(TypeError):
        strategy.step(nodes=nodes, communicator=DummyCommunicator())
