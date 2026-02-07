from __future__ import annotations

import numpy as np
import pytest

from distributed.communicator import SynchronousGossipCommunicator
from distributed.topology import RingTopology


def test_gossip_multi_missing_or_extra_nodes() -> None:
    communicator = SynchronousGossipCommunicator(topology=RingTopology(n=2))

    payloads_missing = {0: {"x": np.zeros(2)}}
    with pytest.raises(ValueError, match="missing nodes"):
        communicator.gossip_multi(payloads_missing)

    payloads_extra = {
        0: {"x": np.zeros(2)},
        1: {"x": np.zeros(2)},
        2: {"x": np.zeros(2)},
    }
    with pytest.raises(ValueError, match="extra nodes"):
        communicator.gossip_multi(payloads_extra)


def test_gossip_multi_channel_mismatches() -> None:
    communicator = SynchronousGossipCommunicator(topology=RingTopology(n=2))

    payloads_keys = {
        0: {"x": np.zeros(2)},
        1: {"y": np.zeros(2)},
    }
    with pytest.raises(ValueError, match="Channel key mismatch"):
        communicator.gossip_multi(payloads_keys)

    payloads_empty = {0: {}, 1: {}}
    with pytest.raises(ValueError, match="at least one channel"):
        communicator.gossip_multi(payloads_empty)

    payloads_shape = {
        0: {"x": np.zeros(2)},
        1: {"x": np.zeros(3)},
    }
    with pytest.raises(ValueError, match="Shape mismatch"):
        communicator.gossip_multi(payloads_shape)
