"""Tests for gossip distributed optimization components.

This module tests:
- Topologies: RingTopology, CompleteTopology
- Weights: Metropolis-Hastings row-stochastic weights
- Communicator: SynchronousGossipCommunicator
"""

from __future__ import annotations

import numpy as np
import pytest

from distributed.communicator import SynchronousGossipCommunicator
from distributed.topology import CompleteTopology, RingTopology
from distributed.weights import metropolis_hastings_rows, row_sums_close_to_one

# =============================================================================
# Tests for RingTopology
# =============================================================================


class TestRingTopology:
    """Tests for ring topology."""

    def test_num_nodes(self) -> None:
        """num_nodes should return n."""
        topo = RingTopology(n=5)
        assert topo.num_nodes() == 5

    def test_neighbors_n4(self) -> None:
        """For n=4, each node should have left and right neighbors."""
        topo = RingTopology(n=4)

        # Node 0: neighbors are 3 (left) and 1 (right)
        assert set(topo.neighbors(0)) == {3, 1}

        # Node 1: neighbors are 0 (left) and 2 (right)
        assert set(topo.neighbors(1)) == {0, 2}

        # Node 2: neighbors are 1 (left) and 3 (right)
        assert set(topo.neighbors(2)) == {1, 3}

        # Node 3: neighbors are 2 (left) and 0 (right)
        assert set(topo.neighbors(3)) == {2, 0}

    def test_neighbors_n2(self) -> None:
        """For n=2, each node has exactly one neighbor (the other)."""
        topo = RingTopology(n=2)

        assert list(topo.neighbors(0)) == [1]
        assert list(topo.neighbors(1)) == [0]

    def test_neighbors_n3(self) -> None:
        """For n=3, each node has two neighbors."""
        topo = RingTopology(n=3)

        assert set(topo.neighbors(0)) == {2, 1}
        assert set(topo.neighbors(1)) == {0, 2}
        assert set(topo.neighbors(2)) == {1, 0}

    def test_n1_raises(self) -> None:
        """n=1 should raise ValueError."""
        with pytest.raises(ValueError, match="n >= 2"):
            RingTopology(n=1)

    def test_neighbors_not_include_self(self) -> None:
        """Neighbors should not include the node itself."""
        topo = RingTopology(n=5)
        for i in range(5):
            assert i not in topo.neighbors(i)

    def test_undirected_symmetry(self) -> None:
        """If j in neighbors(i), then i in neighbors(j)."""
        topo = RingTopology(n=6)
        for i in range(6):
            for j in topo.neighbors(i):
                assert i in topo.neighbors(j)

    def test_neighbors_returns_copy(self) -> None:
        """neighbors() should return a copy (not internal state)."""
        topo = RingTopology(n=4)
        neighbors1 = topo.neighbors(0)
        neighbors2 = topo.neighbors(0)

        # Should be equal but not the same object
        assert neighbors1 == neighbors2
        assert neighbors1 is not neighbors2


# =============================================================================
# Tests for CompleteTopology
# =============================================================================


class TestCompleteTopology:
    """Tests for complete (fully-connected) topology."""

    def test_num_nodes(self) -> None:
        """num_nodes should return n."""
        topo = CompleteTopology(n=5)
        assert topo.num_nodes() == 5

    def test_neighbors_n4(self) -> None:
        """For n=4, each node should have all other nodes as neighbors."""
        topo = CompleteTopology(n=4)

        assert set(topo.neighbors(0)) == {1, 2, 3}
        assert set(topo.neighbors(1)) == {0, 2, 3}
        assert set(topo.neighbors(2)) == {0, 1, 3}
        assert set(topo.neighbors(3)) == {0, 1, 2}

    def test_n1_raises(self) -> None:
        """n=1 should raise ValueError."""
        with pytest.raises(ValueError, match="n >= 2"):
            CompleteTopology(n=1)

    def test_neighbors_not_include_self(self) -> None:
        """Neighbors should not include the node itself."""
        topo = CompleteTopology(n=5)
        for i in range(5):
            assert i not in topo.neighbors(i)

    def test_neighbor_count(self) -> None:
        """Each node should have n-1 neighbors."""
        topo = CompleteTopology(n=6)
        for i in range(6):
            assert len(topo.neighbors(i)) == 5

    def test_undirected_symmetry(self) -> None:
        """If j in neighbors(i), then i in neighbors(j)."""
        topo = CompleteTopology(n=5)
        for i in range(5):
            for j in topo.neighbors(i):
                assert i in topo.neighbors(j)


# =============================================================================
# Tests for Metropolis-Hastings Weights
# =============================================================================


class TestMetropolisHastingsWeights:
    """Tests for Metropolis-Hastings weight computation."""

    def test_row_stochastic_ring(self) -> None:
        """Weights should be row-stochastic (rows sum to 1)."""
        topo = RingTopology(n=5)
        rows = metropolis_hastings_rows(topo)

        for i in range(5):
            row_sum = sum(rows[i].values())
            assert row_sum == pytest.approx(1.0, abs=1e-12)

    def test_row_stochastic_complete(self) -> None:
        """Weights should be row-stochastic for complete graph."""
        topo = CompleteTopology(n=4)
        rows = metropolis_hastings_rows(topo)

        for i in range(4):
            row_sum = sum(rows[i].values())
            assert row_sum == pytest.approx(1.0, abs=1e-12)

    def test_nonnegative_weights(self) -> None:
        """All weights should be non-negative."""
        topo = RingTopology(n=5)
        rows = metropolis_hastings_rows(topo)

        for i in range(5):
            for j, w in rows[i].items():
                assert w >= -1e-15, f"Negative weight w[{i}][{j}] = {w}"

    def test_includes_self_weight(self) -> None:
        """Each row should include self-weight w_ii."""
        topo = RingTopology(n=5)
        rows = metropolis_hastings_rows(topo)

        for i in range(5):
            assert i in rows[i], f"Self-weight missing for node {i}"

    def test_row_sums_close_to_one_helper(self) -> None:
        """row_sums_close_to_one helper should work correctly."""
        topo = RingTopology(n=5)
        rows = metropolis_hastings_rows(topo)

        assert row_sums_close_to_one(rows, tol=1e-12)

        # Test with bad rows
        bad_rows = {0: {0: 0.5, 1: 0.3}}  # Sums to 0.8, not 1.0
        assert not row_sums_close_to_one(bad_rows, tol=1e-12)

    def test_symmetric_weights_ring(self) -> None:
        """For undirected graphs, w_ij should equal w_ji."""
        topo = RingTopology(n=5)
        rows = metropolis_hastings_rows(topo)

        for i in range(5):
            for j in rows[i]:
                if i != j:
                    assert rows[i][j] == pytest.approx(rows[j][i], abs=1e-15)

    def test_symmetric_weights_complete(self) -> None:
        """For complete graph, w_ij should equal w_ji."""
        topo = CompleteTopology(n=4)
        rows = metropolis_hastings_rows(topo)

        for i in range(4):
            for j in rows[i]:
                if i != j:
                    assert rows[i][j] == pytest.approx(rows[j][i], abs=1e-15)


# =============================================================================
# Tests for SynchronousGossipCommunicator
# =============================================================================


class TestSynchronousGossipCommunicator:
    """Tests for synchronous gossip communicator."""

    def test_gossip_basic(self) -> None:
        """Basic gossip operation should work."""
        topo = RingTopology(n=3)
        comm = SynchronousGossipCommunicator(topology=topo)

        payloads = {
            0: np.array([1.0, 0.0]),
            1: np.array([0.0, 1.0]),
            2: np.array([0.0, 0.0]),
        }

        mixed = comm.gossip(payloads)

        assert len(mixed) == 3
        for i in range(3):
            assert mixed[i].shape == (2,)

    def test_average_preservation_ring(self) -> None:
        """Global average should be preserved under gossip (doubly-stochastic)."""
        topo = RingTopology(n=6)
        comm = SynchronousGossipCommunicator(topology=topo)

        # Create random payloads
        rng = np.random.default_rng(42)
        payloads = {i: rng.standard_normal(3) for i in range(6)}

        # Compute mean before
        mean_before = np.mean([payloads[i] for i in range(6)], axis=0)

        # Apply gossip
        mixed = comm.gossip(payloads)

        # Compute mean after
        mean_after = np.mean([mixed[i] for i in range(6)], axis=0)

        # Means should be equal (doubly-stochastic preserves average)
        np.testing.assert_allclose(mean_before, mean_after, atol=1e-12)

    def test_average_preservation_complete(self) -> None:
        """Global average preserved for complete graph."""
        topo = CompleteTopology(n=5)
        comm = SynchronousGossipCommunicator(topology=topo)

        rng = np.random.default_rng(123)
        payloads = {i: rng.standard_normal(4) for i in range(5)}

        mean_before = np.mean([payloads[i] for i in range(5)], axis=0)
        mixed = comm.gossip(payloads)
        mean_after = np.mean([mixed[i] for i in range(5)], axis=0)

        np.testing.assert_allclose(mean_before, mean_after, atol=1e-12)

    def test_determinism(self) -> None:
        """Calling gossip twice with same payloads should give identical results."""
        topo = RingTopology(n=4)
        comm = SynchronousGossipCommunicator(topology=topo)

        rng = np.random.default_rng(42)
        payloads = {i: rng.standard_normal(3) for i in range(4)}

        # First call
        mixed1 = comm.gossip(payloads)

        # Second call with same payloads
        mixed2 = comm.gossip(payloads)

        # Results should be identical
        for i in range(4):
            np.testing.assert_allclose(mixed1[i], mixed2[i])

    def test_missing_node_raises(self) -> None:
        """Missing nodes should raise ValueError."""
        topo = RingTopology(n=4)
        comm = SynchronousGossipCommunicator(topology=topo)

        # Missing node 2
        payloads = {
            0: np.array([1.0]),
            1: np.array([1.0]),
            3: np.array([1.0]),
        }

        with pytest.raises(ValueError, match="missing nodes"):
            comm.gossip(payloads)

    def test_extra_node_raises(self) -> None:
        """Extra nodes should raise ValueError."""
        topo = RingTopology(n=3)
        comm = SynchronousGossipCommunicator(topology=topo)

        # Extra node 3
        payloads = {
            0: np.array([1.0]),
            1: np.array([1.0]),
            2: np.array([1.0]),
            3: np.array([1.0]),
        }

        with pytest.raises(ValueError, match="extra nodes"):
            comm.gossip(payloads)

    def test_shape_mismatch_raises(self) -> None:
        """Mismatched shapes should raise ValueError."""
        topo = RingTopology(n=3)
        comm = SynchronousGossipCommunicator(topology=topo)

        payloads = {
            0: np.array([1.0, 2.0, 3.0]),  # shape (3,)
            1: np.array([1.0, 2.0, 3.0, 4.0]),  # shape (4,) - mismatch!
            2: np.array([1.0, 2.0, 3.0]),
        }

        with pytest.raises(ValueError, match="Shape mismatch"):
            comm.gossip(payloads)

    def test_no_aliasing(self) -> None:
        """Returned vectors should be independent copies."""
        topo = RingTopology(n=3)
        comm = SynchronousGossipCommunicator(topology=topo)

        payloads = {i: np.array([float(i)]) for i in range(3)}
        mixed = comm.gossip(payloads)

        # Mutate returned vector
        mixed[0][0] = 999.0

        # Call gossip again - should not be affected
        mixed2 = comm.gossip(payloads)
        assert mixed2[0][0] != 999.0

    def test_convergence_to_average(self) -> None:
        """Repeated gossip should converge to global average."""
        topo = RingTopology(n=5)
        comm = SynchronousGossipCommunicator(topology=topo)

        rng = np.random.default_rng(42)
        payloads = {i: rng.standard_normal(3) for i in range(5)}

        # Compute global average
        global_avg = np.mean([payloads[i] for i in range(5)], axis=0)

        # Apply many rounds of gossip
        current = payloads
        for _ in range(100):
            current = comm.gossip(current)

        # All nodes should converge to the global average
        for i in range(5):
            np.testing.assert_allclose(current[i], global_avg, atol=1e-6)

    def test_custom_weights(self) -> None:
        """Communicator should accept custom weights."""
        topo = RingTopology(n=3)

        # Custom uniform weights
        custom_rows = {
            0: {0: 1 / 3, 1: 1 / 3, 2: 1 / 3},
            1: {0: 1 / 3, 1: 1 / 3, 2: 1 / 3},
            2: {0: 1 / 3, 1: 1 / 3, 2: 1 / 3},
        }

        comm = SynchronousGossipCommunicator(topology=topo, rows=custom_rows)

        payloads = {
            0: np.array([3.0]),
            1: np.array([6.0]),
            2: np.array([9.0]),
        }

        mixed = comm.gossip(payloads)

        # With uniform weights, all nodes should get the average
        expected = np.array([6.0])  # (3 + 6 + 9) / 3
        for i in range(3):
            np.testing.assert_allclose(mixed[i], expected)


# =============================================================================
# Tests for Multi-Channel Gossip (gossip_multi)
# =============================================================================


class TestMultiChannelGossip:
    """Tests for multi-channel gossip communication."""

    def test_multi_payload_mean_preserved_per_key(self) -> None:
        """Global mean should be preserved for each channel independently."""
        topo = CompleteTopology(n=4)
        comm = SynchronousGossipCommunicator(topology=topo)

        # Create multi-channel payloads
        rng = np.random.default_rng(42)
        payloads = {
            i: {
                "x": rng.standard_normal(5),
                "y": rng.standard_normal(5),
            }
            for i in range(4)
        }

        # Compute initial means per channel
        initial_mean_x = np.mean([payloads[i]["x"] for i in range(4)], axis=0)
        initial_mean_y = np.mean([payloads[i]["y"] for i in range(4)], axis=0)

        # Perform gossip
        mixed = comm.gossip_multi(payloads)

        # Compute final means per channel
        final_mean_x = np.mean([mixed[i]["x"] for i in range(4)], axis=0)
        final_mean_y = np.mean([mixed[i]["y"] for i in range(4)], axis=0)

        # Means should be preserved (doubly-stochastic weights)
        np.testing.assert_allclose(initial_mean_x, final_mean_x, atol=1e-10)
        np.testing.assert_allclose(initial_mean_y, final_mean_y, atol=1e-10)

    def test_multi_payload_mismatched_keys_raises(self) -> None:
        """Mismatched channel keys should raise ValueError."""
        topo = RingTopology(n=3)
        comm = SynchronousGossipCommunicator(topology=topo)

        payloads = {
            0: {"x": np.array([1.0]), "y": np.array([2.0])},
            1: {"x": np.array([1.0]), "y": np.array([2.0])},
            2: {"x": np.array([1.0]), "z": np.array([2.0])},  # Different key!
        }

        with pytest.raises(ValueError, match="Channel key mismatch"):
            comm.gossip_multi(payloads)

    def test_multi_payload_mismatched_shapes_raises(self) -> None:
        """Mismatched shapes within a channel should raise ValueError."""
        topo = RingTopology(n=3)
        comm = SynchronousGossipCommunicator(topology=topo)

        payloads = {
            0: {"x": np.array([1.0, 2.0]), "y": np.array([3.0])},
            1: {"x": np.array([1.0, 2.0]), "y": np.array([3.0])},
            2: {"x": np.array([1.0]), "y": np.array([3.0])},  # Different shape for x!
        }

        with pytest.raises(ValueError, match="Shape mismatch"):
            comm.gossip_multi(payloads)

    def test_multi_payload_deterministic(self) -> None:
        """Multi-channel gossip should be deterministic."""
        topo = RingTopology(n=4)
        comm = SynchronousGossipCommunicator(topology=topo)

        payloads = {i: {"x": np.array([float(i)]), "y": np.array([float(i * 2)])} for i in range(4)}

        mixed1 = comm.gossip_multi(payloads)
        mixed2 = comm.gossip_multi(payloads)

        for i in range(4):
            np.testing.assert_array_equal(mixed1[i]["x"], mixed2[i]["x"])
            np.testing.assert_array_equal(mixed1[i]["y"], mixed2[i]["y"])

    def test_multi_payload_empty_keys_raises(self) -> None:
        """Empty channel keys should raise ValueError."""
        topo = RingTopology(n=2)
        comm = SynchronousGossipCommunicator(topology=topo)

        payloads = {
            0: {},
            1: {},
        }

        with pytest.raises(ValueError, match="at least one channel"):
            comm.gossip_multi(payloads)
