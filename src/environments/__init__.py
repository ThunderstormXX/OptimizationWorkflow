"""Environments module for the benchmark framework.

This package contains environment implementations that orchestrate
optimization loops for various settings (single-process, distributed, etc.).
"""

from __future__ import annotations

from environments.base import BaseEnvironment
from environments.gossip import GossipEnvironment, consensus_error
from environments.single_process import SingleProcessEnvironment

__all__ = [
    "BaseEnvironment",
    "SingleProcessEnvironment",
    "GossipEnvironment",
    "consensus_error",
]
