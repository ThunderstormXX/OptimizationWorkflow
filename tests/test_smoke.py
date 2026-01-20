"""Smoke tests to verify the project setup works correctly."""

from __future__ import annotations

from core import types


def test_import_core_types() -> None:
    """Verify that core.types can be imported successfully."""
    assert types is not None
    assert True
