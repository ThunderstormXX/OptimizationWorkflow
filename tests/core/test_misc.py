from __future__ import annotations

import core.logging as core_logging
import core.rng as core_rng


def test_core_logging_and_rng_modules_importable() -> None:
    assert isinstance(core_logging.__all__, list)
    assert isinstance(core_rng.__all__, list)
