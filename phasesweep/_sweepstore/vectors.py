"""Portable test vectors shipped with the package."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any


def load_vectors(name: str = "staleness_vectors") -> dict[str, Any]:
    """Load a packaged vector set by name (without the .json suffix)."""
    return json.loads((files("phasesweep._sweepstore") / "_vectors" / f"{name}.json").read_text())
