"""Solver and geometry defaults loaded from defaults.toml."""

from __future__ import annotations

import tomllib
from pathlib import Path

_DEFAULTS_PATH = Path(__file__).parent / "defaults.toml"

with open(_DEFAULTS_PATH, "rb") as _f:
    DEFAULTS = tomllib.load(_f)

# Picard iteration (nonlinear FEM)
PICARD_MAX_ITERATIONS: int = DEFAULTS["picard"]["max_iterations"]
PICARD_TOLERANCE: float = DEFAULTS["picard"]["tolerance"]
PICARD_RELAXATION: float = DEFAULTS["picard"]["relaxation"]

# Stepped-slot geometry
SLOT_OPENING_FRACTION: float = DEFAULTS["slots"]["opening_fraction"]

# Copper temperature coefficient (1/K, pure copper)
COPPER_TEMP_COEFF: float = 0.00393
