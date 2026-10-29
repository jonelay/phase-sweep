"""solve_field_fem argument guards that fire before any NGSolve import.

Kept out of test_fem_field.py, whose module-wide fem marker would drop
these from the fast tier.
"""

import pytest

from phasesweep.machines.geometry import inrunner
from phasesweep.solvers.fem_field import solve_field_fem

_KW = dict(r_outer=1.0, r_stator=0.70, r_magnet=0.64, r_rotor=0.30,
           n_slots=12, slot_depth=0.05)


def test_n_p_below_one_raises():
    with pytest.raises(ValueError, match="n_p must be >= 1, got 0"):
        solve_field_fem(inrunner(**_KW), 0, 1.2, 1.05, 1000.0)


def test_n_slots_override_with_opening_width_raises():
    geo = inrunner(**_KW, slot_opening_width=0.02)
    with pytest.raises(ValueError, match="n_slots=24 overrides geometry n_slots=12"):
        solve_field_fem(geo, 2, 1.2, 1.05, 1000.0, n_slots=24)
